"""
pinn_scene_compare.py
=====================
Side-by-side scene overlay: Numerical DRIFT risk field (left) vs PINN
surrogate (right), drawn on the real orthophoto background with actual
vehicle bounding boxes.

Matches the visual style and figsave logic of drift_dataset_visualization.py:
  - Top-level config variables (no argparse)
  - Same pixel-space rendering (bboxVis, _vis_scale_down, _cfg_X_vis/Y_vis)
  - Same viewport logic (VIEW_X/Y clamp around ego)
  - Frame naming: {i}.png  (not frame_0000.png)
  - Single figure reused per frame (fig.clf() each iteration)
  - Progress bar via progress.Bar

Layout per frame
----------------
  ┌──────────────────────────┬──────────────────────────┐
  │  Numerical DRIFT  (PDE)  │  PINN Surrogate  (θ)     │
  │  + orthophoto + vehicles │  + orthophoto + vehicles │
  └──────────────────────────┴──────────────────────────┘

Usage
-----
  Edit DATASET_DIR, RECORDING_ID, N_t below, then run:
      python pinn_scene_compare.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import scienceplots          # noqa: F401
from scipy.ndimage import gaussian_filter as _gf
from progress.bar import Bar

DREAM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DREAM_ROOT)

import torch
from tracks_import import read_from_csv
from config import Config as cfg
from pde_solver import (create_vehicle as drift_create_vehicle,
                        compute_total_Q, compute_velocity_field,
                        compute_diffusion_field)
from Integration.drift_interface import DRIFTInterface
from pinn_risk_field import (Normalizer, FlatSampleCache, RiskFieldNet,
                              PINNTrainer)

# ===========================================================================
# PARAMETERS — edit these
# ===========================================================================

DATASET_DIR    = r"c:\field_modeling\data\rounD"
DATASET = "rounD"
RECORDING_ID   = "01"          # zero-padded recording number, e.g. "00", "18"
EGO_TRACK_ID   = 15          # None = auto-select longest car track
EGO_MIN_FRAMES = 60            # minimum track length for ego candidate

MODEL_PATH     = r"c:\DREAM_final\pinn_rounD_new.pt"          # None = auto-find pinn_rounD_all.pt etc.

dt             = 0.1           # simulation step [s]
N_t            = 200           # number of steps to render
WARMUP_S       = 3.0           # DRIFT warm-up duration [s]

DRIFT_CELL_M   = 2.0           # DRIFT grid cell size [m]
SCENE_MARGIN   = 60.0          # margin beyond track bbox [m]

# Risk visualisation (same as drift_dataset_visualization.py)
RISK_ALPHA        = 0.24
RISK_CMAP         = "jet"
RISK_LEVELS       = 80
RISK_VMAX         = 2.0
RISK_MIN_VIS      = 0.08
RISK_SMOOTH_SIGMA = 2.0
RISK_ALPHA_GAMMA  = 0.8

# Viewport half-extents around ego [m]
VIEW_X = 50.0
VIEW_Y = 28.0

VIS_SCALE_DOWN    = None       # None = auto-detect
VIS_SCALE_PRESETS = {"exid": 6.0, "ind": 12.0, "round": 10.0}

rec      = f"{int(RECORDING_ID):02d}"
save_dir = os.path.join(DREAM_ROOT, f"figsave_PINN_compare_{rec}_DATASET-{DATASET}")
os.makedirs(save_dir, exist_ok=True)

# ===========================================================================
# DEFAULT CFG CENTRE — PINN training used this to shift world coords
# ===========================================================================
_DEFAULT_CFG_X_MID = (-150.0 + 255.2) / 2.0     #  52.6
_DEFAULT_CFG_Y_MID = (-225.2 + -45.3) / 2.0     # -135.25

# Module-level vars set during initialization
_ortho_px_m     = 1.0
_vis_scale_down = 1.0
_cfg_X_vis      = None
_cfg_Y_vis      = None
_ox             = 0.0    # world → PINN coordinate offset x
_oy             = 0.0    # world → PINN coordinate offset y
_track_x_all    = None
_track_y_all    = None


# ===========================================================================
# HELPERS
# ===========================================================================

def _infer_vis_scale(tracks, bg_img, dataset_dir, manual=None):
    if manual is not None and float(manual) > 0.0:
        return float(manual)
    ds_key = os.path.basename(os.path.normpath(dataset_dir)).lower()
    if ds_key in VIS_SCALE_PRESETS:
        return float(VIS_SCALE_PRESETS[ds_key])
    if bg_img is None:
        return 1.0
    try:
        x_vis = np.concatenate([np.asarray(t["xCenterVis"]).ravel() for t in tracks])
        y_vis = np.concatenate([np.asarray(t["yCenterVis"]).ravel() for t in tracks])
        h, w  = bg_img.shape[:2]
        s = max(1.0,
                float(np.nanmax(x_vis)) / max(1.0, 0.98 * w),
                float(np.nanmax(y_vis)) / max(1.0, 0.98 * h))
        return round(s * 2.0) / 2.0 if s > 2.0 else float(s)
    except Exception:
        return 1.0


def build_drift_vehicles(frame_idx, ego_track_id, tracks, tracks_meta, class_map):
    """
    Return (surrounding_vehicles, ego_vehicle, surr_tids) as DRIFT dicts for frame_idx.
    Ego is the dataset track with trackId == ego_track_id.
    surr_tids: list of dataset trackIds in the same order as surrounding_vehicles.
    """
    surrounding, ego_dict = [], None
    surr_tids = []   # dataset trackId for each surrounding vehicle (same order)
    vid = 1
    for tm in tracks_meta:
        tid = tm["trackId"]
        if not (tm["initialFrame"] <= frame_idx <= tm["finalFrame"]):
            continue
        tr  = tracks[tid]
        fi  = frame_idx - tm["initialFrame"]
        cls = class_map.get(tid, "car")

        x     = float(tr["xCenter"][fi])
        y     = float(tr["yCenter"][fi])
        psi   = math.radians(float(tr["heading"][fi]))
        lon_v = float(tr["lonVelocity"][fi])
        vx_g  = lon_v * math.cos(psi)
        vy_g  = lon_v * math.sin(psi)
        lon_a = float(tr["lonAcceleration"][fi]) if "lonAcceleration" in tr else 0.0

        # Skip nearly-stationary non-vehicles (pedestrians, parked)
        if abs(lon_v) < 0.3 and cls not in ("car", "truck", "van"):
            continue

        vclass = "truck" if cls in ("truck", "van") else "car"
        v = drift_create_vehicle(vid=vid, x=x, y=y, vx=vx_g, vy=vy_g, vclass=vclass)
        v["heading"] = psi
        v["a"]       = lon_a

        if tid == ego_track_id:
            ego_dict = v
        else:
            surrounding.append(v)
            surr_tids.append(tid)
            vid += 1

    return surrounding, ego_dict, surr_tids


# ===========================================================================
# VISUALIZATION
# ===========================================================================

def draw_frame_pinn_compare(i, frame_idx, tracks, tracks_meta, class_map,
                             bg_img,
                             risk_field_num, risk_at_ego_num,
                             risk_field_pinn, risk_at_ego_pinn,
                             agent_info=None):
    """
    Two-panel pixel-space rendering — matches draw_frame_drift_overlay() style.
      Panel 0: Numerical solver
      Panel 1: PINN Surrogate

    Uses global _ortho_px_m, _vis_scale_down, _cfg_X_vis, _cfg_Y_vis,
    _track_x_all, _track_y_all.
    """
    fig = plt.gcf()
    fig.clf()

    # Shared vmax derived from numerical field (same scale on both panels)
    vmax = RISK_VMAX
    if risk_field_num is not None:
        R_sm0 = _gf(risk_field_num, sigma=RISK_SMOOTH_SIGMA)
        nz = R_sm0[R_sm0 > RISK_MIN_VIS]
        if nz.size > 50:
            vmax = float(np.percentile(nz, 95))
        vmax = max(vmax, RISK_MIN_VIS + 1e-3)

    panels = [
        (risk_field_num,  risk_at_ego_num,
         r"Numerical solver"),
        (risk_field_pinn, risk_at_ego_pinn,
         r"PINN Surrogate  $\hat{\mathcal{R}}_\theta$"),
    ]

    # ── Ego pixel position for viewport ──────────────────────────────────────
    if EGO_TRACK_ID is not None:
        ego_tm = tracks_meta[EGO_TRACK_ID]
        ego_tr = tracks[EGO_TRACK_ID]
        if ego_tm["initialFrame"] <= frame_idx <= ego_tm["finalFrame"]:
            fi_ego = frame_idx - ego_tm["initialFrame"]
            ex_px = float(ego_tr["xCenterVis"][fi_ego]) / _vis_scale_down
            ey_px = float(ego_tr["yCenterVis"][fi_ego]) / _vis_scale_down
        else:
            ex_px = float(np.mean(_track_x_all) / (_ortho_px_m * _vis_scale_down))
            ey_px = float(np.mean(-_track_y_all) / (_ortho_px_m * _vis_scale_down))
    else:
        ex_px = float(np.mean(_track_x_all) / (_ortho_px_m * _vis_scale_down))
        ey_px = float(np.mean(-_track_y_all) / (_ortho_px_m * _vis_scale_down))

    view_x_px  = VIEW_X / (_ortho_px_m * _vis_scale_down)
    view_y_px  = VIEW_Y / (_ortho_px_m * _vis_scale_down)
    x0_vp, x1_vp       = ex_px - view_x_px, ex_px + view_x_px
    y_top_vp, y_bot_vp = ey_px - view_y_px, ey_px + view_y_px

    if bg_img is not None:
        h_bg, w_bg = bg_img.shape[:2]
        if x0_vp < 0:
            x1_vp -= x0_vp;  x0_vp = 0.0
        if x1_vp > (w_bg - 1):
            x0_vp -= (x1_vp - (w_bg - 1));  x1_vp = float(w_bg - 1)
        if y_top_vp < 0:
            y_bot_vp -= y_top_vp;  y_top_vp = 0.0
        if y_bot_vp > (h_bg - 1):
            y_top_vp -= (y_bot_vp - (h_bg - 1));  y_bot_vp = float(h_bg - 1)
        x0_vp    = max(0.0, x0_vp);  x1_vp    = min(float(w_bg - 1), x1_vp)
        y_top_vp = max(0.0, y_top_vp); y_bot_vp = min(float(h_bg - 1), y_bot_vp)

    # ── Draw each panel ───────────────────────────────────────────────────────
    for panel_idx, (rf, rae, panel_title) in enumerate(panels):
        ax = fig.add_subplot(1, 2, panel_idx + 1)
        ax.cla()

        # 1) Orthophoto background
        if bg_img is not None:
            ax.imshow(bg_img, origin="upper", zorder=0)
        else:
            ax.set_facecolor("#111111")

        # 2) Risk overlay in pixel space
        if rf is not None and _cfg_X_vis is not None and _cfg_Y_vis is not None:
            R_sm = _gf(rf, sigma=RISK_SMOOTH_SIGMA)
            Rn   = ((np.clip(R_sm, RISK_MIN_VIS, vmax) - RISK_MIN_VIS)
                    / max(vmax - RISK_MIN_VIS, 1e-9))
            Rn   = np.power(np.clip(Rn, 0.0, 1.0), RISK_ALPHA_GAMMA)
            R_masked = np.ma.masked_less_equal(Rn, 0.0)
            if np.ma.count(R_masked) > 0:
                ax.contourf(_cfg_X_vis, _cfg_Y_vis, R_masked,
                            levels=np.linspace(0.02, 1.0, RISK_LEVELS),
                            cmap=RISK_CMAP, alpha=RISK_ALPHA,
                            zorder=2, antialiased=True)

        # 3) Vehicles — colour by inclusion status on PINN panel
        # On the numerical panel all agents show normally.
        # On the PINN panel: excluded agents (beyond perception range) are grey
        # with a dashed border so the reader can see what was filtered out.
        _included = agent_info["included"] if agent_info else None
        for tm in tracks_meta:
            tid = tm["trackId"]
            if not (tm["initialFrame"] <= frame_idx <= tm["finalFrame"]):
                continue
            tr   = tracks[tid]
            fi   = frame_idx - tm["initialFrame"]
            cls_ = class_map.get(tid, "car")
            is_ego = (tid == EGO_TRACK_ID)

            # On PINN panel (panel_idx==1) dim out excluded agents
            is_excluded = (panel_idx == 1 and _included is not None
                           and tid not in _included and not is_ego)
            if is_excluded:
                fc = "#888888"; ec = "#555555"; lw = 0.5; av = 0.35; z = 3
                ls = "--"
            else:
                fc = "#F4511E" if is_ego else (
                     "#FF8C00" if cls_ in ("truck", "van") else "#AED6F1")
                ec  = "red" if is_ego else "black"
                lw  = 1.0   if is_ego else 0.5
                av  = 0.95  if is_ego else 0.82
                z   = 5     if is_ego else 4
                ls  = "-"

            if tr.get("bboxVis") is not None:
                bbox = np.asarray(tr["bboxVis"][fi], dtype=float) / _vis_scale_down
                poly = plt.Polygon(bbox, closed=True, facecolor=fc, edgecolor=ec,
                                   linewidth=lw, linestyle=ls,
                                   alpha=av, zorder=z)
                ax.add_patch(poly)
            else:
                cx = float(tr["xCenterVis"][fi]) / _vis_scale_down
                cy = float(tr["yCenterVis"][fi]) / _vis_scale_down
                circ = plt.Circle((cx, cy),
                                  radius=max(1.4, 2.4 / _vis_scale_down),
                                  facecolor=fc, edgecolor=ec,
                                  linewidth=lw, linestyle=ls,
                                  alpha=av, zorder=z)
                ax.add_patch(circ)

        # 4) Viewport
        ax.set_xlim(x0_vp, x1_vp)
        ax.set_ylim(y_bot_vp, y_top_vp)   # pixel Y increases downward
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # 5) Risk badge (bottom-right) + agent count badge (bottom-left, PINN panel only)
        rc = ("red"    if rae > 1.5 else
              "orange" if rae > 0.5 else "lime")
        ax.text(0.985, 0.035, f"R={rae:.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                color=rc, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.55))

        if panel_idx == 1 and agent_info is not None:
            n_in  = agent_info["n_in"]
            n_tot = agent_info["n_tot"]
            pr    = agent_info["perc_range"]
            pr_str = f"  ≤{pr:.0f}m" if pr is not None else "  (all)"
            ac    = "lime" if n_in == n_tot else "orange"
            ax.text(0.015, 0.035, f"Agents: {n_in}/{n_tot}{pr_str}",
                    transform=ax.transAxes, ha="left", va="bottom",
                    color=ac, fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.55))

        ax.set_title(f"{panel_title}  |  t={i * dt:.1f} s  frame={frame_idx}",
                     fontsize=9, fontweight="bold")

        # 6) Colorbar
        _sm = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=RISK_MIN_VIS, vmax=vmax),
            cmap=plt.colormaps[RISK_CMAP])
        _sm.set_array([])
        cbar = fig.colorbar(_sm, ax=ax, fraction=0.018, pad=0.005)
        if panel_idx == 0:
            cbar.set_label(f"Risk  (vmax={vmax:.2f})", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    plt.savefig(os.path.join(save_dir, f"{i}.png"), dpi=150,
                bbox_inches="tight")


# ===========================================================================
# INITIALIZATION
# ===========================================================================

print("=" * 70)
print(f"PINN vs Numerical DRIFT  |  {RECORDING_ID}  |  N_t={N_t}  dt={dt}")
print("=" * 70)

# ── Load dataset ─────────────────────────────────────────────────────────────
tracks_file      = os.path.join(DATASET_DIR, f"{rec}_tracks.csv")
tracks_meta_file = os.path.join(DATASET_DIR, f"{rec}_tracksMeta.csv")
rec_meta_file    = os.path.join(DATASET_DIR, f"{rec}_recordingMeta.csv")

print(f"Loading recording {rec} from {DATASET_DIR} ...")
tracks, tracks_meta, recording_meta = read_from_csv(
    tracks_file, tracks_meta_file, rec_meta_file,
    include_px_coordinates=True)

_ortho_px_m = float(recording_meta["orthoPxToMeter"])
frame_rate   = float(recording_meta["frameRate"])
frame_stride = max(1, round(dt / (1.0 / frame_rate)))   # e.g. 25 Hz → stride 2 for dt=0.08

class_map    = {tm["trackId"]: tm["class"] for tm in tracks_meta}
_track_x_all = np.concatenate([t["xCenter"] for t in tracks])
_track_y_all = np.concatenate([t["yCenter"] for t in tracks])

print(f"  Tracks: {len(tracks)}  |  frameRate={frame_rate} Hz  "
      f"frame_stride={frame_stride}  (dt={dt} s)")

# ── Background image ──────────────────────────────────────────────────────────
bg_path = os.path.join(DATASET_DIR, f"{rec}_background.png")
bg_img  = None
img_h, img_w = 0, 0
if os.path.exists(bg_path):
    _raw   = cv2.imread(bg_path)
    bg_img = cv2.cvtColor(_raw, cv2.COLOR_BGR2RGB)
    img_h, img_w = bg_img.shape[:2]
    print(f"  Background: {img_w}x{img_h} px")
else:
    print(f"  [WARN] Background not found at {bg_path}")

_vis_scale_down = _infer_vis_scale(tracks, bg_img, DATASET_DIR, VIS_SCALE_DOWN)
print(f"  ortho={_ortho_px_m:.6f} m/px  vis_scale_down={_vis_scale_down:.2f}")

# ── Expand DRIFT grid to full scene (same as drift_dataset_visualization.py) ─
cfg.x_min = float(np.min(_track_x_all)) - SCENE_MARGIN
cfg.x_max = float(np.max(_track_x_all)) + SCENE_MARGIN
cfg.y_min = float(np.min(_track_y_all)) - SCENE_MARGIN
cfg.y_max = float(np.max(_track_y_all)) + SCENE_MARGIN
cfg.nx    = int((cfg.x_max - cfg.x_min) / DRIFT_CELL_M) + 2
cfg.ny    = int((cfg.y_max - cfg.y_min) / DRIFT_CELL_M) + 2
cfg.dx    = (cfg.x_max - cfg.x_min) / (cfg.nx - 1)
cfg.dy    = (cfg.y_max - cfg.y_min) / (cfg.ny - 1)
cfg.x     = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
cfg.y     = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
cfg.X, cfg.Y = np.meshgrid(cfg.x, cfg.y)   # world coordinates

_cfg_X_vis = cfg.X / (_ortho_px_m * _vis_scale_down)    # pixel space
_cfg_Y_vis = -cfg.Y / (_ortho_px_m * _vis_scale_down)
print(f"[DRIFT Grid] x=[{cfg.x_min:.0f},{cfg.x_max:.0f}]  "
      f"y=[{cfg.y_min:.0f},{cfg.y_max:.0f}]  "
      f"({cfg.nx}×{cfg.ny}={cfg.nx * cfg.ny // 1000}k cells)")

# ── PINN coordinate offset ────────────────────────────────────────────────────
# ExiDLoader training used: x_pinn = x_world - ox, where ox = median(x) - cfg_default_centre
# At inference we apply the same shift so the PINN input is in-distribution.
_ox = float(np.median(_track_x_all)) - _DEFAULT_CFG_X_MID
_oy = float(np.median(_track_y_all)) - _DEFAULT_CFG_Y_MID
# Grid in PINN coordinate space (for predict_field_from_arrays)
_X_pinn = cfg.X - _ox
_Y_pinn = cfg.Y - _oy
print(f"[PINN offset] ox={_ox:.1f}  oy={_oy:.1f}")

# ── Ego selection ─────────────────────────────────────────────────────────────
if EGO_TRACK_ID is None:
    _best_tid, _best_len = None, 0
    for tm in tracks_meta:
        if class_map.get(tm["trackId"], "") in ("car", "van"):
            if tm["numFrames"] > _best_len:
                _best_len = tm["numFrames"]
                _best_tid = tm["trackId"]
    EGO_TRACK_ID = _best_tid

ego_meta  = tracks_meta[EGO_TRACK_ID]
ego_track = tracks[EGO_TRACK_ID]
ego_fi0   = ego_meta["initialFrame"]
print(f"  Ego trackId={EGO_TRACK_ID}  class={class_map.get(EGO_TRACK_ID)}  "
      f"frames={ego_meta['numFrames']}")

# ── Load PINN model ───────────────────────────────────────────────────────────
print("\nLoading PINN model ...")
model_path = MODEL_PATH
if model_path is None:
    ds_name = os.path.basename(os.path.normpath(DATASET_DIR))
    for cand in [f"pinn_{ds_name}_all.pt",
                 f"pinn_{ds_name}_{rec}.pt",
                 "pinn_risk_field.pt"]:
        fp = os.path.join(DREAM_ROOT, cand)
        if os.path.isfile(fp):
            model_path = fp
            break
if model_path is None or not os.path.isfile(model_path):
    raise FileNotFoundError(
        "No PINN model found. Train first:\n"
        "  python pinn_risk_field.py --dataset inD --recording all")

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt   = torch.load(model_path, map_location=device, weights_only=False)

norm = Normalizer.__new__(Normalizer)
norm.ranges       = ckpt["norm_ranges"]
norm.lambda_decay = cfg.lambda_decay
norm.tau          = cfg.tau

_HIDDEN       = int(ckpt.get("hidden",       128))
_DEPTH        = int(ckpt.get("depth",        6))
_USE_RFF      = bool(ckpt.get("use_rff",     False))
_RFF_FEATURES = int(ckpt.get("rff_features", 64))
_RFF_SCALE    = float(ckpt.get("rff_scale",  10.0))
_USE_CONTEXT  = bool(ckpt.get("use_context", False))
# Perception range used during training.
# Old models (pre-filter) have no key → 0.0 → no filter at inference.
# New models store the actual range → apply same filter at inference.
_PERC_RANGE   = float(ckpt.get("perception_range", 0.0))

_n_cache_cols = len(FlatSampleCache.KEYS)   # 10 with context features
dummy_cache = FlatSampleCache.__new__(FlatSampleCache)
dummy_cache.x_grid = cfg.x
dummy_cache.y_grid = cfg.y
dummy_cache.times  = np.array([0.0, 1.0])
dummy_cache._buf   = np.zeros((2, _n_cache_cols), dtype=np.float32)
dummy_cache._N     = 2

trainer = PINNTrainer(snapshots=[], norm=norm, interp=dummy_cache,
                      hidden=_HIDDEN, depth=_DEPTH,
                      use_rff=_USE_RFF, rff_features=_RFF_FEATURES,
                      rff_scale=_RFF_SCALE,
                      use_context=_USE_CONTEXT,
                      device=device)
trainer.model.load_state_dict(ckpt["model_state"])
trainer.model.eval()
arch_tag = (f"RFF(feat={_RFF_FEATURES},scale={_RFF_SCALE:.0f})"
            if _USE_RFF else "no-RFF")
ctx_tag = "+ctx" if _USE_CONTEXT else ""
print(f"  PINN loaded from {model_path}  "
      f"({_HIDDEN}×{_DEPTH}, {arch_tag}{ctx_tag}), device={device}")

# ── DRIFT warm-up ─────────────────────────────────────────────────────────────
# DRIFTInterface is created AFTER cfg is scene-adapted, so it uses the correct grid.
print("\nDRIFT warm-up ...")
drift = DRIFTInterface()

_surr_init, _ego_v_init, _ = build_drift_vehicles(
    ego_fi0, EGO_TRACK_ID, tracks, tracks_meta, class_map)
if _ego_v_init is None:
    fi0 = ego_fi0 - ego_meta["initialFrame"]
    ex0 = float(ego_track["xCenter"][fi0])
    ey0 = float(ego_track["yCenter"][fi0])
    psi0 = math.radians(float(ego_track["heading"][fi0]))
    ev0  = max(float(ego_track["lonVelocity"][fi0]), 0.5)
    _ego_v_init = drift_create_vehicle(vid=0, x=ex0, y=ey0,
                                       vx=ev0 * math.cos(psi0),
                                       vy=ev0 * math.sin(psi0),
                                       vclass="car")
    _ego_v_init["heading"] = psi0

drift.warmup(_surr_init + [_ego_v_init], _ego_v_init,
             dt=dt, duration=WARMUP_S, substeps=3)
print()

# ===========================================================================
# MAIN RENDER LOOP
# ===========================================================================

print(f"Rendering {N_t} frames → {save_dir}/ ...")
risk_at_ego_num_list  = []
risk_at_ego_pinn_list = []
n_agents_in_list   = []   # N agents included in PINN per step
n_agents_tot_list  = []   # total agents visible per step
max_frame_all = max(tm["finalFrame"] for tm in tracks_meta)

bar = Bar(max=N_t - 1)
plt.figure(figsize=(22, 7))

for i in range(N_t):
    bar.next()
    frame_idx = ego_fi0 + i * frame_stride

    if frame_idx > ego_meta["finalFrame"] or frame_idx > max_frame_all:
        print(f"\n[WARN] End of recording at step {i}.")
        break

    fi_ego = frame_idx - ego_meta["initialFrame"]
    ex     = float(ego_track["xCenter"][fi_ego])
    ey     = float(ego_track["yCenter"][fi_ego])
    epsi   = math.radians(float(ego_track["heading"][fi_ego]))
    ev     = float(ego_track["lonVelocity"][fi_ego])

    surr_vehicles, ego_drift_v, surr_tids = build_drift_vehicles(
        frame_idx, EGO_TRACK_ID, tracks, tracks_meta, class_map)
    if ego_drift_v is None:
        ego_drift_v = drift_create_vehicle(
            vid=0, x=ex, y=ey,
            vx=ev * math.cos(epsi), vy=ev * math.sin(epsi),
            vclass="car")
        ego_drift_v["heading"] = epsi

    # ── Numerical DRIFT step ──────────────────────────────────────────────────
    risk_field_num  = drift.step(surr_vehicles, ego_drift_v, dt=dt, substeps=3)
    risk_at_ego_num = float(drift.get_risk_cartesian(ex, ey))
    risk_at_ego_num_list.append(risk_at_ego_num)

    # ── PINN inference ────────────────────────────────────────────────────────
    # Apply the same perception range that was used during training.
    # _PERC_RANGE == 0 means the model was trained without a filter → use all.
    all_raw = surr_vehicles + [ego_drift_v]
    all_tids_raw = surr_tids + [EGO_TRACK_ID]
    _perc = _PERC_RANGE if _PERC_RANGE > 0 else float('inf')

    # Filter with per-agent distance tracking
    filtered_pairs = [
        (v, tid) for v, tid in zip(all_raw, all_tids_raw)
        if math.hypot(v["x"] - ex, v["y"] - ey) <= _perc
    ]
    all_vehicles = [v for v, _ in filtered_pairs]
    included_tids = {tid for _, tid in filtered_pairs}
    if not all_vehicles:
        all_vehicles = [ego_drift_v]
        included_tids = {EGO_TRACK_ID}

    excluded_tids = set(surr_tids) - included_tids
    n_in  = len(included_tids)
    n_tot = len(surr_vehicles) + 1   # +1 for ego

    # Console log (every 20 steps to avoid spam)
    if i % 20 == 0:
        in_ids  = sorted(included_tids - {EGO_TRACK_ID})
        ex_ids  = sorted(excluded_tids)
        print(f"  [step {i:3d}] agents in PINN: {n_in}/{n_tot} "
              f"| included surr tids={in_ids} | excluded tids={ex_ids}")

    Q, _, _, occ = compute_total_Q(all_vehicles, ego_drift_v, cfg.X, cfg.Y)
    vx_f, vy_f, *_ = compute_velocity_field(all_vehicles, ego_drift_v, cfg.X, cfg.Y)
    D_f = compute_diffusion_field(occ, cfg.X, cfg.Y, all_vehicles, ego_drift_v)

    # Context features (only needed when model was trained with --use_context)
    _N_agents_frame = len(all_vehicles)
    _dist_nearest_frame = None
    if _USE_CONTEXT and _N_agents_frame > 0:
        axy = np.array([[v["x"], v["y"]] for v in all_vehicles], dtype=np.float32)
        dx  = cfg.X[:, :, np.newaxis] - axy[:, 0]
        dy  = cfg.Y[:, :, np.newaxis] - axy[:, 1]
        _dist_nearest_frame = np.sqrt(dx**2 + dy**2).min(axis=2).astype(np.float32)

    # Query PINN in training coordinate space (world - offset)
    t_sim = WARMUP_S + i * dt
    risk_field_pinn = trainer.predict_field_from_arrays(
        _X_pinn, _Y_pinn, t_sim, Q, vx_f, vy_f, D_f,
        N_agents=_N_agents_frame, dist_nearest=_dist_nearest_frame)

    # Risk at ego: nearest grid cell
    i_y = int(np.argmin(np.abs(cfg.y - ey)))
    i_x = int(np.argmin(np.abs(cfg.x - ex)))
    i_y = max(0, min(i_y, cfg.ny - 1))
    i_x = max(0, min(i_x, cfg.nx - 1))
    risk_at_ego_pinn = float(risk_field_pinn[i_y, i_x])
    risk_at_ego_pinn_list.append(risk_at_ego_pinn)
    n_agents_in_list.append(n_in)
    n_agents_tot_list.append(n_tot)

    # ── Draw frame ────────────────────────────────────────────────────────────
    agent_info = dict(included=included_tids, excluded=excluded_tids,
                      n_in=n_in, n_tot=n_tot,
                      perc_range=_perc if math.isfinite(_perc) else None)
    draw_frame_pinn_compare(
        i, frame_idx, tracks, tracks_meta, class_map, bg_img,
        risk_field_num,  risk_at_ego_num,
        risk_field_pinn, risk_at_ego_pinn,
        agent_info=agent_info)

bar.finish()
print()
print(f"Rendering complete — {N_t} frames saved to {save_dir}/")

# ── Summary metrics plot ──────────────────────────────────────────────────────
if len(risk_at_ego_num_list) > 1:
    _t = np.arange(len(risk_at_ego_num_list)) * dt
    with plt.style.context(["science", "no-latex"]):
        fig_m, ax_m = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
        ax_m.plot(_t, risk_at_ego_num_list,  color="C3", lw=1.4,
                  label="Numerical DRIFT")
        ax_m.plot(_t, risk_at_ego_pinn_list, color="C5", lw=1.4, ls="--",
                  label=r"PINN $\hat{\mathcal{R}}_\theta$")
        ax_m.fill_between(_t, risk_at_ego_num_list,  alpha=0.15, color="C3")
        ax_m.fill_between(_t, risk_at_ego_pinn_list, alpha=0.15, color="C5")
        ax_m.set_xlabel("t [s]")
        ax_m.set_ylabel("Risk at ego $R$")
        ax_m.set_title(
            f"DRIFT vs PINN Risk at Ego  |  rec {rec}  track {EGO_TRACK_ID}",
            fontsize=10)
        ax_m.legend(fontsize=8)
        ax_m.grid(True, lw=0.4, alpha=0.4)
        fig_m.savefig(os.path.join(save_dir, "risk_at_ego.png"),
                      dpi=150, bbox_inches="tight")
        plt.close(fig_m)
    print(f"Risk-at-ego plot → {save_dir}/risk_at_ego.png")

# ── Agent selection summary plot ──────────────────────────────────────────────
if len(n_agents_in_list) > 1:
    _t = np.arange(len(n_agents_in_list)) * dt
    _n_in  = np.array(n_agents_in_list,  dtype=float)
    _n_tot = np.array(n_agents_tot_list, dtype=float)
    _frac  = np.where(_n_tot > 0, _n_in / _n_tot, 1.0)

    with plt.style.context(["science", "no-latex"]):
        fig_a, (ax_a1, ax_a2) = plt.subplots(
            2, 1, figsize=(10, 5), constrained_layout=True,
            gridspec_kw={"height_ratios": [2, 1]})

        # Top: absolute counts
        ax_a1.plot(_t, _n_tot, color="C0", lw=1.2, label="Total agents visible")
        ax_a1.plot(_t, _n_in,  color="C2", lw=1.4, label="Agents in PINN (≤range)")
        ax_a1.fill_between(_t, _n_in, _n_tot, alpha=0.20, color="C3",
                           label="Excluded agents")
        ax_a1.set_ylabel("Agent count")
        ax_a1.set_title(
            f"Agent selection  |  rec {rec}  ego={EGO_TRACK_ID}"
            + (f"  perc_range={_PERC_RANGE:.0f} m" if _PERC_RANGE > 0 else "  (no filter)"),
            fontsize=9)
        ax_a1.legend(fontsize=7, ncol=3)
        ax_a1.grid(True, lw=0.4, alpha=0.4)
        ax_a1.set_xlim(_t[0], _t[-1])

        # Bottom: inclusion fraction
        ax_a2.plot(_t, _frac * 100, color="C2", lw=1.2)
        ax_a2.axhline(100, color="grey", lw=0.6, ls="--")
        ax_a2.set_ylim(0, 110)
        ax_a2.set_xlabel("t [s]")
        ax_a2.set_ylabel("% included")
        ax_a2.set_xlim(_t[0], _t[-1])
        ax_a2.grid(True, lw=0.4, alpha=0.4)

        fig_a.savefig(os.path.join(save_dir, "agent_selection.png"),
                      dpi=150, bbox_inches="tight")
        plt.close(fig_a)
    print(f"Agent selection plot → {save_dir}/agent_selection.png")

    # Print per-step summary to console
    print("\n--- Agent selection summary ---")
    print(f"  Mean included:  {_n_in.mean():.1f} / {_n_tot.mean():.1f}  "
          f"({_frac.mean()*100:.1f}%)")
    print(f"  Min  included:  {int(_n_in.min())} / {int(_n_tot.max())}  "
          f"(step {int(np.argmin(_n_in))})")
    print(f"  Steps w/ excluded agents: "
          f"{int((_n_in < _n_tot).sum())} / {len(_n_in)}")

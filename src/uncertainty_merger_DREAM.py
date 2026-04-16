"""
Uncertainty Merger Scenario: Baseline IDEAM vs DREAM
====================================================

Top subplot (baseline):
- Ego: IDEAM (MPC-CBF + IDEAM decision).
- Merger: IDEAM (right -> centre lane change).

Bottom subplot (ours):
- Ego: DREAM controller.
- Merger: IDEAM (right -> centre lane change).

Blocker placement and lane-change timing are tuned so ego/merger do not start
LC before overtaking the truck.
"""

import argparse
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import math
import sys
import time
import traceback
import numpy as np
try:
    import torch
    _TORCH_OK = True
except ImportError:
    torch = None
    _TORCH_OK = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from scipy.ndimage import gaussian_filter as _gf
import scienceplots  # noqa: F401

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from Control.MPC import *  # noqa: F403
from Control.constraint_params import *  # noqa: F403
from Model.Dynamical_model import *  # noqa: F403
from Model.params import *  # noqa: F403
from Model.surrounding_params import *  # noqa: F403
from Model.Surrounding_model import *  # noqa: F403
from Control.HOCBF import *  # noqa: F403
from DecisionMaking.decision_params import *  # noqa: F403
from DecisionMaking.give_desired_path import *  # noqa: F403
from DecisionMaking.util import *  # noqa: F403
from DecisionMaking.util_params import *  # noqa: F403
from DecisionMaking.decision import *  # noqa: F403
from Prediction.surrounding_prediction import *  # noqa: F403
from progress.bar import Bar

from config import Config as cfg
from pde_solver import create_vehicle as drift_create_vehicle
from Integration.prideam_controller import create_prideam_controller
from Integration.drift_interface import DRIFTInterface
from Integration.integration_config import get_preset
from rl.config.rl_config import DEFAULT_CONFIG as RL_DEFAULT_CONFIG
if _TORCH_OK:
    from rl.train import PolicyNet
    try:
        from rl.policy.decision_policy import DecisionPolicy
        from rl.policy.decision_inference import (
            build_simulator_obs as _build_dec_obs,
            decide_and_setup as _decide_and_setup,
        )
        from rl.train_bc import load_decision_policy as _load_bc_policy
        _RL_DECISION_OK = True
    except Exception as _dec_err:
        print(f"[warn] decision-level RL unavailable: {_dec_err}")
        _RL_DECISION_OK = False
else:
    PolicyNet = None
    _RL_DECISION_OK = False

# ADA source adapter (DREAM-ADA arm)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "Aggressiveness_Modeling"))
from Aggressiveness_Modeling.ADA_drift_source import compute_Q_ADA  # noqa: E402

# APF source adapter (DREAM-APF arm)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "APF_Modeling"))
from APF_Modeling.APF_drift_source import compute_Q_APF  # noqa: E402

# OA-CMPC source adapter (Zheng et al. 2026, arXiv:2503.04563)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "OA_CMPC"))
from OA_CMPC.oa_cmpc_source import compute_Q_OACMPC  # noqa: E402


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run uncertainty merger benchmark: IDEAM baseline vs DREAM."
    )
    parser.add_argument(
        "--integration-mode",
        default=os.environ.get("DREAM_INTEGRATION_MODE", "conservative"),
        help="Integration preset name for PRIDEAM controller.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("DREAM_STEPS", "100")),
        help="Number of simulation steps.",
    )
    parser.add_argument(
        "--save-dpi",
        type=int,
        default=int(os.environ.get("DREAM_DPI", "300")),
        help="DPI used when saving frame images.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=int(os.environ.get("DREAM_LOG_EVERY", "20")),
        help="Logging interval in steps.",
    )
    parser.add_argument(
        "--save-frames",
        type=_str2bool,
        default=_str2bool(os.environ.get("DREAM_SAVE_FRAMES", "1")),
        help="Whether to save per-step visualization frames.",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join(SCRIPT_DIR, "figsave_uncertainty_merger_v6"),
        help="Directory where frames and metrics are saved.",
    )
    parser.add_argument(
        "--run-mode",
        default=os.environ.get("DREAM_RUN_MODE", "single"),
        choices=["single", "ablation"],
        help="Run mode: single (default, full sim+figures) or ablation (DRIFT parameter sensitivity).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        metavar="MODEL",
        help="Planners to run (space-separated): all DREAM RL-PPO ADA APF OA-CMPC IDEAM",
    )
    parser.add_argument(
        "--rl-checkpoint",
        default=None,
        help="PPO checkpoint for RL-PPO benchmark arm. Default: rl/checkpoints/ppo_best.pt or ppo_final.pt",
    )
    parser.add_argument(
        "--rl-policy-mode",
        default=os.environ.get("DREAM_RL_MODE", "ppo"),
        choices=["ppo", "decision"],
        help=("Mode for the RL-PPO arm: 'ppo' = legacy continuous controller, "
              "'decision' = BC/PPO-trained DecisionPolicy commanding lane + "
              "cruise speed through the MPC+CBF tracker."),
    )
    parser.add_argument(
        "--rl-decision-checkpoint",
        default=os.environ.get("DREAM_RL_DEC_CKPT",
                               "rl/checkpoints/decision_policy_bc.pt"),
        help="Checkpoint path for the decision-level RL policy.",
    )
    return parser.parse_args()


CLI_ARGS = _parse_cli_args()

# Resolve selected planners from --models argument
_ALL_PLANNERS_M = {'DREAM', 'RL-PPO', 'ADA', 'APF', 'OA-CMPC', 'IDEAM'}
_NORM_MAP_M = {p.lower(): p for p in _ALL_PLANNERS_M}
_NORM_MAP_M.update({'oacmpc': 'OA-CMPC', 'dream': 'DREAM', 'ada': 'ADA',
                    'apf': 'APF', 'ideam': 'IDEAM', 'rlppo': 'RL-PPO', 'rl': 'RL-PPO'})
if 'all' in [m.lower() for m in CLI_ARGS.models]:
    RUN_PLANNERS = _ALL_PLANNERS_M.copy()
else:
    RUN_PLANNERS = set()
    for _m in CLI_ARGS.models:
        _key = _m.lower().replace('-', '')
        _mapped = _NORM_MAP_M.get(_key) or _NORM_MAP_M.get(_m.lower())
        if _mapped is None:
            raise SystemExit(f"Unknown planner '{_m}'. Choose from: "
                             + " ".join(sorted(_ALL_PLANNERS_M)) + " all")
        RUN_PLANNERS.add(_mapped)

# ============================================================================
# CONFIGURATION
# ============================================================================

INTEGRATION_MODE = CLI_ARGS.integration_mode
config_integration = get_preset(INTEGRATION_MODE)
config_integration.apply_mode()

N_t = max(1, CLI_ARGS.steps)
SAVE_DPI = CLI_ARGS.save_dpi
LOG_EVERY = max(1, CLI_ARGS.log_every)
SAVE_FRAMES = CLI_ARGS.save_frames
_ABLATION_MODE = (CLI_ARGS.run_mode == "ablation")
dt = 0.1
boundary = 1.0

# Vehicle geometry
CAR_LENGTH = 3.5
CAR_WIDTH = 1.2
TRUCK_LENGTH = 12.0
TRUCK_WIDTH = 2.0

# Colors
EGO_IDEAM_COLOR = "#4CAF50"
EGO_DREAM_COLOR = "#2196F3"
EGO_RL_COLOR    = "#795548"
EGO_ADA_COLOR   = "#9C27B0"
EGO_APF_COLOR   = "#009688"
EGO_OACMPC_COLOR = "#FF5722"
TRUCK_COLOR = "#FF6F00"
MERGER_COLOR = "#E91E63"
SHADOW_COLOR = "#4A4A4A"

# DRIFT visualization
RISK_ALPHA = 0.65
RISK_CMAP = "jet"
RISK_LEVELS = 40
RISK_VMAX = 2.0

# Metric thresholds
NEAR_COLLISION_DIST = 3.0
COLLISION_DIST = 1.0

# Scenario parameters
TRUCK_VD = 5.3
TRUCK_S0 = 45.0
TRUCK_V0 = 5.3

EGO_S0 = 20.0
EGO_V0 = 10.2

# Left-lane blocker (IDM-controlled slow car).
BLOCKER_S_INIT = 80.0
BLOCKER_VD = 10.0

# Mergers: IDEAM-controlled in both baseline and DREAM worlds.
MERGER_BASE_S0 = 24.0
MERGER_BASE_V0 = 10.5
MERGER_DREAM_S0 = 24.0
MERGER_DREAM_V0 = 10.5

# Scenario-level lane-change forcing for reproducibility.
EGO_FORCE_CENTER_MIN_STEP = 0
MERGER_FORCE_CENTER_MIN_STEP = 0
EGO_LC_OVERTAKE_MARGIN = 4.0
MERGER_LC_OVERTAKE_MARGIN = 4.0
EGO_BLOCKER_TRIGGER_GAP = 45.0
EGO_BASE_ASSIST_BLEND = 0.20
EGO_DREAM_ASSIST_BLEND = 0.20
MERGER_BASE_ASSIST_BLEND = 0.24
MERGER_DREAM_ASSIST_BLEND = 0.24
KEEP_LANE_ASSIST_BLEND = 0.30

RL_OBS_DIM = 22
RL_MAX_ACCEL_CONT = 1.5
RL_NORM_RISK = 5.0
RL_NORM_GRAD = 2.0

x_area = 50.0
y_area = 15.0
steer_range = [math.radians(-8.0), math.radians(8.0)]

save_dir = os.path.abspath(CLI_ARGS.save_dir)
os.makedirs(save_dir, exist_ok=True)


# ============================================================================
# SYNTHETIC AGENTS
# ============================================================================

class LeftLaneBlocker:
    """Slow IDM-controlled car in left lane — always visible to both planners."""

    def __init__(self, path, path_data, s_init, vd, dt, steer_range):
        self.path        = path
        self.path_data   = path_data   # (x, y, s) arrays for find_frenet_coord
        self.s           = float(s_init)
        self.v           = float(vd)
        self.vd          = float(vd)
        self.dt          = dt
        self.steer_range = steer_range
        self.a           = 0.0
        self.x, self.y   = path(self.s)
        self.yaw         = path.get_theta_r(self.s)
        self._ctrl = Curved_Road_Vehicle(
            a_max=2.0, delta=4, s0=2.0, b=1.5, T=1.5,
            K_P=1.0, K_D=0.1, K_I=0.01,
            dt=dt, lf=1.5, lr=1.5, length=CAR_LENGTH)

    def update(self):
        self.x, self.y, self.yaw, v_next, _, self.a = \
            self._ctrl.update_states(
                self.s, self.v, self.vd, None, None, self.path, self.steer_range)
        self.s += self.v * self.dt
        self.v  = max(0.0, v_next)

    def to_mpc_row(self):
        px, py, ps = self.path_data
        try:
            s_f, ey_f, epsi_f = find_frenet_coord(
                self.path, px, py, ps, [self.x, self.y, self.yaw])
        except Exception:
            s_f, ey_f, epsi_f = self.s, 0.0, 0.0
        return np.array([s_f, ey_f, epsi_f, self.x, self.y, self.yaw,
                         max(0.0, self.v), self.a])

    def to_drift_vehicle(self, vid=3):
        psi = self.yaw
        v   = drift_create_vehicle(vid=vid, x=self.x, y=self.y,
                                   vx=self.v * math.cos(psi),
                                   vy=self.v * math.sin(psi), vclass="car")
        v["heading"] = psi
        v["a"]       = self.a
        return v


# ============================================================================
# VISUAL HELPERS
# ============================================================================

def draw_vehicle_rect(ax, x, y, yaw_rad, length, width, facecolor,
                      edgecolor="black", lw=0.8, zorder=3, alpha=1.0,
                      linestyle="-"):
    rect = mpatches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle="round,pad=0.05",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=zorder,
        linestyle=linestyle
    )
    t = Affine2D().rotate(yaw_rad).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    return rect


def compute_truck_shadow(ego_x, ego_y, truck_row, shadow_length=55.0):
    tx, ty, yaw = truck_row[3], truck_row[4], truck_row[5]
    dx, dy = tx - ego_x, ty - ego_y
    dist = math.sqrt(dx**2 + dy**2)
    if dist < 2.0:
        return None

    corners_local = np.array([
        [-TRUCK_LENGTH / 2, -TRUCK_WIDTH / 2],
        [TRUCK_LENGTH / 2, -TRUCK_WIDTH / 2],
        [TRUCK_LENGTH / 2, TRUCK_WIDTH / 2],
        [-TRUCK_LENGTH / 2, TRUCK_WIDTH / 2],
    ])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    corners = (rot @ corners_local.T).T + np.array([tx, ty])

    angles = np.arctan2(corners[:, 1] - ego_y, corners[:, 0] - ego_x)
    left_corner = corners[np.argmax(angles)]
    right_corner = corners[np.argmin(angles)]

    l_dir = left_corner - np.array([ego_x, ego_y])
    l_dir /= (np.linalg.norm(l_dir) + 1e-9)
    r_dir = right_corner - np.array([ego_x, ego_y])
    r_dir /= (np.linalg.norm(r_dir) + 1e-9)

    return np.array([
        left_corner,
        left_corner + l_dir * shadow_length,
        right_corner + r_dir * shadow_length,
        right_corner
    ])


def draw_shadow_polygon(ax, shadow_polygon, alpha=0.30):
    if shadow_polygon is None:
        return
    patch = plt.Polygon(
        shadow_polygon,
        facecolor=SHADOW_COLOR, alpha=alpha,
        edgecolor="red", linewidth=1.6, linestyle="--", zorder=2
    )
    ax.add_patch(patch)


def draw_panel(ax, ego_X0, ego_X0_g, truck_row, merger_row, blocker_row,
               title, x_range, y_range,
               risk_field=None, horizon=None, ego_color=EGO_IDEAM_COLOR):
    plt.sca(ax)
    ax.cla()
    plot_env()

    contourf_obj = None
    if risk_field is not None:
        r_sm = _gf(risk_field, sigma=0.8)
        r_sm = np.clip(r_sm, 0, RISK_VMAX)
        contourf_obj = ax.contourf(
            cfg.X, cfg.Y, r_sm,
            levels=RISK_LEVELS, cmap=RISK_CMAP,
            alpha=RISK_ALPHA, vmin=0, vmax=RISK_VMAX,
            zorder=1, extend="max"
        )
        ax.contour(
            cfg.X, cfg.Y, r_sm,
            levels=np.linspace(0.2, RISK_VMAX, 8),
            colors="darkred", linewidths=0.5, alpha=0.4, zorder=1
        )

    if horizon is not None and len(horizon) > 0:
        h = np.asarray(horizon)
        if h.ndim == 2 and h.shape[1] >= 2:
            ax.plot(h[:, 0], h[:, 1], color="#00BCD4", lw=1.8, ls="--", zorder=7)
            ax.scatter(h[:, 0], h[:, 1], color="#00BCD4", s=6, zorder=7)

    # Truck + shadow
    draw_shadow_polygon(ax, compute_truck_shadow(ego_X0_g[0], ego_X0_g[1], truck_row))
    draw_vehicle_rect(
        ax, truck_row[3], truck_row[4], truck_row[5],
        TRUCK_LENGTH, TRUCK_WIDTH, TRUCK_COLOR,
        edgecolor="darkred", lw=1.2, zorder=5
    )
    ax.text(
        truck_row[3] - 2.5, truck_row[4] + 2.8, f"Truck {truck_row[6]:.1f} m/s",
        rotation=np.rad2deg(truck_row[5]), c="darkred",
        fontsize=5, style="oblique", fontweight="bold"
    )

    # Merger
    draw_vehicle_rect(
        ax, merger_row[3], merger_row[4], merger_row[5],
        CAR_LENGTH, CAR_WIDTH, MERGER_COLOR,
        edgecolor="black", lw=0.9, zorder=5
    )
    ax.text(
        merger_row[3] - 1.5, merger_row[4] + 1.1, "Merger",
        rotation=np.rad2deg(merger_row[5]), c="black",
        fontsize=5, style="oblique", fontweight="bold"
    )

    # Left-lane blocker
    if blocker_row is not None:
        draw_vehicle_rect(
            ax, blocker_row[3], blocker_row[4], blocker_row[5],
            CAR_LENGTH, CAR_WIDTH, "#616161",
            edgecolor="black", lw=0.9, zorder=5
        )
        ax.text(
            blocker_row[3] - 1.7, blocker_row[4] + 1.1, "Blocker",
            rotation=np.rad2deg(blocker_row[5]), c="black",
            fontsize=5, style="oblique", fontweight="bold"
        )

    # Ego
    draw_vehicle_rect(
        ax, ego_X0_g[0], ego_X0_g[1], ego_X0_g[2],
        CAR_LENGTH, CAR_WIDTH, ego_color,
        edgecolor="navy", lw=1.0, zorder=6
    )
    ax.text(
        ego_X0_g[0] - 2.0, ego_X0_g[1] - 2.0, f"{ego_X0[0]:.1f} m/s",
        c="black", fontsize=6
    )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    return contourf_obj


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

path_center = np.array([path1c, path2c, path3c], dtype=object)
sample_center = np.array([samples1c, samples2c, samples3c], dtype=object)
x_center = [x1c, x2c, x3c]
y_center = [y1c, y2c, y3c]
x_bound = [x1, x2]
y_bound = [y1, y2]
path_bound = [path1, path2]
path_bound_sample = [samples1, samples2]


def lane_from_global(X0_g):
    return judge_current_position(X0_g[0:2], x_bound, y_bound, path_bound, path_bound_sample)


def empty_lane():
    return np.zeros((0, 8), dtype=float)


def sort_lane(arr):
    if arr is None or len(arr) == 0:
        return empty_lane()
    idx = np.argsort(arr[:, 0])
    return np.asarray(arr[idx], dtype=float)


def stack_rows(rows):
    valid = [np.asarray(r, dtype=float).reshape(1, 8) for r in rows if r is not None]
    if not valid:
        return empty_lane()
    return sort_lane(np.vstack(valid))


def state_to_row(X0, X0_g):
    return np.array([X0[3], X0[4], X0[5], X0_g[0], X0_g[1], X0_g[2], X0[0], 0.0], dtype=float)


def row_to_drift_vehicle(row, vid, vclass="car"):
    psi = float(row[5])
    v_lon = float(row[6])
    v = drift_create_vehicle(
        vid=vid,
        x=float(row[3]), y=float(row[4]),
        vx=v_lon * math.cos(psi), vy=v_lon * math.sin(psi),
        vclass=vclass
    )
    v["heading"] = psi
    v["a"] = float(row[7]) if len(row) > 7 else 0.0
    return v


def build_horizon(X0_seed, X0_g_seed, oa_cmd, od_cmd, path_d, sample, x_list, y_list,
                  dyn_obj, dt_, boundary_):
    if oa_cmd is None or od_cmd is None:
        return None
    Xv = list(X0_seed)
    Xgv = list(X0_g_seed)
    out = [list(Xgv)]
    n = max(0, len(oa_cmd) - 1)
    for k in range(n):
        u = [oa_cmd[k + 1], od_cmd[k + 1]]
        Xv, Xgv, _, _ = dyn_obj.propagate(Xv, u, dt_, Xgv, path_d, sample, x_list, y_list, boundary_)
        out.append(list(Xgv))
    return np.array(out)


def pick_group_for_lane(group_dict, lane_index):
    lane_to_groups = {
        0: ("L1", "L2"),
        1: ("C1", "C2"),
        2: ("R1", "R2"),
    }
    for name in lane_to_groups.get(lane_index, ()):
        if name in group_dict:
            return group_dict[name]
    return group_dict[next(iter(group_dict))]


def force_path_target(path_now, target_lane, X0, X0_g):
    target_lane = int(target_lane)
    path_d = [path1c, path2c, path3c][target_lane]
    _, x_list, y_list, sample = get_path_info(target_lane)

    if target_lane == path_now:
        C_label = "K"
    else:
        C_label = "R" if target_lane > path_now else "L"
    X0_forced = repropagate(path_d, sample, x_list, y_list, X0_g, X0)
    return path_d, target_lane, C_label, sample, x_list, y_list, X0_forced


def blend_state_toward_lane(X0, X0_g, target_lane, blend=0.25):
    blend = float(np.clip(blend, 0.0, 1.0))
    path_t = [path1c, path2c, path3c][int(target_lane)]
    s_ref = float(X0[3])
    x_t, y_t = path_t(s_ref)
    psi_t = path_t.get_theta_r(s_ref)

    X0_g_new = list(X0_g)
    X0_g_new[0] = (1.0 - blend) * X0_g[0] + blend * x_t
    X0_g_new[1] = (1.0 - blend) * X0_g[1] + blend * y_t
    dpsi = math.atan2(math.sin(psi_t - X0_g[2]), math.cos(psi_t - X0_g[2]))
    X0_g_new[2] = X0_g[2] + blend * dpsi

    _, x_list, y_list, sample = get_path_info(int(target_lane))
    X0_new = repropagate(path_t, sample, x_list, y_list, X0_g_new, list(X0))
    return X0_new, X0_g_new


def progress_on_reference(X0_g):
    """Project global pose to a common reference for fair s(t) comparison."""
    candidates = (
        (path1c, x1c, y1c, samples1c),
        (path2c, x2c, y2c, samples2c),
        (path3c, x3c, y3c, samples3c),
    )
    for path_ref, xr, yr, sr in candidates:
        try:
            s_ref, _, _ = find_frenet_coord(path_ref, xr, yr, sr, X0_g)
            if np.isfinite(s_ref):
                return float(s_ref)
        except Exception:
            continue
    return float("nan")


def min_center_distance(ego_g, rows):
    """Minimum Euclidean centre distance from ego to a list of vehicle rows."""
    ex, ey = float(ego_g[0]), float(ego_g[1])
    dists = []
    for row in rows:
        if row is None:
            continue
        rx, ry = float(row[3]), float(row[4])
        d = math.hypot(rx - ex, ry - ey)
        if np.isfinite(d):
            dists.append(d)
    return float(min(dists)) if dists else float("nan")


def _default_rl_checkpoint():
    for name in ("ppo_best.pt", "ppo_final.pt"):
        path = os.path.join(SCRIPT_DIR, "rl", "checkpoints", name)
        if os.path.isfile(path):
            return path
    return None


def _load_rl_decision_policy(checkpoint_path):
    """Load a decision-level BC/PPO policy checkpoint, or return None.

    Looks for the given path (default: rl/checkpoints/decision_policy_bc.pt).
    Falls back gracefully if torch or the checkpoint is unavailable — the
    caller then skips the RL-decision arm.
    """
    if not (_TORCH_OK and _RL_DECISION_OK):
        return None, None
    candidates = []
    if checkpoint_path:
        candidates.append(checkpoint_path)
    candidates += [
        os.path.join(SCRIPT_DIR, "rl", "checkpoints", "decision_policy_bc.pt"),
        os.path.join(SCRIPT_DIR, "rl", "checkpoints", "decision_policy_ppo.pt"),
    ]
    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if path is None:
        print(f"[rl-dec] no decision-policy checkpoint found — looked at: "
              f"{[c for c in candidates if c]}")
        return None, None
    try:
        policy = _load_bc_policy(path, device="cpu")
    except Exception as e:
        print(f"[rl-dec] failed to load decision policy from {path}: {e}")
        return None, None
    print(f"[rl-dec] decision policy loaded: {path}")
    return policy, path


def _load_rl_policy(checkpoint_path):
    if not _TORCH_OK or PolicyNet is None:
        raise SystemExit("RL-PPO selected but PyTorch is unavailable in this environment.")
    path = checkpoint_path or _default_rl_checkpoint()
    if path is None or not os.path.isfile(path):
        raise SystemExit(
            "RL-PPO selected but no PPO checkpoint was found. "
            "Train with `python rl/train.py` or pass --rl-checkpoint."
        )
    ckpt = torch.load(path, map_location="cpu")
    policy = PolicyNet(obs_dim=RL_OBS_DIM, action_dim=2)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(f"RL-PPO checkpoint: {path}")
    return policy, path


def _lane_gap_and_dv(ego_s, ego_v, lane_rows, default_gap=60.0):
    if lane_rows is None or len(lane_rows) == 0:
        return float(default_gap), 0.0, 0.0
    best_gap = float(default_gap)
    best_dv = 0.0
    best_v = 0.0
    for row in np.asarray(lane_rows):
        ds = float(row[0]) - float(ego_s)
        if ds > 0.0 and ds < best_gap:
            best_gap = ds
            best_v = float(row[6])
            best_dv = float(ego_v) - best_v
    return best_gap, best_dv, best_v


def _risk_sample_along_path(drift_iface, path_obj, s_ref, offsets):
    vals = []
    for ds in offsets:
        x_q, y_q = path_obj(float(s_ref) + float(ds))
        vals.append(float(drift_iface.get_risk_cartesian(float(x_q), float(y_q))))
    return vals


def build_rl_observation(X0, X0_g, oa_prev, od_prev, drift_iface,
                         lane_left, lane_center, lane_right,
                         force_target_lane=None, cbf_active=0.0):
    lane_now = lane_from_global(X0_g)
    if force_target_lane is not None:
        lane_idx = int(np.clip(force_target_lane, 0, 2))
    else:
        lane_idx = int(np.clip(lane_now, 0, 2))

    path_d = [path1c, path2c, path3c][lane_idx]
    _, x_list, y_list, sample = get_path_info(lane_idx)
    s_ref, ey_ref, epsi_ref = find_frenet_coord(path_d, x_list, y_list, sample, X0_g)

    ego_v = float(X0[0])
    last_a = float(oa_prev[0]) if hasattr(oa_prev, "__len__") else float(oa_prev)
    last_delta = float(od_prev[0]) if hasattr(od_prev, "__len__") else float(od_prev)

    lane_map = {0: lane_left, 1: lane_center, 2: lane_right}
    ds_curr, dv_curr, _ = _lane_gap_and_dv(s_ref, ego_v, lane_map.get(lane_idx))
    if lane_idx > 0:
        ds_left, dv_left, _ = _lane_gap_and_dv(s_ref, ego_v, lane_map.get(lane_idx - 1))
    else:
        ds_left, dv_left = 0.0, 0.0
    if lane_idx < 2:
        ds_right, dv_right, _ = _lane_gap_and_dv(s_ref, ego_v, lane_map.get(lane_idx + 1))
    else:
        ds_right, dv_right = 0.0, 0.0

    r_ego = float(drift_iface.get_risk_cartesian(float(X0_g[0]), float(X0_g[1])))
    r_5m, r_10m, r_20m = _risk_sample_along_path(drift_iface, path_d, s_ref, (5.0, 10.0, 20.0))
    grad_x, grad_y = drift_iface.get_risk_gradient_cartesian(float(X0_g[0]), float(X0_g[1]))
    grad_x = float(grad_x)
    grad_y = float(grad_y)

    if lane_idx > 0:
        x_l, y_l = [path1c, path2c, path3c][lane_idx - 1](s_ref)
        r_left = float(drift_iface.get_risk_cartesian(float(x_l), float(y_l)))
    else:
        r_left = 0.0
    if lane_idx < 2:
        x_r, y_r = [path1c, path2c, path3c][lane_idx + 1](s_ref)
        r_right = float(drift_iface.get_risk_cartesian(float(x_r), float(y_r)))
    else:
        r_right = 0.0

    in_merge = 1.0 if (30.0 <= float(s_ref) <= 70.0) else 0.0
    obs = np.array([
        ego_v / RL_DEFAULT_CONFIG.NORM_V,
        float(ey_ref) / RL_DEFAULT_CONFIG.NORM_EY,
        float(epsi_ref) / RL_DEFAULT_CONFIG.NORM_EPSI,
        last_a / RL_DEFAULT_CONFIG.NORM_A,
        last_delta / RL_DEFAULT_CONFIG.NORM_EPSI,
        ds_curr / RL_DEFAULT_CONFIG.NORM_DS - 1.0,
        dv_curr / RL_DEFAULT_CONFIG.NORM_DV,
        0.0,
        ds_left / RL_DEFAULT_CONFIG.NORM_DS - 1.0,
        dv_left / RL_DEFAULT_CONFIG.NORM_DV,
        ds_right / RL_DEFAULT_CONFIG.NORM_DS - 1.0,
        dv_right / RL_DEFAULT_CONFIG.NORM_DV,
        r_ego / RL_NORM_RISK,
        r_5m / RL_NORM_RISK,
        r_10m / RL_NORM_RISK,
        r_20m / RL_NORM_RISK,
        grad_x / RL_NORM_GRAD,
        grad_y / RL_NORM_GRAD,
        r_left / RL_NORM_RISK,
        r_right / RL_NORM_RISK,
        in_merge,
        float(cbf_active),
    ], dtype=np.float32)
    return np.clip(obs, -3.0, 3.0), path_d, sample, x_list, y_list, lane_idx


def rl_agent_step(X0, X0_g, oa_prev, od_prev, policy, drift_iface, dynamics_obj,
                  lane_left, lane_center, lane_right, force_target_lane=None):
    try:
        t0 = time.time()
        obs, path_d, sample, x_list, y_list, lane_idx = build_rl_observation(
            X0, X0_g, oa_prev, od_prev, drift_iface,
            lane_left, lane_center, lane_right,
            force_target_lane=force_target_lane, cbf_active=0.0
        )
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _ = policy(obs_t)
        action = mean.squeeze(0).cpu().numpy()
        a_cmd = float(np.clip(action[0], RL_DEFAULT_CONFIG.MIN_ACCEL, RL_MAX_ACCEL_CONT))
        d_cmd = float(np.clip(action[1], -RL_DEFAULT_CONFIG.MAX_STEER, RL_DEFAULT_CONFIG.MAX_STEER))
        t1 = time.time()

        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0, [a_cmd, d_cmd], dt, X0_g, path_d, sample, x_list, y_list, boundary
        )

        return {
            "ok": True,
            "X0": X0_new,
            "X0_g": X0_g_new,
            "oa": [a_cmd],
            "od": [d_cmd],
            "last_X": None,
            "path_changed": lane_idx,
            "path_d": path_d,
            "sample": sample,
            "x_list": x_list,
            "y_list": y_list,
            "path_now": lane_idx,
            "path_dindex": lane_idx,
            "forced": force_target_lane is not None,
            "t_decision": t1 - t0,
            "t_mpc": float("nan"),
            "obs": obs,
        }
    except Exception as e:
        lane_now = lane_from_global(X0_g)
        path_d = [path1c, path2c, path3c][lane_now]
        _, x_list, y_list, sample = get_path_info(lane_now)
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0, [0.0, 0.0], dt, X0_g, path_d, sample, x_list, y_list, boundary
        )
        return {
            "ok": False,
            "error": f"{e}\n{traceback.format_exc()}",
            "X0": X0_new,
            "X0_g": X0_g_new,
            "oa": [0.0],
            "od": [0.0],
            "last_X": None,
            "path_changed": lane_now,
            "path_d": path_d,
            "sample": sample,
            "x_list": x_list,
            "y_list": y_list,
            "path_now": lane_now,
            "path_dindex": lane_now,
            "forced": force_target_lane is not None,
            "t_decision": float("nan"),
            "t_mpc": float("nan"),
        }


def rl_decision_step(
    X0, X0_g, oa_prev, od_prev, last_X_prev, path_changed_prev,
    decision_policy, mpc_ctrl, util_obj, decision_obj, dynamics_obj,
    lane_left, lane_center, lane_right,
    force_target_lane=None, bypass_probe_guard=False,
    v_ref=None, lane_rel_start=0,
):
    """Decision-level RL step.

    Runs the :class:`rl.policy.decision_policy.DecisionPolicy` to select a
    (lane_delta, speed_mode) action, then delegates low-level tracking to
    the IDEAM MPC+CBF pipeline. The policy output can be overridden by
    `force_target_lane` for the same scenario-level forcing used by other
    arms (keeps the comparison fair).
    """
    try:
        _t0 = time.time()
        lane_now = lane_from_global(X0_g)
        v_ref_eff = float(v_ref) if v_ref is not None else float(RL_DEFAULT_CONFIG.TARGET_SPEED)

        # 1) Build the 17-dim decision observation from simulator state.
        obs = _build_dec_obs(
            X0, X0_g,
            lane_left, lane_center, lane_right,
            lane_now=lane_now,
            lane_rel_start=lane_rel_start,
            ey=0.0, epsi=0.0,
        )

        # 2) Run the policy to get (force_target_lane, v_target) unless the
        #    scenario is forcing a lane — in that case we still call the
        #    policy to set the speed override but use the forced lane.
        ft_lane_pol, v_target = _decide_and_setup(
            obs, decision_policy, mpc_ctrl, lane_now, v_ref_eff,
            deterministic=True,
        )
        ft_lane = int(force_target_lane) if force_target_lane is not None else ft_lane_pol

        _t_decision = time.time() - _t0

        # 3) Delegate to the IDEAM low-level pipeline with the chosen lane.
        out = ideam_agent_step(
            X0, X0_g, oa_prev, od_prev, last_X_prev, path_changed_prev,
            mpc_ctrl, util_obj, decision_obj, dynamics_obj,
            lane_left, lane_center, lane_right,
            force_target_lane=ft_lane,
            bypass_probe_guard=bypass_probe_guard,
        )

        # 4) Clear the override so downstream callers of the same mpc_ctrl
        #    (unlikely in benchmark but safer) don't inherit our speed.
        try:
            mpc_ctrl.set_target_speed_override(None)
        except Exception:
            pass

        out["t_decision"] = _t_decision + out.get("t_decision", 0.0)
        out["rl_action_obs"] = obs
        out["rl_target_speed"] = v_target
        out["rl_forced_lane"] = ft_lane
        return out
    except Exception as e:
        # Fall back to a zero-action propagate so the simulation keeps running.
        path_now = lane_from_global(X0_g)
        path_ego = [path1c, path2c, path3c][path_now]
        _, x_list, y_list, sample = get_path_info(path_now)
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0, [0.0, 0.0], dt, X0_g, path_ego, sample, x_list, y_list, boundary
        )
        return {
            "ok": False,
            "error": f"{e}\n{traceback.format_exc()}",
            "X0": X0_new,
            "X0_g": X0_g_new,
            "oa": [0.0] * mpc_ctrl.T,
            "od": [0.0] * mpc_ctrl.T,
            "last_X": last_X_prev,
            "path_changed": path_now,
            "path_d": path_ego,
            "sample": sample,
            "x_list": x_list,
            "y_list": y_list,
            "path_now": path_now,
            "path_dindex": path_now,
            "forced": force_target_lane is not None,
            "t_decision": float("nan"),
            "t_mpc": float("nan"),
        }


def ideam_agent_step(
    X0, X0_g, oa_prev, od_prev, last_X_prev, path_changed_prev,
    mpc_ctrl, util_obj, decision_obj, dynamics_obj,
    lane_left, lane_center, lane_right,
    force_target_lane=None, bypass_probe_guard=False
):
    path_now = lane_from_global(X0_g)
    path_ego = [path1c, path2c, path3c][path_now]
    start_group_str = {0: "L1", 1: "C1", 2: "R1"}[path_now]
    force_active = force_target_lane is not None

    try:
        if last_X_prev is None:
            ovx, ovy, owz, oS, oey, oepsi = clac_last_X(
                oa_prev, od_prev, mpc_ctrl.T, path_ego, dt, 6, X0, X0_g)
            last_X_prev = [ovx, ovy, owz, oS, oey, oepsi]

        _td0 = time.time()
        all_info = util_obj.get_alllane_lf(path_ego, X0_g, path_now, lane_left, lane_center, lane_right)
        group_dict, ego_group = util_obj.formulate_gap_group(
            path_now, last_X_prev, all_info, lane_left, lane_center, lane_right)

        desired_group = decision_obj.decision_making(group_dict, start_group_str)
        if force_active:
            desired_group = pick_group_for_lane(group_dict, int(force_target_lane))
        path_d, path_dindex, C_label, sample, x_list, y_list, X0 = Decision_info(
            X0, X0_g, path_center, sample_center, x_center, y_center,
            boundary, desired_group, path_ego, path_now
        )
        if force_active and path_dindex != int(force_target_lane):
            path_d, path_dindex, C_label, sample, x_list, y_list, X0 = force_path_target(
                path_now, int(force_target_lane), X0, X0_g
            )

        C_label_additive = util_obj.inquire_C_state(C_label, desired_group)
        C_label_virtual = C_label

        if C_label_additive == "Probe" and not (force_active and bypass_probe_guard):
            path_d, path_dindex, C_label_virtual = path_ego, path_now, "K"
            _, xc, yc, samplesc = get_path_info(path_dindex)
            X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)

        if path_changed_prev != path_dindex and last_X_prev is not None:
            mpc_ctrl.get_path_curvature(path=path_d)
            proj_s, proj_ey = path_to_path_proj(last_X_prev[3], last_X_prev[4], path_changed_prev, path_dindex)
            last_X_prev = [last_X_prev[0], last_X_prev[1], last_X_prev[2], proj_s, proj_ey, last_X_prev[5]]
        path_changed_prev = path_dindex
        _td1 = time.time()   # decision phase complete

        _tm0 = time.time()
        res = mpc_ctrl.iterative_linear_mpc_control(
            X0, oa_prev, od_prev, dt, None, None, C_label, X0_g, path_d, last_X_prev,
            path_now, ego_group, path_ego, desired_group,
            lane_left, lane_center, lane_right, path_dindex, C_label_additive, C_label_virtual
        )
        _tm1 = time.time()
        if res is None:
            raise RuntimeError("IDEAM MPC returned None")

        oa_cmd, od_cmd, ovx, ovy, owz, oS, oey, oepsi = res
        if oa_cmd is None or od_cmd is None:
            raise RuntimeError("IDEAM MPC returned empty controls")

        last_X = [ovx, ovy, owz, oS, oey, oepsi]
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0, [oa_cmd[0], od_cmd[0]], dt, X0_g, path_d, sample, x_list, y_list, boundary
        )

        return {
            "ok": True,
            "X0": X0_new,
            "X0_g": X0_g_new,
            "oa": oa_cmd,
            "od": od_cmd,
            "last_X": last_X,
            "path_changed": path_changed_prev,
            "path_d": path_d,
            "sample": sample,
            "x_list": x_list,
            "y_list": y_list,
            "t_decision": _td1 - _td0,
            "t_mpc":      _tm1 - _tm0,
            "path_now": path_now,
            "path_dindex": path_dindex,
            "forced": force_active,
        }
    except Exception as e:
        # Robust fallback to keep simulation running
        _, x_list, y_list, sample = get_path_info(path_now)
        oa_cmd = [0.0] * mpc_ctrl.T
        od_cmd = [0.0] * mpc_ctrl.T
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0, [0.0, 0.0], dt, X0_g, path_ego, sample, x_list, y_list, boundary
        )
        return {
            "ok": False,
            "error": f"{e}\n{traceback.format_exc()}",
            "X0": X0_new,
            "X0_g": X0_g_new,
            "oa": oa_cmd,
            "od": od_cmd,
            "last_X": last_X_prev,
            "path_changed": path_now,
            "path_d": path_ego,
            "sample": sample,
            "x_list": x_list,
            "y_list": y_list,
            "path_now": path_now,
            "path_dindex": path_now,
            "forced": force_active,
        }


def dream_agent_step(
    X0, X0_g, oa_prev, od_prev, last_X_prev, path_changed_prev,
    controller_obj, util_obj, decision_obj, dynamics_obj,
    lane_left, lane_center, lane_right, enable_decision_veto=True,
    force_target_lane=None, bypass_probe_guard=False, force_ignore_veto=True
):
    path_now = lane_from_global(X0_g)
    path_ego = [path1c, path2c, path3c][path_now]
    start_group_str = {0: "L1", 1: "C1", 2: "R1"}[path_now]
    force_active = force_target_lane is not None

    mpc_ctrl = controller_obj.mpc

    try:
        if last_X_prev is None:
            ovx, ovy, owz, oS, oey, oepsi = clac_last_X(
                oa_prev, od_prev, mpc_ctrl.T, path_ego, dt, 6, X0, X0_g)
            last_X_prev = [ovx, ovy, owz, oS, oey, oepsi]

        _td0_dr = time.time()
        all_info = util_obj.get_alllane_lf(path_ego, X0_g, path_now, lane_left, lane_center, lane_right)
        group_dict, ego_group = util_obj.formulate_gap_group(
            path_now, last_X_prev, all_info, lane_left, lane_center, lane_right)

        desired_group = decision_obj.decision_making(group_dict, start_group_str)
        if force_active:
            desired_group = pick_group_for_lane(group_dict, int(force_target_lane))
        path_d, path_dindex, C_label, sample, x_list, y_list, X0 = Decision_info(
            X0, X0_g, path_center, sample_center, x_center, y_center,
            boundary, desired_group, path_ego, path_now
        )
        if force_active and path_dindex != int(force_target_lane):
            path_d, path_dindex, C_label, sample, x_list, y_list, X0 = force_path_target(
                path_now, int(force_target_lane), X0, X0_g
            )

        C_label_additive = util_obj.inquire_C_state(C_label, desired_group)
        C_label_virtual = C_label

        if C_label_additive == "Probe" and not (force_active and bypass_probe_guard):
            path_d, path_dindex, C_label_virtual = path_ego, path_now, "K"
            _, xc, yc, samplesc = get_path_info(path_dindex)
            X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)

        if enable_decision_veto and C_label != "K" and not (force_active and force_ignore_veto):
            risk_score, allow, _ = controller_obj.evaluate_decision_risk(list(X0), path_now, path_dindex)
            if not allow:
                _ = risk_score  # silence lint intent
                path_d, path_dindex, C_label_virtual = path_ego, path_now, "K"
                _, xc, yc, samplesc = get_path_info(path_dindex)
                X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)

        if path_changed_prev != path_dindex and last_X_prev is not None:
            controller_obj.get_path_curvature(path=path_d)
            proj_s, proj_ey = path_to_path_proj(last_X_prev[3], last_X_prev[4], path_changed_prev, path_dindex)
            last_X_prev = [last_X_prev[0], last_X_prev[1], last_X_prev[2], proj_s, proj_ey, last_X_prev[5]]
        path_changed_prev = path_dindex
        _td1_dr = time.time()   # decision + risk-veto phase complete

        _tm0_dr = time.time()
        oa_cmd, od_cmd, ovx, ovy, owz, oS, oey, oepsi = controller_obj.solve_with_risk(
            X0, oa_prev, od_prev, dt, None, None, C_label, X0_g, path_d, last_X_prev,
            path_now, ego_group, path_ego, desired_group,
            lane_left, lane_center, lane_right,
            path_dindex, C_label_additive, C_label_virtual
        )
        _tm1_dr = time.time()
        if oa_cmd is None or od_cmd is None:
            raise RuntimeError("DREAM MPC returned empty controls")

        last_X = [ovx, ovy, owz, oS, oey, oepsi]
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0, [oa_cmd[0], od_cmd[0]], dt, X0_g, path_d, sample, x_list, y_list, boundary
        )

        return {
            "ok": True,
            "X0": X0_new,
            "X0_g": X0_g_new,
            "oa": oa_cmd,
            "od": od_cmd,
            "last_X": last_X,
            "path_changed": path_changed_prev,
            "path_d": path_d,
            "sample": sample,
            "x_list": x_list,
            "y_list": y_list,
            "path_now": path_now,
            "path_dindex": path_dindex,
            "forced": force_active,
            "t_decision": _td1_dr - _td0_dr,
            "t_mpc":      _tm1_dr - _tm0_dr,
        }
    except Exception as e:
        _, x_list, y_list, sample = get_path_info(path_now)
        oa_cmd = [0.0] * mpc_ctrl.T
        od_cmd = [0.0] * mpc_ctrl.T
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0, [0.0, 0.0], dt, X0_g, path_ego, sample, x_list, y_list, boundary
        )
        return {
            "ok": False,
            "error": f"{e}\n{traceback.format_exc()}",
            "X0": X0_new,
            "X0_g": X0_g_new,
            "oa": oa_cmd,
            "od": od_cmd,
            "last_X": last_X_prev,
            "path_changed": path_now,
            "path_d": path_ego,
            "sample": sample,
            "x_list": x_list,
            "y_list": y_list,
            "path_now": path_now,
            "path_dindex": path_now,
            "forced": force_active,
        }


def update_truck_state(truck_row, truck_dyn, leaders_center):
    prev = np.asarray(truck_row, dtype=float).copy()
    s = float(prev[0])
    v = max(0.0, float(prev[6]))

    s_ahead = None
    v_ahead = None
    ahead = [r for r in leaders_center if r is not None and float(r[0]) > s + 1.0]
    if ahead:
        ahead = sorted(ahead, key=lambda r: float(r[0]))
        s_ahead = float(ahead[0][0])
        v_ahead = float(ahead[0][6])

    def _fallback_row(v_hint=None, a_hint=0.0):
        s_fb = s + v * dt
        x_fb, y_fb = path2c(s_fb)
        psi_fb = path2c.get_theta_r(s_fb)
        v_fb = v if v_hint is None else max(0.0, float(v_hint))
        a_fb = float(a_hint) if np.isfinite(a_hint) else 0.0
        return np.array([s_fb, 0.0, 0.0, x_fb, y_fb, psi_fb, v_fb, a_fb], dtype=float)

    try:
        x_next, y_next, psi_next, v_next, _, a = truck_dyn.update_states(
            s, v, TRUCK_VD, s_ahead, v_ahead, path2c, steer_range
        )
        raw_vals = np.array([x_next, y_next, psi_next, v_next, a], dtype=float)
        if not np.all(np.isfinite(raw_vals)):
            return _fallback_row()

        s_next, ey_next, epsi_next = find_frenet_coord(
            path2c, x2c, y2c, samples2c, [x_next, y_next, psi_next]
        )
        row = np.array([s_next, ey_next, epsi_next, x_next, y_next, psi_next, v_next, a], dtype=float)
        if not np.all(np.isfinite(row)):
            return _fallback_row(v_hint=v_next, a_hint=a)

        # Keep truck anchored to centre lane in this synthetic scenario.
        if abs(float(ey_next)) > 2.0:
            return _fallback_row(v_hint=v_next, a_hint=a)
        return row
    except Exception:
        return _fallback_row()


# ============================================================================
# INITIALIZATION
# ============================================================================

print("=" * 72)
print("UNCERTAINTY MERGER DREAM (baseline IDEAM vs DREAM)")
print("=" * 72)
print(f"Integration mode: {INTEGRATION_MODE}")
print(f"Steps: {N_t}")
print(f"Active planners: {' | '.join(sorted(RUN_PLANNERS))}")
print()

Params = params()
dynamics = Dynamic(**Params)
decision_param = decision_params()

decision_base = decision(**decision_param)
decision_merger_base = decision(**decision_param)
decision_dream = decision(**decision_param)
decision_merger_dream = decision(**decision_param)
decision_merger_rl = decision(**decision_param)

util_cfg = util_params()

base_mpc = LMPC(**constraint_params())
base_utils = LeaderFollower_Uitl(**util_cfg)
base_mpc.set_util(base_utils)
base_mpc.get_path_curvature(path=path1c)

dream_controller = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        "mpc_cost": config_integration.mpc_risk_weight,
        "cbf_modulation": config_integration.cbf_alpha,
        "decision_threshold": config_integration.decision_risk_threshold,
    }
)
dream_utils = LeaderFollower_Uitl(**util_cfg)
dream_controller.set_util(dream_utils)
dream_controller.get_path_curvature(path=path1c)

# Baseline merger controller (IDEAM in top subplot).
base_merger_mpc = LMPC(**constraint_params())
base_merger_utils = LeaderFollower_Uitl(**util_cfg)
base_merger_mpc.set_util(base_merger_utils)
base_merger_mpc.get_path_curvature(path=path3c)

# DREAM-world merger controller (IDEAM).
dream_merger_mpc = LMPC(**constraint_params())
dream_merger_utils = LeaderFollower_Uitl(**util_cfg)
dream_merger_mpc.set_util(dream_merger_utils)
dream_merger_mpc.get_path_curvature(path=path3c)

RL_POLICY_MODE = str(CLI_ARGS.rl_policy_mode).lower()
if 'RL-PPO' in RUN_PLANNERS:
    if RL_POLICY_MODE == "decision":
        rl_policy, rl_ckpt_path = _load_rl_decision_policy(
            CLI_ARGS.rl_decision_checkpoint
        )
        if rl_policy is None:
            print("[rl-dec] decision checkpoint unavailable — falling back to PPO mode")
            RL_POLICY_MODE = "ppo"
            rl_policy, rl_ckpt_path = _load_rl_policy(CLI_ARGS.rl_checkpoint)
    else:
        rl_policy, rl_ckpt_path = _load_rl_policy(CLI_ARGS.rl_checkpoint)
    drift_rl = DRIFTInterface({0: path1c, 1: path2c, 2: path3c})
else:
    rl_policy, rl_ckpt_path, drift_rl = None, None, None

# RL-world merger controller (IDEAM).
rl_merger_mpc = LMPC(**constraint_params())
rl_merger_utils = LeaderFollower_Uitl(**util_cfg)
rl_merger_mpc.set_util(rl_merger_utils)
rl_merger_mpc.get_path_curvature(path=path3c)

# RL-decision ego-lane controller — only used when --rl-policy-mode=decision.
# The RL decision policy commands lane + cruise speed and delegates low-level
# tracking to this LMPC via ideam_agent_step (inside rl_decision_step).
if 'RL-PPO' in RUN_PLANNERS and RL_POLICY_MODE == "decision":
    rl_ego_mpc = LMPC(**constraint_params())
    rl_ego_utils = LeaderFollower_Uitl(**util_cfg)
    rl_ego_mpc.set_util(rl_ego_utils)
    rl_ego_mpc.get_path_curvature(path=path1c)
    decision_rl_ego = decision(**decision_param)
else:
    rl_ego_mpc = None
    rl_ego_utils = None
    decision_rl_ego = None

# DREAM-ADA controller — same MPC/CBF/decision logic, ADA source field.
ada_controller = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        "mpc_cost": config_integration.mpc_risk_weight,
        "cbf_modulation": config_integration.cbf_alpha,
        "decision_threshold": config_integration.decision_risk_threshold,
    }
)
ada_utils = LeaderFollower_Uitl(**util_cfg)
ada_controller.set_util(ada_utils)
ada_controller.get_path_curvature(path=path1c)
decision_ada = decision(**decision_param)

# ADA-world merger controller (IDEAM).
ada_merger_mpc = LMPC(**constraint_params())
ada_merger_utils = LeaderFollower_Uitl(**util_cfg)
ada_merger_mpc.set_util(ada_merger_utils)
ada_merger_mpc.get_path_curvature(path=path3c)
decision_merger_ada = decision(**decision_param)

# DREAM-APF controller — same MPC/CBF/decision logic, APF source field.
apf_controller = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        "mpc_cost": config_integration.mpc_risk_weight,
        "cbf_modulation": config_integration.cbf_alpha,
        "decision_threshold": config_integration.decision_risk_threshold,
    }
)
apf_utils = LeaderFollower_Uitl(**util_cfg)
apf_controller.set_util(apf_utils)
apf_controller.get_path_curvature(path=path1c)
decision_apf = decision(**decision_param)

# APF-world merger controller (IDEAM).
apf_merger_mpc = LMPC(**constraint_params())
apf_merger_utils = LeaderFollower_Uitl(**util_cfg)
apf_merger_mpc.set_util(apf_merger_utils)
apf_merger_mpc.get_path_curvature(path=path3c)
decision_merger_apf = decision(**decision_param)

# OA-CMPC standalone planner (MPC cost only — no CBF, no veto).
# Mirrors the paper's pure-MPC design: occlusion geometry informs the MPC
# cost penalty but no CBF safety filter or decision veto are applied.
oacmpc_controller = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        "mpc_cost": config_integration.mpc_risk_weight,
        "cbf_modulation": 0.0,           # no CBF
        "decision_threshold": float('inf'),  # no decision veto
    }
)
oacmpc_utils = LeaderFollower_Uitl(**util_cfg)
oacmpc_controller.set_util(oacmpc_utils)
oacmpc_controller.get_path_curvature(path=path1c)
decision_oacmpc = decision(**decision_param)

# OA-CMPC-world merger controller (IDEAM).
oacmpc_merger_mpc = LMPC(**constraint_params())
oacmpc_merger_utils = LeaderFollower_Uitl(**util_cfg)
oacmpc_merger_mpc.set_util(oacmpc_merger_utils)
oacmpc_merger_mpc.get_path_curvature(path=path3c)
decision_merger_oacmpc = decision(**decision_param)

# Ego states
X0_base = [EGO_V0, 0.0, 0.0, EGO_S0, 0.0, 0.0]
X0_g_base = [path1c(EGO_S0)[0], path1c(EGO_S0)[1], path1c.get_theta_r(EGO_S0)]
oa_base, od_base = 0.0, 0.0
last_X_base = None
path_changed_base = 0

X0_dream = [EGO_V0, 0.0, 0.0, EGO_S0, 0.0, 0.0]
X0_g_dream = [path1c(EGO_S0)[0], path1c(EGO_S0)[1], path1c.get_theta_r(EGO_S0)]
oa_dream, od_dream = 0.0, 0.0
last_X_dream = None
path_changed_dream = 0

X0_rl = [EGO_V0, 0.0, 0.0, EGO_S0, 0.0, 0.0]
X0_g_rl = [path1c(EGO_S0)[0], path1c(EGO_S0)[1], path1c.get_theta_r(EGO_S0)]
oa_rl, od_rl = 0.0, 0.0
last_X_rl = None
path_changed_rl = 0

# Baseline merger states (IDEAM).
X0_merger_base = [MERGER_BASE_V0, 0.0, 0.0, MERGER_BASE_S0, 0.0, 0.0]
X0_g_merger_base = [path3c(MERGER_BASE_S0)[0], path3c(MERGER_BASE_S0)[1], path3c.get_theta_r(MERGER_BASE_S0)]
oa_merger_base, od_merger_base = 0.0, 0.0
last_X_merger_base = None
path_changed_merger_base = 2

# DREAM-world merger states (IDEAM).
X0_merger_dream = [MERGER_DREAM_V0, 0.0, 0.0, MERGER_DREAM_S0, 0.0, 0.0]
X0_g_merger_dream = [path3c(MERGER_DREAM_S0)[0], path3c(MERGER_DREAM_S0)[1], path3c.get_theta_r(MERGER_DREAM_S0)]
oa_merger_dream, od_merger_dream = 0.0, 0.0
last_X_merger_dream = None
path_changed_merger_dream = 2

X0_merger_rl = [MERGER_DREAM_V0, 0.0, 0.0, MERGER_DREAM_S0, 0.0, 0.0]
X0_g_merger_rl = [path3c(MERGER_DREAM_S0)[0], path3c(MERGER_DREAM_S0)[1], path3c.get_theta_r(MERGER_DREAM_S0)]
oa_merger_rl, od_merger_rl = 0.0, 0.0
last_X_merger_rl = None
path_changed_merger_rl = 2

# ADA-world ego states (same initial conditions as DREAM world).
X0_ada = [EGO_V0, 0.0, 0.0, EGO_S0, 0.0, 0.0]
X0_g_ada = [path1c(EGO_S0)[0], path1c(EGO_S0)[1], path1c.get_theta_r(EGO_S0)]
oa_ada, od_ada = 0.0, 0.0
last_X_ada = None
path_changed_ada = 0

# ADA-world merger states (same initial as DREAM merger).
X0_merger_ada = [MERGER_DREAM_V0, 0.0, 0.0, MERGER_DREAM_S0, 0.0, 0.0]
X0_g_merger_ada = [path3c(MERGER_DREAM_S0)[0], path3c(MERGER_DREAM_S0)[1], path3c.get_theta_r(MERGER_DREAM_S0)]
oa_merger_ada, od_merger_ada = 0.0, 0.0
last_X_merger_ada = None
path_changed_merger_ada = 2

# APF-world ego states (same initial conditions as DREAM world).
X0_apf = [EGO_V0, 0.0, 0.0, EGO_S0, 0.0, 0.0]
X0_g_apf = [path1c(EGO_S0)[0], path1c(EGO_S0)[1], path1c.get_theta_r(EGO_S0)]
oa_apf, od_apf = 0.0, 0.0
last_X_apf = None
path_changed_apf = 0

# APF-world merger states (same initial as DREAM merger).
X0_merger_apf = [MERGER_DREAM_V0, 0.0, 0.0, MERGER_DREAM_S0, 0.0, 0.0]
X0_g_merger_apf = [path3c(MERGER_DREAM_S0)[0], path3c(MERGER_DREAM_S0)[1], path3c.get_theta_r(MERGER_DREAM_S0)]
oa_merger_apf, od_merger_apf = 0.0, 0.0
last_X_merger_apf = None
path_changed_merger_apf = 2

# OA-CMPC-world ego states (same initial conditions as DREAM world).
X0_oacmpc = [EGO_V0, 0.0, 0.0, EGO_S0, 0.0, 0.0]
X0_g_oacmpc = [path1c(EGO_S0)[0], path1c(EGO_S0)[1], path1c.get_theta_r(EGO_S0)]
oa_oacmpc, od_oacmpc = 0.0, 0.0
last_X_oacmpc = None
path_changed_oacmpc = 0

# OA-CMPC-world merger states (same initial as DREAM merger).
X0_merger_oacmpc = [MERGER_DREAM_V0, 0.0, 0.0, MERGER_DREAM_S0, 0.0, 0.0]
X0_g_merger_oacmpc = [path3c(MERGER_DREAM_S0)[0], path3c(MERGER_DREAM_S0)[1], path3c.get_theta_r(MERGER_DREAM_S0)]
oa_merger_oacmpc, od_merger_oacmpc = 0.0, 0.0
last_X_merger_oacmpc = None
path_changed_merger_oacmpc = 2

# Trucks (separate baseline and dream worlds, same dynamics/initial states).
truck_dyn = Curved_Road_Vehicle(**surrounding_params())
truck_x0 = path2c(TRUCK_S0)
truck_psi0 = path2c.get_theta_r(TRUCK_S0)
truck_row_base  = np.array([TRUCK_S0, 0.0, 0.0, truck_x0[0], truck_x0[1], truck_psi0, TRUCK_V0, 0.0], dtype=float)
truck_row_dream = np.array([TRUCK_S0, 0.0, 0.0, truck_x0[0], truck_x0[1], truck_psi0, TRUCK_V0, 0.0], dtype=float)
truck_row_rl    = np.array([TRUCK_S0, 0.0, 0.0, truck_x0[0], truck_x0[1], truck_psi0, TRUCK_V0, 0.0], dtype=float)
truck_row_ada   = np.array([TRUCK_S0, 0.0, 0.0, truck_x0[0], truck_x0[1], truck_psi0, TRUCK_V0, 0.0], dtype=float)
truck_row_apf   = np.array([TRUCK_S0, 0.0, 0.0, truck_x0[0], truck_x0[1], truck_psi0, TRUCK_V0, 0.0], dtype=float)
truck_row_oacmpc = np.array([TRUCK_S0, 0.0, 0.0, truck_x0[0], truck_x0[1], truck_psi0, TRUCK_V0, 0.0], dtype=float)

# Left-lane blocker: slow IDM car forcing both planners to consider LC to centre
blocker = LeftLaneBlocker(
    path=path1c, path_data=(x1c, y1c, samples1c),
    s_init=BLOCKER_S_INIT, vd=BLOCKER_VD, dt=dt, steer_range=steer_range)

# DRIFT setup for DREAM
drift = dream_controller.drift
print("Computing DRIFT road mask...")
step_mask = 50
left_edge = np.column_stack([x[::step_mask], y[::step_mask]])
right_edge = np.column_stack([x3[::step_mask], y3[::step_mask]])
road_polygon = np.vstack([left_edge, right_edge[::-1]])
road_path = MplPath(road_polygon)
grid_pts = np.column_stack([cfg.X.ravel(), cfg.Y.ravel()])
inside = road_path.contains_points(grid_pts).reshape(cfg.X.shape).astype(float)
road_mask = np.clip(_gf(inside, sigma=1.5), 0, 1)
drift.set_road_mask(road_mask)

print("DRIFT warm-up (3 s)...")
ego_drift_init = drift_create_vehicle(
    vid=0, x=X0_g_dream[0], y=X0_g_dream[1],
    vx=X0_dream[0] * math.cos(X0_g_dream[2]),
    vy=X0_dream[0] * math.sin(X0_g_dream[2]),
    vclass="car"
)
ego_drift_init["heading"] = X0_g_dream[2]
warm_vehicles = [
    row_to_drift_vehicle(truck_row_dream, vid=1, vclass="truck"),
    row_to_drift_vehicle(state_to_row(X0_merger_dream, X0_g_merger_dream), vid=2, vclass="car"),
    blocker.to_drift_vehicle(vid=3),   # blocker in DRIFT
]
drift.warmup(warm_vehicles, ego_drift_init, dt=dt, duration=3.0, substeps=3)

if 'RL-PPO' in RUN_PLANNERS:
    drift_rl.set_road_mask(road_mask)
    print("RL-PPO DRIFT warm-up (3 s)...")
    ego_drift_init_rl = drift_create_vehicle(
        vid=0, x=X0_g_rl[0], y=X0_g_rl[1],
        vx=X0_rl[0] * math.cos(X0_g_rl[2]),
        vy=X0_rl[0] * math.sin(X0_g_rl[2]),
        vclass="car"
    )
    ego_drift_init_rl["heading"] = X0_g_rl[2]
    warm_vehicles_rl = [
        row_to_drift_vehicle(truck_row_rl, vid=1, vclass="truck"),
        row_to_drift_vehicle(state_to_row(X0_merger_rl, X0_g_merger_rl), vid=2, vclass="car"),
        blocker.to_drift_vehicle(vid=3),
    ]
    drift_rl.warmup(warm_vehicles_rl, ego_drift_init_rl, dt=dt, duration=3.0, substeps=3)
else:
    print("RL-PPO warm-up skipped (arm not selected).")
print()

# DRIFT (ADA source)
drift_ada = ada_controller.drift
if 'ADA' in RUN_PLANNERS:
    drift_ada.set_road_mask(road_mask)
    print("ADA-DRIFT warm-up (3 s)...")
    ego_drift_init_ada = drift_create_vehicle(
        vid=0, x=X0_g_ada[0], y=X0_g_ada[1],
        vx=X0_ada[0] * math.cos(X0_g_ada[2]),
        vy=X0_ada[0] * math.sin(X0_g_ada[2]),
        vclass="car"
    )
    ego_drift_init_ada["heading"] = X0_g_ada[2]
    warm_vehicles_ada = [
        row_to_drift_vehicle(truck_row_ada, vid=1, vclass="truck"),
        row_to_drift_vehicle(state_to_row(X0_merger_ada, X0_g_merger_ada), vid=2, vclass="car"),
        blocker.to_drift_vehicle(vid=3),
    ]
    drift_ada.warmup(warm_vehicles_ada, ego_drift_init_ada, dt=dt, duration=3.0, substeps=3,
                     source_fn=compute_Q_ADA)
else:
    print("ADA warm-up skipped (arm not selected).")
print()

# DRIFT (APF source)
drift_apf = apf_controller.drift
if 'APF' in RUN_PLANNERS:
    drift_apf.set_road_mask(road_mask)
    print("APF-DRIFT warm-up (3 s)...")
    ego_drift_init_apf = drift_create_vehicle(
        vid=0, x=X0_g_apf[0], y=X0_g_apf[1],
        vx=X0_apf[0] * math.cos(X0_g_apf[2]),
        vy=X0_apf[0] * math.sin(X0_g_apf[2]),
        vclass="car"
    )
    ego_drift_init_apf["heading"] = X0_g_apf[2]
    warm_vehicles_apf = [
        row_to_drift_vehicle(truck_row_apf, vid=1, vclass="truck"),
        row_to_drift_vehicle(state_to_row(X0_merger_apf, X0_g_merger_apf), vid=2, vclass="car"),
        blocker.to_drift_vehicle(vid=3),
    ]
    drift_apf.warmup(warm_vehicles_apf, ego_drift_init_apf, dt=dt, duration=3.0, substeps=3,
                     source_fn=compute_Q_APF)
else:
    print("APF warm-up skipped (arm not selected).")
print()

# DRIFT (OA-CMPC source — MPC cost only, no CBF/veto)
drift_oacmpc = oacmpc_controller.drift
if 'OA-CMPC' in RUN_PLANNERS:
    drift_oacmpc.set_road_mask(road_mask)
    print("OA-CMPC-DRIFT warm-up (3 s)...")
    ego_drift_init_oacmpc = drift_create_vehicle(
        vid=0, x=X0_g_oacmpc[0], y=X0_g_oacmpc[1],
        vx=X0_oacmpc[0] * math.cos(X0_g_oacmpc[2]),
        vy=X0_oacmpc[0] * math.sin(X0_g_oacmpc[2]),
        vclass="car"
    )
    ego_drift_init_oacmpc["heading"] = X0_g_oacmpc[2]
    warm_vehicles_oacmpc = [
        row_to_drift_vehicle(truck_row_oacmpc, vid=1, vclass="truck"),
        row_to_drift_vehicle(state_to_row(X0_merger_oacmpc, X0_g_merger_oacmpc), vid=2, vclass="car"),
        blocker.to_drift_vehicle(vid=3),
    ]
    drift_oacmpc.warmup(warm_vehicles_oacmpc, ego_drift_init_oacmpc, dt=dt, duration=3.0, substeps=3,
                        source_fn=compute_Q_OACMPC)
else:
    print("OA-CMPC warm-up skipped (arm not selected).")
print()

print(f"Ego init left lane     s={EGO_S0:.1f}  vx={EGO_V0:.1f}")
print(f"Truck init center lane s={TRUCK_S0:.1f}  vx={TRUCK_V0:.1f}  vd={TRUCK_VD:.1f}")
print(f"Blocker init left lane s={blocker.s:.1f}  vd={BLOCKER_VD:.1f}")
print(f"Baseline merger init right lane s={MERGER_BASE_S0:.1f}  vx={MERGER_BASE_V0:.1f}")
print(f"DREAM merger init right lane s={MERGER_DREAM_S0:.1f}  vx={MERGER_DREAM_V0:.1f}")
print()


# ============================================================================
# MAIN LOOP
# ============================================================================

bar = Bar(max=max(1, N_t - 1))
risk_field = risk_field_rl = risk_field_ada = risk_field_apf = risk_field_oacmpc = None

# Dynamic panel layout — one column per selected arm
_PANEL_ORDER_M = ['IDEAM', 'DREAM', 'RL-PPO', 'OA-CMPC', 'ADA', 'APF']
_active_panels_m = [p for p in _PANEL_ORDER_M if p in RUN_PLANNERS]
_n_pan_m = max(1, len(_active_panels_m))
plt.figure(figsize=(6 * _n_pan_m, 8))

# ============================================================================
# METRIC BUFFERS
# ============================================================================
time_hist = []

base_s = []
base_vx = []
base_acc = []
base_lane_hist = []
base_dist_merger = []
base_min_dist_sur = []

dream_s = []
dream_vx = []
dream_acc = []
dream_lane_hist = []
dream_dist_merger = []
dream_min_dist_sur = []

rl_s = []
rl_vx = []
rl_acc = []
rl_lane_hist = []
rl_dist_merger = []
rl_min_dist_sur = []

base_merger_lane_hist = []
dream_merger_lane_hist = []
rl_merger_lane_hist = []
risk_at_ego_hist = []
risk_at_ego_rl_hist = []

ada_s = []
ada_vx = []
ada_acc = []
ada_lane_hist = []
ada_dist_merger = []
ada_min_dist_sur = []
ada_merger_lane_hist = []
risk_at_ego_ada_hist = []

apf_s = []
apf_vx = []
apf_acc = []
apf_lane_hist = []
apf_dist_merger = []
apf_min_dist_sur = []
apf_merger_lane_hist = []
risk_at_ego_apf_hist = []

oacmpc_s = []
oacmpc_vx = []
oacmpc_acc = []
oacmpc_lane_hist = []
oacmpc_dist_merger = []
oacmpc_min_dist_sur = []
oacmpc_merger_lane_hist = []
risk_at_ego_oacmpc_hist = []

# TTC to each arm's OccludedMerger
base_ttc  = []
dream_ttc = []
rl_ttc    = []
ada_ttc   = []
apf_ttc   = []
oacmpc_ttc = []

# REI integrand: R(ego, t) * v_x(t) evaluated on DREAM GVF field (common field)
base_rei_integrand  = []
dream_rei_integrand = []
rl_rei_integrand    = []
ada_rei_integrand   = []
apf_rei_integrand   = []
oacmpc_rei_integrand = []

# Reveal event: first step when base-arm merger enters centre lane
_merger_reveal_step = None

# ── Computational efficiency timing ──────────────────────────────────────────
# Per-step wall-clock times broken down into:
#   t_drift    : DRIFT PDE step (DREAM/ADA/APF only)
#   t_decision : gap formulation + decision making (+ risk veto for DREAM arms)
#   t_mpc      : CasADi/LMPC NLP solve (dominant cost)
#   t_total    : full agent step wall-clock
_t_ideam_total    = [];  _t_ideam_decision = [];  _t_ideam_mpc = []
_t_dream_total    = [];  _t_dream_decision = [];  _t_dream_mpc = [];  _t_dream_drift = []
_t_rl_total       = [];  _t_rl_decision    = [];  _t_rl_mpc    = [];  _t_rl_drift = []
_t_ada_total      = [];  _t_ada_decision   = [];  _t_ada_mpc   = [];  _t_ada_drift   = []
_t_apf_total      = [];  _t_apf_decision   = [];  _t_apf_mpc   = [];  _t_apf_drift   = []
_t_oacmpc_total   = [];  _t_oacmpc_decision = [];  _t_oacmpc_mpc = [];  _t_oacmpc_drift = []

# ── TTC helper ────────────────────────────────────────────────────────────
_CAR_LEN_TTC = 5.0
_TTC_CAP     = 60.0

def _ttc_to_agent(ego_pos, ego_vx, ag_x, ag_y, ag_vx):
    """Point-mass longitudinal TTC. All inputs cast to float (blocks CasADi)."""
    dx   = float(ag_x) - float(ego_pos[0])
    dy   = float(ag_y) - float(ego_pos[1])
    dist = math.hypot(dx, dy)
    dist_bumper = max(dist - _CAR_LEN_TTC, 0.0)
    if dist_bumper <= 0.0:
        return 0.0
    cos_th  = dx / max(dist, 0.1)
    v_close = (float(ego_vx) - float(ag_vx)) * cos_th
    if v_close <= 1e-3:
        return _TTC_CAP
    return float(min(dist_bumper / v_close, _TTC_CAP))


for i in range(N_t):
    bar.next()

    # ------------------------------------------------------------------------
    # Advance synthetic agents
    # ------------------------------------------------------------------------
    blocker.update()

    # ------------------------------------------------------------------------
    # Build baseline world lanes (all IDEAM-controlled).
    # ------------------------------------------------------------------------
    _bl_row = blocker.to_mpc_row()
    merger_base_row = state_to_row(X0_merger_base, X0_g_merger_base)
    merger_base_lane = lane_from_global(X0_g_merger_base)

    lane_left_base = stack_rows([_bl_row] + ([merger_base_row] if merger_base_lane == 0 else []))
    lane_center_base = stack_rows([truck_row_base] + ([merger_base_row] if merger_base_lane == 1 else []))
    lane_right_base = stack_rows([merger_base_row] if merger_base_lane == 2 else [])

    # ------------------------------------------------------------------------
    # Baseline ego (IDEAM).
    # ------------------------------------------------------------------------
    ego_blocker_gap_base = float(blocker.s) - float(X0_base[3])
    ego_base_ready_for_center = (
        i >= EGO_FORCE_CENTER_MIN_STEP and
        X0_base[3] > float(truck_row_base[0]) + EGO_LC_OVERTAKE_MARGIN and
        0.0 < ego_blocker_gap_base < EGO_BLOCKER_TRIGGER_GAP
    )
    lane_base_now = lane_from_global(X0_g_base)
    if lane_base_now == 0:
        force_ego_base = 1 if ego_base_ready_for_center else 0
    else:
        force_ego_base = None
    _t0_ideam = time.time()
    out_base = ideam_agent_step(
        X0_base, X0_g_base, oa_base, od_base, last_X_base, path_changed_base,
        base_mpc, base_utils, decision_base, dynamics,
        lane_left_base, lane_center_base, lane_right_base,
        force_target_lane=force_ego_base,
        bypass_probe_guard=True
    )
    _t_ideam_total.append(time.time() - _t0_ideam)
    _t_ideam_decision.append(out_base.get("t_decision", float("nan")))
    _t_ideam_mpc.append(out_base.get("t_mpc", float("nan")))
    X0_base, X0_g_base = out_base["X0"], out_base["X0_g"]
    oa_base, od_base = out_base["oa"], out_base["od"]
    last_X_base, path_changed_base = out_base["last_X"], out_base["path_changed"]
    if force_ego_base is not None:
        tgt = int(force_ego_base)
        if lane_from_global(X0_g_base) != tgt:
            blend = EGO_BASE_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
            X0_base, X0_g_base = blend_state_toward_lane(X0_base, X0_g_base, tgt, blend)

    # ------------------------------------------------------------------------
    # Baseline merger (IDEAM).
    # ------------------------------------------------------------------------
    base_row = state_to_row(X0_base, X0_g_base)
    base_lane = lane_from_global(X0_g_base)
    lane_left_merger = stack_rows([_bl_row] + ([base_row] if base_lane == 0 else []))
    lane_center_merger = stack_rows([truck_row_base] + ([base_row] if base_lane == 1 else []))
    lane_right_merger = stack_rows([base_row] if base_lane == 2 else [])

    merger_base_ready_for_center = (
        i >= MERGER_FORCE_CENTER_MIN_STEP and
        X0_merger_base[3] > float(truck_row_base[0]) + MERGER_LC_OVERTAKE_MARGIN and
        lane_from_global(X0_g_base) == 1
    )
    lane_merger_base_now = lane_from_global(X0_g_merger_base)
    if lane_merger_base_now == 2:
        force_merger_base = 1 if merger_base_ready_for_center else 2
    else:
        force_merger_base = None
    out_merger_base = ideam_agent_step(
        X0_merger_base, X0_g_merger_base,
        oa_merger_base, od_merger_base, last_X_merger_base, path_changed_merger_base,
        base_merger_mpc, base_merger_utils, decision_merger_base, dynamics,
        lane_left_merger, lane_center_merger, lane_right_merger,
        force_target_lane=force_merger_base,
        bypass_probe_guard=True
    )
    X0_merger_base, X0_g_merger_base = out_merger_base["X0"], out_merger_base["X0_g"]
    oa_merger_base, od_merger_base = out_merger_base["oa"], out_merger_base["od"]
    last_X_merger_base, path_changed_merger_base = out_merger_base["last_X"], out_merger_base["path_changed"]
    if force_merger_base is not None:
        tgt = int(force_merger_base)
        if lane_from_global(X0_g_merger_base) != tgt:
            blend = MERGER_BASE_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
            X0_merger_base, X0_g_merger_base = blend_state_toward_lane(
                X0_merger_base, X0_g_merger_base, tgt, blend
            )

    # ------------------------------------------------------------------------
    # Build DREAM world lanes (DREAM ego + IDEAM merger).
    # ------------------------------------------------------------------------
    merger_dream_row = state_to_row(X0_merger_dream, X0_g_merger_dream)
    merger_dream_lane = lane_from_global(X0_g_merger_dream)
    lane_left_dream = stack_rows([_bl_row] + ([merger_dream_row] if merger_dream_lane == 0 else []))
    lane_center_dream = stack_rows([truck_row_dream] + ([merger_dream_row] if merger_dream_lane == 1 else []))
    lane_right_dream = stack_rows([merger_dream_row] if merger_dream_lane == 2 else [])

    # ------------------------------------------------------------------------
    # DRIFT step (GVF) for DREAM world.
    # ------------------------------------------------------------------------
    ego_drift = drift_create_vehicle(
        vid=0, x=X0_g_dream[0], y=X0_g_dream[1],
        vx=X0_dream[0] * math.cos(X0_g_dream[2]) - X0_dream[1] * math.sin(X0_g_dream[2]),
        vy=X0_dream[0] * math.sin(X0_g_dream[2]) + X0_dream[1] * math.cos(X0_g_dream[2]),
        vclass="car"
    )
    ego_drift["heading"] = X0_g_dream[2]

    drift_vehicles = [
        row_to_drift_vehicle(truck_row_dream, vid=1, vclass="truck"),
        row_to_drift_vehicle(merger_dream_row, vid=2, vclass="car"),
        blocker.to_drift_vehicle(vid=3),
    ]
    _t_drift0 = time.time()
    risk_field = drift.step(drift_vehicles, ego_drift, dt=dt, substeps=3)
    _t_dream_drift.append(time.time() - _t_drift0)

    # ------------------------------------------------------------------------
    # Build RL-PPO world lanes.
    # ------------------------------------------------------------------------
    merger_rl_row = state_to_row(X0_merger_rl, X0_g_merger_rl)
    merger_rl_lane = lane_from_global(X0_g_merger_rl)
    lane_left_rl = stack_rows([_bl_row] + ([merger_rl_row] if merger_rl_lane == 0 else []))
    lane_center_rl = stack_rows([truck_row_rl] + ([merger_rl_row] if merger_rl_lane == 1 else []))
    lane_right_rl = stack_rows([merger_rl_row] if merger_rl_lane == 2 else [])

    # ------------------------------------------------------------------------
    # DRIFT step (GVF) for RL-PPO world.
    # ------------------------------------------------------------------------
    if 'RL-PPO' in RUN_PLANNERS:
        ego_drift_rl = drift_create_vehicle(
            vid=0, x=X0_g_rl[0], y=X0_g_rl[1],
            vx=X0_rl[0] * math.cos(X0_g_rl[2]) - X0_rl[1] * math.sin(X0_g_rl[2]),
            vy=X0_rl[0] * math.sin(X0_g_rl[2]) + X0_rl[1] * math.cos(X0_g_rl[2]),
            vclass="car"
        )
        ego_drift_rl["heading"] = X0_g_rl[2]
        drift_vehicles_rl = [
            row_to_drift_vehicle(truck_row_rl, vid=1, vclass="truck"),
            row_to_drift_vehicle(merger_rl_row, vid=2, vclass="car"),
            blocker.to_drift_vehicle(vid=3),
        ]
        _t_rl_drift0 = time.time()
        risk_field_rl = drift_rl.step(drift_vehicles_rl, ego_drift_rl, dt=dt, substeps=3)
        _t_rl_drift.append(time.time() - _t_rl_drift0)
    else:
        _t_rl_drift.append(0.0)
        risk_field_rl = None

    # ------------------------------------------------------------------------
    # Build ADA world lanes.
    # ------------------------------------------------------------------------
    merger_ada_row = state_to_row(X0_merger_ada, X0_g_merger_ada)
    merger_ada_lane = lane_from_global(X0_g_merger_ada)
    lane_left_ada    = stack_rows([_bl_row] + ([merger_ada_row] if merger_ada_lane == 0 else []))
    lane_center_ada  = stack_rows([truck_row_ada] + ([merger_ada_row] if merger_ada_lane == 1 else []))
    lane_right_ada   = stack_rows([merger_ada_row] if merger_ada_lane == 2 else [])

    # ------------------------------------------------------------------------
    # DRIFT step (ADA) for ADA world.
    # ------------------------------------------------------------------------
    if 'ADA' in RUN_PLANNERS:
        ego_drift_ada_v = drift_create_vehicle(
            vid=0, x=X0_g_ada[0], y=X0_g_ada[1],
            vx=X0_ada[0] * math.cos(X0_g_ada[2]) - X0_ada[1] * math.sin(X0_g_ada[2]),
            vy=X0_ada[0] * math.sin(X0_g_ada[2]) + X0_ada[1] * math.cos(X0_g_ada[2]),
            vclass="car"
        )
        ego_drift_ada_v["heading"] = X0_g_ada[2]
        drift_vehicles_ada = [
            row_to_drift_vehicle(truck_row_ada, vid=1, vclass="truck"),
            row_to_drift_vehicle(merger_ada_row, vid=2, vclass="car"),
            blocker.to_drift_vehicle(vid=3),
        ]
        _t_ada_drift0 = time.time()
        risk_field_ada = drift_ada.step(drift_vehicles_ada, ego_drift_ada_v,
                                        dt=dt, substeps=3, source_fn=compute_Q_ADA)
        _t_ada_drift.append(time.time() - _t_ada_drift0)
    else:
        _t_ada_drift.append(0.0)

    # ------------------------------------------------------------------------
    # DREAM ego (risk-aware).
    # ------------------------------------------------------------------------
    ego_blocker_gap_dream = float(blocker.s) - float(X0_dream[3])
    ego_dream_ready_for_center = (
        i >= EGO_FORCE_CENTER_MIN_STEP and
        X0_dream[3] > float(truck_row_dream[0]) + EGO_LC_OVERTAKE_MARGIN and
        0.0 < ego_blocker_gap_dream < EGO_BLOCKER_TRIGGER_GAP
    )
    lane_dream_now = lane_from_global(X0_g_dream)
    if lane_dream_now == 0:
        force_ego_dream = 1 if ego_dream_ready_for_center else 0
    else:
        force_ego_dream = None
    _t0_dream = time.time()
    out_dream = dream_agent_step(
        X0_dream, X0_g_dream, oa_dream, od_dream, last_X_dream, path_changed_dream,
        dream_controller, dream_utils, decision_dream, dynamics,
        lane_left_dream, lane_center_dream, lane_right_dream,
        enable_decision_veto=config_integration.enable_decision_veto,
        force_target_lane=force_ego_dream,
        bypass_probe_guard=True,
        force_ignore_veto=True
    )
    _t_dream_total.append(time.time() - _t0_dream)
    _t_dream_decision.append(out_dream.get("t_decision", float("nan")))
    _t_dream_mpc.append(out_dream.get("t_mpc", float("nan")))
    X0_dream, X0_g_dream = out_dream["X0"], out_dream["X0_g"]
    oa_dream, od_dream = out_dream["oa"], out_dream["od"]
    last_X_dream, path_changed_dream = out_dream["last_X"], out_dream["path_changed"]
    if force_ego_dream is not None:
        tgt = int(force_ego_dream)
        if lane_from_global(X0_g_dream) != tgt:
            blend = EGO_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
            X0_dream, X0_g_dream = blend_state_toward_lane(X0_dream, X0_g_dream, tgt, blend)

    # ------------------------------------------------------------------------
    # RL-PPO ego (learned low-level policy on DRIFT observation).
    # ------------------------------------------------------------------------
    if 'RL-PPO' in RUN_PLANNERS:
        ego_blocker_gap_rl = float(blocker.s) - float(X0_rl[3])
        ego_rl_ready_for_center = (
            i >= EGO_FORCE_CENTER_MIN_STEP and
            X0_rl[3] > float(truck_row_rl[0]) + EGO_LC_OVERTAKE_MARGIN and
            0.0 < ego_blocker_gap_rl < EGO_BLOCKER_TRIGGER_GAP
        )
        lane_rl_now = lane_from_global(X0_g_rl)
        if lane_rl_now == 0:
            force_ego_rl = 1 if ego_rl_ready_for_center else 0
        else:
            force_ego_rl = None
        _t0_rl = time.time()
        if RL_POLICY_MODE == "decision" and rl_ego_mpc is not None and rl_policy is not None:
            out_rl = rl_decision_step(
                X0_rl, X0_g_rl, oa_rl, od_rl, last_X_rl, path_changed_rl,
                rl_policy, rl_ego_mpc, rl_ego_utils, decision_rl_ego, dynamics,
                lane_left_rl, lane_center_rl, lane_right_rl,
                force_target_lane=force_ego_rl,
                bypass_probe_guard=True,
                v_ref=float(RL_DEFAULT_CONFIG.TARGET_SPEED),
            )
        else:
            out_rl = rl_agent_step(
                X0_rl, X0_g_rl, oa_rl, od_rl, rl_policy, drift_rl, dynamics,
                lane_left_rl, lane_center_rl, lane_right_rl,
                force_target_lane=force_ego_rl
            )
        _t_rl_total.append(time.time() - _t0_rl)
        _t_rl_decision.append(out_rl.get("t_decision", float("nan")))
        _t_rl_mpc.append(out_rl.get("t_mpc", float("nan")))
        X0_rl, X0_g_rl = out_rl["X0"], out_rl["X0_g"]
        oa_rl, od_rl = out_rl["oa"], out_rl["od"]
        last_X_rl, path_changed_rl = out_rl["last_X"], out_rl["path_changed"]
        if force_ego_rl is not None:
            tgt = int(force_ego_rl)
            if lane_from_global(X0_g_rl) != tgt:
                blend = EGO_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
                X0_rl, X0_g_rl = blend_state_toward_lane(X0_rl, X0_g_rl, tgt, blend)
    else:
        _t_rl_total.append(0.0)
        _t_rl_decision.append(float("nan"))
        _t_rl_mpc.append(float("nan"))
        out_rl = {"ok": True, "error": ""}

    # ------------------------------------------------------------------------
    # DREAM-world merger (IDEAM).
    # ------------------------------------------------------------------------
    dream_row = state_to_row(X0_dream, X0_g_dream)
    dream_lane = lane_from_global(X0_g_dream)
    lane_left_merger_dream = stack_rows([_bl_row] + ([dream_row] if dream_lane == 0 else []))
    lane_center_merger_dream = stack_rows([truck_row_dream] + ([dream_row] if dream_lane == 1 else []))
    lane_right_merger_dream = stack_rows([dream_row] if dream_lane == 2 else [])

    merger_dream_ready_for_center = (
        i >= MERGER_FORCE_CENTER_MIN_STEP and
        X0_merger_dream[3] > float(truck_row_dream[0]) + MERGER_LC_OVERTAKE_MARGIN and
        lane_from_global(X0_g_dream) == 1
    )
    lane_merger_dream_now = lane_from_global(X0_g_merger_dream)
    if lane_merger_dream_now == 2:
        force_merger_dream = 1 if merger_dream_ready_for_center else 2
    else:
        force_merger_dream = None
    out_merger_dream = ideam_agent_step(
        X0_merger_dream, X0_g_merger_dream,
        oa_merger_dream, od_merger_dream, last_X_merger_dream, path_changed_merger_dream,
        dream_merger_mpc, dream_merger_utils, decision_merger_dream, dynamics,
        lane_left_merger_dream, lane_center_merger_dream, lane_right_merger_dream,
        force_target_lane=force_merger_dream,
        bypass_probe_guard=True
    )
    X0_merger_dream, X0_g_merger_dream = out_merger_dream["X0"], out_merger_dream["X0_g"]
    oa_merger_dream, od_merger_dream = out_merger_dream["oa"], out_merger_dream["od"]
    last_X_merger_dream, path_changed_merger_dream = out_merger_dream["last_X"], out_merger_dream["path_changed"]
    if force_merger_dream is not None:
        tgt = int(force_merger_dream)
        if lane_from_global(X0_g_merger_dream) != tgt:
            blend = MERGER_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
            X0_merger_dream, X0_g_merger_dream = blend_state_toward_lane(
                X0_merger_dream, X0_g_merger_dream, tgt, blend
            )

    # ------------------------------------------------------------------------
    # RL-world merger (IDEAM).
    # ------------------------------------------------------------------------
    if 'RL-PPO' in RUN_PLANNERS:
        rl_row = state_to_row(X0_rl, X0_g_rl)
        rl_lane = lane_from_global(X0_g_rl)
        lane_left_merger_rl = stack_rows([_bl_row] + ([rl_row] if rl_lane == 0 else []))
        lane_center_merger_rl = stack_rows([truck_row_rl] + ([rl_row] if rl_lane == 1 else []))
        lane_right_merger_rl = stack_rows([rl_row] if rl_lane == 2 else [])

        merger_rl_ready_for_center = (
            i >= MERGER_FORCE_CENTER_MIN_STEP and
            X0_merger_rl[3] > float(truck_row_rl[0]) + MERGER_LC_OVERTAKE_MARGIN and
            lane_from_global(X0_g_rl) == 1
        )
        lane_merger_rl_now = lane_from_global(X0_g_merger_rl)
        if lane_merger_rl_now == 2:
            force_merger_rl = 1 if merger_rl_ready_for_center else 2
        else:
            force_merger_rl = None
        out_merger_rl = ideam_agent_step(
            X0_merger_rl, X0_g_merger_rl,
            oa_merger_rl, od_merger_rl, last_X_merger_rl, path_changed_merger_rl,
            rl_merger_mpc, rl_merger_utils, decision_merger_rl, dynamics,
            lane_left_merger_rl, lane_center_merger_rl, lane_right_merger_rl,
            force_target_lane=force_merger_rl,
            bypass_probe_guard=True
        )
        X0_merger_rl, X0_g_merger_rl = out_merger_rl["X0"], out_merger_rl["X0_g"]
        oa_merger_rl, od_merger_rl = out_merger_rl["oa"], out_merger_rl["od"]
        last_X_merger_rl, path_changed_merger_rl = out_merger_rl["last_X"], out_merger_rl["path_changed"]
        if force_merger_rl is not None:
            tgt = int(force_merger_rl)
            if lane_from_global(X0_g_merger_rl) != tgt:
                blend = MERGER_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
                X0_merger_rl, X0_g_merger_rl = blend_state_toward_lane(
                    X0_merger_rl, X0_g_merger_rl, tgt, blend
                )
    else:
        out_merger_rl = {"ok": True, "error": ""}

    # ------------------------------------------------------------------------
    # ADA ego (risk-aware, ADA source).
    # ------------------------------------------------------------------------
    if 'ADA' in RUN_PLANNERS:
        ego_blocker_gap_ada = float(blocker.s) - float(X0_ada[3])
        ego_ada_ready_for_center = (
            i >= EGO_FORCE_CENTER_MIN_STEP and
            X0_ada[3] > float(truck_row_ada[0]) + EGO_LC_OVERTAKE_MARGIN and
            0.0 < ego_blocker_gap_ada < EGO_BLOCKER_TRIGGER_GAP
        )
        lane_ada_now = lane_from_global(X0_g_ada)
        if lane_ada_now == 0:
            force_ego_ada = 1 if ego_ada_ready_for_center else 0
        else:
            force_ego_ada = None
        _t0_ada = time.time()
        out_ada = dream_agent_step(
            X0_ada, X0_g_ada, oa_ada, od_ada, last_X_ada, path_changed_ada,
            ada_controller, ada_utils, decision_ada, dynamics,
            lane_left_ada, lane_center_ada, lane_right_ada,
            enable_decision_veto=config_integration.enable_decision_veto,
            force_target_lane=force_ego_ada,
            bypass_probe_guard=True,
            force_ignore_veto=True
        )
        _t_ada_total.append(time.time() - _t0_ada)
        _t_ada_decision.append(out_ada.get("t_decision", float("nan")))
        _t_ada_mpc.append(out_ada.get("t_mpc", float("nan")))
        X0_ada, X0_g_ada = out_ada["X0"], out_ada["X0_g"]
        oa_ada, od_ada = out_ada["oa"], out_ada["od"]
        last_X_ada, path_changed_ada = out_ada["last_X"], out_ada["path_changed"]
        if force_ego_ada is not None:
            tgt = int(force_ego_ada)
            if lane_from_global(X0_g_ada) != tgt:
                blend = EGO_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
                X0_ada, X0_g_ada = blend_state_toward_lane(X0_ada, X0_g_ada, tgt, blend)
    else:
        _t_ada_total.append(0.0)
        _t_ada_decision.append(float("nan"))
        _t_ada_mpc.append(float("nan"))
        out_ada = {"ok": True, "error": ""}

    # ------------------------------------------------------------------------
    # ADA-world merger (IDEAM).
    # ------------------------------------------------------------------------
    if 'ADA' in RUN_PLANNERS:
        ada_row = state_to_row(X0_ada, X0_g_ada)
        ada_lane = lane_from_global(X0_g_ada)
        lane_left_merger_ada   = stack_rows([_bl_row] + ([ada_row] if ada_lane == 0 else []))
        lane_center_merger_ada = stack_rows([truck_row_ada] + ([ada_row] if ada_lane == 1 else []))
        lane_right_merger_ada  = stack_rows([ada_row] if ada_lane == 2 else [])

        merger_ada_ready_for_center = (
            i >= MERGER_FORCE_CENTER_MIN_STEP and
            X0_merger_ada[3] > float(truck_row_ada[0]) + MERGER_LC_OVERTAKE_MARGIN and
            lane_from_global(X0_g_ada) == 1
        )
        lane_merger_ada_now = lane_from_global(X0_g_merger_ada)
        if lane_merger_ada_now == 2:
            force_merger_ada = 1 if merger_ada_ready_for_center else 2
        else:
            force_merger_ada = None
        out_merger_ada = ideam_agent_step(
            X0_merger_ada, X0_g_merger_ada,
            oa_merger_ada, od_merger_ada, last_X_merger_ada, path_changed_merger_ada,
            ada_merger_mpc, ada_merger_utils, decision_merger_ada, dynamics,
            lane_left_merger_ada, lane_center_merger_ada, lane_right_merger_ada,
            force_target_lane=force_merger_ada,
            bypass_probe_guard=True
        )
        X0_merger_ada, X0_g_merger_ada = out_merger_ada["X0"], out_merger_ada["X0_g"]
        oa_merger_ada, od_merger_ada = out_merger_ada["oa"], out_merger_ada["od"]
        last_X_merger_ada, path_changed_merger_ada = out_merger_ada["last_X"], out_merger_ada["path_changed"]
        if force_merger_ada is not None:
            tgt = int(force_merger_ada)
            if lane_from_global(X0_g_merger_ada) != tgt:
                blend = MERGER_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
                X0_merger_ada, X0_g_merger_ada = blend_state_toward_lane(
                    X0_merger_ada, X0_g_merger_ada, tgt, blend)
    else:
        out_merger_ada = {"ok": True, "error": ""}

    # ------------------------------------------------------------------------
    # Build APF world lanes.
    # ------------------------------------------------------------------------
    merger_apf_row = state_to_row(X0_merger_apf, X0_g_merger_apf)
    merger_apf_lane = lane_from_global(X0_g_merger_apf)
    lane_left_apf    = stack_rows([_bl_row] + ([merger_apf_row] if merger_apf_lane == 0 else []))
    lane_center_apf  = stack_rows([truck_row_apf] + ([merger_apf_row] if merger_apf_lane == 1 else []))
    lane_right_apf   = stack_rows([merger_apf_row] if merger_apf_lane == 2 else [])

    # ------------------------------------------------------------------------
    # DRIFT step (APF) for APF world.
    # ------------------------------------------------------------------------
    if 'APF' in RUN_PLANNERS:
        ego_drift_apf_v = drift_create_vehicle(
            vid=0, x=X0_g_apf[0], y=X0_g_apf[1],
            vx=X0_apf[0] * math.cos(X0_g_apf[2]) - X0_apf[1] * math.sin(X0_g_apf[2]),
            vy=X0_apf[0] * math.sin(X0_g_apf[2]) + X0_apf[1] * math.cos(X0_g_apf[2]),
            vclass="car"
        )
        ego_drift_apf_v["heading"] = X0_g_apf[2]
        drift_vehicles_apf = [
            row_to_drift_vehicle(truck_row_apf, vid=1, vclass="truck"),
            row_to_drift_vehicle(merger_apf_row, vid=2, vclass="car"),
            blocker.to_drift_vehicle(vid=3),
        ]
        _t_apf_drift0 = time.time()
        risk_field_apf = drift_apf.step(drift_vehicles_apf, ego_drift_apf_v,
                                        dt=dt, substeps=3, source_fn=compute_Q_APF)
        _t_apf_drift.append(time.time() - _t_apf_drift0)
    else:
        _t_apf_drift.append(0.0)

    # ------------------------------------------------------------------------
    # APF ego (risk-aware, APF source).
    # ------------------------------------------------------------------------
    if 'APF' in RUN_PLANNERS:
        ego_blocker_gap_apf = float(blocker.s) - float(X0_apf[3])
        ego_apf_ready_for_center = (
            i >= EGO_FORCE_CENTER_MIN_STEP and
            X0_apf[3] > float(truck_row_apf[0]) + EGO_LC_OVERTAKE_MARGIN and
            0.0 < ego_blocker_gap_apf < EGO_BLOCKER_TRIGGER_GAP
        )
        lane_apf_now = lane_from_global(X0_g_apf)
        if lane_apf_now == 0:
            force_ego_apf = 1 if ego_apf_ready_for_center else 0
        else:
            force_ego_apf = None
        _t0_apf = time.time()
        out_apf = dream_agent_step(
            X0_apf, X0_g_apf, oa_apf, od_apf, last_X_apf, path_changed_apf,
            apf_controller, apf_utils, decision_apf, dynamics,
            lane_left_apf, lane_center_apf, lane_right_apf,
            enable_decision_veto=config_integration.enable_decision_veto,
            force_target_lane=force_ego_apf,
            bypass_probe_guard=True,
            force_ignore_veto=True
        )
        _t_apf_total.append(time.time() - _t0_apf)
        _t_apf_decision.append(out_apf.get("t_decision", float("nan")))
        _t_apf_mpc.append(out_apf.get("t_mpc", float("nan")))
        X0_apf, X0_g_apf = out_apf["X0"], out_apf["X0_g"]
        oa_apf, od_apf = out_apf["oa"], out_apf["od"]
        last_X_apf, path_changed_apf = out_apf["last_X"], out_apf["path_changed"]
        if force_ego_apf is not None:
            tgt = int(force_ego_apf)
            if lane_from_global(X0_g_apf) != tgt:
                blend = EGO_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
                X0_apf, X0_g_apf = blend_state_toward_lane(X0_apf, X0_g_apf, tgt, blend)
    else:
        _t_apf_total.append(0.0)
        _t_apf_decision.append(float("nan"))
        _t_apf_mpc.append(float("nan"))
        out_apf = {"ok": True, "error": ""}

    # ------------------------------------------------------------------------
    # APF-world merger (IDEAM).
    # ------------------------------------------------------------------------
    if 'APF' in RUN_PLANNERS:
        apf_row = state_to_row(X0_apf, X0_g_apf)
        apf_lane = lane_from_global(X0_g_apf)
        lane_left_merger_apf   = stack_rows([_bl_row] + ([apf_row] if apf_lane == 0 else []))
        lane_center_merger_apf = stack_rows([truck_row_apf] + ([apf_row] if apf_lane == 1 else []))
        lane_right_merger_apf  = stack_rows([apf_row] if apf_lane == 2 else [])

        merger_apf_ready_for_center = (
            i >= MERGER_FORCE_CENTER_MIN_STEP and
            X0_merger_apf[3] > float(truck_row_apf[0]) + MERGER_LC_OVERTAKE_MARGIN and
            lane_from_global(X0_g_apf) == 1
        )
        lane_merger_apf_now = lane_from_global(X0_g_merger_apf)
        if lane_merger_apf_now == 2:
            force_merger_apf = 1 if merger_apf_ready_for_center else 2
        else:
            force_merger_apf = None
        out_merger_apf = ideam_agent_step(
            X0_merger_apf, X0_g_merger_apf,
            oa_merger_apf, od_merger_apf, last_X_merger_apf, path_changed_merger_apf,
            apf_merger_mpc, apf_merger_utils, decision_merger_apf, dynamics,
            lane_left_merger_apf, lane_center_merger_apf, lane_right_merger_apf,
            force_target_lane=force_merger_apf,
            bypass_probe_guard=True
        )
        X0_merger_apf, X0_g_merger_apf = out_merger_apf["X0"], out_merger_apf["X0_g"]
        oa_merger_apf, od_merger_apf = out_merger_apf["oa"], out_merger_apf["od"]
        last_X_merger_apf, path_changed_merger_apf = out_merger_apf["last_X"], out_merger_apf["path_changed"]
        if force_merger_apf is not None:
            tgt = int(force_merger_apf)
            if lane_from_global(X0_g_merger_apf) != tgt:
                blend = MERGER_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
                X0_merger_apf, X0_g_merger_apf = blend_state_toward_lane(
                    X0_merger_apf, X0_g_merger_apf, tgt, blend)
    else:
        out_merger_apf = {"ok": True, "error": ""}

    # ------------------------------------------------------------------------
    # Build OA-CMPC world lanes.
    # ------------------------------------------------------------------------
    merger_oacmpc_row = state_to_row(X0_merger_oacmpc, X0_g_merger_oacmpc)
    merger_oacmpc_lane = lane_from_global(X0_g_merger_oacmpc)
    lane_left_oacmpc    = stack_rows([_bl_row] + ([merger_oacmpc_row] if merger_oacmpc_lane == 0 else []))
    lane_center_oacmpc  = stack_rows([truck_row_oacmpc] + ([merger_oacmpc_row] if merger_oacmpc_lane == 1 else []))
    lane_right_oacmpc   = stack_rows([merger_oacmpc_row] if merger_oacmpc_lane == 2 else [])

    # ------------------------------------------------------------------------
    # DRIFT step (OA-CMPC source) for OA-CMPC world.
    # ------------------------------------------------------------------------
    if 'OA-CMPC' in RUN_PLANNERS:
        ego_drift_oacmpc_v = drift_create_vehicle(
            vid=0, x=X0_g_oacmpc[0], y=X0_g_oacmpc[1],
            vx=X0_oacmpc[0] * math.cos(X0_g_oacmpc[2]) - X0_oacmpc[1] * math.sin(X0_g_oacmpc[2]),
            vy=X0_oacmpc[0] * math.sin(X0_g_oacmpc[2]) + X0_oacmpc[1] * math.cos(X0_g_oacmpc[2]),
            vclass="car"
        )
        ego_drift_oacmpc_v["heading"] = X0_g_oacmpc[2]
        drift_vehicles_oacmpc = [
            row_to_drift_vehicle(truck_row_oacmpc, vid=1, vclass="truck"),
            row_to_drift_vehicle(merger_oacmpc_row, vid=2, vclass="car"),
            blocker.to_drift_vehicle(vid=3),
        ]
        _t_oacmpc_drift0 = time.time()
        risk_field_oacmpc = drift_oacmpc.step(drift_vehicles_oacmpc, ego_drift_oacmpc_v,
                                              dt=dt, substeps=3, source_fn=compute_Q_OACMPC)
        _t_oacmpc_drift.append(time.time() - _t_oacmpc_drift0)
    else:
        _t_oacmpc_drift.append(0.0)

    # ------------------------------------------------------------------------
    # OA-CMPC ego (risk-aware, OA-CMPC source).
    # ------------------------------------------------------------------------
    if 'OA-CMPC' in RUN_PLANNERS:
        ego_blocker_gap_oacmpc = float(blocker.s) - float(X0_oacmpc[3])
        ego_oacmpc_ready_for_center = (
            i >= EGO_FORCE_CENTER_MIN_STEP and
            X0_oacmpc[3] > float(truck_row_oacmpc[0]) + EGO_LC_OVERTAKE_MARGIN and
            0.0 < ego_blocker_gap_oacmpc < EGO_BLOCKER_TRIGGER_GAP
        )
        lane_oacmpc_now = lane_from_global(X0_g_oacmpc)
        if lane_oacmpc_now == 0:
            force_ego_oacmpc = 1 if ego_oacmpc_ready_for_center else 0
        else:
            force_ego_oacmpc = None
        _t0_oacmpc = time.time()
        out_oacmpc = dream_agent_step(
            X0_oacmpc, X0_g_oacmpc, oa_oacmpc, od_oacmpc, last_X_oacmpc, path_changed_oacmpc,
            oacmpc_controller, oacmpc_utils, decision_oacmpc, dynamics,
            lane_left_oacmpc, lane_center_oacmpc, lane_right_oacmpc,
            enable_decision_veto=config_integration.enable_decision_veto,
            force_target_lane=force_ego_oacmpc,
            bypass_probe_guard=True,
            force_ignore_veto=True
        )
        _t_oacmpc_total.append(time.time() - _t0_oacmpc)
        _t_oacmpc_decision.append(out_oacmpc.get("t_decision", float("nan")))
        _t_oacmpc_mpc.append(out_oacmpc.get("t_mpc", float("nan")))
        X0_oacmpc, X0_g_oacmpc = out_oacmpc["X0"], out_oacmpc["X0_g"]
        oa_oacmpc, od_oacmpc = out_oacmpc["oa"], out_oacmpc["od"]
        last_X_oacmpc, path_changed_oacmpc = out_oacmpc["last_X"], out_oacmpc["path_changed"]
        if force_ego_oacmpc is not None:
            tgt = int(force_ego_oacmpc)
            if lane_from_global(X0_g_oacmpc) != tgt:
                blend = EGO_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
                X0_oacmpc, X0_g_oacmpc = blend_state_toward_lane(X0_oacmpc, X0_g_oacmpc, tgt, blend)
    else:
        _t_oacmpc_total.append(0.0)
        _t_oacmpc_decision.append(float("nan"))
        _t_oacmpc_mpc.append(float("nan"))
        out_oacmpc = {"ok": True, "error": ""}

    # ------------------------------------------------------------------------
    # OA-CMPC-world merger (IDEAM).
    # ------------------------------------------------------------------------
    if 'OA-CMPC' in RUN_PLANNERS:
        oacmpc_row = state_to_row(X0_oacmpc, X0_g_oacmpc)
        oacmpc_lane = lane_from_global(X0_g_oacmpc)
        lane_left_merger_oacmpc   = stack_rows([_bl_row] + ([oacmpc_row] if oacmpc_lane == 0 else []))
        lane_center_merger_oacmpc = stack_rows([truck_row_oacmpc] + ([oacmpc_row] if oacmpc_lane == 1 else []))
        lane_right_merger_oacmpc  = stack_rows([oacmpc_row] if oacmpc_lane == 2 else [])

        merger_oacmpc_ready_for_center = (
            i >= MERGER_FORCE_CENTER_MIN_STEP and
            X0_merger_oacmpc[3] > float(truck_row_oacmpc[0]) + MERGER_LC_OVERTAKE_MARGIN and
            lane_from_global(X0_g_oacmpc) == 1
        )
        lane_merger_oacmpc_now = lane_from_global(X0_g_merger_oacmpc)
        if lane_merger_oacmpc_now == 2:
            force_merger_oacmpc = 1 if merger_oacmpc_ready_for_center else 2
        else:
            force_merger_oacmpc = None
        out_merger_oacmpc = ideam_agent_step(
            X0_merger_oacmpc, X0_g_merger_oacmpc,
            oa_merger_oacmpc, od_merger_oacmpc, last_X_merger_oacmpc, path_changed_merger_oacmpc,
            oacmpc_merger_mpc, oacmpc_merger_utils, decision_merger_oacmpc, dynamics,
            lane_left_merger_oacmpc, lane_center_merger_oacmpc, lane_right_merger_oacmpc,
            force_target_lane=force_merger_oacmpc,
            bypass_probe_guard=True
        )
        X0_merger_oacmpc, X0_g_merger_oacmpc = out_merger_oacmpc["X0"], out_merger_oacmpc["X0_g"]
        oa_merger_oacmpc, od_merger_oacmpc = out_merger_oacmpc["oa"], out_merger_oacmpc["od"]
        last_X_merger_oacmpc, path_changed_merger_oacmpc = out_merger_oacmpc["last_X"], out_merger_oacmpc["path_changed"]
        if force_merger_oacmpc is not None:
            tgt = int(force_merger_oacmpc)
            if lane_from_global(X0_g_merger_oacmpc) != tgt:
                blend = MERGER_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
                X0_merger_oacmpc, X0_g_merger_oacmpc = blend_state_toward_lane(
                    X0_merger_oacmpc, X0_g_merger_oacmpc, tgt, blend)
    else:
        out_merger_oacmpc = {"ok": True, "error": ""}

    # ------------------------------------------------------------------------
    # Update trucks in their own worlds.
    # ------------------------------------------------------------------------
    leaders_center_base = []
    if lane_from_global(X0_g_base) == 1:
        leaders_center_base.append(state_to_row(X0_base, X0_g_base))
    if lane_from_global(X0_g_merger_base) == 1:
        leaders_center_base.append(state_to_row(X0_merger_base, X0_g_merger_base))
    truck_row_base = update_truck_state(truck_row_base, truck_dyn, leaders_center_base)

    leaders_center_dream = []
    if lane_from_global(X0_g_dream) == 1:
        leaders_center_dream.append(state_to_row(X0_dream, X0_g_dream))
    if lane_from_global(X0_g_merger_dream) == 1:
        leaders_center_dream.append(state_to_row(X0_merger_dream, X0_g_merger_dream))
    truck_row_dream = update_truck_state(truck_row_dream, truck_dyn, leaders_center_dream)

    leaders_center_rl = []
    if lane_from_global(X0_g_rl) == 1:
        leaders_center_rl.append(state_to_row(X0_rl, X0_g_rl))
    if lane_from_global(X0_g_merger_rl) == 1:
        leaders_center_rl.append(state_to_row(X0_merger_rl, X0_g_merger_rl))
    truck_row_rl = update_truck_state(truck_row_rl, truck_dyn, leaders_center_rl)

    leaders_center_ada = []
    if lane_from_global(X0_g_ada) == 1:
        leaders_center_ada.append(state_to_row(X0_ada, X0_g_ada))
    if lane_from_global(X0_g_merger_ada) == 1:
        leaders_center_ada.append(state_to_row(X0_merger_ada, X0_g_merger_ada))
    truck_row_ada = update_truck_state(truck_row_ada, truck_dyn, leaders_center_ada)

    leaders_center_apf = []
    if lane_from_global(X0_g_apf) == 1:
        leaders_center_apf.append(state_to_row(X0_apf, X0_g_apf))
    if lane_from_global(X0_g_merger_apf) == 1:
        leaders_center_apf.append(state_to_row(X0_merger_apf, X0_g_merger_apf))
    truck_row_apf = update_truck_state(truck_row_apf, truck_dyn, leaders_center_apf)

    leaders_center_oacmpc = []
    if lane_from_global(X0_g_oacmpc) == 1:
        leaders_center_oacmpc.append(state_to_row(X0_oacmpc, X0_g_oacmpc))
    if lane_from_global(X0_g_merger_oacmpc) == 1:
        leaders_center_oacmpc.append(state_to_row(X0_merger_oacmpc, X0_g_merger_oacmpc))
    truck_row_oacmpc = update_truck_state(truck_row_oacmpc, truck_dyn, leaders_center_oacmpc)

    # ------------------------------------------------------------------------
    # Metrics (sample after all agent/state updates).
    # ------------------------------------------------------------------------
    blocker_row_now = blocker.to_mpc_row()
    merger_row_now_base   = state_to_row(X0_merger_base, X0_g_merger_base)
    merger_row_now_dream  = state_to_row(X0_merger_dream, X0_g_merger_dream)
    merger_row_now_rl     = state_to_row(X0_merger_rl, X0_g_merger_rl)
    merger_row_now_ada    = state_to_row(X0_merger_ada, X0_g_merger_ada)
    merger_row_now_apf    = state_to_row(X0_merger_apf, X0_g_merger_apf)
    merger_row_now_oacmpc = state_to_row(X0_merger_oacmpc, X0_g_merger_oacmpc)

    time_hist.append(i * dt)

    base_s.append(progress_on_reference(X0_g_base))
    base_vx.append(float(X0_base[0]))
    base_lane_hist.append(int(lane_from_global(X0_g_base)))
    base_merger_lane_hist.append(int(lane_from_global(X0_g_merger_base)))
    base_dist_merger.append(float(math.hypot(
        float(X0_g_base[0]) - float(merger_row_now_base[3]),
        float(X0_g_base[1]) - float(merger_row_now_base[4]),
    )))
    base_min_dist_sur.append(min_center_distance(
        X0_g_base, [truck_row_base, merger_row_now_base, blocker_row_now]
    ))
    if hasattr(oa_base, "__len__"):
        base_acc.append(float(oa_base[0]) if len(oa_base) > 0 else 0.0)
    else:
        base_acc.append(float(oa_base))

    dream_s.append(progress_on_reference(X0_g_dream))
    dream_vx.append(float(X0_dream[0]))
    dream_lane_hist.append(int(lane_from_global(X0_g_dream)))
    dream_merger_lane_hist.append(int(lane_from_global(X0_g_merger_dream)))
    dream_dist_merger.append(float(math.hypot(
        float(X0_g_dream[0]) - float(merger_row_now_dream[3]),
        float(X0_g_dream[1]) - float(merger_row_now_dream[4]),
    )))
    dream_min_dist_sur.append(min_center_distance(
        X0_g_dream, [truck_row_dream, merger_row_now_dream, blocker_row_now]
    ))
    if hasattr(oa_dream, "__len__"):
        dream_acc.append(float(oa_dream[0]) if len(oa_dream) > 0 else 0.0)
    else:
        dream_acc.append(float(oa_dream))

    try:
        risk_at_ego_hist.append(float(drift.get_risk_cartesian(float(X0_g_dream[0]), float(X0_g_dream[1]))))
    except Exception:
        risk_at_ego_hist.append(float("nan"))

    if 'RL-PPO' in RUN_PLANNERS:
        rl_s.append(progress_on_reference(X0_g_rl))
        rl_vx.append(float(X0_rl[0]))
        rl_lane_hist.append(int(lane_from_global(X0_g_rl)))
        rl_merger_lane_hist.append(int(lane_from_global(X0_g_merger_rl)))
        rl_dist_merger.append(float(math.hypot(
            float(X0_g_rl[0]) - float(merger_row_now_rl[3]),
            float(X0_g_rl[1]) - float(merger_row_now_rl[4]),
        )))
        rl_min_dist_sur.append(min_center_distance(
            X0_g_rl, [truck_row_rl, merger_row_now_rl, blocker_row_now]
        ))
        if hasattr(oa_rl, "__len__"):
            rl_acc.append(float(oa_rl[0]) if len(oa_rl) > 0 else 0.0)
        else:
            rl_acc.append(float(oa_rl))
        try:
            risk_at_ego_rl_hist.append(float(drift_rl.get_risk_cartesian(
                float(X0_g_rl[0]), float(X0_g_rl[1]))))
        except Exception:
            risk_at_ego_rl_hist.append(float("nan"))
    else:
        rl_s.append(float('nan')); rl_vx.append(float('nan'))
        rl_acc.append(float('nan')); rl_lane_hist.append(0)
        rl_merger_lane_hist.append(0)
        rl_dist_merger.append(float('nan')); rl_min_dist_sur.append(float('nan'))
        risk_at_ego_rl_hist.append(float('nan'))

    if 'ADA' in RUN_PLANNERS:
        ada_s.append(progress_on_reference(X0_g_ada))
        ada_vx.append(float(X0_ada[0]))
        ada_lane_hist.append(int(lane_from_global(X0_g_ada)))
        ada_merger_lane_hist.append(int(lane_from_global(X0_g_merger_ada)))
        ada_dist_merger.append(float(math.hypot(
            float(X0_g_ada[0]) - float(merger_row_now_ada[3]),
            float(X0_g_ada[1]) - float(merger_row_now_ada[4]),
        )))
        ada_min_dist_sur.append(min_center_distance(
            X0_g_ada, [truck_row_ada, merger_row_now_ada, blocker_row_now]
        ))
        if hasattr(oa_ada, "__len__"):
            ada_acc.append(float(oa_ada[0]) if len(oa_ada) > 0 else 0.0)
        else:
            ada_acc.append(float(oa_ada))
        try:
            risk_at_ego_ada_hist.append(float(drift_ada.get_risk_cartesian(
                float(X0_g_ada[0]), float(X0_g_ada[1]))))
        except Exception:
            risk_at_ego_ada_hist.append(float("nan"))
    else:
        ada_s.append(float('nan')); ada_vx.append(float('nan'))
        ada_acc.append(float('nan')); ada_lane_hist.append(0)
        ada_merger_lane_hist.append(0)
        ada_dist_merger.append(float('nan')); ada_min_dist_sur.append(float('nan'))
        risk_at_ego_ada_hist.append(float('nan'))

    if 'APF' in RUN_PLANNERS:
        apf_s.append(progress_on_reference(X0_g_apf))
        apf_vx.append(float(X0_apf[0]))
        apf_lane_hist.append(int(lane_from_global(X0_g_apf)))
        apf_merger_lane_hist.append(int(lane_from_global(X0_g_merger_apf)))
        apf_dist_merger.append(float(math.hypot(
            float(X0_g_apf[0]) - float(merger_row_now_apf[3]),
            float(X0_g_apf[1]) - float(merger_row_now_apf[4]),
        )))
        apf_min_dist_sur.append(min_center_distance(
            X0_g_apf, [truck_row_apf, merger_row_now_apf, blocker_row_now]
        ))
        if hasattr(oa_apf, "__len__"):
            apf_acc.append(float(oa_apf[0]) if len(oa_apf) > 0 else 0.0)
        else:
            apf_acc.append(float(oa_apf))
        try:
            risk_at_ego_apf_hist.append(float(drift_apf.get_risk_cartesian(
                float(X0_g_apf[0]), float(X0_g_apf[1]))))
        except Exception:
            risk_at_ego_apf_hist.append(float("nan"))
    else:
        apf_s.append(float('nan')); apf_vx.append(float('nan'))
        apf_acc.append(float('nan')); apf_lane_hist.append(0)
        apf_merger_lane_hist.append(0)
        apf_dist_merger.append(float('nan')); apf_min_dist_sur.append(float('nan'))
        risk_at_ego_apf_hist.append(float('nan'))

    if 'OA-CMPC' in RUN_PLANNERS:
        oacmpc_s.append(progress_on_reference(X0_g_oacmpc))
        oacmpc_vx.append(float(X0_oacmpc[0]))
        oacmpc_lane_hist.append(int(lane_from_global(X0_g_oacmpc)))
        oacmpc_merger_lane_hist.append(int(lane_from_global(X0_g_merger_oacmpc)))
        oacmpc_dist_merger.append(float(math.hypot(
            float(X0_g_oacmpc[0]) - float(merger_row_now_oacmpc[3]),
            float(X0_g_oacmpc[1]) - float(merger_row_now_oacmpc[4]),
        )))
        oacmpc_min_dist_sur.append(min_center_distance(
            X0_g_oacmpc, [truck_row_oacmpc, merger_row_now_oacmpc, blocker_row_now]
        ))
        if hasattr(oa_oacmpc, "__len__"):
            oacmpc_acc.append(float(oa_oacmpc[0]) if len(oa_oacmpc) > 0 else 0.0)
        else:
            oacmpc_acc.append(float(oa_oacmpc))
        try:
            risk_at_ego_oacmpc_hist.append(float(drift_oacmpc.get_risk_cartesian(
                float(X0_g_oacmpc[0]), float(X0_g_oacmpc[1]))))
        except Exception:
            risk_at_ego_oacmpc_hist.append(float("nan"))
    else:
        oacmpc_s.append(float('nan')); oacmpc_vx.append(float('nan'))
        oacmpc_acc.append(float('nan')); oacmpc_lane_hist.append(0)
        oacmpc_merger_lane_hist.append(0)
        oacmpc_dist_merger.append(float('nan')); oacmpc_min_dist_sur.append(float('nan'))
        risk_at_ego_oacmpc_hist.append(float('nan'))

    # -- Reveal tracking (base-arm merger enters centre lane) ----------------
    if _merger_reveal_step is None and lane_from_global(X0_g_merger_base) == 1:
        _merger_reveal_step = i

    # -- TTC to each arm's OccludedMerger ------------------------------------
    base_ttc.append(_ttc_to_agent(
        X0_g_base,  float(X0_base[0]),
        float(merger_row_now_base[3]),  float(merger_row_now_base[4]),
        float(X0_merger_base[0])))
    dream_ttc.append(_ttc_to_agent(
        X0_g_dream, float(X0_dream[0]),
        float(merger_row_now_dream[3]), float(merger_row_now_dream[4]),
        float(X0_merger_dream[0])))
    rl_ttc.append(_ttc_to_agent(
        X0_g_rl, float(X0_rl[0]),
        float(merger_row_now_rl[3]), float(merger_row_now_rl[4]),
        float(X0_merger_rl[0])) if 'RL-PPO' in RUN_PLANNERS else float('nan'))
    ada_ttc.append(_ttc_to_agent(
        X0_g_ada, float(X0_ada[0]),
        float(merger_row_now_ada[3]), float(merger_row_now_ada[4]),
        float(X0_merger_ada[0])) if 'ADA' in RUN_PLANNERS else float('nan'))
    apf_ttc.append(_ttc_to_agent(
        X0_g_apf, float(X0_apf[0]),
        float(merger_row_now_apf[3]), float(merger_row_now_apf[4]),
        float(X0_merger_apf[0])) if 'APF' in RUN_PLANNERS else float('nan'))
    oacmpc_ttc.append(_ttc_to_agent(
        X0_g_oacmpc, float(X0_oacmpc[0]),
        float(merger_row_now_oacmpc[3]), float(merger_row_now_oacmpc[4]),
        float(X0_merger_oacmpc[0])) if 'OA-CMPC' in RUN_PLANNERS else float('nan'))

    # -- REI integrand: GVF risk at each ego × speed (common evaluation field)
    _r_base  = float(drift.get_risk_cartesian(float(X0_g_base[0]),  float(X0_g_base[1])))
    _r_dream = risk_at_ego_hist[-1]   # already computed above
    _r_rl = (float(drift_rl.get_risk_cartesian(float(X0_g_rl[0]), float(X0_g_rl[1])))
             if 'RL-PPO' in RUN_PLANNERS else float('nan'))
    _r_ada  = (float(drift.get_risk_cartesian(float(X0_g_ada[0]),   float(X0_g_ada[1])))
               if 'ADA' in RUN_PLANNERS else float('nan'))
    _r_apf  = (float(drift.get_risk_cartesian(float(X0_g_apf[0]),   float(X0_g_apf[1])))
               if 'APF' in RUN_PLANNERS else float('nan'))
    _r_oacmpc_rei = (float(drift_oacmpc.get_risk_cartesian(float(X0_g_oacmpc[0]), float(X0_g_oacmpc[1])))
                     if 'OA-CMPC' in RUN_PLANNERS else float('nan'))
    base_rei_integrand.append(_r_base  * float(X0_base[0]))
    dream_rei_integrand.append(_r_dream * float(X0_dream[0]))
    rl_rei_integrand.append(float('nan') if 'RL-PPO' not in RUN_PLANNERS else _r_rl * float(X0_rl[0]))
    ada_rei_integrand.append(float('nan') if 'ADA' not in RUN_PLANNERS else _r_ada * float(X0_ada[0]))
    apf_rei_integrand.append(float('nan') if 'APF' not in RUN_PLANNERS else _r_apf * float(X0_apf[0]))
    oacmpc_rei_integrand.append(float('nan') if 'OA-CMPC' not in RUN_PLANNERS else _r_oacmpc_rei * float(X0_oacmpc[0]))

    # ------------------------------------------------------------------------
    # Horizons for plotting
    # ------------------------------------------------------------------------
    h_base = build_horizon(
        X0_base, X0_g_base, oa_base, od_base,
        out_base["path_d"], out_base["sample"], out_base["x_list"], out_base["y_list"],
        dynamics, dt, boundary
    )
    h_dream = build_horizon(
        X0_dream, X0_g_dream, oa_dream, od_dream,
        out_dream["path_d"], out_dream["sample"], out_dream["x_list"], out_dream["y_list"],
        dynamics, dt, boundary
    )
    if 'RL-PPO' in RUN_PLANNERS and out_rl.get("ok", False):
        h_rl = build_horizon(
            X0_rl, X0_g_rl, oa_rl, od_rl,
            out_rl["path_d"], out_rl["sample"], out_rl["x_list"], out_rl["y_list"],
            dynamics, dt, boundary
        )
    else:
        h_rl = None
    if 'ADA' in RUN_PLANNERS:
        h_ada = build_horizon(
            X0_ada, X0_g_ada, oa_ada, od_ada,
            out_ada["path_d"], out_ada["sample"], out_ada["x_list"], out_ada["y_list"],
            dynamics, dt, boundary
        )
    else:
        h_ada = None
    if 'APF' in RUN_PLANNERS:
        h_apf = build_horizon(
            X0_apf, X0_g_apf, oa_apf, od_apf,
            out_apf["path_d"], out_apf["sample"], out_apf["x_list"], out_apf["y_list"],
            dynamics, dt, boundary
        )
    else:
        h_apf = None
    if 'OA-CMPC' in RUN_PLANNERS:
        h_oacmpc = build_horizon(
            X0_oacmpc, X0_g_oacmpc, oa_oacmpc, od_oacmpc,
            out_oacmpc["path_d"], out_oacmpc["sample"], out_oacmpc["x_list"], out_oacmpc["y_list"],
            dynamics, dt, boundary
        )
    else:
        h_oacmpc = None

    # ------------------------------------------------------------------------
    # Plot — dynamic panels for selected arms only
    # ------------------------------------------------------------------------
    fig = plt.gcf()
    fig.clf()

    # Pre-compute per-arm data (always computed; unused if arm not selected)
    xr_base   = [X0_g_base[0]   - x_area, X0_g_base[0]   + x_area]
    yr_base   = [X0_g_base[1]   - y_area, X0_g_base[1]   + y_area]
    xr_dream  = [X0_g_dream[0]  - x_area, X0_g_dream[0]  + x_area]
    yr_dream  = [X0_g_dream[1]  - y_area, X0_g_dream[1]  + y_area]
    xr_rl     = [X0_g_rl[0]     - x_area, X0_g_rl[0]     + x_area]
    yr_rl     = [X0_g_rl[1]     - y_area, X0_g_rl[1]     + y_area]
    xr_ada    = [X0_g_ada[0]    - x_area, X0_g_ada[0]    + x_area]
    yr_ada    = [X0_g_ada[1]    - y_area, X0_g_ada[1]    + y_area]
    xr_apf    = [X0_g_apf[0]    - x_area, X0_g_apf[0]    + x_area]
    yr_apf    = [X0_g_apf[1]    - y_area, X0_g_apf[1]    + y_area]
    xr_oacmpc = [X0_g_oacmpc[0] - x_area, X0_g_oacmpc[0] + x_area]
    yr_oacmpc = [X0_g_oacmpc[1] - y_area, X0_g_oacmpc[1] + y_area]

    merger_row_plot_base   = state_to_row(X0_merger_base,   X0_g_merger_base)
    merger_row_plot_dream  = state_to_row(X0_merger_dream,  X0_g_merger_dream)
    merger_row_plot_rl     = state_to_row(X0_merger_rl,     X0_g_merger_rl)
    merger_row_plot_ada    = state_to_row(X0_merger_ada,    X0_g_merger_ada)
    merger_row_plot_apf    = state_to_row(X0_merger_apf,    X0_g_merger_apf)
    merger_row_plot_oacmpc = state_to_row(X0_merger_oacmpc, X0_g_merger_oacmpc)

    _cf_last = None
    _ax_last = None
    for _pi_m, _pname_m in enumerate(_active_panels_m):
        _ax_pm = fig.add_subplot(1, _n_pan_m, _pi_m + 1)
        _ax_last = _ax_pm
        if _pname_m == 'IDEAM':
            draw_panel(
                _ax_pm, X0_base, X0_g_base, truck_row_base, merger_row_plot_base, _bl_row,
                title="IDEAM",
                x_range=xr_base, y_range=yr_base,
                risk_field=None, horizon=h_base, ego_color=EGO_IDEAM_COLOR
            )
        elif _pname_m == 'DREAM':
            _cf = draw_panel(
                _ax_pm, X0_dream, X0_g_dream, truck_row_dream, merger_row_plot_dream, _bl_row,
                title="DREAM",
                x_range=xr_dream, y_range=yr_dream,
                risk_field=risk_field, horizon=h_dream, ego_color=EGO_DREAM_COLOR
            )
            if _cf is not None: _cf_last = _cf
        elif _pname_m == 'RL-PPO':
            _cf = draw_panel(
                _ax_pm, X0_rl, X0_g_rl, truck_row_rl, merger_row_plot_rl, _bl_row,
                title="RL-PPO",
                x_range=xr_rl, y_range=yr_rl,
                risk_field=risk_field_rl, horizon=h_rl, ego_color=EGO_RL_COLOR
            )
            if _cf is not None: _cf_last = _cf
        elif _pname_m == 'OA-CMPC':
            # OA-CMPC: no risk field overlay (pure-MPC, no DRIFT visualisation)
            draw_panel(
                _ax_pm, X0_oacmpc, X0_g_oacmpc, truck_row_oacmpc, merger_row_plot_oacmpc, _bl_row,
                title="OA-CMPC",
                x_range=xr_oacmpc, y_range=yr_oacmpc,
                risk_field=None, horizon=h_oacmpc, ego_color=EGO_OACMPC_COLOR
            )
        elif _pname_m == 'ADA':
            _cf = draw_panel(
                _ax_pm, X0_ada, X0_g_ada, truck_row_ada, merger_row_plot_ada, _bl_row,
                title="ADA",
                x_range=xr_ada, y_range=yr_ada,
                risk_field=risk_field_ada, horizon=h_ada, ego_color=EGO_ADA_COLOR
            )
            if _cf is not None: _cf_last = _cf
        elif _pname_m == 'APF':
            _cf = draw_panel(
                _ax_pm, X0_apf, X0_g_apf, truck_row_apf, merger_row_plot_apf, _bl_row,
                title="APF",
                x_range=xr_apf, y_range=yr_apf,
                risk_field=risk_field_apf, horizon=h_apf, ego_color=EGO_APF_COLOR
            )
            if _cf is not None: _cf_last = _cf

    if _cf_last is not None and _ax_last is not None:
        cbar = fig.colorbar(_cf_last, ax=_ax_last, orientation="vertical", pad=0.02, fraction=0.035)
        cbar.set_label("Risk Level", fontsize=9, weight="bold")
        cbar.ax.tick_params(labelsize=8, colors="black")

    if SAVE_FRAMES:
        plt.savefig(os.path.join(save_dir, f"{i}.png"), dpi=SAVE_DPI)

    if i % max(1, LOG_EVERY) == 0:
        print(
            f"[{i:03d}] base_lane={lane_from_global(X0_g_base)} "
            f"dream_lane={lane_from_global(X0_g_dream)} "
            f"rl_lane={lane_from_global(X0_g_rl)} "
            f"ada_lane={lane_from_global(X0_g_ada)} "
            f"apf_lane={lane_from_global(X0_g_apf)} "
            f"oacmpc_lane={lane_from_global(X0_g_oacmpc)} "
            f"base_merger={lane_from_global(X0_g_merger_base)} "
            f"dream_merger={lane_from_global(X0_g_merger_dream)} "
            f"rl_merger={lane_from_global(X0_g_merger_rl)} "
            f"ada_merger={lane_from_global(X0_g_merger_ada)} "
            f"apf_merger={lane_from_global(X0_g_merger_apf)} "
            f"oacmpc_merger={lane_from_global(X0_g_merger_oacmpc)}"
        )
        if not out_base["ok"]:
            print(f"  [warn] baseline fallback: {out_base.get('error','')}")
        if not out_dream["ok"]:
            print(f"  [warn] dream fallback: {out_dream.get('error','')}")
        if not out_rl["ok"]:
            print(f"  [warn] rl fallback: {out_rl.get('error','')}")
        if not out_ada["ok"]:
            print(f"  [warn] ada fallback: {out_ada.get('error','')}")
        if not out_apf["ok"]:
            print(f"  [warn] apf fallback: {out_apf.get('error','')}")
        if not out_oacmpc["ok"]:
            print(f"  [warn] oacmpc fallback: {out_oacmpc.get('error','')}")
        if not out_merger_base["ok"]:
            print(f"  [warn] base merger fallback: {out_merger_base.get('error','')}")
        if not out_merger_dream["ok"]:
            print(f"  [warn] dream merger fallback: {out_merger_dream.get('error','')}")
        if not out_merger_rl["ok"]:
            print(f"  [warn] rl merger fallback: {out_merger_rl.get('error','')}")
        if not out_merger_ada["ok"]:
            print(f"  [warn] ada merger fallback: {out_merger_ada.get('error','')}")
        if not out_merger_apf["ok"]:
            print(f"  [warn] apf merger fallback: {out_merger_apf.get('error','')}")
        if not out_merger_oacmpc["ok"]:
            print(f"  [warn] oacmpc merger fallback: {out_merger_oacmpc.get('error','')}")

bar.finish()
print()
print("Simulation complete.")
if SAVE_FRAMES:
    print(f"Frames saved to: {save_dir}")
else:
    print("Frame saving disabled (--save-frames false).")

# ============================================================================
# POST-LOOP METRICS
# ============================================================================
_t        = np.asarray(time_hist, dtype=float)
_dt       = float(_t[1] - _t[0]) if len(_t) > 1 else 0.1
_tau_c    = 3.0    # critical TTC [s]
_W1, _W2  = 100, 150   # reveal window half-widths [steps]
_delta    = 10         # post-reveal SM offset [steps]
_TTC_CAP  = 60.0
_N        = len(_t)

_lc_step  = _merger_reveal_step   # t=0 anchor for mechanism figure

_C  = {"IDEAM": EGO_IDEAM_COLOR, "DREAM": EGO_DREAM_COLOR, "RL-PPO": EGO_RL_COLOR,
       "ADA":   EGO_ADA_COLOR,   "APF":   EGO_APF_COLOR,
       "OACMPC": EGO_OACMPC_COLOR}
_LS = {"IDEAM": "--",  "DREAM": "-", "RL-PPO": (0, (5, 1, 1, 1)),
       "ADA": "-.",  "APF": ":",  "OACMPC": (0, (3, 1, 1, 1))}
_bar_labels = ["DREAM", "RL-PPO", "ADA", "APF", "OA-CMPC", "IDEAM"]
_bar_colors = [EGO_DREAM_COLOR, EGO_RL_COLOR, EGO_ADA_COLOR, EGO_APF_COLOR, EGO_OACMPC_COLOR, EGO_IDEAM_COLOR]
_bar_x      = np.arange(len(_bar_labels))

# ── Scalar metrics ────────────────────────────────────────────────────────
def _safe_min(arr, lo, hi):
    sl = [float(v) for v in arr[lo:hi]]
    return min(sl) if sl else float("nan")

def _safe_val(arr, idx):
    return float(arr[idx]) if (idx is not None and 0 <= idx < len(arr)) else float("nan")

if _lc_step is not None and not _ABLATION_MODE:
    _w0, _w1  = max(0, _lc_step - _W1), min(_N, _lc_step + _W2)
    dream_ttc_min_rev  = _safe_min(dream_ttc,  _w0, _w1)
    rl_ttc_min_rev     = _safe_min(rl_ttc,     _w0, _w1)
    base_ttc_min_rev   = _safe_min(base_ttc,   _w0, _w1)
    ada_ttc_min_rev    = _safe_min(ada_ttc,    _w0, _w1)
    apf_ttc_min_rev    = _safe_min(apf_ttc,    _w0, _w1)
    oacmpc_ttc_min_rev = _safe_min(oacmpc_ttc, _w0, _w1)
    dream_sm_rev   = _safe_val(dream_ttc,  _lc_step + _delta)
    rl_sm_rev      = _safe_val(rl_ttc,     _lc_step + _delta)
    base_sm_rev    = _safe_val(base_ttc,   _lc_step + _delta)
    ada_sm_rev     = _safe_val(ada_ttc,    _lc_step + _delta)
    apf_sm_rev     = _safe_val(apf_ttc,    _lc_step + _delta)
    oacmpc_sm_rev  = _safe_val(oacmpc_ttc, _lc_step + _delta)
    _occ_end = _lc_step   # merger was occluded before it entered centre
    _f = lambda arr: sum(1 for v in arr[:_occ_end] if float(v) < _tau_c) / max(1, _occ_end)
    dream_cmr  = float(_f(dream_ttc));  rl_cmr     = float(_f(rl_ttc));   base_cmr   = float(_f(base_ttc))
    ada_cmr    = float(_f(ada_ttc));    apf_cmr    = float(_f(apf_ttc))
    oacmpc_cmr = float(_f(oacmpc_ttc))
else:
    dream_ttc_min_rev = rl_ttc_min_rev = base_ttc_min_rev = ada_ttc_min_rev = apf_ttc_min_rev = oacmpc_ttc_min_rev = float("nan")
    dream_sm_rev = rl_sm_rev = base_sm_rev = ada_sm_rev = apf_sm_rev = oacmpc_sm_rev = float("nan")
    dream_cmr = rl_cmr = base_cmr = ada_cmr = apf_cmr = oacmpc_cmr = float("nan")

dream_rei  = float(np.sum(dream_rei_integrand)  * _dt)
rl_rei     = float(np.sum(rl_rei_integrand)     * _dt)
base_rei   = float(np.sum(base_rei_integrand)   * _dt)
ada_rei    = float(np.sum(ada_rei_integrand)    * _dt)
apf_rei    = float(np.sum(apf_rei_integrand)    * _dt)
oacmpc_rei = float(np.sum(oacmpc_rei_integrand) * _dt)

_mean_v_base = float(np.mean(base_vx)) if base_vx else 0.0
dream_ct_v  = _mean_v_base - float(np.mean(dream_vx))
rl_ct_v     = _mean_v_base - float(np.mean(rl_vx))
ada_ct_v    = _mean_v_base - float(np.mean(ada_vx))
apf_ct_v    = _mean_v_base - float(np.mean(apf_vx))
oacmpc_ct_v = _mean_v_base - float(np.mean(oacmpc_vx))

def _mean_jerk(acc_arr):
    a = np.array(acc_arr, dtype=float)
    return float(np.mean(np.abs(np.diff(a) / _dt))) if len(a) > 1 else 0.0

dream_mean_jerk  = _mean_jerk(dream_acc);  rl_mean_jerk   = _mean_jerk(rl_acc);  base_mean_jerk = _mean_jerk(base_acc)
ada_mean_jerk    = _mean_jerk(ada_acc);    apf_mean_jerk  = _mean_jerk(apf_acc)
oacmpc_mean_jerk = _mean_jerk(oacmpc_acc)

# ── Computational efficiency summary ─────────────────────────────────────────
def _timing_stats(t_arr, ref_dt=0.1):
    """Return (mean, max, r_RT) where r_RT = fraction of steps exceeding ref_dt."""
    if not t_arr:
        return float("nan"), float("nan"), float("nan")
    a = np.array(t_arr, dtype=float)
    return float(np.mean(a)), float(np.max(a)), float(np.mean(a > ref_dt))

def _ms(arr):
    """Mean of finite values in arr, in seconds."""
    a = np.array([v for v in arr if not math.isnan(v)], dtype=float)
    return float(np.mean(a)) if len(a) > 0 else float("nan")


def _fmt_ms(val):
    return "   N/A   " if math.isnan(val) else f"{val*1000:.1f} ms"

_tplan_dream, _tmax_dream, _rRT_dream = _timing_stats(_t_dream_total)
_tplan_rl, _tmax_rl, _rRT_rl = _timing_stats(_t_rl_total)
_tplan_ideam, _tmax_ideam, _rRT_ideam = _timing_stats(_t_ideam_total)
_tplan_ada,   _tmax_ada,   _rRT_ada   = _timing_stats(_t_ada_total)
_tplan_apf,   _tmax_apf,   _rRT_apf   = _timing_stats(_t_apf_total)
_tplan_oacmpc, _tmax_oacmpc, _rRT_oacmpc = _timing_stats(_t_oacmpc_total)

print()
print("=" * 100)
print("  Computational Efficiency - per-step breakdown (wall-clock, single scenario)")
print("  Hardware: Intel Core Ultra 9 285K  |  24 cores @ 3.7 GHz  |  64 GB RAM  |  CPU-only (no GPU offload)")
print("=" * 100)
_hd = (f"  {'Arm':<20}  {'t_drift(mean)':>13}  {'t_decision(mean)':>16}  "
       f"{'t_MPC(mean)':>12}  {'t_total(mean)':>13}  {'t_total(max)':>13}  {'r_RT':>6}")
print(_hd)
print("=" * 100)
for _arm, _td_arr, _tm_arr, _tdr_arr, _tmean, _tmax_val, _rRT_val in [
    ("DREAM",              _t_dream_decision,  _t_dream_mpc,  _t_dream_drift,  _tplan_dream,  _tmax_dream,  _rRT_dream),
    ("RL-PPO",             _t_rl_decision,     _t_rl_mpc,     _t_rl_drift,     _tplan_rl,     _tmax_rl,     _rRT_rl),
    ("IDEAM",              _t_ideam_decision,  _t_ideam_mpc,  None,            _tplan_ideam,  _tmax_ideam,  _rRT_ideam),
    ("ADA-based planner",  _t_ada_decision,    _t_ada_mpc,    _t_ada_drift,    _tplan_ada,    _tmax_ada,    _rRT_ada),
    ("APF-based planner",  _t_apf_decision,    _t_apf_mpc,    _t_apf_drift,    _tplan_apf,    _tmax_apf,    _rRT_apf),
    ("OA-CMPC",            _t_oacmpc_decision, _t_oacmpc_mpc, _t_oacmpc_drift, _tplan_oacmpc, _tmax_oacmpc, _rRT_oacmpc),
]:
    _drift_s = _fmt_ms(_ms(_tdr_arr)) if _tdr_arr is not None else "   N/A   "
    _dec_s   = _fmt_ms(_ms(_td_arr))
    _mpc_s   = _fmt_ms(_ms(_tm_arr))
    _tot_s   = _fmt_ms(_tmean)
    _max_s   = _fmt_ms(_tmax_val)
    _rr_s    = f"{_rRT_val*100:.1f}%"
    print(f"  {_arm:<20}  {_drift_s:>13}  {_dec_s:>16}  {_mpc_s:>12}  {_tot_s:>13}  {_max_s:>13}  {_rr_s:>6}")
print("=" * 100)
print(f"  dt = {dt*1000:.0f} ms  |  r_RT = fraction of steps where t_total > dt")
print("  t_drift   : DRIFT PDE step (NumPy grid advection-diffusion, CPU)")
print("  t_decision: gap-group formulation + behaviour decision + CBF risk veto (DREAM arms)")
print("  t_MPC     : CasADi NLP solve (LMPC, warm-started)")
print("  t_total   : full per-step planning cycle = t_drift + t_decision + t_MPC + overhead")
print()

# ── FIGURE A: Mechanism figure (event-aligned) ────────────────────────────
if _lc_step is not None and not _ABLATION_MODE:
    _t0_idx = max(0, _lc_step - _W1)
    _t1_idx = min(_N, _lc_step + _W2)
    _t_win  = _t[_t0_idx:_t1_idx] - _lc_step * _dt
    _sl     = slice(_t0_idx, _t1_idx)

    with plt.style.context(["science", "no-latex"]):
        fig_mech, axes_mech = plt.subplots(3, 1, figsize=(7, 9), constrained_layout=True)
        fig_mech.suptitle(
            "Mechanism Figure — Blocker-Forced Merge Conflict\n"
            r"$(t = 0$ = OccludedMerger enters centre lane$)$", fontsize=11)

        def _vm(ax, lbl=True):
            ax.axvline(0, color="magenta", lw=1.4, ls="--",
                       label="Merger enters centre" if lbl else None)
            if _t0_idx < _lc_step:
                ax.axvspan(_t_win[0], 0, color="lightyellow",
                           alpha=0.55, label="Occlusion window" if lbl else None)

        # (a) GVF risk at ego
        ax0 = axes_mech[0]
        for _k, _r in [("DREAM", risk_at_ego_hist), ("RL-PPO", risk_at_ego_rl_hist), ("ADA", risk_at_ego_ada_hist),
                        ("APF", risk_at_ego_apf_hist), ("OACMPC", risk_at_ego_oacmpc_hist)]:
            _rs = _r[_t0_idx:_t1_idx]
            if _rs:
                _lbl = _k if _k != "OACMPC" else "OA-CMPC"
                ax0.plot(_t_win[:len(_rs)], _rs, color=_C[_k], ls=_LS[_k], lw=1.5, label=_lbl)
                if _k == "DREAM":
                    ax0.fill_between(_t_win[:len(_rs)], _rs, alpha=0.18, color=_C[_k])
        _vm(ax0)
        ax0.set_ylabel("$R_{\\mathrm{ego}}$"); ax0.set_title("(a) GVF Risk at Ego Corridor")
        ax0.legend(fontsize=7, loc="upper left"); ax0.set_xticklabels([])

        # (b) Speed
        ax1 = axes_mech[1]
        for _k, _vx in [("DREAM", dream_vx), ("RL-PPO", rl_vx), ("ADA", ada_vx),
                         ("APF", apf_vx), ("OACMPC", oacmpc_vx), ("IDEAM", base_vx)]:
            _lbl = _k if _k != "OACMPC" else "OA-CMPC"
            ax1.plot(_t_win, np.array(_vx)[_sl], color=_C[_k], ls=_LS[_k], lw=1.5, label=_lbl)
        _vm(ax1, lbl=False)
        ax1.set_ylabel("$v_x$ [m/s]"); ax1.set_title("(b) Longitudinal Speed")
        ax1.legend(fontsize=7, loc="upper left"); ax1.set_xticklabels([])

        # (c) TTC to merger
        ax2 = axes_mech[2]
        for _k, _tc in [("DREAM", dream_ttc), ("RL-PPO", rl_ttc), ("ADA", ada_ttc),
                         ("APF", apf_ttc), ("OACMPC", oacmpc_ttc), ("IDEAM", base_ttc)]:
            _lbl = _k if _k != "OACMPC" else "OA-CMPC"
            ax2.plot(_t_win, np.array(_tc)[_sl], color=_C[_k], ls=_LS[_k], lw=1.5, label=_lbl)
        ax2.axhline(_tau_c, color="red", lw=0.9, ls=":", label=f"TTC crit ({_tau_c}s)")
        _vm(ax2, lbl=False)
        ax2.set_ylabel("TTC [s]"); ax2.set_xlabel("Time relative to reveal [s]")
        ax2.set_title("(c) TTC to OccludedMerger")
        ax2.legend(fontsize=7, loc="upper left"); ax2.set_ylim(0, min(30, _TTC_CAP))

        plt.savefig(os.path.join(save_dir, "metrics_mechanism.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig_mech)

if not _ABLATION_MODE:
    # ── FIGURE B: Summary scalar metrics bar chart ────────────────────────────
    def _bar_panel(ax, values, title, ylabel, lower_is_better=True, ymin=None):
        values = [float(v) for v in values]   # block CasADi leakage
        bars = ax.bar(_bar_x, values, color=_bar_colors, width=0.55, zorder=3)
        ax.set_xticks(_bar_x); ax.set_xticklabels(_bar_labels, fontsize=8)
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", lw=0.5, zorder=0)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        _valid = [(v, j) for j, v in enumerate(values) if not math.isnan(v)]
        if _valid:
            _bv, _bj = (min(_valid) if lower_is_better else max(_valid))
            bars[_bj].set_edgecolor("gold"); bars[_bj].set_linewidth(2)
        for bar, val in zip(bars, values):
            if not math.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f" {val:.2f}", ha="center", va="bottom", fontsize=7)

    with plt.style.context(["science", "no-latex"]):
        fig_s, axes_s = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        fig_s.suptitle("DREAM vs Baselines — Scalar Metrics Summary (Merger Conflict)",
                       fontsize=10)

        _bar_panel(axes_s[0, 0],
                   [dream_ttc_min_rev, rl_ttc_min_rev, ada_ttc_min_rev, apf_ttc_min_rev, oacmpc_ttc_min_rev, base_ttc_min_rev],
                   r"$\mathrm{TTC}^\mathrm{rev}_{\min}$ — Event-conditioned min TTC",
                   "TTC [s]", lower_is_better=False, ymin=0)
        _bar_panel(axes_s[0, 1],
                   [dream_cmr, rl_cmr, ada_cmr, apf_cmr, oacmpc_cmr, base_cmr],
                   r"$\mathrm{CMR}_\mathrm{occ}$ — Critical-moment rate (TTC < "
                   f"{_tau_c:.0f}s)", "Fraction", lower_is_better=True, ymin=0)
        _bar_panel(axes_s[0, 2],
                   [dream_sm_rev, rl_sm_rev, ada_sm_rev, apf_sm_rev, oacmpc_sm_rev, base_sm_rev],
                   r"$\mathrm{SM}_\mathrm{rev}$ — Post-reveal TTC (+1 s)",
                   "TTC [s]", lower_is_better=False, ymin=0)
        _bar_panel(axes_s[1, 0],
                   [dream_rei, rl_rei, ada_rei, apf_rei, oacmpc_rei, base_rei],
                   r"$\mathrm{REI}$ — Risk Exposure Integral",
                   r"$\Sigma\,R\cdot v_x\cdot\Delta t$", lower_is_better=True, ymin=0)
        _bar_panel(axes_s[1, 1],
                   [dream_ct_v, rl_ct_v, ada_ct_v, apf_ct_v, oacmpc_ct_v, 0.0],
                   r"$\mathrm{CT}_v$ — Conservatism tax vs IDEAM",
                   r"$\bar{v}_\mathrm{IDEAM}-\bar{v}_\mathrm{planner}$ [m/s]",
                   lower_is_better=True)
        _bar_panel(axes_s[1, 2],
                   [dream_mean_jerk, rl_mean_jerk, ada_mean_jerk, apf_mean_jerk, oacmpc_mean_jerk, base_mean_jerk],
                   r"Mean $|\dot{a}_x|$ — Motion stability",
                   r"m/s$^3$", lower_is_better=True, ymin=0)

        plt.savefig(os.path.join(save_dir, "metrics_summary.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig_s)

    # ── FIGURE C: Time-series overview (3×2) ─────────────────────────────────
    with plt.style.context(["science", "no-latex"]):
        fig_m, axes_m = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
        fig_m.suptitle("Uncertainty Merger: IDEAM vs DREAM vs RL-PPO vs ADA vs APF vs OA-CMPC",
                       fontsize=13)

        _risk_t     = _t[:len(risk_at_ego_hist)]
        _risk_rl_t  = _t[:len(risk_at_ego_rl_hist)]
        _risk_ada_t = _t[:len(risk_at_ego_ada_hist)]
        _risk_apf_t = _t[:len(risk_at_ego_apf_hist)]
        _reveal_t   = (_lc_step * _dt) if _lc_step is not None else None

        def _shade(ax):
            if _reveal_t is not None:
                ax.axvline(_reveal_t, color="magenta", lw=1.2, ls="--",
                           label="Merger enters centre")

        ax = axes_m[0, 0]
        ax.plot(_t, base_s,    color=_C["IDEAM"],  ls=_LS["IDEAM"],  label="IDEAM")
        ax.plot(_t, dream_s,   color=_C["DREAM"],  ls=_LS["DREAM"],  label="DREAM")
        ax.plot(_t, rl_s,      color=_C["RL-PPO"], ls=_LS["RL-PPO"], label="RL-PPO")
        ax.plot(_t, ada_s,     color=_C["ADA"],    ls=_LS["ADA"],    label="ADA")
        ax.plot(_t, apf_s,     color=_C["APF"],    ls=_LS["APF"],    label="APF")
        ax.plot(_t, oacmpc_s,  color=_C["OACMPC"], ls=_LS["OACMPC"], label="OA-CMPC")
        _shade(ax); ax.set_xlabel("t [s]"); ax.set_ylabel("s [m]")
        ax.set_title("Progress s(t)"); ax.legend(fontsize=8)

        ax = axes_m[0, 1]
        ax.plot(_t, base_vx,    color=_C["IDEAM"],  ls=_LS["IDEAM"],  label="IDEAM")
        ax.plot(_t, dream_vx,   color=_C["DREAM"],  ls=_LS["DREAM"],  label="DREAM")
        ax.plot(_t, rl_vx,      color=_C["RL-PPO"], ls=_LS["RL-PPO"], label="RL-PPO")
        ax.plot(_t, ada_vx,     color=_C["ADA"],    ls=_LS["ADA"],    label="ADA")
        ax.plot(_t, apf_vx,     color=_C["APF"],    ls=_LS["APF"],    label="APF")
        ax.plot(_t, oacmpc_vx,  color=_C["OACMPC"], ls=_LS["OACMPC"], label="OA-CMPC")
        _shade(ax); ax.set_xlabel("t [s]"); ax.set_ylabel("$v_x$ [m/s]")
        ax.set_title("Speed $v_x$(t)"); ax.legend(fontsize=8)

        ax = axes_m[1, 0]
        ax.plot(_t, base_ttc,    color=_C["IDEAM"],  ls=_LS["IDEAM"],  label="IDEAM")
        ax.plot(_t, dream_ttc,   color=_C["DREAM"],  ls=_LS["DREAM"],  label="DREAM")
        ax.plot(_t, rl_ttc,      color=_C["RL-PPO"], ls=_LS["RL-PPO"], label="RL-PPO")
        ax.plot(_t, ada_ttc,     color=_C["ADA"],    ls=_LS["ADA"],    label="ADA")
        ax.plot(_t, apf_ttc,     color=_C["APF"],    ls=_LS["APF"],    label="APF")
        ax.plot(_t, oacmpc_ttc,  color=_C["OACMPC"], ls=_LS["OACMPC"], label="OA-CMPC")
        ax.axhline(_tau_c, color="red", lw=0.9, ls=":", label=f"TTC crit ({_tau_c}s)")
        _shade(ax); ax.set_xlabel("t [s]"); ax.set_ylabel("TTC [s]")
        ax.set_title("TTC to OccludedMerger"); ax.set_ylim(0, min(30, _TTC_CAP))
        ax.legend(fontsize=7)

        _risk_oacmpc_t = _t[:len(risk_at_ego_oacmpc_hist)]
        ax = axes_m[1, 1]
        ax.plot(_risk_t, risk_at_ego_hist,
                color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
        ax.fill_between(_risk_t, risk_at_ego_hist, color=_C["DREAM"], alpha=0.20)
        ax.plot(_risk_rl_t,     risk_at_ego_rl_hist,    color=_C["RL-PPO"], ls=_LS["RL-PPO"], label="RL-PPO")
        ax.plot(_risk_ada_t,    risk_at_ego_ada_hist,   color=_C["ADA"],    ls=_LS["ADA"],    label="ADA")
        ax.plot(_risk_apf_t,    risk_at_ego_apf_hist,   color=_C["APF"],    ls=_LS["APF"],    label="APF")
        ax.plot(_risk_oacmpc_t, risk_at_ego_oacmpc_hist, color=_C["OACMPC"], ls=_LS["OACMPC"], label="OA-CMPC")
        _shade(ax); ax.set_xlabel("t [s]"); ax.set_ylabel("R(ego)")
        ax.set_title("GVF Risk at Ego Corridor"); ax.legend(fontsize=8)

        ax = axes_m[2, 0]
        ax.plot(_t, base_acc,    color=_C["IDEAM"],  ls=_LS["IDEAM"],  label="IDEAM")
        ax.plot(_t, dream_acc,   color=_C["DREAM"],  ls=_LS["DREAM"],  label="DREAM")
        ax.plot(_t, rl_acc,      color=_C["RL-PPO"], ls=_LS["RL-PPO"], label="RL-PPO")
        ax.plot(_t, ada_acc,     color=_C["ADA"],    ls=_LS["ADA"],    label="ADA")
        ax.plot(_t, apf_acc,     color=_C["APF"],    ls=_LS["APF"],    label="APF")
        ax.plot(_t, oacmpc_acc,  color=_C["OACMPC"], ls=_LS["OACMPC"], label="OA-CMPC")
        ax.axhline(0.0, color="black", lw=0.6); _shade(ax)
        ax.set_xlabel("t [s]"); ax.set_ylabel("$a_x$ [m/s²]")
        ax.set_title("Longitudinal Acceleration"); ax.legend(fontsize=8)

        ax = axes_m[2, 1]
        ax.plot(_t, base_min_dist_sur,    color=_C["IDEAM"],  ls=_LS["IDEAM"],  label="IDEAM")
        ax.plot(_t, dream_min_dist_sur,   color=_C["DREAM"],  ls=_LS["DREAM"],  label="DREAM")
        ax.plot(_t, rl_min_dist_sur,      color=_C["RL-PPO"], ls=_LS["RL-PPO"], label="RL-PPO")
        ax.plot(_t, ada_min_dist_sur,     color=_C["ADA"],    ls=_LS["ADA"],    label="ADA")
        ax.plot(_t, apf_min_dist_sur,     color=_C["APF"],    ls=_LS["APF"],    label="APF")
        ax.plot(_t, oacmpc_min_dist_sur,  color=_C["OACMPC"], ls=_LS["OACMPC"], label="OA-CMPC")
        ax.axhline(NEAR_COLLISION_DIST, color="orange", lw=0.9, ls=":")
        _shade(ax); ax.set_xlabel("t [s]"); ax.set_ylabel("$S_o$ [m]")
        ax.set_title("Min Distance to Nearby Vehicles")
        ax.set_ylim(bottom=0.0); ax.legend(fontsize=8)

        metrics_png = os.path.join(save_dir, "metrics_uncertainty_merger.png")
        plt.savefig(metrics_png, dpi=300, bbox_inches="tight")
        plt.close(fig_m)

    # ── Save numeric metrics ──────────────────────────────────────────────────
    metrics_npy = os.path.join(save_dir, "metrics_uncertainty_merger.npy")
    np.save(metrics_npy, {
        "time":               np.asarray(time_hist,             dtype=float),
        "base_s":             np.asarray(base_s,                dtype=float),
        "dream_s":            np.asarray(dream_s,               dtype=float),
        "rl_s":               np.asarray(rl_s,                  dtype=float),
        "ada_s":              np.asarray(ada_s,                 dtype=float),
        "apf_s":              np.asarray(apf_s,                 dtype=float),
        "oacmpc_s":           np.asarray(oacmpc_s,              dtype=float),
        "base_vx":            np.asarray(base_vx,               dtype=float),
        "dream_vx":           np.asarray(dream_vx,              dtype=float),
        "rl_vx":              np.asarray(rl_vx,                 dtype=float),
        "ada_vx":             np.asarray(ada_vx,                dtype=float),
        "apf_vx":             np.asarray(apf_vx,                dtype=float),
        "oacmpc_vx":          np.asarray(oacmpc_vx,             dtype=float),
        "base_acc":           np.asarray(base_acc,              dtype=float),
        "dream_acc":          np.asarray(dream_acc,             dtype=float),
        "rl_acc":             np.asarray(rl_acc,                dtype=float),
        "ada_acc":            np.asarray(ada_acc,               dtype=float),
        "apf_acc":            np.asarray(apf_acc,               dtype=float),
        "oacmpc_acc":         np.asarray(oacmpc_acc,            dtype=float),
        "base_lane":          np.asarray(base_lane_hist,        dtype=int),
        "dream_lane":         np.asarray(dream_lane_hist,       dtype=int),
        "rl_lane":            np.asarray(rl_lane_hist,          dtype=int),
        "ada_lane":           np.asarray(ada_lane_hist,         dtype=int),
        "apf_lane":           np.asarray(apf_lane_hist,         dtype=int),
        "oacmpc_lane":        np.asarray(oacmpc_lane_hist,      dtype=int),
        "base_merger_lane":   np.asarray(base_merger_lane_hist,  dtype=int),
        "dream_merger_lane":  np.asarray(dream_merger_lane_hist, dtype=int),
        "rl_merger_lane":     np.asarray(rl_merger_lane_hist,   dtype=int),
        "ada_merger_lane":    np.asarray(ada_merger_lane_hist,  dtype=int),
        "apf_merger_lane":    np.asarray(apf_merger_lane_hist,  dtype=int),
        "oacmpc_merger_lane": np.asarray(oacmpc_merger_lane_hist, dtype=int),
        "base_dist_merger":   np.asarray(base_dist_merger,      dtype=float),
        "dream_dist_merger":  np.asarray(dream_dist_merger,     dtype=float),
        "rl_dist_merger":     np.asarray(rl_dist_merger,        dtype=float),
        "ada_dist_merger":    np.asarray(ada_dist_merger,       dtype=float),
        "apf_dist_merger":    np.asarray(apf_dist_merger,       dtype=float),
        "oacmpc_dist_merger": np.asarray(oacmpc_dist_merger,    dtype=float),
        "base_min_dist_sur":  np.asarray(base_min_dist_sur,     dtype=float),
        "dream_min_dist_sur": np.asarray(dream_min_dist_sur,    dtype=float),
        "rl_min_dist_sur":    np.asarray(rl_min_dist_sur,       dtype=float),
        "ada_min_dist_sur":   np.asarray(ada_min_dist_sur,      dtype=float),
        "apf_min_dist_sur":   np.asarray(apf_min_dist_sur,      dtype=float),
        "oacmpc_min_dist_sur":np.asarray(oacmpc_min_dist_sur,   dtype=float),
        "base_ttc":           np.asarray(base_ttc,              dtype=float),
        "dream_ttc":          np.asarray(dream_ttc,             dtype=float),
        "rl_ttc":             np.asarray(rl_ttc,                dtype=float),
        "ada_ttc":            np.asarray(ada_ttc,               dtype=float),
        "apf_ttc":            np.asarray(apf_ttc,               dtype=float),
        "oacmpc_ttc":         np.asarray(oacmpc_ttc,            dtype=float),
        "base_rei_integrand":  np.asarray(base_rei_integrand,   dtype=float),
        "dream_rei_integrand": np.asarray(dream_rei_integrand,  dtype=float),
        "rl_rei_integrand":    np.asarray(rl_rei_integrand,     dtype=float),
        "ada_rei_integrand":   np.asarray(ada_rei_integrand,    dtype=float),
        "apf_rei_integrand":   np.asarray(apf_rei_integrand,    dtype=float),
        "oacmpc_rei_integrand":np.asarray(oacmpc_rei_integrand, dtype=float),
        "risk_at_ego":         np.asarray(risk_at_ego_hist,     dtype=float),
        "risk_at_ego_rl":      np.asarray(risk_at_ego_rl_hist,  dtype=float),
        "risk_at_ego_ada":     np.asarray(risk_at_ego_ada_hist, dtype=float),
        "risk_at_ego_apf":     np.asarray(risk_at_ego_apf_hist, dtype=float),
        "risk_at_ego_oacmpc":  np.asarray(risk_at_ego_oacmpc_hist, dtype=float),
        # Scalar metrics
        "ttc_min_rev": {"DREAM": dream_ttc_min_rev, "RL-PPO": rl_ttc_min_rev, "ADA": ada_ttc_min_rev,
                        "APF":   apf_ttc_min_rev,   "OACMPC": oacmpc_ttc_min_rev,
                        "IDEAM": base_ttc_min_rev},
        "cmr_occ":     {"DREAM": dream_cmr,          "RL-PPO": rl_cmr, "ADA":  ada_cmr,
                        "APF":   apf_cmr,            "OACMPC": oacmpc_cmr,
                        "IDEAM": base_cmr},
        "sm_rev":      {"DREAM": dream_sm_rev,        "RL-PPO": rl_sm_rev, "ADA":  ada_sm_rev,
                        "APF":   apf_sm_rev,          "OACMPC": oacmpc_sm_rev,
                        "IDEAM": base_sm_rev},
        "rei":         {"DREAM": dream_rei,           "RL-PPO": rl_rei, "ADA":  ada_rei,
                        "APF":   apf_rei,             "OACMPC": oacmpc_rei,
                        "IDEAM": base_rei},
        "ct_v":        {"DREAM": dream_ct_v,          "RL-PPO": rl_ct_v, "ADA":  ada_ct_v,
                        "APF":   apf_ct_v,            "OACMPC": oacmpc_ct_v,
                        "IDEAM": 0.0},
        "mean_jerk":   {"DREAM": dream_mean_jerk,     "RL-PPO": rl_mean_jerk, "ADA":  ada_mean_jerk,
                        "APF":   apf_mean_jerk,       "OACMPC": oacmpc_mean_jerk,
                        "IDEAM": base_mean_jerk},
        "near_collision_dist": float(NEAR_COLLISION_DIST),
        "collision_dist":      float(COLLISION_DIST),
        "tau_c":               _tau_c,
        "merger_reveal_step":  _lc_step,
    })

    print(f"Metrics plot:      {metrics_png}")
    print(f"Mechanism figure:  {os.path.join(save_dir, 'metrics_mechanism.png')}"
          f"{'  (skipped - no reveal)' if _lc_step is None else ''}")
    print(f"Summary metrics:   {os.path.join(save_dir, 'metrics_summary.png')}")
    print(f"Metrics data:      {metrics_npy}")
    print(f"\nScalar metrics:")
    print(f"  TTC_min_rev : DREAM={dream_ttc_min_rev:.2f}  RL-PPO={rl_ttc_min_rev:.2f}  ADA={ada_ttc_min_rev:.2f}"
          f"  APF={apf_ttc_min_rev:.2f}  OA-CMPC={oacmpc_ttc_min_rev:.2f}  IDEAM={base_ttc_min_rev:.2f}  [s, higher is better]")
    print(f"  CMR_occ     : DREAM={dream_cmr:.3f}  RL-PPO={rl_cmr:.3f}  ADA={ada_cmr:.3f}"
          f"  APF={apf_cmr:.3f}  OA-CMPC={oacmpc_cmr:.3f}  IDEAM={base_cmr:.3f}  [frac, lower is better]")
    print(f"  REI         : DREAM={dream_rei:.2f}  RL-PPO={rl_rei:.2f}  ADA={ada_rei:.2f}"
          f"  APF={apf_rei:.2f}  OA-CMPC={oacmpc_rei:.2f}  IDEAM={base_rei:.2f}  [lower is better]")
    print(f"  CT_v        : DREAM={dream_ct_v:.2f}  RL-PPO={rl_ct_v:.2f}  ADA={ada_ct_v:.2f}"
          f"  APF={apf_ct_v:.2f}  OA-CMPC={oacmpc_ct_v:.2f}  IDEAM=0.00  [m/s, lower is better]")
    print(f"  Mean |jerk| : DREAM={dream_mean_jerk:.3f}  RL-PPO={rl_mean_jerk:.3f}  ADA={ada_mean_jerk:.3f}"
          f"  APF={apf_mean_jerk:.3f}  OA-CMPC={oacmpc_mean_jerk:.3f}  IDEAM={base_mean_jerk:.3f}  [m/s^3, lower is better]")


# ===========================================================================
# ABLATION MODE RUNNER  (--run-mode ablation)
# ===========================================================================
# Runs the DREAM ego (GVF arm) four times with different DRIFT configs to
# match the paper ablation table (Table ablation_compact):
#
#   Full DREAM      — unmodified GVF source (reference row)
#   No advection    — transport velocity c = 0
#   Weak occ-prop   — D_occ x0.15, G_occ x0.15  (weaker hidden-risk)
#   No occ-coupling — Q_occ = 0, occ_mask = False (no occlusion coupling)
#
# Metrics: REI, S_o_min, mean|jerk|, CT_v vs IDEAM baseline.
# CT_v uses IDEAM mean speed (base_vx) collected in the main loop.
# ===========================================================================

if CLI_ARGS.run_mode == "ablation":
    import Integration.drift_interface as _di_mod
    from pde_solver import compute_Q_vehicle, compute_Q_merge, compute_Q_occlusion

    _WEAK_G_SCALE = 0.15                        # G_occ: 5.0 -> 0.75
    _WEAK_D_OCC   = cfg.D_occ * _WEAK_G_SCALE  # D_occ: 6.0 -> 0.90

    # ── Ablation source functions ────────────────────────────────────────────

    def _src_weak_occ(vehicles, ego, X, Y):
        """G_occ x0.15; occ_mask kept so D_occ patch applies too."""
        Q_veh = compute_Q_vehicle(vehicles, ego, X, Y)
        Q_occ_raw, occ_mask = compute_Q_occlusion(vehicles, ego, X, Y)
        Q_merge = compute_Q_merge(vehicles, ego, X, Y)
        Q_occ = Q_occ_raw * _WEAK_G_SCALE
        return Q_veh + Q_occ + Q_merge, Q_veh, Q_occ, occ_mask

    def _src_no_occ(vehicles, ego, X, Y):
        """No occlusion source; zero occ_mask disables D_occ boost too."""
        Q_veh  = compute_Q_vehicle(vehicles, ego, X, Y)
        Q_merge = compute_Q_merge(vehicles, ego, X, Y)
        z = np.zeros_like(X)
        return Q_veh + Q_merge, Q_veh, z, z.astype(bool)

    def _zero_vel_field(vehicles, ego, X, Y):
        """Zero advection velocity: c = 0."""
        z = np.zeros_like(X)
        return z, z, z, z, z, z

    # ── Mini-simulation runner ───────────────────────────────────────────────
    _ABL_LC_START = 35    # step merger begins right->centre LC
    _ABL_LC_DUR   = 20    # steps for full LC completion
    _TTC_CAP_ABL  = 60.0
    _TAU_C_ABL    = 3.0   # critical TTC [s] for REI

    def _run_ablation(label, source_fn=None, zero_adv=False, d_occ_val=None):
        """
        Run DREAM ego alone for N_t steps.
        Uses a deterministic kinematic merger (reproducible across variants).
        Returns scalar-metric dict.
        """
        # Patch DRIFT parameters
        _orig_docc = cfg.D_occ
        _orig_vel  = _di_mod.compute_velocity_field
        if d_occ_val is not None:
            cfg.D_occ = float(d_occ_val)
        if zero_adv:
            _di_mod.compute_velocity_field = _zero_vel_field
        try:
            # Fresh ego state
            _X0  = [EGO_V0, 0.0, 0.0, EGO_S0, 0.0, 0.0]
            _X0g = [path1c(EGO_S0)[0], path1c(EGO_S0)[1],
                    path1c.get_theta_r(EGO_S0)]
            _oa, _od, _lX, _pc = 0.0, 0.0, None, 0

            # Fresh truck (constant speed, centre lane)
            _trk = np.array([
                TRUCK_S0, 0.0, 0.0,
                path2c(TRUCK_S0)[0], path2c(TRUCK_S0)[1],
                path2c.get_theta_r(TRUCK_S0), TRUCK_V0, 0.0,
            ], dtype=float)
            _truck_dyn_abl = Curved_Road_Vehicle(
                a_max=2.0, delta=4, s0=2.0, b=1.5, T=1.5,
                K_P=1.0, K_D=0.1, K_I=0.01,
                dt=dt, lf=1.5, lr=1.5, length=TRUCK_LENGTH)

            # Fresh blocker
            _blk = LeftLaneBlocker(
                path=path1c, path_data=(x1c, y1c, samples1c),
                s_init=BLOCKER_S_INIT, vd=BLOCKER_VD, dt=dt, steer_range=steer_range)

            # Kinematic merger (starts right lane, LC at step _ABL_LC_START)
            _mg_s = float(MERGER_DREAM_S0)
            _mg_v = float(MERGER_DREAM_V0)

            # DRIFT reset + warm-up
            _ctrl = dream_controller
            _ctrl.drift.reset()
            _mg_xwm, _mg_ywm   = path3c(_mg_s)
            _mg_psi_wm          = path3c.get_theta_r(_mg_s)
            _mg_wm_row = np.array([_mg_s, 0.0, 0.0,
                                   _mg_xwm, _mg_ywm, _mg_psi_wm, _mg_v, 0.0])
            _ego_iv = drift_create_vehicle(
                vid=0, x=_X0g[0], y=_X0g[1],
                vx=_X0[0] * math.cos(_X0g[2]),
                vy=_X0[0] * math.sin(_X0g[2]), vclass="car")
            _ego_iv["heading"] = _X0g[2]
            _ctrl.drift.warmup(
                [row_to_drift_vehicle(_trk, vid=1, vclass="truck"),
                 row_to_drift_vehicle(_mg_wm_row, vid=2, vclass="car"),
                 _blk.to_drift_vehicle(vid=3)],
                _ego_iv, dt=dt, duration=3.0, substeps=3, source_fn=source_fn)

            # Fresh decision + utils
            _dec   = decision(**decision_params())
            _utils = LeaderFollower_Uitl(**util_cfg)
            _ctrl.set_util(_utils)

            # Accumulators
            _vx_hist, _acc_hist, _ttc_hist, _dist_hist, _rei_hist = [], [], [], [], []
            _abl_t_steps = []   # per-step wall-clock time for this variant

            for _si in range(N_t):
                _trk = update_truck_state(_trk, _truck_dyn_abl, [])
                _blk.update()
                _blk_row = _blk.to_mpc_row()

                # Kinematic merger position
                _mg_s += _mg_v * dt
                if _si < _ABL_LC_START:
                    _mg_path, _mg_lane = path3c, 2
                elif _si < _ABL_LC_START + _ABL_LC_DUR:
                    _frac = (_si - _ABL_LC_START) / _ABL_LC_DUR
                    _mg_path = path2c if _frac >= 0.5 else path3c
                    _mg_lane = 1     if _frac >= 0.5 else 2
                else:
                    _mg_path, _mg_lane = path2c, 1
                _mg_x, _mg_y = _mg_path(_mg_s)
                _mg_psi      = _mg_path.get_theta_r(_mg_s)
                _mg_row = np.array([_mg_s, 0.0, 0.0,
                                    _mg_x, _mg_y, _mg_psi, _mg_v, 0.0])

                # DRIFT step (timed — parameter variant affects this cost)
                _t0_abl = time.time()
                _ego_d = drift_create_vehicle(
                    vid=0, x=_X0g[0], y=_X0g[1],
                    vx=_X0[0] * math.cos(_X0g[2]),
                    vy=_X0[0] * math.sin(_X0g[2]), vclass="car")
                _ego_d["heading"] = _X0g[2]
                _ctrl.drift.step(
                    [row_to_drift_vehicle(_trk, vid=1, vclass="truck"),
                     row_to_drift_vehicle(_mg_row, vid=2, vclass="car"),
                     _blk.to_drift_vehicle(vid=3)],
                    _ego_d, dt=dt, substeps=3, source_fn=source_fn)

                # MPC lane arrays
                _ll = stack_rows([_blk_row] + ([_mg_row] if _mg_lane == 0 else []))
                _lc_lane = stack_rows([_trk] + ([_mg_row] if _mg_lane == 1 else []))
                _lr = stack_rows([_mg_row] if _mg_lane == 2 else [])

                # Lane-change forcing (same logic as main loop)
                _cur_lane = lane_from_global(_X0g)
                _ego_ready = (
                    float(_X0[3]) > float(_trk[0]) + EGO_LC_OVERTAKE_MARGIN and
                    0 < float(_blk.s) - float(_X0[3]) < EGO_BLOCKER_TRIGGER_GAP
                )
                _force = (1 if (_cur_lane == 0 and _ego_ready)
                          else (None if _cur_lane != 0 else 0))

                # DREAM ego step
                _res = dream_agent_step(
                    _X0, _X0g, _oa, _od, _lX, _pc,
                    _ctrl, _utils, _dec, dynamics,
                    _ll, _lc_lane, _lr,
                    enable_decision_veto=config_integration.enable_decision_veto,
                    force_target_lane=_force,
                    bypass_probe_guard=True,
                    force_ignore_veto=True,
                )
                _abl_t_steps.append(time.time() - _t0_abl)
                if _res["ok"]:
                    _X0, _X0g = _res["X0"], _res["X0_g"]
                    _oa, _od  = _res["oa"], _res["od"]
                    _lX, _pc  = _res["last_X"], _res["path_changed"]
                    if _force is not None and lane_from_global(_X0g) != int(_force):
                        _X0, _X0g = blend_state_toward_lane(
                            _X0, _X0g, int(_force), EGO_DREAM_ASSIST_BLEND)

                # Metrics
                _vx_hist.append(float(_X0[0]))
                _a_cmd = (float(_oa[0]) if hasattr(_oa, "__len__") else float(_oa))
                _acc_hist.append(_a_cmd)

                # TTC / spacing to nearest vehicle
                _ego_s  = float(_X0[3])
                _cands  = [(max(0.01, abs(float(r[0]) - _ego_s) - 2.5), float(r[6]))
                           for r in [_trk, _mg_row, _blk_row]]
                _min_d, _lead_v = min(_cands, key=lambda x: x[0])
                _v_close = max(float(_X0[0]) - _lead_v, 0.001)
                _ttc     = min(_min_d / _v_close, _TTC_CAP_ABL)
                _ttc_hist.append(_ttc)
                _dist_hist.append(_min_d)
                _rei_hist.append(max(0.0, _TAU_C_ABL - _ttc) / _TAU_C_ABL)

            _rei    = float(np.sum(_rei_hist) * dt)
            _so_min = float(np.min(_dist_hist)) if _dist_hist else float("nan")
            _mj     = (float(np.mean(np.abs(np.diff(_acc_hist) / dt)))
                       if len(_acc_hist) > 1 else 0.0)
            _mv     = float(np.mean(_vx_hist)) if _vx_hist else 0.0
            _tp_m, _tp_x, _tp_rr = _timing_stats(_abl_t_steps)
            return {"label": label, "rei": _rei, "so_min": _so_min,
                    "mean_jerk": _mj, "mean_vx": _mv,
                    "tplan_mean": _tp_m, "tplan_max": _tp_x, "r_RT": _tp_rr}

        finally:
            cfg.D_occ = _orig_docc
            if zero_adv:
                _di_mod.compute_velocity_field = _orig_vel
            dream_controller.set_util(dream_utils)   # restore main ref

    # ── Run four ablation variants ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ABLATION  --  DRIFT parameter sensitivity  (Table ablation_compact)")
    print("=" * 72)

    _abl_configs = [
        # (label,                  source_fn,     zero_adv, d_occ_val)
        ("Full DREAM",             None,           False,    None),
        ("No advection (c=0)",     None,           True,     None),
        ("Weak occ-prop (x0.15)",  _src_weak_occ,  False,    _WEAK_D_OCC),
        ("No occ-coupling",        _src_no_occ,    False,    None),
    ]

    _abl_results = []
    for _abl_lbl, _abl_src, _abl_za, _abl_docc in _abl_configs:
        print(f"  [{_abl_lbl}]  {N_t} steps ... ", end="", flush=True)
        _r = _run_ablation(_abl_lbl, _abl_src, _abl_za, _abl_docc)
        _abl_results.append(_r)
        print(f"done  REI={_r['rei']:.3f}  So_min={_r['so_min']:.2f}m  "
              f"|j|={_r['mean_jerk']:.4f}  t_plan={_r['tplan_mean']:.4f}s")

    # CT_v vs IDEAM baseline (base_vx from main loop always runs)
    _base_mv_abl = float(np.mean(base_vx)) if base_vx else float("nan")
    _full_rei    = _abl_results[0]["rei"]

    print()
    _sep = "-" * 80
    print(_sep)
    print(f"  {'Setting':<24}  {'REI':>7}  {'So_min':>7}  "
          f"{'|j|':>7}  {'CT_v':>7}  {'t_plan':>8}  {'t_max':>8}")
    print(_sep)
    for _r in _abl_results:
        _ct  = _base_mv_abl - _r["mean_vx"]
        _tag = ("REF" if _r["label"] == "Full DREAM"
                else ("worse" if _r["rei"] > _full_rei * 1.03 else "similar"))
        _tp = f"{_r['tplan_mean']:.4f}s" if not math.isnan(_r['tplan_mean']) else "N/A"
        _tx = f"{_r['tplan_max']:.4f}s"  if not math.isnan(_r['tplan_max'])  else "N/A"
        print(f"  {_r['label']:<24}  {_r['rei']:>7.3f}  {_r['so_min']:>6.2f}m  "
              f"{_r['mean_jerk']:>7.4f}  {_ct:>7.3f}  {_tp:>8}  {_tx:>8}   [{_tag}]")
    print(_sep)
    print("  REI, So_min, |j|: compared to Full DREAM row.")
    print("  CT_v: +ve = ego slower than IDEAM baseline (more conservative).")
    print("  t_plan: wall-clock per MPC step; t_max: worst-case step time.")
    print("  Expected ablation trend vs Full DREAM: REI up, So_min down, |j| up.")

    _abl_npy = os.path.join(save_dir, "ablation_metrics.npy")
    np.save(_abl_npy, {r["label"]: r for r in _abl_results})
    print(f"\nAblation metrics -> {_abl_npy}")

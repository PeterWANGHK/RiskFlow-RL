"""
DREAM Uncertainty Test — Two-Threat Occlusion Scenario (Proactive Risk)
========================================================================
Demonstrates DREAM's core advantage: proactive, uncertainty-aware risk planning
against two simultaneous occluded threats that only DRIFT can "see".

Scenario
--------
1. EGO starts in the LEFT lane (path1c, lane 0) at s = 20 m, vx = 8 m/s.
   A slow TRUCK in the CENTRE lane (vehicle_centre[1], vd = 5.5 m/s, s ≈ 51 m)
   is overtaken from the left while the merger approaches from the right.

2. An OCCLUDED MERGER starts in the RIGHT lane alongside the truck (same
   Frenet-s, different lane).  It moves slightly faster (MERGER_VD > truck vd)
   so it naturally overtakes the truck while hidden.  Always fed to DRIFT so
   risk accumulates in the right→centre merge zone — but NEVER in IDEAM's MPC
   arrays until its LC is ≥ 50 % complete.

3. After overtaking the centre-lane truck on the right, the OccludedMerger
   performs its own LC right → centre.  If one of the ego planners also chooses
   a LEFT → CENTRE lane change around the same time, a surprise conflict can
   emerge: ego arrives in centre from the left, merger from the right.
     · IDEAM:  may commit to centre lane → merger mid-LC → tight encounter.
     · DREAM:  DRIFT risk pre-built in merge zone → decision veto →
               holds left lane or slows → no conflict.

4. A fast HIDDEN CAR (LeftLaneFastCar, vd = 14 m/s) lurks in the LEFT lane,
   always in DRIFT, optionally revealed to IDEAM — adds a second hidden
   threat demonstrating DRIFT's broader uncertainty awareness.

DREAM vs IDEAM contrast
-----------------------
* IDEAM (baseline):  No risk field → blind overtake attempt on centre →
  encounters merger → near-conflict, hard braking.

* DREAM (ours):  DRIFT risk in right→centre zone → decision veto →
  holds left lane or slows → smooth speed profile, safe gap.

Key metrics
-----------
  · Min TTC with merger   (IDEAM << DREAM)
  · Peak deceleration     (IDEAM > DREAM)
  · Min gap to merger     (IDEAM < DREAM)
  · Speed profile         (DREAM smoother)
"""

# ===========================================================================
# IMPORTS
# ===========================================================================
import sys
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import math
import time
import copy
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from scipy.ndimage import gaussian_filter as _gf
import scienceplots  # noqa: F401 — registers "science" style with matplotlib

sys.path.append("C:\\IDEAM_implementation-main\\decision_improve")
from Control.MPC import *
from Control.constraint_params import *
from Model.Dynamical_model import *
from Model.params import *
from Model.surrounding_params import *
from Model.Surrounding_model import *
from Model.surrounding_vehicles import *
from Control.HOCBF import *
from DecisionMaking.decision_params import *
from DecisionMaking.give_desired_path import *
from DecisionMaking.util import *
from DecisionMaking.util_params import *
from DecisionMaking.decision import *
from Prediction.surrounding_prediction import *
from progress.bar import Bar

# DREAM
from config import Config as cfg
from pde_solver import (PDESolver, compute_total_Q, compute_velocity_field,
                        compute_diffusion_field, create_vehicle as drift_create_vehicle)
from Integration.drift_interface import DRIFTInterface
from Integration.prideam_controller import create_prideam_controller
from Integration.integration_config import get_preset

# ADA + APF source adapters (comparison arms)
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "Aggressiveness_Modeling"))
from Aggressiveness_Modeling.ADA_drift_source import compute_Q_ADA  # noqa: E402
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "APF_Modeling"))
from APF_Modeling.APF_drift_source import compute_Q_APF  # noqa: E402

# OA-CMPC source adapter
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "OA_CMPC"))
from OA_CMPC.oa_cmpc_source import compute_Q_OACMPC  # noqa: E402

# RL decision-policy (optional — graceful if torch not available)
_RL_DECISION_OK = False
try:
    import torch as _torch
    from rl.policy.decision_policy import (
        DEC_OBS_DIM, DEC_N_ACTIONS, DecisionPolicy,
        build_decision_obs, decode_action, target_speed_from_action,
    )
    from rl.policy.decision_inference import build_simulator_obs as _build_dec_obs
    from rl.train_bc import load_decision_policy as _load_bc_policy
    _RL_DECISION_OK = True
except Exception as _e:
    print(f"[rl-dec] optional RL imports failed: {_e}")
    _RL_DECISION_OK = False

# ===========================================================================
# INTEGRATION MODE
# ===========================================================================
INTEGRATION_MODE = "conservative"
config_integration = get_preset(INTEGRATION_MODE)
config_integration.apply_mode()

# ===========================================================================
# COMMAND-LINE MODEL SELECTION
# ===========================================================================
# Usage examples:
#   python uncertainty_test_DREAM.py                          → all 5 arms
#   python uncertainty_test_DREAM.py --models all             → all 5 arms
#   python uncertainty_test_DREAM.py --models DREAM IDEAM     → 2 arms only
#   python uncertainty_test_DREAM.py --models OA-CMPC IDEAM   → 2 arms only
#
# Skipped arms: DRIFT warmup is skipped (saves ~5 s each); main loop step
# is skipped and nan placeholders fill all metric lists; plot lines omitted.
import argparse as _argparse
_ALL_PLANNERS = {'DREAM', 'ADA', 'APF', 'OA-CMPC', 'IDEAM', 'RL-PPO'}
_ap = _argparse.ArgumentParser(
    description="DREAM uncertainty test — multi-planner benchmark",
    formatter_class=_argparse.RawDescriptionHelpFormatter,
    epilog="Available planners: " + "  ".join(sorted(_ALL_PLANNERS)) + "  all")
_ap.add_argument(
    '--models', nargs='+', default=['all'], metavar='MODEL',
    help="Planners to run (space-separated). Use 'all' for all six.")
_ap.add_argument(
    '--mode', choices=['single', 'batch'], default=None,
    help="Override run mode (default: use RUN_MODE constant below).")
_ap.add_argument(
    '--rl-policy-mode', default='decision', choices=['decision'],
    help="RL decision policy mode (only 'decision' is supported here).")
_ap.add_argument(
    '--rl-decision-checkpoint',
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'rl', 'checkpoints', 'decision_policy_bc.pt'),
    help="Path to a BC or PPO decision-policy checkpoint.")
_ap_args, _ = _ap.parse_known_args()

# Normalise model names (case-insensitive, allow 'oacmpc' → 'OA-CMPC')
_NORM_MAP = {p.lower(): p for p in _ALL_PLANNERS}
_NORM_MAP.update({'oacmpc': 'OA-CMPC', 'dream': 'DREAM', 'ada': 'ADA',
                  'apf': 'APF', 'ideam': 'IDEAM',
                  'rlppo': 'RL-PPO', 'rl': 'RL-PPO', 'rl-ppo': 'RL-PPO'})
if 'all' in [m.lower() for m in _ap_args.models]:
    RUN_PLANNERS = _ALL_PLANNERS.copy()
else:
    RUN_PLANNERS = set()
    for _m in _ap_args.models:
        _key = _m.lower().replace('-', '')
        _mapped = _NORM_MAP.get(_key) or _NORM_MAP.get(_m.lower())
        if _mapped is None:
            _ap.error(f"Unknown planner '{_m}'. Choose from: "
                      + " ".join(sorted(_ALL_PLANNERS)) + " all")
        RUN_PLANNERS.add(_mapped)

# ===========================================================================
# RUN MODE
# ===========================================================================
# "single" : run one scenario with full visualisation (existing behaviour)
# "batch"  : iterate over all BATCH_SUFFIX files in BATCH_DIR, no per-frame
#             figures, aggregate statistics at the end.
RUN_MODE          = "single"           # "single" | "batch"
if _ap_args.mode is not None:
    RUN_MODE = _ap_args.mode
SINGLE_PARAMS_DIR = r"C:\DREAM_final\file_save\10_400"
BATCH_DIR         = r"C:\DREAM_final\file_save"
BATCH_SUFFIX      = "_100"            # 200 unique scenarios (0_100 .. 199_100)
BATCH_N_T         = 100               # steps per batch episode (= 10 s)
BATCH_OUT         = r"C:\DREAM_final\batch_results"

print("=" * 70)
print(f"UNCERTAINTY TEST  |  DREAM mode: {INTEGRATION_MODE.upper()}  |  RUN: {RUN_MODE.upper()}")
print(f"  Active planners: {' | '.join(sorted(RUN_PLANNERS))}")
print("=" * 70)

# ===========================================================================
# SCENARIO PARAMETERS
# ===========================================================================

# -- Truck designation (lane_idx, vehicle_idx): desired_speed [m/s] -----------
# (1, 1) = centre-lane slow truck.  Ego approaches from the left lane and the
# merger from the right lane; conflict can appear if ego later chooses centre.
# From 120_400 init: vehicle_centre[1] at s ≈ 51 m.
TRUCK_DESIGNATIONS = {
    (1, 1): 7.0,
    (1, 3): 6.5     # centre-lane slow truck (vd = 5.5 m/s)
}
_AGENT_TRUCK_LANE = 1   # lane 1 = centre
_AGENT_TRUCK_IDX  = 3   # vehicle_centre[2] at s ≈ 51 m

# -- Left-lane fast car (occluded, DRIFT-only until revealed) ---------------
# Hidden by the truck occlusion geometry.  Always in DRIFT (builds risk in the
# left shadow zone) but EXCLUDED from IDEAM's vehicle arrays until it exits
# the truck's occlusion shadow (geometric reveal) or fallback step fires.
LEFT_FAST_CAR_S_INIT          = 78.0   # [m] start ~20 m ahead of left truck
LEFT_FAST_CAR_VD              = 14.0   # [m/s] fast car desired speed
LEFT_FAST_CAR_REVEAL_FALLBACK = 90     # step — fires if occlusion never clears

# -- DRIFT risk boost around truck -----------------------------------------
TRUCK_RISK_BOOST    = 2.5
TRUCK_RISK_SIGMA    = 12.0
TRUCK_WEIGHT_SCALE  = 4.0
TRUCK_INFLUENCE_DIST = 70.0
TRUCK_PROACTIVE_DIST = 55.0
TRUCK_PROACTIVE_RISK = 0.35
TRUCK_SAFE_SPEED    = 7.5

# Asymmetric constraint scaling
TRUCK_LONG_EXTRA   = 8.0
TRUCK_TH_EXTRA     = 1.0
TRUCK_AL_SCALE     = 0.5
TRUCK_BL_SCALE     = 0.7
TRUCK_CENTER_RELAX = 4.0

# Squeeze avoidance
SQUEEZE_LON_DIST   = 25.0
SQUEEZE_MIN_ACCEL  = 0.8

# -- Occluded Merger (right lane → centre lane) ----------------------------
# IDEAM-controlled car in the right lane.  It starts alongside the centre-lane
# truck, stays in the right lane until it has safely overtaken that truck, and
# then performs a right→centre lane change using the same IDEAM controller
# structure as ``uncertainty_merger_DREAM.py``.
# Always in DRIFT; hidden from IDEAM until it is physically at least halfway
# across the lane change.
MERGER_VD                 = 10.5    # [m/s] desired speed (slightly > truck 5.5)
MERGER_LC_RELEASE_MARGIN  = 9.0    # [m] release centre-LC to baseline only after body clearance
MERGER_LANE_BLEND         = 0.24   # lane-change assist, matches merger benchmark
KEEP_LANE_ASSIST_BLEND    = 0.30

# Collision / near-collision thresholds
NEAR_COLLISION_DIST = 8.0          # [m]
COLLISION_DIST      = 3.0          # [m]

# -- Startup geometry constraints (checked/printed at init) ----------------
# Violation warnings are printed but do NOT halt the simulation.
EGO_FRONT_CLEAR_M    = 50.0  # [m] no centre-lane vehicle in (ego_s, ego_s+50m)
TRUCK_SIDE_EXCL_M    = 25.0  # [m] no lane-0/2 vehicle within ±25m of truck s

# -- Visualization ---------------------------------------------------------
CAR_LENGTH   = 3.5
CAR_WIDTH    = 1.2
TRUCK_LENGTH = 12.0
TRUCK_WIDTH  = 2.0
MERGER_SPAWN_MIN_GAP = 8.0  # [m] min longitudinal gap to existing right-lane cars
MERGER_SPAWN_TRUCK_BACKOFF = 5.0  # [m] prefer merger to start a bit behind the truck

EGO_DREAM_COLOR   = '#2196F3'   # Blue  (GVF-DRIFT, ours)
EGO_IDEAM_COLOR   = '#4CAF50'   # Green (baseline)
EGO_ADA_COLOR     = '#9C27B0'   # Purple
EGO_APF_COLOR     = '#009688'   # Teal
EGO_OACMPC_COLOR  = '#FF5722'   # Deep orange (OA-CMPC)
SURROUND_COLOR    = '#FFD600'   # Yellow
TRUCK_COLOR       = '#FF6F00'   # Dark orange
SHADOW_COLOR      = '#4A4A4A'   # Dark grey
AGENT_OCC_COLOR   = '#9C27B0'   # Purple (ghost / occluded)
AGENT_VIS_COLOR   = '#E91E63'   # Magenta (visible)

RISK_ALPHA  = 0.65
RISK_CMAP   = 'jet'
RISK_LEVELS = 40
RISK_VMAX   = 2.0

N_t = 250
dt  = 0.1
x_area = 50.0
y_area = 15.0

save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        "figsave_RL_v2")
os.makedirs(save_dir, exist_ok=True)

# ===========================================================================
# HELPER: draw a rotated rectangle for one vehicle
# ===========================================================================

def draw_vehicle_rect(ax, x, y, yaw_rad, length, width,
                      facecolor, edgecolor='black', lw=0.8,
                      zorder=3, alpha=1.0, linestyle='-'):
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


# ===========================================================================
# HELPER: truck occlusion shadow polygon (same as emergency_test_prideam.py)
# ===========================================================================

def compute_truck_shadow(ego_x, ego_y, truck_state, shadow_length=55.0):
    """Shadow polygon from ego through truck corners — returns Nx2 array or None."""
    tx, ty, yaw = truck_state[3], truck_state[4], truck_state[5]
    dx, dy = tx - ego_x, ty - ego_y
    dist = math.sqrt(dx**2 + dy**2)
    if dist < 3:
        return None

    L, W = TRUCK_LENGTH, TRUCK_WIDTH
    corners_local = np.array([[-L/2, -W/2], [L/2, -W/2],
                               [L/2,  W/2], [-L/2,  W/2]])
    cos_h, sin_h = math.cos(yaw), math.sin(yaw)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = (rot @ corners_local.T).T + np.array([tx, ty])

    angles = np.arctan2(corners[:, 1] - ego_y, corners[:, 0] - ego_x)
    left_corner  = corners[np.argmax(angles)]
    right_corner = corners[np.argmin(angles)]

    l_dir = left_corner  - np.array([ego_x, ego_y])
    l_dir /= (np.linalg.norm(l_dir) + 1e-9)
    r_dir = right_corner - np.array([ego_x, ego_y])
    r_dir /= (np.linalg.norm(r_dir) + 1e-9)

    return np.array([left_corner,
                     left_corner  + l_dir * shadow_length,
                     right_corner + r_dir * shadow_length,
                     right_corner])


def draw_shadow_polygon(ax, shadow_polygon, alpha=0.30):
    if shadow_polygon is None:
        return
    patch = plt.Polygon(shadow_polygon,
                        facecolor=SHADOW_COLOR, alpha=alpha,
                        edgecolor='red', linewidth=1.8,
                        linestyle='--', zorder=2)
    ax.add_patch(patch)


def empty_lane():
    return np.zeros((0, 8), dtype=float)


def sort_lane(arr):
    if arr is None:
        return empty_lane()
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return empty_lane()
    arr = arr.reshape(-1, 8)
    idx = np.argsort(arr[:, 0])
    return np.asarray(arr[idx], dtype=float)


def append_lane_row(lane_arr, row):
    if lane_arr is None:
        lane_base = empty_lane()
    else:
        lane_base = np.asarray(lane_arr, dtype=float)
        if lane_base.size == 0:
            lane_base = empty_lane()
        else:
            lane_base = lane_base.reshape(-1, 8)
    if row is None:
        return lane_base
    row_arr = np.asarray(row, dtype=float).reshape(1, 8)
    if lane_base.size == 0:
        return row_arr
    return np.vstack([lane_base, row_arr])


def nearest_longitudinal_neighbor(s_query, lane_arr):
    """Closest same-lane vehicle to ``s_query`` in Frenet-s."""
    lane = sort_lane(lane_arr)
    if lane.size == 0:
        return None, float("inf"), None
    s_vals = lane[:, 0].astype(float)
    idx = int(np.argmin(np.abs(s_vals - float(s_query))))
    s_near = float(s_vals[idx])
    return idx, abs(s_near - float(s_query)), s_near


def find_safe_merger_spawn_s(truck_s, right_lane_rows, min_gap=MERGER_SPAWN_MIN_GAP):
    """
    Pick the nearest non-overlapping right-lane spawn near the truck.

    If ``truck_s`` falls in an occupied region, project it to the nearest free
    interval boundary, with a slight preference for a behind-the-truck spawn on
    exact ties so the merger still overtakes from the right.
    """
    s_ref = max(0.0, float(truck_s))
    gap_req = max(0.0, float(min_gap))
    lane = sort_lane(right_lane_rows)

    if lane.size == 0:
        return {
            "s": s_ref,
            "offset": 0.0,
            "nearest_gap": float("inf"),
            "nearest_idx": None,
            "nearest_s": None,
        }

    s_vals = lane[:, 0].astype(float)
    intervals = []
    lo = 0.0
    for s_occ in s_vals:
        hi = float(s_occ) - gap_req
        if hi >= lo:
            intervals.append((lo, hi))
        lo = float(s_occ) + gap_req
    intervals.append((lo, float("inf")))

    best = None
    best_key = None
    for lo, hi in intervals:
        if s_ref < lo:
            cand = lo
        elif np.isfinite(hi) and s_ref > hi:
            cand = hi
        else:
            cand = s_ref

        cand = max(0.0, float(cand))
        offset = cand - s_ref
        bias = 0 if offset <= 0.0 else 1
        key = (abs(offset), bias)
        if best is None or key < best_key:
            best = cand
            best_key = key

    near_idx, near_gap, near_s = nearest_longitudinal_neighbor(best, lane)
    return {
        "s": best,
        "offset": float(best - s_ref),
        "nearest_gap": float(near_gap),
        "nearest_idx": near_idx,
        "nearest_s": near_s,
    }


# ===========================================================================
# LEFT LANE FAST CAR  (occluded, DRIFT-only until geometrically revealed)
# ===========================================================================

class LeftLaneFastCar:
    """
    A fast car in the left lane hidden by the truck occlusion geometry.

    Pre-reveal:
        Rides at ``s_init`` in the left lane at high ``vd``, updated by IDM
        using left-lane vehicles ahead as leaders.  Always fed to DRIFT
        (builds risk in the truck shadow zone) but EXCLUDED from IDEAM's
        vehicle arrays → IDEAM sees the left lane as clear.

    Reveal trigger (primary — geometric):
        Each step ``check_occlusion(ego_x, ego_y, truck_state)`` tests whether
        the car is still inside the left-truck's sight-line shadow polygon.
        Once it exits (ego shifts laterally during LC, widening the gap), the
        car is marked ``revealed = True`` and injected into ``vl_mpc``.

    Reveal trigger (fallback):
        If the geometric check never fires, forced at
        ``LEFT_FAST_CAR_REVEAL_FALLBACK`` step.
    """

    def __init__(self, paths, path_data, steer_range, s_init, vd, lane_idx=0):
        self.paths       = paths
        self.path_data   = path_data
        self.steer_range = steer_range
        self.lane_idx    = lane_idx
        self.vd          = vd
        self.model       = Curved_Road_Vehicle(**surrounding_params())

        path = paths[lane_idx]
        x0, y0   = path(s_init)
        psi0     = path.get_theta_r(s_init)
        xc, yc, samps = path_data[lane_idx]
        try:
            s0, ey0, epsi0 = find_frenet_coord(path, xc, yc, samps,
                                               [x0, y0, psi0])
        except Exception:
            s0, ey0, epsi0 = s_init, 0.0, 0.0
        self.state = np.array([s0, ey0, epsi0, x0, y0, psi0, vd, 0.0])

        self.revealed    = False
        self.reveal_step = None
        self.occluded    = True   # start inside truck shadow

    # ------------------------------------------------------------------
    def update(self, vehicle_left):
        """Advance one step using IDM (left-lane vehicles as leaders)."""
        s  = self.state[0]
        vx = self.state[6]
        path = self.paths[self.lane_idx]

        # Find nearest left-lane vehicle ahead
        s_ahead, v_ahead = None, None
        for veh in vehicle_left:
            gap = veh[0] - s - 3.5
            if gap > 2.0 and (s_ahead is None or veh[0] < s_ahead):
                s_ahead, v_ahead = veh[0], veh[6]

        x_n, y_n, psi_n, vx_n, _, a = self.model.update_states(
            s, vx, self.vd, s_ahead, v_ahead, path, self.steer_range)

        xc, yc, samps = self.path_data[self.lane_idx]
        try:
            s_n, ey_n, epsi_n = find_frenet_coord(
                path, xc, yc, samps, [x_n, y_n, psi_n])
        except Exception:
            s_n, ey_n, epsi_n = s + vx * dt, 0.0, 0.0
        self.state = np.array([s_n, ey_n, epsi_n, x_n, y_n, psi_n,
                               max(vx_n, 0.0), a])

    # ------------------------------------------------------------------
    def check_occlusion(self, ego_x, ego_y, truck_state):
        """Update self.occluded using left-truck sight-line shadow."""
        shadow_poly = compute_truck_shadow(ego_x, ego_y, truck_state)
        if shadow_poly is None:
            self.occluded = False
            return
        poly_path     = MplPath(shadow_poly)
        ax, ay        = float(self.state[3]), float(self.state[4])
        self.occluded = bool(poly_path.contains_point([ax, ay]))

    # ------------------------------------------------------------------
    def to_drift_vehicle(self, vid=997):
        """Always provide to DRIFT (builds risk even when occluded)."""
        psi = self.state[5]
        vx  = self.state[6]
        v   = drift_create_vehicle(
            vid=vid,
            x=float(self.state[3]), y=float(self.state[4]),
            vx=vx * math.cos(psi), vy=vx * math.sin(psi),
            vclass='car')
        v['heading'] = psi
        v['a']       = float(self.state[7])
        return v

    # ------------------------------------------------------------------
    def to_ideam_row(self):
        """Vehicle array row [s, ey, epsi, x, y, psi, vx, a]."""
        return self.state.copy()

    # ------------------------------------------------------------------
    @property
    def x(self):   return float(self.state[3])
    @property
    def y(self):   return float(self.state[4])
    @property
    def psi(self): return float(self.state[5])
    @property
    def vx(self):  return float(self.state[6])


# ===========================================================================
# OCCLUDED MERGER  (right lane → centre lane, simultaneous merge conflict)
# ===========================================================================

class OccludedMerger:
    """
    IDEAM-controlled car in the RIGHT lane that starts alongside the slow truck,
    stays hidden from MPC while it remains mostly in the right lane, and only
    releases centre-lane authority back to the baseline planner once it has
    safely cleared the truck.
    """

    def __init__(self, paths, path_data, s_init, vd, lane_blend=MERGER_LANE_BLEND):
        self.paths = paths
        self.path_data = path_data
        self.vd = float(vd)
        self.lane_blend = float(lane_blend)

        self.X0 = [self.vd, 0.0, 0.0, float(s_init), 0.0, 0.0]
        self.X0_g = [paths[2](s_init)[0], paths[2](s_init)[1], paths[2].get_theta_r(s_init)]
        self.oa = 0.0
        self.od = 0.0
        self.last_X = None
        self.path_changed = 2
        self.a = 0.0

        self.request_step = None
        self.request_reason = None
        self.lc_started_step = None
        self.lc_progress = 0.0
        self.last_update_ok = True
        self.last_error = None

        self.mpc = LMPC(**constraint_params())
        self.utils = LeaderFollower_Uitl(**util_params())
        self.mpc.set_util(self.utils)
        self.mpc.get_path_curvature(path=paths[2])
        self.decision = decision(**decision_params())

    # ------------------------------------------------------------------
    def trigger_lc(self, step_idx, reason="overtake-complete"):
        """Register that baseline control may now choose the centre lane."""
        if self.request_step is None:
            self.request_step = int(step_idx)
            self.request_reason = str(reason)

    # ------------------------------------------------------------------
    def update(self, vehicle_left, vehicle_centre, vehicle_right,
               truck_state, ego_state=None, ego_global=None, left_fast_row=None,
               step_idx=0):
        """Advance one timestep using IDEAM, not manual lane interpolation."""
        ego_row = None
        ego_lane = None
        if ego_state is not None and ego_global is not None:
            ego_row = state_to_row(ego_state, ego_global)
            ego_lane = lane_from_global(ego_global)

        lane_left = append_lane_row(vehicle_left, ego_row if ego_lane == 0 else None)
        lane_centre = append_lane_row(vehicle_centre, ego_row if ego_lane == 1 else None)
        lane_right = append_lane_row(vehicle_right, ego_row if ego_lane == 2 else None)
        lane_left = append_lane_row(lane_left, left_fast_row)
        lane_left = sort_lane(lane_left)
        lane_centre = sort_lane(lane_centre)
        lane_right = sort_lane(lane_right)

        lane_now = lane_from_global(self.X0_g)
        prev_progress = float(self.lc_progress)
        truck_s = float(truck_state[0]) if truck_state is not None else float("-inf")
        if (self.request_step is None and
                float(self.X0[3]) > truck_s + MERGER_LC_RELEASE_MARGIN):
            self.trigger_lc(step_idx, reason="truck-clearance-ready")

        ready_for_centre = self.request_step is not None

        force_target_lane = None
        if lane_now == 2 and not ready_for_centre:
            force_target_lane = 2

        out = ideam_agent_step(
            self.X0, self.X0_g, self.oa, self.od, self.last_X, self.path_changed,
            self.mpc, self.utils, self.decision, dynamics,
            lane_left, lane_centre, lane_right,
            force_target_lane=force_target_lane,
            bypass_probe_guard=True,
        )

        self.last_update_ok = bool(out.get("ok", False))
        self.last_error = out.get("error")
        self.X0 = list(out["X0"])
        self.X0_g = list(out["X0_g"])
        self.oa = out["oa"]
        self.od = out["od"]
        self.last_X = out["last_X"]
        self.path_changed = out["path_changed"]
        self.a = float(self.oa[0]) if hasattr(self.oa, "__len__") and len(self.oa) > 0 else 0.0

        if force_target_lane is not None:
            tgt = int(force_target_lane)
            if lane_from_global(self.X0_g) != tgt:
                blend = self.lane_blend if tgt == 1 else KEEP_LANE_ASSIST_BLEND
                self.X0, self.X0_g = blend_state_toward_lane(self.X0, self.X0_g, tgt, blend)

        self.lc_progress = lane_transition_progress(self.X0, self.X0_g, from_lane=2, to_lane=1)
        if (ready_for_centre and self.lc_started_step is None and
                (lane_from_global(self.X0_g) != 2 or self.lc_progress > max(0.02, prev_progress + 1e-3))):
            self.lc_started_step = int(step_idx)

    # ------------------------------------------------------------------
    def in_centre_lane(self):
        """True once the body has physically crossed the lane midpoint."""
        return self.lc_progress >= 0.5 or lane_from_global(self.X0_g) == 1

    def is_occluded(self):
        """Occluded from IDEAM while still primarily in right lane."""
        return not self.in_centre_lane()

    # ------------------------------------------------------------------
    def to_drift_vehicle(self, vid=998):
        """Always provided to DRIFT to build risk in the merge zone."""
        return row_to_drift_vehicle(self.to_ideam_row(), vid=vid, vclass='car')

    # ------------------------------------------------------------------
    def to_ideam_row(self):
        """Vehicle array row [s, ey, epsi, x, y, psi, vx, a]."""
        return state_to_row(self.X0, self.X0_g, accel=self.a)

    def to_center_lane_row(self):
        """Project onto centre-lane Frenet coordinates for IDEAM visibility."""
        try:
            s_c, ey_c, epsi_c = find_frenet_coord(
                path2c, x2c, y2c, samples2c, self.X0_g)
            return np.array([
                s_c, ey_c, epsi_c,
                self.x, self.y, self.psi, self.vx, self.a
            ], dtype=float)
        except Exception:
            return self.to_ideam_row()

    # ------------------------------------------------------------------
    @property
    def state(self):
        return self.to_ideam_row()

    @property
    def x(self):
        return float(self.X0_g[0])

    @property
    def y(self):
        return float(self.X0_g[1])

    @property
    def psi(self):
        return float(self.X0_g[2])

    @property
    def vx(self):
        return float(self.X0[0])


# ===========================================================================
# INITIALIZATION
# ===========================================================================

bar = Bar(max=N_t - 1)
boundary    = 1.0
steer_range = [math.radians(-8.0), math.radians(8.0)]

X0   = [8.0, 0.0, 0.0, 20.0, 0.0, 0.0]
X0_g = [path1c(X0[3])[0], path1c(X0[3])[1], path1c.get_theta_r(X0[3])]  # left lane

path_center  = np.array([path1c, path2c, path3c], dtype=object)
sample_center = np.array([samples1c, samples2c, samples3c], dtype=object)
x_center = [x1c, x2c, x3c]
y_center = [y1c, y2c, y3c]
x_bound  = [x1, x2]
y_bound  = [y1, y2]
path_bound        = [path1, path2]
path_bound_sample = [samples1, samples2]


def lane_from_global(X0_g_):
    return judge_current_position(X0_g_[0:2], x_bound, y_bound, path_bound, path_bound_sample)


def state_to_row(X0_, X0_g_, accel=0.0):
    return np.array([
        float(X0_[3]), float(X0_[4]), float(X0_[5]),
        float(X0_g_[0]), float(X0_g_[1]), float(X0_g_[2]),
        float(X0_[0]), float(accel)
    ], dtype=float)


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


def pick_group_for_lane(group_dict, lane_index):
    lane_to_groups = {
        0: ("L1", "L2"),
        1: ("C1", "C2"),
        2: ("R1", "R2"),
    }
    for name in lane_to_groups.get(int(lane_index), ()):
        if name in group_dict:
            return group_dict[name]
    return group_dict[next(iter(group_dict))]


def force_path_target(path_now, target_lane, X0_, X0_g_):
    target_lane = int(target_lane)
    path_d = [path1c, path2c, path3c][target_lane]
    _, x_list, y_list, sample = get_path_info(target_lane)

    if target_lane == path_now:
        C_label = "K"
    else:
        C_label = "R" if target_lane > path_now else "L"
    X0_forced = repropagate(path_d, sample, x_list, y_list, X0_g_, X0_)
    return path_d, target_lane, C_label, sample, x_list, y_list, X0_forced


def blend_state_toward_lane(X0_, X0_g_, target_lane, blend=0.25):
    blend = float(np.clip(blend, 0.0, 1.0))
    path_t = [path1c, path2c, path3c][int(target_lane)]
    s_ref = float(X0_[3])
    x_t, y_t = path_t(s_ref)
    psi_t = path_t.get_theta_r(s_ref)

    X0_g_new = list(X0_g_)
    X0_g_new[0] = (1.0 - blend) * X0_g_[0] + blend * x_t
    X0_g_new[1] = (1.0 - blend) * X0_g_[1] + blend * y_t
    dpsi = math.atan2(math.sin(psi_t - X0_g_[2]), math.cos(psi_t - X0_g_[2]))
    X0_g_new[2] = X0_g_[2] + blend * dpsi

    _, x_list, y_list, sample = get_path_info(int(target_lane))
    X0_new = repropagate(path_t, sample, x_list, y_list, X0_g_new, list(X0_))
    return X0_new, X0_g_new


def lane_transition_progress(X0_, X0_g_, from_lane=2, to_lane=1):
    path_from = [path1c, path2c, path3c][int(from_lane)]
    path_to = [path1c, path2c, path3c][int(to_lane)]
    s_ref = float(X0_[3])
    p_from = np.asarray(path_from(s_ref), dtype=float)
    p_to = np.asarray(path_to(s_ref), dtype=float)
    p_now = np.asarray([float(X0_g_[0]), float(X0_g_[1])], dtype=float)
    span = p_to - p_from
    denom = float(np.dot(span, span))
    if denom < 1e-9:
        return 1.0 if lane_from_global(X0_g_) == int(to_lane) else 0.0
    prog = float(np.dot(p_now - p_from, span) / denom)
    return float(np.clip(prog, 0.0, 1.0))

X_traj = [X0_g[0]]
Y_traj = [X0_g[1]]
oa, od  = 0.0, 0.0
path_desired = []
pathRecord   = [0]   # ego starts in left lane (index 0)

Params           = params()
Constraint_params = constraint_params()
dynamics         = Dynamic(**Params)
decision_param   = decision_params()
decision_maker   = decision(**decision_param)


def ideam_agent_step(
    X0_, X0_g_, oa_prev, od_prev, last_X_prev, path_changed_prev,
    mpc_ctrl, util_obj, decision_obj, dynamics_obj,
    lane_left, lane_center, lane_right,
    force_target_lane=None, bypass_probe_guard=False
):
    path_now = lane_from_global(X0_g_)
    path_ego = [path1c, path2c, path3c][path_now]
    start_group_str = {0: "L1", 1: "C1", 2: "R1"}[path_now]
    force_active = force_target_lane is not None

    try:
        if last_X_prev is None:
            ovx, ovy, owz, oS, oey, oepsi = clac_last_X(
                oa_prev, od_prev, mpc_ctrl.T, path_ego, dt, 6, X0_, X0_g_)
            last_X_prev = [ovx, ovy, owz, oS, oey, oepsi]

        all_info = util_obj.get_alllane_lf(path_ego, X0_g_, path_now, lane_left, lane_center, lane_right)
        group_dict, ego_group = util_obj.formulate_gap_group(
            path_now, last_X_prev, all_info, lane_left, lane_center, lane_right)

        desired_group = decision_obj.decision_making(group_dict, start_group_str)
        if force_active:
            desired_group = pick_group_for_lane(group_dict, int(force_target_lane))
        path_d, path_dindex, C_label, sample, x_list, y_list, X0_ = Decision_info(
            X0_, X0_g_, path_center, sample_center, x_center, y_center,
            boundary, desired_group, path_ego, path_now
        )
        if force_active and path_dindex != int(force_target_lane):
            path_d, path_dindex, C_label, sample, x_list, y_list, X0_ = force_path_target(
                path_now, int(force_target_lane), X0_, X0_g_
            )

        C_label_additive = util_obj.inquire_C_state(C_label, desired_group)
        C_label_virtual = C_label

        if C_label_additive == "Probe" and not (force_active and bypass_probe_guard):
            path_d, path_dindex, C_label_virtual = path_ego, path_now, "K"
            _, xc, yc, samplesc = get_path_info(path_dindex)
            X0_ = repropagate(path_d, samplesc, xc, yc, X0_g_, X0_)

        if path_changed_prev != path_dindex and last_X_prev is not None:
            mpc_ctrl.get_path_curvature(path=path_d)
            proj_s, proj_ey = path_to_path_proj(
                last_X_prev[3], last_X_prev[4], path_changed_prev, path_dindex)
            last_X_prev = [
                last_X_prev[0], last_X_prev[1], last_X_prev[2],
                proj_s, proj_ey, last_X_prev[5]
            ]
        path_changed_prev = path_dindex

        res = mpc_ctrl.iterative_linear_mpc_control(
            X0_, oa_prev, od_prev, dt, None, None, C_label, X0_g_, path_d, last_X_prev,
            path_now, ego_group, path_ego, desired_group,
            lane_left, lane_center, lane_right, path_dindex, C_label_additive, C_label_virtual
        )
        if res is None:
            raise RuntimeError("IDEAM MPC returned None")

        oa_cmd, od_cmd, ovx, ovy, owz, oS, oey, oepsi = res
        if oa_cmd is None or od_cmd is None:
            raise RuntimeError("IDEAM MPC returned empty controls")

        last_X = [ovx, ovy, owz, oS, oey, oepsi]
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0_, [oa_cmd[0], od_cmd[0]], dt, X0_g_, path_d, sample, x_list, y_list, boundary
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
        }
    except Exception as e:
        _, x_list, y_list, sample = get_path_info(path_now)
        oa_cmd = [0.0] * mpc_ctrl.T
        od_cmd = [0.0] * mpc_ctrl.T
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0_, [0.0, 0.0], dt, X0_g_, path_ego, sample, x_list, y_list, boundary
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


def rl_decision_step_test(
    X0_, X0_g_, oa_prev, od_prev, last_X_prev, path_changed_prev,
    policy, mpc_ctrl, util_obj, decision_obj, dynamics_obj,
    lane_left, lane_center, lane_right,
    force_target_lane=None, v_ref=8.0,
):
    """RL decision step for uncertainty_test_DREAM: policy picks lane+speed,
    MPC+CBF tracks."""
    try:
        lane_now = lane_from_global(X0_g_)
        obs = _build_dec_obs(
            X0_, X0_g_, lane_left, lane_center, lane_right,
            lane_now=lane_now, lane_rel_start=0, ey=0.0, epsi=0.0,
        )
        a = int(policy.act(obs, deterministic=True))
        lane_delta, _speed_mode = decode_action(a)
        v_target = target_speed_from_action(a, v_ref)

        ft_lane = int(np.clip(lane_now + lane_delta, 0, 2))
        if force_target_lane is not None:
            ft_lane = int(force_target_lane)

        mpc_ctrl.set_target_speed_override(float(v_target))
        out = ideam_agent_step(
            X0_, X0_g_, oa_prev, od_prev, last_X_prev, path_changed_prev,
            mpc_ctrl, util_obj, decision_obj, dynamics_obj,
            lane_left, lane_center, lane_right,
            force_target_lane=ft_lane, bypass_probe_guard=True,
        )
        try:
            mpc_ctrl.set_target_speed_override(None)
        except Exception:
            pass
        out["rl_action"] = a
        out["rl_target_speed"] = v_target
        out["rl_forced_lane"] = ft_lane
        return out
    except Exception as e:
        path_now = lane_from_global(X0_g_)
        path_ego = [path1c, path2c, path3c][path_now]
        _, x_list, y_list, sample = get_path_info(path_now)
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0_, [0.0, 0.0], dt, X0_g_, path_ego, sample, x_list, y_list, boundary
        )
        return {
            "ok": False, "error": str(e),
            "X0": X0_new, "X0_g": X0_g_new,
            "oa": [0.0] * mpc_ctrl.T, "od": [0.0] * mpc_ctrl.T,
            "last_X": last_X_prev, "path_changed": path_now,
            "path_d": path_ego, "sample": sample,
            "x_list": x_list, "y_list": y_list,
            "path_now": path_now, "path_dindex": path_now, "forced": False,
        }


# -- Expand DRIFT grid to cover full road extent ----------------------
# The default grid (x∈[-150, 255.2]) only covers part of the road.
# We compute the true road bounding box from loaded path coordinates
# and expand cfg before the PDESolver is instantiated.
_road_x_all = np.concatenate([x1c, x2c, x3c])
_road_y_all = np.concatenate([y1c, y2c, y3c])
_margin_m   = 25.0          # extra margin beyond road edge [m]
_dx_orig = (255.2 - (-150.0)) / (250 - 1)   # ≈ 1.62 m/cell (original)
_dy_orig = ((-45.3) - (-225.2)) / (80 - 1)  # ≈ 2.28 m/cell (original)

_new_x_min = min(float(np.min(_road_x_all)) - _margin_m, cfg.x_min)
_new_x_max = max(float(np.max(_road_x_all)) + _margin_m, cfg.x_max)
_new_y_min = min(float(np.min(_road_y_all)) - _margin_m, cfg.y_min)
_new_y_max = max(float(np.max(_road_y_all)) + _margin_m, cfg.y_max)

cfg.x_min, cfg.x_max = _new_x_min, _new_x_max
cfg.y_min, cfg.y_max = _new_y_min, _new_y_max
cfg.nx = max(250, int((_new_x_max - _new_x_min) / _dx_orig) + 2)
cfg.ny = max(80,  int((_new_y_max - _new_y_min) / _dy_orig) + 2)
cfg.dx = (_new_x_max - _new_x_min) / (cfg.nx - 1)
cfg.dy = (_new_y_max - _new_y_min) / (cfg.ny - 1)
cfg.x  = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
cfg.y  = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
cfg.X, cfg.Y = np.meshgrid(cfg.x, cfg.y)
print(f"[DRIFT Grid] x∈[{cfg.x_min:.0f}, {cfg.x_max:.0f}] m  "
      f"y∈[{cfg.y_min:.0f}, {cfg.y_max:.0f}] m  "
      f"nx={cfg.nx}, ny={cfg.ny}  "
      f"({cfg.nx*cfg.ny/1e3:.0f}k cells)")

# -- PRIDEAM controller ------------------------------------------------
controller = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        'mpc_cost':           config_integration.mpc_risk_weight,
        'cbf_modulation':     config_integration.cbf_alpha,
        'decision_threshold': config_integration.decision_risk_threshold,
    }
)
controller.get_path_curvature(path=path1c)   # ego starts in left lane

# -- ADA-source controller --------------------------------------------------
controller_ada = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        'mpc_cost':           config_integration.mpc_risk_weight,
        'cbf_modulation':     config_integration.cbf_alpha,
        'decision_threshold': config_integration.decision_risk_threshold,
    }
)
utils_ada = LeaderFollower_Uitl(**util_params())
controller_ada.set_util(utils_ada)
controller_ada.get_path_curvature(path=path1c)
decision_maker_ada = decision(**decision_params())

# -- APF-source controller --------------------------------------------------
controller_apf = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        'mpc_cost':           config_integration.mpc_risk_weight,
        'cbf_modulation':     config_integration.cbf_alpha,
        'decision_threshold': config_integration.decision_risk_threshold,
    }
)
utils_apf = LeaderFollower_Uitl(**util_params())
controller_apf.set_util(utils_apf)
controller_apf.get_path_curvature(path=path1c)
decision_maker_apf = decision(**decision_params())

# -- OA-CMPC standalone planner (MPC cost only — no CBF, no veto) -----------
# OA-CMPC uses its geometric occlusion risk as an MPC cost penalty but does
# not apply a CBF safety filter or a decision veto, mirroring the paper's
# pure-MPC design (ADMM multi-branch MPC without explicit CBF).
controller_oacmpc = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        'mpc_cost':           config_integration.mpc_risk_weight,
        'cbf_modulation':     0.0,           # no CBF
        'decision_threshold': float('inf'),  # no decision veto
    }
)
utils_oacmpc = LeaderFollower_Uitl(**util_params())
controller_oacmpc.set_util(utils_oacmpc)
controller_oacmpc.get_path_curvature(path=path1c)
decision_maker_oacmpc = decision(**decision_params())

# -- RL-decision controller ------------------------------------------------
# Uses its own PRIDEAM controller + DRIFT (no risk veto — RL picks the lane).
# The DecisionPolicy selects (lane, speed); the LMPC inside tracks.
controller_rl = None
utils_rl = decision_maker_rl = drift_rl = None
_rl_policy = None
EGO_RL_COLOR = "#795548"

if 'RL-PPO' in RUN_PLANNERS:
    if not _RL_DECISION_OK:
        print("[rl-dec] WARNING: RL-PPO selected but RL imports failed — skipping.")
        RUN_PLANNERS.discard('RL-PPO')
    else:
        # Load policy
        _rl_ckpt = _ap_args.rl_decision_checkpoint
        _rl_cands = [_rl_ckpt,
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'rl', 'checkpoints', 'decision_policy_bc.pt'),
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'rl', 'checkpoints', 'decision_policy_ppo.pt')]
        _rl_path = next((p for p in _rl_cands if p and os.path.isfile(p)), None)
        if _rl_path is None:
            print(f"[rl-dec] no checkpoint found — disabling RL-PPO arm")
            RUN_PLANNERS.discard('RL-PPO')
        else:
            _rl_policy = _load_bc_policy(_rl_path, device='cpu')
            print(f"[rl-dec] loaded: {_rl_path}")
            controller_rl = create_prideam_controller(
                paths={0: path1c, 1: path2c, 2: path3c},
                risk_weights={
                    'mpc_cost':           0.0,
                    'cbf_modulation':     0.0,
                    'decision_threshold': float('inf'),
                })
            utils_rl = LeaderFollower_Uitl(**util_params())
            controller_rl.set_util(utils_rl)
            controller_rl.get_path_curvature(path=path1c)
            decision_maker_rl = decision(**decision_params())

# ===========================================================================
# SINGLE-MODE SIMULATION  (skipped entirely when RUN_MODE == "batch")
# ===========================================================================
# -- Surrounding vehicles (from pickle for reproducibility) -----------
params_dir   = SINGLE_PARAMS_DIR
surroundings = Surrounding_Vehicles(steer_range, dt, boundary, params_dir)

util_params_ = util_params()
utils        = LeaderFollower_Uitl(**util_params_)
controller.set_util(utils)
mpc_controller = controller.mpc

# -- Baseline IDEAM MPC (independent parallel planner) ----------------
baseline_mpc_viz  = LMPC(**constraint_params())
utils_ideam_viz   = LeaderFollower_Uitl(**util_params_)
baseline_mpc_viz.set_util(utils_ideam_viz)
baseline_mpc_viz.get_path_curvature(path=path1c)   # same lane as ego: left

# IDEAM and DREAM must share identical initial ego state and lane so that
# the only performance difference comes from the planner, not from state bias.
X0_ideam   = [8.0, 0.0, 0.0, 20.0, 0.0, 0.0]      # identical to DREAM X0
X0_g_ideam = [path1c(X0_ideam[3])[0],               # ← path1c (left lane)
               path1c(X0_ideam[3])[1],
               path1c.get_theta_r(X0_ideam[3])]
oa_ideam_viz, od_ideam_viz  = 0.0, 0.0
last_X_ideam_viz             = None
last_ideam_panel_horizon_viz = None
path_changed_ideam_viz       = 0   # starts in left lane (index 0)

# Panel snapshots (safe on frame 0)
ideam_panel_X0   = list(X0_ideam)
ideam_panel_X0_g = list(X0_g_ideam)

path_changed = 0   # ego starts in left lane (index 0)

# ── Speed designations ───────────────────────────────────────────────────
# Applied BEFORE truck designations so truck is not accidentally overwritten.
#
# Exclusion-zone clearance:
#   Truck is now vehicle_centre[1] (s≈51 m).  Nearby centre/right vehicles can
#   still occupy the merge corridor around the truck, so boost their vd so they
#   clear the scene during pre-advance and the first few main-loop steps.
#   the first few main-loop steps, keeping the centre/right lanes open for
#   IDEAM's overtake and the merger's LC.
#   NOTE: _N_PREADVANCE=5 (0.5 s): centre[0]/right[0] only move ~4 m → safe.
surroundings.vd_center_all[1] = 30.0   # clears centre[1] (s≈50.5) — truck's centre-flank
surroundings.vd_right_all[1]  = 30.0   # clears right[1]  (s≈53)   — truck's right-flank

# -- Designate trucks & set desired speed (must come AFTER general boost) --
for (_tl, _tv), _tvd in TRUCK_DESIGNATIONS.items():
    if _tl == 0:   surroundings.vd_left_all[_tv]   = _tvd
    elif _tl == 1: surroundings.vd_center_all[_tv] = _tvd
    elif _tl == 2: surroundings.vd_right_all[_tv]  = _tvd

# Path data tuple list for find_frenet_coord
_path_data = [
    (x1c, y1c, samples1c),
    (x2c, y2c, samples2c),
    (x3c, y3c, samples3c),
]

# Construct LeftLaneFastCar — fast occluded car in left lane hidden by the
# truck geometry.  Always in DRIFT, revealed to IDEAM geometrically.
left_fast_car = LeftLaneFastCar(
    paths       = [path1c, path2c, path3c],
    path_data   = _path_data,
    steer_range = steer_range,
    s_init      = LEFT_FAST_CAR_S_INIT,
    vd          = LEFT_FAST_CAR_VD,
    lane_idx    = 0,   # left lane
)

# OccludedMerger is constructed AFTER the pre-advance loop (below) so its
# initial s can be set relative to the post-pre-advance truck position.

# ===========================================================================
# CONVERT IDEAM VEHICLES → DRIFT FORMAT
# ===========================================================================

def convert_to_drift(vehicle_left, vehicle_centre, vehicle_right,
                     truck_set=None, max_per_lane=5, exclude_set=None):
    """exclude_set: set of (lane_idx, vehicle_idx) to skip (LC agent slots)."""
    vehicles = []
    vid = 1
    for lane_idx, arr in enumerate([vehicle_left, vehicle_centre, vehicle_right]):
        if arr is None:
            continue
        count = 0
        for ri, row in enumerate(arr):
            if exclude_set and (lane_idx, ri) in exclude_set:
                continue   # this slot is now managed by an IdeamDecisionLCAgent
            is_truck = truck_set is not None and (lane_idx, ri) in truck_set
            if max_per_lane is not None and count >= max_per_lane and not is_truck:
                continue
            if len(row) < 7:
                continue
            psi   = row[5]
            v_lon = row[6]
            accel = row[7] if len(row) > 7 else 0.0
            vclass = 'truck' if is_truck else 'car'
            v = drift_create_vehicle(
                vid=vid,
                x=row[3], y=row[4],
                vx=v_lon * math.cos(psi),
                vy=v_lon * math.sin(psi),
                vclass=vclass)
            v['a']       = accel
            v['heading'] = psi
            vehicles.append(v)
            vid += 1
            count += 1
    return vehicles


# ===========================================================================
# DRIFT INITIALIZATION + ROAD MASK
# ===========================================================================

drift = controller.drift

print("Computing road boundary mask...")
_step    = 50
left_edge  = np.column_stack([x[::_step],  y[::_step]])
right_edge = np.column_stack([x3[::_step], y3[::_step]])
road_polygon  = np.vstack([left_edge, right_edge[::-1]])
road_mpl_path = MplPath(road_polygon)

grid_pts = np.column_stack([cfg.X.ravel(), cfg.Y.ravel()])
inside   = road_mpl_path.contains_points(grid_pts).reshape(cfg.X.shape).astype(float)
road_mask = np.clip(_gf(inside, sigma=1.5), 0, 1)
drift.set_road_mask(road_mask)
print(f"  Road mask: {np.sum(inside > 0.5)} / {inside.size} on-road "
      f"({100*np.mean(inside > 0.5):.1f}%)")

# ===========================================================================
# PRE-ADVANCE SURROUNDINGS — clear exclusion-zone positions before warm-up
# ===========================================================================
# Root cause of position bug: surroundings loads positions from the file
# (120_400) at init.  Setting vd_left_all[1]=30 and vd_right_all[1]=30
# only changes the IDM desired speed; it does NOT move left[1] (s≈51 m) and
# right[1] (s≈53 m) away from the truck (s≈50 m) at t=0.
# Fix: run total_update_emergency() for _N_PREADVANCE steps so that the two
# fast vehicles (vd=30 m/s) physically clear the truck's flanks before the
# DRIFT warm-up begins and before the main loop reads their positions.
_N_PREADVANCE = 5    # 0.5 s at dt=0.1 s; left[1]/right[1] at vd=30 move ≈15 m clear;
                     # centre[0] at v≈8 only moves ≈4 m → stays at s≈5 m, well behind ego
print(f"Pre-advancing surroundings {_N_PREADVANCE} steps "
      f"({_N_PREADVANCE * dt:.1f} s) to clear exclusion zones ...")
for _pre in range(_N_PREADVANCE):
    surroundings.total_update_emergency(_pre)
print("  Pre-advance complete.")

# Construct OccludedMerger using post-pre-advance truck position, then
# run DRIFT warm-up for 5 s so risk accumulates before t=0.
print("DRIFT warm-up (5 s)...")
vl0, vc0, vr0 = surroundings.get_vehicles_states()

_init_truck_state = [vl0, vc0, vr0][_AGENT_TRUCK_LANE][_AGENT_TRUCK_IDX]
_init_truck_s     = float(_init_truck_state[0])
_merger_spawn     = find_safe_merger_spawn_s(
    _init_truck_s - MERGER_SPAWN_TRUCK_BACKOFF, vr0)
_merger_s_init    = float(_merger_spawn["s"])
merger = OccludedMerger(
    paths       = [path1c, path2c, path3c],
    path_data   = _path_data,
    s_init      = _merger_s_init,
    vd          = MERGER_VD,
)
_spawn_neighbor_idx = _merger_spawn["nearest_idx"]
_spawn_neighbor_txt = (
    "none" if _spawn_neighbor_idx is None
    else f"right[{_spawn_neighbor_idx}] at s={_merger_spawn['nearest_s']:.1f} m"
)
print(f"[MERGER INIT] truck_s={_init_truck_s:.1f} m  "
      f"→ merger safe-spawn at s={merger.state[0]:.1f} m  "
      f"(offset={merger.state[0]-_init_truck_s:+.1f} m from truck, "
      f"nearest-gap={_merger_spawn['nearest_gap']:.1f} m to {_spawn_neighbor_txt}, "
      f"vd={MERGER_VD} m/s)")
print(f"[LEFT_FAST_CAR INIT] left lane  s={left_fast_car.state[0]:.1f} m  "
      f"vd={LEFT_FAST_CAR_VD} m/s")

vd_init = convert_to_drift(vl0, vc0, vr0,
                           truck_set=set(TRUCK_DESIGNATIONS.keys()))
vd_init.append(merger.to_drift_vehicle(vid=998))        # merger always in DRIFT
vd_init.append(left_fast_car.to_drift_vehicle(vid=997)) # fast car always in DRIFT

_psi0 = X0_g[2]
ego_init = drift_create_vehicle(
    vid=0,
    x=X0_g[0], y=X0_g[1],
    vx=X0[0] * math.cos(_psi0) - X0[1] * math.sin(_psi0),
    vy=X0[0] * math.sin(_psi0) + X0[1] * math.cos(_psi0),
    vclass='car')
ego_init['heading'] = _psi0

drift.warmup(vd_init, ego_init, dt=dt, duration=5.0, substeps=3)

# ADA-source DRIFT
drift_ada = controller_ada.drift
if 'ADA' in RUN_PLANNERS:
    drift_ada.set_road_mask(road_mask)
    print("ADA-DRIFT warm-up (5 s)...")
    drift_ada.warmup(vd_init, ego_init, dt=dt, duration=5.0, substeps=3,
                     source_fn=compute_Q_ADA)
else:
    print("ADA warm-up skipped (arm not selected).")

# APF-source DRIFT
drift_apf = controller_apf.drift
if 'APF' in RUN_PLANNERS:
    drift_apf.set_road_mask(road_mask)
    print("APF-DRIFT warm-up (5 s)...")
    drift_apf.warmup(vd_init, ego_init, dt=dt, duration=5.0, substeps=3,
                     source_fn=compute_Q_APF)
else:
    print("APF warm-up skipped (arm not selected).")

# OA-CMPC standalone planner DRIFT (MPC cost only, no CBF/veto)
drift_oacmpc = controller_oacmpc.drift
if 'OA-CMPC' in RUN_PLANNERS:
    drift_oacmpc.set_road_mask(road_mask)
    print("OA-CMPC-DRIFT warm-up (5 s)...")
    drift_oacmpc.warmup(vd_init, ego_init, dt=dt, duration=5.0, substeps=3,
                        source_fn=compute_Q_OACMPC)
else:
    print("OA-CMPC warm-up skipped (arm not selected).")
print()

# RL-PPO DRIFT warm-up (no custom source — standard DRIFT physics)
if 'RL-PPO' in RUN_PLANNERS and drift_rl is not None:
    print("RL-PPO DRIFT warm-up (5 s)...")
    drift_rl.warmup(vd_init, ego_init, dt=dt, duration=5.0, substeps=3)
else:
    print("RL-PPO warm-up skipped (arm not selected).")
print()

# ===========================================================================
# STEP 1 — SCENARIO VERIFICATION  (printed once at startup)
# ===========================================================================

_lane_names = {0: 'left', 1: 'centre', 2: 'right'}
_truck_vd   = TRUCK_DESIGNATIONS[(_AGENT_TRUCK_LANE, _AGENT_TRUCK_IDX)]
_truck_arr0 = [vl0, vc0, vr0][_AGENT_TRUCK_LANE]
_truck_s    = float(_truck_arr0[_AGENT_TRUCK_IDX][0])
_ego_s      = float(X0[3])
_mg_right_idx, _mg_right_gap, _mg_right_s = nearest_longitudinal_neighbor(
    merger.state[0], vr0)

print("=" * 70)
print("SCENARIO GEOMETRY VERIFICATION")
print("=" * 70)

# ── Actors ────────────────────────────────────────────────────────────────
print(f"  EGO          : left lane    s={_ego_s:.1f} m  vx={X0[0]:.1f} m/s")
print(f"  TRUCK        : {_lane_names[_AGENT_TRUCK_LANE]} lane  idx={_AGENT_TRUCK_IDX}"
      f"  s={_truck_s:.1f} m  vd={_truck_vd} m/s")
print(f"  MERGER       : right lane  s={merger.state[0]:.1f} m"
      f"  (truck offset={merger.state[0]-_truck_s:+.1f} m, "
      f"nearest right gap={_mg_right_gap:.1f} m, vd={MERGER_VD} m/s, "
      f"LC=IDEAM-controlled)")
print(f"  LEFT_FAST_CAR: left lane   s={left_fast_car.state[0]:.1f} m  "
      f"vd={LEFT_FAST_CAR_VD:.1f} m/s  (occluded={left_fast_car.occluded})")

# ── Constraint checks ─────────────────────────────────────────────────────
_violations = []

# Check 1: merger is near the truck and not overlapping right-lane traffic
_mg_s = float(merger.state[0])
print(f"\n  Constraint 1 - merger safe-spawn near truck "
      f"(truck s={_truck_s:.1f} m, merger s={_mg_s:.1f} m):")
if _mg_right_gap >= MERGER_SPAWN_MIN_GAP and abs(_mg_s - _truck_s) < 12.0:
    print(f"    OK  merger at s={_mg_s:.1f} m  "
          f"(truck offset={_mg_s-_truck_s:+.1f} m, "
          f"nearest right gap={_mg_right_gap:.1f} m)")
else:
    _mg_right_tag = "none" if _mg_right_idx is None else f"right[{_mg_right_idx}]"
    _mg_right_s_txt = "" if _mg_right_s is None else f" at s={_mg_right_s:.1f} m"
    _msg = (f"    WARN  merger at s={_mg_s:.1f} m  "
            f"(truck offset={_mg_s-_truck_s:+.1f} m, "
            f"nearest right gap={_mg_right_gap:.1f} m to {_mg_right_tag}"
            f"{_mg_right_s_txt})")
    print(_msg)
    _violations.append(_msg)

# Check 2: left fast car is ahead of ego
print(f"\n  Constraint 2 — left fast car ahead of ego "
      f"(s_fc={left_fast_car.state[0]:.1f} m > ego s={_ego_s:.1f} m):")
if left_fast_car.state[0] > _ego_s:
    print(f"    OK  fast car s={left_fast_car.state[0]:.1f} m  "
          f"(+{left_fast_car.state[0]-_ego_s:.1f} m ahead)")
else:
    _msg = f"    WARN  fast car s={left_fast_car.state[0]:.1f} m behind ego"
    print(_msg)
    _violations.append(_msg)

# Check 3: (placeholder for truck-side exclusion zone)
_ego_front_end = _ego_s + EGO_FRONT_CLEAR_M  # kept for compatibility
_lane_specs_for_check = [
    (0, vl0, 'left', surroundings.vd_left_all),
    (1, vc0, 'centre', surroundings.vd_center_all),
    (2, vr0, 'right', surroundings.vd_right_all),
]
_lane_specs_for_check = [spec for spec in _lane_specs_for_check if spec[0] != _AGENT_TRUCK_LANE]
print(f"\n  Constraint 3 — truck-side exclusion zone "
      f"(±{TRUCK_SIDE_EXCL_M:.0f} m of truck s={_truck_s:.1f} m, non-truck lanes):")
_excl_lo, _excl_hi = _truck_s - TRUCK_SIDE_EXCL_M, _truck_s + TRUCK_SIDE_EXCL_M
for _li, _lr, _lname, _vd_arr in _lane_specs_for_check:
    for _vi, _veh in enumerate(_lr):
        _vs = float(_veh[0])
        if _excl_lo < _vs < _excl_hi:
            _vd = _vd_arr[_vi]
            _ok = _vd >= 20.0
            _tag = "OK (fast, will clear)" if _ok else "WARN"
            _msg = (f"    {_tag}  {_lname}[{_vi}] at s={_vs:.1f} m"
                    f"  (Δ={_vs-_truck_s:+.1f} m)  vd={_vd:.1f} m/s")
            print(_msg)
            if not _ok:
                _violations.append(_msg)

# Check 4: sight-line shadow (informational)
_shadow_init = compute_truck_shadow(X0_g[0], X0_g[1],
                                    _truck_arr0[_AGENT_TRUCK_IDX])
if _shadow_init is not None:
    _mg_in_sl = bool(MplPath(_shadow_init).contains_point([merger.x, merger.y]))
else:
    _mg_in_sl = False
print(f"\n  Constraint 4 — ego sight-line shadow covers merger: {_mg_in_sl}")
if not _mg_in_sl:
    print("    (DRIFT Q_occlusion cone is typically wider than the geometric shadow)")

# ── Summary ───────────────────────────────────────────────────────────────
print()
if _violations:
    print(f"  *** {len(_violations)} geometry violation(s) detected — "
          f"review vd settings or vehicle positions ***")
else:
    print("  All geometry constraints satisfied.")

_drift_trucks = [v for v in vd_init if v.get('class') == 'truck']
print(f"  DRIFT trucks: {len(_drift_trucks)} tagged vclass='truck'")
print("=" * 70)
print()

# ===========================================================================
# METRIC RECORDS
# ===========================================================================

risk_at_ego_list      = []
risk_at_ego_ada_list  = []
risk_at_ego_apf_list  = []
risk_at_ego_oacmpc_list = []

# DREAM (GVF-DRIFT, ours)
dream_s          = []
dream_vx         = []
dream_vy         = []
dream_acc        = []
dream_s_obs      = []
dream_dist_agent = []

# IDEAM (baseline)
ideam_s          = []
ideam_vx         = []
ideam_vy         = []
ideam_acc        = []
ideam_s_obs      = []
ideam_dist_agent = []

# ADA-DRIFT (static)
ada_s            = []
ada_vx           = []
ada_acc          = []
ada_s_obs        = []
ada_dist_agent   = []

# APF-DRIFT (static)
apf_s            = []
apf_vx           = []
apf_acc          = []
apf_s_obs        = []
apf_dist_agent   = []

# OA-CMPC-DRIFT
oacmpc_s         = []
oacmpc_vx        = []
oacmpc_acc       = []
oacmpc_s_obs     = []
oacmpc_dist_agent = []

# RL-PPO decision
rl_s             = []
rl_vx            = []
rl_acc           = []
rl_s_obs         = []
rl_dist_agent    = []

# Phantom occlusion flag per step
agent_occluded_record = []
merger_truck_dist = []

# TTC to OccludedMerger (per step, all planners)
dream_ttc  = []
ideam_ttc  = []
rl_ttc     = []
ada_ttc    = []
apf_ttc    = []
oacmpc_ttc = []

# Risk-exposure integrand: R(ego, t) * v_x(t)  [units: (risk) * m/s]
# All planners evaluated on the DREAM GVF field for fair comparison.
dream_rei_integrand  = []
ideam_rei_integrand  = []
rl_rei_integrand     = []
ada_rei_integrand    = []
apf_rei_integrand    = []
oacmpc_rei_integrand = []

# Per-step planning times [s] — used for t_plan_mean, t_plan_max, r_RT
dream_step_times  = []
ideam_step_times  = []
rl_step_times     = []
ada_step_times    = []
apf_step_times    = []
oacmpc_step_times = []

# MPC failure counters
dream_mpc_fail = 0
ideam_mpc_fail = 0
ada_mpc_fail   = 0
apf_mpc_fail   = 0

# Risk-crossing step (first step DREAM risk exceeds decision threshold)
_risk_cross_step = None

# -- ADA/APF arm state (mirrors DREAM state variables) ----------------------
X0_ada   = [8.0, 0.0, 0.0, 20.0, 0.0, 0.0]
X0_g_ada = [path1c(X0_ada[3])[0], path1c(X0_ada[3])[1], path1c.get_theta_r(X0_ada[3])]  # left
X0_apf   = [8.0, 0.0, 0.0, 20.0, 0.0, 0.0]
X0_g_apf = [path1c(X0_apf[3])[0], path1c(X0_apf[3])[1], path1c.get_theta_r(X0_apf[3])]  # left

_arm_ada_state = dict(
    X0=X0_ada, X0_g=X0_g_ada, oa=0.0, od=0.0, last_X=None,
    ovx=0.0, ovy=0.0, owz=0.0, oS=X0_ada[3], oey=0.0, oepsi=0.0,
    path_changed=0, path_d=path1c,
    controller=controller_ada, drift_obj=drift_ada, source_fn=compute_Q_ADA,
    utils_obj=utils_ada, decision_obj=decision_maker_ada,
    d0_base=utils_ada.d0, Th_base=utils_ada.Th,
    al_base=controller_ada.mpc.a_l, bl_base=controller_ada.mpc.b_l,
    P_base=controller_ada.mpc.P.copy(), proactive_cooldown=0,
)
_arm_apf_state = dict(
    X0=X0_apf, X0_g=X0_g_apf, oa=0.0, od=0.0, last_X=None,
    ovx=0.0, ovy=0.0, owz=0.0, oS=X0_apf[3], oey=0.0, oepsi=0.0,
    path_changed=0, path_d=path1c,
    controller=controller_apf, drift_obj=drift_apf, source_fn=compute_Q_APF,
    utils_obj=utils_apf, decision_obj=decision_maker_apf,
    d0_base=utils_apf.d0, Th_base=utils_apf.Th,
    al_base=controller_apf.mpc.a_l, bl_base=controller_apf.mpc.b_l,
    P_base=controller_apf.mpc.P.copy(), proactive_cooldown=0,
)

X0_oacmpc   = [8.0, 0.0, 0.0, 20.0, 0.0, 0.0]
X0_g_oacmpc = [path1c(X0_oacmpc[3])[0], path1c(X0_oacmpc[3])[1], path1c.get_theta_r(X0_oacmpc[3])]
_arm_oacmpc_state = dict(
    X0=X0_oacmpc, X0_g=X0_g_oacmpc, oa=0.0, od=0.0, last_X=None,
    ovx=0.0, ovy=0.0, owz=0.0, oS=X0_oacmpc[3], oey=0.0, oepsi=0.0,
    path_changed=0, path_d=path1c,
    controller=controller_oacmpc, drift_obj=drift_oacmpc, source_fn=compute_Q_OACMPC,
    utils_obj=utils_oacmpc, decision_obj=decision_maker_oacmpc,
    d0_base=utils_oacmpc.d0, Th_base=utils_oacmpc.Th,
    al_base=controller_oacmpc.mpc.a_l, bl_base=controller_oacmpc.mpc.b_l,
    P_base=controller_oacmpc.mpc.P.copy(), proactive_cooldown=0,
)

# -- RL-decision arm state -------------------------------------------------
if 'RL-PPO' in RUN_PLANNERS and controller_rl is not None:
    X0_rl   = [8.0, 0.0, 0.0, 20.0, 0.0, 0.0]
    X0_g_rl = [path1c(X0_rl[3])[0], path1c(X0_rl[3])[1], path1c.get_theta_r(X0_rl[3])]
    oa_rl, od_rl = 0.0, 0.0
    last_X_rl = None
    path_changed_rl = 0
    drift_rl = controller_rl.drift
    drift_rl.set_road_mask(road_mask)
else:
    X0_rl = [8.0, 0.0, 0.0, 20.0, 0.0, 0.0]
    X0_g_rl = [path1c(X0_rl[3])[0], path1c(X0_rl[3])[1], path1c.get_theta_r(X0_rl[3])]
    oa_rl = od_rl = 0.0
    last_X_rl = None
    path_changed_rl = 0

# -- TTC helper (used every step) ------------------------------------------
_CAR_LEN_TTC = 5.0   # effective bumper-to-bumper subtraction [m]
_TTC_CAP     = 60.0  # cap for "no closing / no conflict" [s]

def _ttc_to_agent(ego_pos, ego_vx, ag_x, ag_y, ag_vx):
    """
    Conservative point-mass longitudinal TTC.

    Closing rate = ego_vx - agent_vx (projected onto ego→agent axis).
    Subtracted CAR length from centre-to-centre distance.
    """
    # Explicit float casts prevent CasADi symbolic values leaking in
    dx   = float(ag_x)     - float(ego_pos[0])
    dy   = float(ag_y)     - float(ego_pos[1])
    dist = math.hypot(dx, dy)
    dist_bumper = max(dist - _CAR_LEN_TTC, 0.0)
    if dist_bumper <= 0.0:
        return 0.0
    cos_th  = dx / max(dist, 0.1)
    v_close = (float(ego_vx) - float(ag_vx)) * cos_th
    if v_close <= 1e-3:
        return _TTC_CAP
    return float(min(dist_bumper / v_close, _TTC_CAP))


# IDEAM diagnostic tracking
_prev_path_di_i          = None   # IDEAM desired-path index from previous step
_prev_path_now_i         = None   # IDEAM actual lane from previous step
_ideam_first_lcc_step    = None   # diagnostic: first IDEAM centre-lane command
_merger_ttc_at_trigger   = None   # TTC (IDEAM-to-merger) at LC trigger moment

# ===========================================================================
# HELPER: plot risk overlay (same as emergency_test_prideam.py)
# ===========================================================================

def plot_risk_overlay(ax):
    """Overlay DRIFT risk field on current axis (using module-level risk_field)."""
    R_sm = _gf(risk_field, sigma=0.8)
    R_sm = np.clip(R_sm, 0, RISK_VMAX)
    if True:
        cf = ax.contourf(cfg.X, cfg.Y, R_sm,
                         levels=RISK_LEVELS, cmap=RISK_CMAP,
                         alpha=RISK_ALPHA, vmin=0, vmax=RISK_VMAX,
                         zorder=1, extend='max')
    ax.contour(cfg.X, cfg.Y, R_sm,
               levels=np.linspace(0.2, RISK_VMAX, 8),
               colors='darkred', linewidths=0.5, alpha=0.4, zorder=1)
    return cf


# ===========================================================================
# HELPER: draw one scenario panel
# ===========================================================================

def draw_panel(ax, ego_global, ego_state, vehicle_left, vehicle_centre,
               vehicle_right, x_range, y_range, title,
               horizon=None, risk_f=None, risk_val=None,
               show_merger=True, ego_color=None):
    """Draw DREAM or IDEAM panel with road, vehicles, ego, and optional risk overlay."""
    plt.sca(ax)
    ax.cla()

    # Road geometry
    plot_env()

    # Risk overlay (DREAM panel only)
    cf = None
    if risk_f is not None:
        R_sm = _gf(risk_f, sigma=0.8)
        R_sm = np.clip(R_sm, 0, RISK_VMAX)
        cf = ax.contourf(cfg.X, cfg.Y, R_sm,
                         levels=RISK_LEVELS, cmap=RISK_CMAP,
                         alpha=RISK_ALPHA, vmin=0, vmax=RISK_VMAX,
                         zorder=1, extend='max')
        ax.contour(cfg.X, cfg.Y, R_sm,
                   levels=np.linspace(0.2, RISK_VMAX, 8),
                   colors='darkred', linewidths=0.5, alpha=0.4, zorder=1)

    # Horizon
    if horizon is not None and len(horizon) > 0:
        h = np.asarray(horizon)
        if h.ndim == 2 and h.shape[1] >= 2:
            ax.plot(h[:, 0], h[:, 1], color='#00BCD4', lw=1.8, ls='--', zorder=7)
            ax.scatter(h[:, 0], h[:, 1], color='#00BCD4', s=6, zorder=7)

    # Surrounding vehicles
    for li, (lane_arr, pr, xr, yr, sr) in enumerate([
        (vehicle_left,   path1c, x1c, y1c, samples1c),
        (vehicle_centre, path2c, x2c, y2c, samples2c),
        (vehicle_right,  path3c, x3c, y3c, samples3c),
    ]):
        for vi in range(len(lane_arr)):
            is_truck = (li, vi) in TRUCK_DESIGNATIONS
            veh = lane_arr[vi]
            in_view = (x_range[0] <= veh[3] <= x_range[1] and
                       y_range[0] <= veh[4] <= y_range[1])
            if not in_view:
                continue
            if is_truck:
                shadow_p = compute_truck_shadow(ego_global[0], ego_global[1], veh)
                draw_shadow_polygon(ax, shadow_p)
                draw_vehicle_rect(ax, veh[3], veh[4], veh[5],
                                  TRUCK_LENGTH, TRUCK_WIDTH, TRUCK_COLOR,
                                  edgecolor='darkred', lw=1.2, zorder=5)
                ax.text(veh[3] - 2, veh[4] + 2.5,
                        f"Truck {veh[6]:.1f} m/s",
                        rotation=np.rad2deg(veh[5]),
                        c='darkred', fontsize=5, style='oblique',
                        fontweight='bold')
            else:
                draw_vehicle_rect(ax, veh[3], veh[4], veh[5],
                                  CAR_LENGTH, CAR_WIDTH, SURROUND_COLOR,
                                  edgecolor='black', lw=0.6, zorder=4)
            if in_view:
                sx, ey_, _ = find_frenet_coord(pr, xr, yr, sr,
                                               [veh[3], veh[4], veh[5]])
                tx, ty = pr.get_cartesian_coords(sx - 4.75, ey_ - 1.0)
                ax.text(tx, ty, f"{veh[6]:.1f}", rotation=np.rad2deg(veh[5]),
                        c='k', fontsize=4, style='oblique')

    # OccludedMerger — draw as a plain solid vehicle without extra annotations.
    if show_merger:
        _mx, _my = merger.x, merger.y
        _m_in_view = (x_range[0] <= _mx <= x_range[1] and
                      y_range[0] <= _my <= y_range[1])
        if _m_in_view:
            draw_vehicle_rect(ax, _mx, _my, merger.psi,
                              CAR_LENGTH, CAR_WIDTH, AGENT_VIS_COLOR,
                              edgecolor='darkred', lw=1.2, zorder=5)

    # Ego vehicle
    if ego_color is not None:
        facecolor = ego_color
    else:
        facecolor = EGO_DREAM_COLOR if risk_f is not None else EGO_IDEAM_COLOR
    draw_vehicle_rect(ax, ego_global[0], ego_global[1], ego_global[2],
                      CAR_LENGTH, CAR_WIDTH, facecolor,
                      edgecolor='navy', lw=1.0, zorder=6)
    try:
        ego_sp = find_frenet_coord(path1c, x1c, y1c, samples1c, ego_global)
        _ref_path, _ref_s = path1c, ego_sp[0]
    except Exception:
        ego_sp = find_frenet_coord(path2c, x2c, y2c, samples2c, ego_global)
        _ref_path, _ref_s = path2c, ego_sp[0]
    tx, ty = _ref_path.get_cartesian_coords(_ref_s - 5.1, ego_sp[1] - 1.0)
    ax.text(tx, ty, f"{ego_state[0]:.1f} m/s",
            rotation=np.rad2deg(_ref_path.get_theta_r(_ref_s)),
            c='black', fontsize=5, style='oblique')

    # Risk value badge
    if risk_val is not None:
        col = 'red' if risk_val > 1.5 else ('orange' if risk_val > 0.5 else 'green')
        ax.text(0.985, 0.965, f"R={risk_val:.2f}",
                transform=ax.transAxes, ha='right', va='top',
                c=col, fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    return cf


# ===========================================================================
# HELPER: build MPC horizon for visualization
# ===========================================================================

def build_horizon(X0_seed, X0_g_seed, ctrl_a, ctrl_d,
                  path_d, sample, x_list, y_list):
    X_v = list(X0_seed)
    Xg_v = list(X0_g_seed)
    horizon = [list(Xg_v)]
    n = max(0, len(ctrl_a) - 1) if ctrl_a is not None else 0
    for k in range(n):
        if ctrl_a is not None and ctrl_d is not None:
            u = [ctrl_a[k + 1], ctrl_d[k + 1]]
        else:
            u = [0.0, 0.0]
        X_v, Xg_v, _, _ = dynamics.propagate(
            X_v, u, dt, Xg_v, path_d, sample, x_list, y_list, boundary)
        horizon.append(list(Xg_v))
    return np.array(horizon)


# ===========================================================================
# BASELINE PARAMETERS (restored each step)
# ===========================================================================

_d0_base  = utils.d0
_Th_base  = utils.Th
_al_base  = controller.mpc.a_l
_bl_base  = controller.mpc.b_l
_P_base   = controller.mpc.P.copy()

_proactive_lc_cooldown = 0


# ===========================================================================
# HELPER: run one step for an ADA/APF DREAM arm (same planner, swapped DRIFT)
# ===========================================================================

def _run_dream_arm(arm, vehicles_drift_list, vl_mpc, vc_mpc, vr_mpc):
    """
    Execute one full planner step (DRIFT + truck boost + decision + MPC) for a
    comparison arm (ADA or APF).  Mirrors DREAM sections 5-9 but parameterised
    so all three arms share identical planner logic; only the DRIFT source differs.

    arm : dict — holds all per-arm state and controller references.
    Returns (arm, risk_field, risk_at_ego, horizon_array).
    """
    X0           = list(arm['X0'])
    X0_g         = list(arm['X0_g'])
    oa           = arm['oa']
    od           = arm['od']
    last_X       = arm['last_X']
    ovx          = arm.get('ovx', 0.0);  ovy   = arm.get('ovy', 0.0)
    owz          = arm.get('owz', 0.0)
    oS           = arm.get('oS', X0[3]); oey   = arm.get('oey', 0.0)
    oepsi        = arm.get('oepsi', 0.0)
    path_changed = arm['path_changed']
    ctrl         = arm['controller']
    drift_obj    = arm['drift_obj']
    source_fn    = arm['source_fn']
    utils_obj    = arm['utils_obj']
    dec_obj      = arm['decision_obj']
    d0_b         = arm['d0_base'];  Th_b = arm['Th_base']
    al_b         = arm['al_base'];  bl_b = arm['bl_base']
    P_b          = arm['P_base']
    cooldown     = arm.get('proactive_cooldown', 0)
    mpc_ctrl     = ctrl.mpc

    # -- DRIFT step ----------------------------------------------------------
    ego_psi = X0_g[2]
    ego_dv  = drift_create_vehicle(
        vid=0, x=X0_g[0], y=X0_g[1],
        vx=X0[0]*math.cos(ego_psi) - X0[1]*math.sin(ego_psi),
        vy=X0[0]*math.sin(ego_psi) + X0[1]*math.cos(ego_psi),
        vclass='car')
    ego_dv['heading'] = ego_psi
    risk_field  = drift_obj.step(vehicles_drift_list, ego_dv,
                                 dt=dt, substeps=3, source_fn=source_fn)
    risk_at_ego = float(drift_obj.get_risk_cartesian(X0_g[0], X0_g[1]))

    # -- Truck boost (identical to GVF arm) ----------------------------------
    _lane_arrs       = [vl_mpc, vc_mpc, vr_mpc]
    _ego_truck_dist  = float('inf')
    _nearest_truck_k = None
    _boost_f         = np.ones_like(risk_field)
    _ego_boost       = 1.0
    for (_tkl, _tkv) in TRUCK_DESIGNATIONS.keys():
        _tv  = _lane_arrs[_tkl][_tkv]
        _tx, _ty = float(_tv[3]), float(_tv[4])
        _d = float(np.sqrt((_tx - X0_g[0])**2 + (_ty - X0_g[1])**2))
        if _d < _ego_truck_dist:
            _ego_truck_dist = _d;  _nearest_truck_k = (_tkl, _tkv)
        _dsq = (cfg.X - _tx)**2 + (cfg.Y - _ty)**2
        _boost_f   += TRUCK_RISK_BOOST * np.exp(-_dsq / (2*TRUCK_RISK_SIGMA**2))
        _ego_boost += TRUCK_RISK_BOOST * np.exp(-_d**2  / (2*TRUCK_RISK_SIGMA**2))
    risk_field  = risk_field * _boost_f
    risk_at_ego = risk_at_ego * _ego_boost

    _prox = max(0.0, 1.0 - _ego_truck_dist / TRUCK_INFLUENCE_DIST)
    ctrl.weights.mpc_cost = (config_integration.mpc_risk_weight *
                              (1.0 + (TRUCK_WEIGHT_SCALE - 1.0) * _prox))

    # -- Decision logic (identical to GVF arm) -------------------------------
    path_now  = judge_current_position(
        X0_g[0:2], x_bound, y_bound, path_bound, path_bound_sample)
    path_ego  = surroundings.get_path_ego(path_now)

    _truck_in_lane = (_nearest_truck_k is not None and _nearest_truck_k[0] == path_now)
    if _truck_in_lane:
        utils_obj.d0 = d0_b + TRUCK_LONG_EXTRA * _prox
        utils_obj.Th = Th_b + TRUCK_TH_EXTRA  * _prox
    else:
        utils_obj.d0 = d0_b;  utils_obj.Th = Th_b
    mpc_ctrl.a_l = al_b * (1.0 + TRUCK_AL_SCALE  * _prox)
    mpc_ctrl.b_l = bl_b * (1.0 + TRUCK_BL_SCALE  * _prox)
    mpc_ctrl.P   = P_b  / max(1.0, 1.0 + TRUCK_CENTER_RELAX * _prox)

    _left_adj = _right_adj = float('inf')
    _ego_s = X0[3]
    if path_now == 1:
        for _v in vl_mpc: _left_adj  = min(_left_adj,  abs(_v[0] - _ego_s))
        for _v in vr_mpc: _right_adj = min(_right_adj, abs(_v[0] - _ego_s))
    elif path_now == 0:
        for _v in vc_mpc: _right_adj = min(_right_adj, abs(_v[0] - _ego_s))
    elif path_now == 2:
        for _v in vc_mpc: _left_adj  = min(_left_adj,  abs(_v[0] - _ego_s))
    _squeezed = (_left_adj < SQUEEZE_LON_DIST and _right_adj < SQUEEZE_LON_DIST
                 and _ego_s > 35.0)
    _sq_esc = False

    sgs = {0: "L1", 1: "C1", 2: "R1"}[path_now]
    if last_X is None:
        ovx, ovy, owz, oS, oey, oepsi = clac_last_X(
            oa, od, mpc_ctrl.T, path_ego, dt, 6, X0, X0_g)
        last_X = [ovx, ovy, owz, oS, oey, oepsi]

    all_info   = utils_obj.get_alllane_lf(path_ego, X0_g, path_now, vl_mpc, vc_mpc, vr_mpc)
    group_dict, ego_grp = utils_obj.formulate_gap_group(
        path_now, last_X, all_info, vl_mpc, vc_mpc, vr_mpc)

    des_grp = dec_obj.decision_making(group_dict, sgs)
    path_d, path_di, C_lbl, sample, x_list, y_list, X0 = Decision_info(
        X0, X0_g, path_center, sample_center, x_center, y_center,
        boundary, des_grp, path_ego, path_now)

    C_lbl_add = utils_obj.inquire_C_state(C_lbl, des_grp)
    if C_lbl_add == "Probe":
        path_d, path_di, C_lbl_v = path_ego, path_now, "K"
        _, xc, yc, sc = get_path_info(path_di)
        X0 = repropagate(path_d, sc, xc, yc, X0_g, X0)
    else:
        C_lbl_v = C_lbl

    if config_integration.enable_decision_veto and C_lbl != "K":
        _, _allow, _ = ctrl.evaluate_decision_risk(list(X0), path_now, path_di)
        if not _allow:
            path_d, path_di, C_lbl_v = path_ego, path_now, "K"
            _, xc, yc, sc = get_path_info(path_di)
            X0 = repropagate(path_d, sc, xc, yc, X0_g, X0)

    if cooldown > 0:
        cooldown -= 1
    if (TRUCK_DESIGNATIONS and _ego_truck_dist < TRUCK_PROACTIVE_DIST
            and risk_at_ego > TRUCK_PROACTIVE_RISK
            and C_lbl_v == "K" and cooldown == 0):
        _tk_ln = _nearest_truck_k[0]
        if path_now == _tk_ln:
            _alt = 0 if _tk_ln == 1 else (1 if _tk_ln != 1 else 0)
        elif path_now == 1:
            _alt = 2 if _tk_ln == 0 else 0
        else:
            _alt = path_now
        if _alt != path_now:
            path_d, path_di = path_center[_alt], _alt
            C_lbl_v = "L" if _alt < path_now else "R"
            _, xc, yc, sc = get_path_info(path_di)
            X0 = repropagate(path_d, sc, xc, yc, X0_g, X0)
            cooldown = 30

    if _squeezed and C_lbl_v == "K" and cooldown == 0:
        _esc = -1
        if path_now == 1:
            _esc = 0 if _left_adj >= _right_adj else 2
        if _esc != -1 and _esc != path_now:
            path_d, path_di = path_center[_esc], _esc
            C_lbl_v = "L" if _esc < path_now else "R"
            _, xc, yc, sc = get_path_info(path_di)
            X0 = repropagate(path_d, sc, xc, yc, X0_g, X0)
            mpc_ctrl.b_l = bl_b;  cooldown = 30;  _sq_esc = True
        elif path_now in (0, 2):
            _sq_esc = True
        if _sq_esc:
            ctrl.weights.mpc_cost = 0.0
    if _squeezed and not _sq_esc:
        ctrl.weights.mpc_cost = 0.0

    if path_changed != path_di:
        ctrl.get_path_curvature(path=path_d)
        oS, oey = path_to_path_proj(oS, oey, path_changed, path_di)
        last_X = [ovx, ovy, owz, oS, oey, oepsi]
    path_changed = path_di

    # -- MPC solve -----------------------------------------------------------
    oa, od, ovx, ovy, owz, oS, oey, oepsi = ctrl.solve_with_risk(
        X0, oa, od, dt, None, None, C_lbl, X0_g, path_d, last_X,
        path_now, ego_grp, path_ego, des_grp,
        vl_mpc, vc_mpc, vr_mpc, path_di, C_lbl_add, C_lbl_v)

    if (_truck_in_lane and C_lbl_v == "K" and not _sq_esc
            and _ego_truck_dist < TRUCK_INFLUENCE_DIST
            and X0[0] > TRUCK_SAFE_SPEED):
        _os = X0[0] - TRUCK_SAFE_SPEED
        oa = list(oa);  oa[0] = min(oa[0], -min(2.0, _os * 0.5))
    if _squeezed and oa[0] < SQUEEZE_MIN_ACCEL:
        oa = list(oa);  oa[0] = SQUEEZE_MIN_ACCEL

    last_X = [ovx, ovy, owz, oS, oey, oepsi]
    X0, X0_g, _, _ = dynamics.propagate(
        X0, [oa[0], od[0]], dt, X0_g, path_d, sample, x_list, y_list, boundary)

    # -- Horizon for visualization -------------------------------------------
    Xv, Xgv = list(X0), list(X0_g)
    horiz = [list(Xgv)]
    for _k in range(len(oa) - 1):
        Xv, Xgv, _, _ = dynamics.propagate(
            Xv, [oa[_k+1], od[_k+1]], dt, Xgv, path_d, sample, x_list, y_list, boundary)
        horiz.append(list(Xgv))

    # -- Update arm state ----------------------------------------------------
    arm.update(dict(X0=X0, X0_g=X0_g, oa=oa, od=od, last_X=last_X,
                    ovx=ovx, ovy=ovy, owz=owz, oS=oS, oey=oey, oepsi=oepsi,
                    path_changed=path_changed, path_d=path_d,
                    proactive_cooldown=cooldown))
    return arm, risk_field, risk_at_ego, np.array(horiz)

if RUN_MODE == "single":
    # ===========================================================================
    # MAIN SIMULATION LOOP
    # ===========================================================================

    print(f"Running uncertainty test: {N_t} steps, dt={dt}s")
    print()

    risk_field = risk_field_ada = risk_field_apf = risk_field_oacmpc = risk_field_rl = None
    _horiz_ada = _horiz_apf = _horiz_oacmpc = None
    _rae_ada = _rae_apf = _rae_oacmpc = _rae_rl = 0.0
    current_colorbar = None

    # Dynamic panel layout — one column per selected arm
    _PANEL_ORDER = ['IDEAM', 'DREAM', 'RL-PPO', 'OA-CMPC', 'ADA', 'APF']
    _active_panels = [p for p in _PANEL_ORDER if p in RUN_PLANNERS]
    _n_pan = max(1, len(_active_panels))
    plt.figure(figsize=(7 * _n_pan, 7))

    for i in range(N_t):
        bar.next()

        # ──────────────────────────────────────────────────────────────────────
        # 1. GET CURRENT SURROUNDING VEHICLE STATES
        # ──────────────────────────────────────────────────────────────────────
        vehicle_left, vehicle_centre, vehicle_right = surroundings.get_vehicles_states()

        # Clamp truck speeds to their designated desired speed every step.
        # get_vehicles_states() returns REFERENCES to internal arrays, so this
        # also fixes the IDM input for the next surroundings.total_update_emergency().
        for (_tk_l, _tk_v), _tk_vd in TRUCK_DESIGNATIONS.items():
            _tk_arrs = [vehicle_left, vehicle_centre, vehicle_right]
            if len(_tk_arrs[_tk_l]) > _tk_v:
                _tk_arrs[_tk_l][_tk_v][6] = min(float(_tk_arrs[_tk_l][_tk_v][6]),
                                                 float(_tk_vd))

        # ──────────────────────────────────────────────────────────────────────
        # 2. UPDATE SYNTHETIC AGENTS (left_fast_car + merger, every step)
        # ──────────────────────────────────────────────────────────────────────
        # 2a. Get truck state for ghost-tracking and occlusion check.
        _tk_arrs_cur  = [vehicle_left, vehicle_centre, vehicle_right]
        _truck_state_i = _tk_arrs_cur[_AGENT_TRUCK_LANE][_AGENT_TRUCK_IDX]

        # 2b. Advance the left-lane fast car (IDM on path1c, revealed when shadow clears).
        left_fast_car.update(vehicle_left)
        left_fast_car.check_occlusion(X0_g_ideam[0], X0_g_ideam[1], _truck_state_i)
        if not left_fast_car.revealed and not left_fast_car.occluded:
            left_fast_car.revealed    = True
            left_fast_car.reveal_step = i
            print(f"[FAST CAR REVEALED] Step {i} (t={i*dt:.1f}s) geometric trigger")
        elif not left_fast_car.revealed and i >= LEFT_FAST_CAR_REVEAL_FALLBACK:
            left_fast_car.revealed    = True
            left_fast_car.reveal_step = i
            print(f"[FAST CAR REVEALED] Step {i} (t={i*dt:.1f}s) fallback trigger")

        # 2c. Advance merger using its own IDEAM controller.  Keep it in the right
        #     lane until it has physically cleared the truck, then release LC
        #     authority back to the baseline planner.
        _left_fast_row_for_merger = left_fast_car.to_ideam_row() if left_fast_car.revealed else None
        merger.update(
            vehicle_left, vehicle_centre, vehicle_right,
            truck_state=_truck_state_i,
            ego_state=X0_ideam, ego_global=X0_g_ideam,
            left_fast_row=_left_fast_row_for_merger,
            step_idx=i,
        )
        if not merger.last_update_ok and i % 50 == 0:
            print(f"[MERGER] Frame {i}: IDEAM fallback used")
        if merger.request_step == i:
            _truck_gap_req = merger.state[0] - float(_truck_state_i[0])
            print(f"[MERGER LC REQUEST] Step {i} (t={i*dt:.1f}s)  "
                  f"source={merger.request_reason}  truck_gap={_truck_gap_req:.1f}m")
        if merger.lc_started_step == i:
            _mg_pos  = np.array([merger.x, merger.y])
            _id_pos  = np.array([X0_g_ideam[0], X0_g_ideam[1]])
            _mg_dist = float(np.linalg.norm(_mg_pos - _id_pos))
            _rel_spd = max(0.5, abs(merger.vx - float(X0_ideam[0])))
            _merger_ttc_at_trigger = _mg_dist / _rel_spd
            _truck_gap = merger.state[0] - float(_truck_state_i[0])
            print(f"[MERGER LC STARTED] Step {i} (t={i*dt:.1f}s)  "
                  f"request={merger.request_reason}  truck_gap={_truck_gap:.1f}m")
            print(f"  merger: s={merger.state[0]:.1f}m  "
                  f"pos=({merger.x:.1f},{merger.y:.1f})  vx={merger.vx:.1f} m/s")
            print(f"  IDEAM:  pos=({X0_g_ideam[0]:.1f},{X0_g_ideam[1]:.1f})  "
                  f"vx={float(X0_ideam[0]):.1f} m/s")
            print(f"  dist_to_IDEAM={_mg_dist:.1f}m  TTC≈{_merger_ttc_at_trigger:.1f}s")
        agent_occluded_record.append(merger.is_occluded())
        merger_truck_dist.append(float(math.hypot(
            merger.x - float(_truck_state_i[3]),
            merger.y - float(_truck_state_i[4]),
        )))

        # ──────────────────────────────────────────────────────────────────────
        # 3. BUILD DRIFT VEHICLE LIST (merger + left_fast_car ALWAYS included)
        # ──────────────────────────────────────────────────────────────────────
        vehicles_drift = convert_to_drift(
            vehicle_left, vehicle_centre, vehicle_right,
            truck_set=set(TRUCK_DESIGNATIONS.keys()))
        vehicles_drift.append(merger.to_drift_vehicle(vid=998))           # always in DRIFT
        vehicles_drift.append(left_fast_car.to_drift_vehicle(vid=997))    # always in DRIFT

        # ──────────────────────────────────────────────────────────────────────
        # 4. BUILD MPC VEHICLE LISTS
        #    left_fast_car injected into vl_mpc only after reveal.
        #    Merger: excluded while in right lane (lc_progress < 0.5); injected
        #    into vc_mpc once substantially in centre lane (lc_progress >= 0.5).
        # ──────────────────────────────────────────────────────────────────────
        vl_mpc = vehicle_left.copy()
        vc_mpc = vehicle_centre.copy()
        vr_mpc = vehicle_right.copy()

        # Inject fast car into left-lane array after reveal so IDEAM/DREAM react.
        if left_fast_car.revealed:
            vl_mpc = append_lane_row(vl_mpc, left_fast_car.to_ideam_row())

        if merger.in_centre_lane():
            vc_mpc = append_lane_row(vc_mpc, merger.to_center_lane_row())

        # ──────────────────────────────────────────────────────────────────────
        # 5. STEP DRIFT
        # ──────────────────────────────────────────────────────────────────────
        ego_psi = X0_g[2]
        ego_drift = drift_create_vehicle(
            vid=0, x=X0_g[0], y=X0_g[1],
            vx=X0[0] * math.cos(ego_psi) - X0[1] * math.sin(ego_psi),
            vy=X0[0] * math.sin(ego_psi) + X0[1] * math.cos(ego_psi),
            vclass='car')
        ego_drift['heading'] = ego_psi

        risk_field = drift.step(vehicles_drift, ego_drift, dt=dt, substeps=3)
        risk_at_ego = drift.get_risk_cartesian(X0_g[0], X0_g[1])
        risk_at_ego_list.append(risk_at_ego)

        # ──────────────────────────────────────────────────────────────────────
        # 6. TRUCK RISK BOOST + DYNAMIC MPC WEIGHT
        # NOTE (architectural limitation): the local `risk_field` and `risk_at_ego`
        # variables are boosted here for visualization and the decision-veto check
        # (evaluate_decision_risk reads risk_at_ego via controller.drift.get_risk).
        # However solve_with_risk() (Section 9) internally queries controller.drift,
        # which holds the raw un-boosted DRIFT field from drift.step().
        # Consequence: MPC trajectory cost uses un-boosted risk; only the veto gate
        # and visualization use the boosted value.  To fully align them the boosted
        # field would need to be written back to drift.risk_field before Section 9.
        # ──────────────────────────────────────────────────────────────────────
        _ego_truck_dist   = float('inf')
        _nearest_truck_key = None
        if TRUCK_DESIGNATIONS:
            _lane_arrays = [vl_mpc, vc_mpc, vr_mpc]
            _boost_field = np.ones_like(risk_field)
            _ego_boost   = 1.0
            for (_tk_l, _tk_v) in TRUCK_DESIGNATIONS.keys():
                _tv = _lane_arrays[_tk_l][_tk_v]
                _tx, _ty = _tv[3], _tv[4]
                _d = float(np.sqrt((_tx - X0_g[0])**2 + (_ty - X0_g[1])**2))
                if _d < _ego_truck_dist:
                    _ego_truck_dist    = _d
                    _nearest_truck_key = (_tk_l, _tk_v)
                _dsq = (cfg.X - _tx)**2 + (cfg.Y - _ty)**2
                _boost_field += TRUCK_RISK_BOOST * np.exp(-_dsq / (2*TRUCK_RISK_SIGMA**2))
                _ego_boost   += TRUCK_RISK_BOOST * np.exp(-_d**2 / (2*TRUCK_RISK_SIGMA**2))
            risk_field  = risk_field * _boost_field
            risk_at_ego = risk_at_ego * _ego_boost

        _prox = max(0.0, 1.0 - _ego_truck_dist / TRUCK_INFLUENCE_DIST)
        controller.weights.mpc_cost = (
            config_integration.mpc_risk_weight * (
                1.0 + (TRUCK_WEIGHT_SCALE - 1.0) * _prox))

        # ──────────────────────────────────────────────────────────────────────
        # 7. DREAM PATH DECISION LOGIC
        # ──────────────────────────────────────────────────────────────────────
        path_now  = judge_current_position(
            X0_g[0:2], x_bound, y_bound, path_bound, path_bound_sample)
        path_ego  = surroundings.get_path_ego(path_now)

        # Asymmetric constraint scaling
        _truck_in_ego_lane = (_nearest_truck_key is not None and
                              _nearest_truck_key[0] == path_now)
        if _truck_in_ego_lane:
            utils.d0 = _d0_base + TRUCK_LONG_EXTRA * _prox
            utils.Th = _Th_base + TRUCK_TH_EXTRA  * _prox
        else:
            utils.d0 = _d0_base
            utils.Th = _Th_base
        controller.mpc.a_l = _al_base * (1.0 + TRUCK_AL_SCALE * _prox)
        controller.mpc.b_l = _bl_base * (1.0 + TRUCK_BL_SCALE * _prox)
        _relax = max(1.0, 1.0 + TRUCK_CENTER_RELAX * _prox)
        controller.mpc.P  = _P_base / _relax

        # Lateral squeeze detection
        _left_adj  = float('inf')
        _right_adj = float('inf')
        _ego_s     = X0[3]
        if path_now == 1:
            for _v in vl_mpc:
                _left_adj  = min(_left_adj,  abs(_v[0] - _ego_s))
            for _v in vr_mpc:
                _right_adj = min(_right_adj, abs(_v[0] - _ego_s))
        elif path_now == 0:
            for _v in vc_mpc:
                _right_adj = min(_right_adj, abs(_v[0] - _ego_s))
        elif path_now == 2:
            for _v in vc_mpc:
                _left_adj  = min(_left_adj,  abs(_v[0] - _ego_s))
        _squeezed            = (_left_adj < SQUEEZE_LON_DIST and
                                _right_adj < SQUEEZE_LON_DIST and
                                _ego_s > 35.0)
        _squeeze_escape      = False

        start_group_str = {0: "L1", 1: "C1", 2: "R1"}[path_now]

        if i == 0:
            ovx, ovy, owz, oS, oey, oepsi = clac_last_X(
                oa, od, mpc_controller.T, path_ego, dt, 6, X0, X0_g)
            last_X = [ovx, ovy, owz, oS, oey, oepsi]

        all_info   = utils.get_alllane_lf(
            path_ego, X0_g, path_now, vl_mpc, vc_mpc, vr_mpc)
        group_dict, ego_group = utils.formulate_gap_group(
            path_now, last_X, all_info, vl_mpc, vc_mpc, vr_mpc)

        _t_dec = time.time()
        desired_group = decision_maker.decision_making(group_dict, start_group_str)
        decision_dur  = time.time() - _t_dec

        path_d, path_dindex, C_label, sample, x_list, y_list, X0 = Decision_info(
            X0, X0_g, path_center, sample_center, x_center, y_center,
            boundary, desired_group, path_ego, path_now)

        C_label_additive = utils.inquire_C_state(C_label, desired_group)
        if C_label_additive == "Probe":
            path_d, path_dindex, C_label_virtual = path_ego, path_now, "K"
            _, xc, yc, samplesc = get_path_info(path_dindex)
            X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)
        else:
            C_label_virtual = C_label

        # Decision veto
        if config_integration.enable_decision_veto and C_label != "K":
            ego_state_veto = list(X0)
            _rs, _allow, _ = controller.evaluate_decision_risk(
                ego_state_veto, path_now, path_dindex)
            if not _allow:
                path_d, path_dindex, C_label_virtual = path_ego, path_now, "K"
                _, xc, yc, samplesc = get_path_info(path_dindex)
                X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)

        # Proactive LC away from truck
        if _proactive_lc_cooldown > 0:
            _proactive_lc_cooldown -= 1

        if (TRUCK_DESIGNATIONS and
                _ego_truck_dist < TRUCK_PROACTIVE_DIST and
                risk_at_ego > TRUCK_PROACTIVE_RISK and
                C_label_virtual == "K" and _proactive_lc_cooldown == 0):
            _tk_lane = _nearest_truck_key[0]
            if path_now == _tk_lane:
                # Ego is in same lane as truck — move away to overtake.
                # Truck in centre (1) → go LEFT (0), NOT right (merger is there).
                _alt = 0 if _tk_lane == 1 else (1 if _tk_lane != 1 else 0)
            elif path_now == 1:
                _alt = 2 if _tk_lane == 0 else 0
            else:
                _alt = path_now
            if _alt != path_now:
                path_d, path_dindex = path_center[_alt], _alt
                C_label_virtual = "L" if _alt < path_now else "R"
                _, xc, yc, samplesc = get_path_info(path_dindex)
                X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)
                _proactive_lc_cooldown = 30

        # Squeeze escape
        if (_squeezed and C_label_virtual == "K" and _proactive_lc_cooldown == 0):
            _esc = -1
            if path_now == 1:
                _esc = 0 if _left_adj >= _right_adj else 2
            if _esc != -1 and _esc != path_now:
                path_d, path_dindex = path_center[_esc], _esc
                C_label_virtual = "L" if _esc < path_now else "R"
                _, xc, yc, samplesc = get_path_info(path_dindex)
                X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)
                controller.mpc.b_l = _bl_base
                _proactive_lc_cooldown = 30
                _squeeze_escape = True
            elif path_now in (0, 2):
                _squeeze_escape = True
            if _squeeze_escape:
                controller.weights.mpc_cost = 0.0
        if _squeezed and not _squeeze_escape:
            controller.weights.mpc_cost = 0.0

        path_desired.append(path_d)
        if path_changed != path_dindex:
            controller.get_path_curvature(path=path_d)
            oS, oey = path_to_path_proj(oS, oey, path_changed, path_dindex)
            last_X = [ovx, ovy, owz, oS, oey, oepsi]
        path_changed = path_dindex

        # ──────────────────────────────────────────────────────────────────────
        # 8. IDEAM PARALLEL SIMULATION
        # ──────────────────────────────────────────────────────────────────────
        ideam_panel_X0    = list(X0_ideam)
        ideam_panel_X0_g  = list(X0_g_ideam)
        ideam_panel_horizon = None
        _t_ideam_start = time.time()

        try:
            path_now_i = judge_current_position(
                X0_g_ideam[0:2], x_bound, y_bound, path_bound, path_bound_sample)
            path_ego_i = surroundings.get_path_ego(path_now_i)
            sgs_i = {0: "L1", 1: "C1", 2: "R1"}[path_now_i]

            if i == 0:
                _ovx_i, _ovy_i, _owz_i, _oS_i, _oey_i, _oepsi_i = clac_last_X(
                    oa_ideam_viz, od_ideam_viz,
                    baseline_mpc_viz.T, path_ego_i, dt, 6, X0_ideam, X0_g_ideam)
                last_X_ideam_viz = [_ovx_i, _ovy_i, _owz_i, _oS_i, _oey_i, _oepsi_i]

            all_info_i = utils_ideam_viz.get_alllane_lf(
                path_ego_i, X0_g_ideam, path_now_i, vl_mpc, vc_mpc, vr_mpc)
            gd_i, ego_grp_i = utils_ideam_viz.formulate_gap_group(
                path_now_i, last_X_ideam_viz, all_info_i, vl_mpc, vc_mpc, vr_mpc)
            dg_i = decision_maker.decision_making(gd_i, sgs_i)

            path_d_i, path_di_i, Cl_i, samp_i, xl_i, yl_i, X0_ideam = Decision_info(
                X0_ideam, X0_g_ideam, path_center, sample_center,
                x_center, y_center, boundary, dg_i, path_ego_i, path_now_i)

            Cla_i = utils_ideam_viz.inquire_C_state(Cl_i, dg_i)
            if Cla_i == "Probe":
                path_d_i, path_di_i, Clv_i = path_ego_i, path_now_i, "K"
                _, xci, yci, sci = get_path_info(path_di_i)
                X0_ideam = repropagate(path_d_i, sci, xci, yci, X0_g_ideam, X0_ideam)
            else:
                Clv_i = Cl_i

            # Diagnostic: first true left→centre LC intent observed from IDEAM
            if _ideam_first_lcc_step is None and path_now_i == 0 and path_di_i == 1:
                _ideam_first_lcc_step = i
                print(f"[IDEAM LC-CENTRE] Step {i} (t={i*dt:.1f}s): "
                      f"IDEAM first commands centre lane (path {path_now_i}→{path_di_i})")

            if (path_changed_ideam_viz != path_di_i and last_X_ideam_viz is not None):
                _oS_i, _oey_i = path_to_path_proj(
                    last_X_ideam_viz[3], last_X_ideam_viz[4],
                    path_changed_ideam_viz, path_di_i)
                last_X_ideam_viz = [last_X_ideam_viz[0], last_X_ideam_viz[1],
                                     last_X_ideam_viz[2], _oS_i, _oey_i,
                                     last_X_ideam_viz[5]]
            baseline_mpc_viz.get_path_curvature(path=path_d_i)
            path_changed_ideam_viz = path_di_i

            res_i = baseline_mpc_viz.iterative_linear_mpc_control(
                X0_ideam, oa_ideam_viz, od_ideam_viz, dt,
                None, None, Cl_i, X0_g_ideam, path_d_i, last_X_ideam_viz,
                path_now_i, ego_grp_i, path_ego_i, dg_i,
                vl_mpc, vc_mpc, vr_mpc, path_di_i, Cla_i, Clv_i)

            if res_i is not None:
                oa_i, od_i, _ovx_i, _ovy_i, _owz_i, _oS_i, _oey_i, _oepsi_i = res_i
                last_X_ideam_viz = [_ovx_i, _ovy_i, _owz_i, _oS_i, _oey_i, _oepsi_i]
                oa_ideam_viz, od_ideam_viz = oa_i, od_i
                X0_ideam, X0_g_ideam, _, _ = dynamics.propagate(
                    list(X0_ideam), [oa_i[0], od_i[0]], dt,
                    list(X0_g_ideam), path_d_i, samp_i, xl_i, yl_i, boundary)
                ideam_panel_X0   = list(X0_ideam)
                ideam_panel_X0_g = list(X0_g_ideam)
                ideam_panel_horizon = build_horizon(
                    ideam_panel_X0, ideam_panel_X0_g, oa_i, od_i,
                    path_d_i, samp_i, xl_i, yl_i)
                last_ideam_panel_horizon_viz = ideam_panel_horizon

        except Exception as e:
            ideam_panel_horizon = last_ideam_panel_horizon_viz
            ideam_mpc_fail += 1
            if i % 50 == 0:
                print(f"[IDEAM] Frame {i}: {e}")
        ideam_step_times.append(time.time() - _t_ideam_start)

        # Store IDEAM's actual lane and desired path for merger LC trigger next step.
        # Both are needed to detect a true left→centre transition event.
        try:
            _prev_path_di_i  = path_di_i
            _prev_path_now_i = path_now_i
        except NameError:
            _prev_path_di_i  = None
            _prev_path_now_i = None

        # ──────────────────────────────────────────────────────────────────────
        # 9. DREAM MPC SOLVE WITH RISK
        # ──────────────────────────────────────────────────────────────────────
        _t_solve = time.time()
        try:
            oa, od, ovx, ovy, owz, oS, oey, oepsi = controller.solve_with_risk(
                X0, oa, od, dt, None, None, C_label, X0_g, path_d, last_X,
                path_now, ego_group, path_ego, desired_group,
                vl_mpc, vc_mpc, vr_mpc, path_dindex, C_label_additive, C_label_virtual)
        except Exception as _dream_exc:
            dream_mpc_fail += 1
            if i % 50 == 0:
                print(f"[DREAM] Frame {i}: {_dream_exc}")
        dream_step_times.append(time.time() - _t_solve)

        # Speed cap and squeeze floor
        if (_truck_in_ego_lane and C_label_virtual == "K" and
                not _squeeze_escape and
                _ego_truck_dist < TRUCK_INFLUENCE_DIST and
                X0[0] > TRUCK_SAFE_SPEED):
            _os = X0[0] - TRUCK_SAFE_SPEED
            oa = list(oa)
            oa[0] = min(oa[0], -min(2.0, _os * 0.5))

        if _squeezed and oa[0] < SQUEEZE_MIN_ACCEL:
            oa = list(oa)
            oa[0] = SQUEEZE_MIN_ACCEL

        # Risk threshold crossing for ALT metric
        if _risk_cross_step is None and float(risk_at_ego) > config_integration.decision_risk_threshold:
            _risk_cross_step = i

        last_X = [ovx, ovy, owz, oS, oey, oepsi]
        X0, X0_g, _, _ = dynamics.propagate(
            X0, [oa[0], od[0]], dt, X0_g, path_d, sample, x_list, y_list, boundary)

        # ──────────────────────────────────────────────────────────────────────
        # 9b. ADA + APF DREAM arms (same planner, static-field DRIFT source)
        # ──────────────────────────────────────────────────────────────────────
        if 'ADA' in RUN_PLANNERS:
            _t_ada_start = time.time()
            _arm_ada_state, risk_field_ada, _rae_ada, _horiz_ada = _run_dream_arm(
                _arm_ada_state, vehicles_drift, vl_mpc, vc_mpc, vr_mpc)
            ada_step_times.append(time.time() - _t_ada_start)
            risk_at_ego_ada_list.append(_rae_ada)
        else:
            ada_step_times.append(0.0); risk_at_ego_ada_list.append(float('nan'))
            _rae_ada = float('nan')

        if 'APF' in RUN_PLANNERS:
            _t_apf_start = time.time()
            _arm_apf_state, risk_field_apf, _rae_apf, _horiz_apf = _run_dream_arm(
                _arm_apf_state, vehicles_drift, vl_mpc, vc_mpc, vr_mpc)
            apf_step_times.append(time.time() - _t_apf_start)
            risk_at_ego_apf_list.append(_rae_apf)
        else:
            apf_step_times.append(0.0); risk_at_ego_apf_list.append(float('nan'))
            _rae_apf = float('nan')

        if 'OA-CMPC' in RUN_PLANNERS:
            _t_oacmpc_start = time.time()
            _arm_oacmpc_state, risk_field_oacmpc, _rae_oacmpc, _horiz_oacmpc = _run_dream_arm(
                _arm_oacmpc_state, vehicles_drift, vl_mpc, vc_mpc, vr_mpc)
            oacmpc_step_times.append(time.time() - _t_oacmpc_start)
            risk_at_ego_oacmpc_list.append(_rae_oacmpc)
        else:
            oacmpc_step_times.append(0.0); risk_at_ego_oacmpc_list.append(float('nan'))
            _rae_oacmpc = float('nan')

        # 9c. RL-PPO decision arm
        _horiz_rl = None
        if 'RL-PPO' in RUN_PLANNERS and _rl_policy is not None:
            _t_rl_start = time.time()
            out_rl = rl_decision_step_test(
                X0_rl, X0_g_rl, oa_rl, od_rl, last_X_rl, path_changed_rl,
                _rl_policy, controller_rl.mpc, utils_rl, decision_maker_rl, dynamics,
                vl_mpc, vc_mpc, vr_mpc,
                force_target_lane=None, v_ref=8.0,
            )
            X0_rl, X0_g_rl = out_rl["X0"], out_rl["X0_g"]
            oa_rl, od_rl = out_rl["oa"], out_rl["od"]
            last_X_rl = out_rl["last_X"]
            path_changed_rl = out_rl["path_changed"]
            # Build planned-trajectory horizon for drawing
            _horiz_rl = build_horizon(
                X0_rl, X0_g_rl, oa_rl, od_rl,
                out_rl["path_d"], out_rl["sample"], out_rl["x_list"], out_rl["y_list"])
            rl_step_times.append(time.time() - _t_rl_start)
            # DRIFT step for the RL arm's own risk field
            _psi_rl = X0_g_rl[2]
            _ego_dv_rl = drift_create_vehicle(
                vid=0, x=X0_g_rl[0], y=X0_g_rl[1],
                vx=X0_rl[0]*math.cos(_psi_rl) - X0_rl[1]*math.sin(_psi_rl),
                vy=X0_rl[0]*math.sin(_psi_rl) + X0_rl[1]*math.cos(_psi_rl),
                vclass='car')
            _ego_dv_rl['heading'] = _psi_rl
            risk_field_rl = drift_rl.step(vehicles_drift, _ego_dv_rl, dt=dt, substeps=3)
            _rae_rl = float(drift_rl.get_risk_cartesian(X0_g_rl[0], X0_g_rl[1]))
        else:
            rl_step_times.append(0.0)
            risk_field_rl = None; _rae_rl = 0.0

        # ──────────────────────────────────────────────────────────────────────
        # 10. METRICS
        # ──────────────────────────────────────────────────────────────────────
        # DREAM — use path1c (ego's starting lane) for consistent Frenet progress
        try:
            _s_d, _, _ = find_frenet_coord(path1c, x1c, y1c, samples1c, X0_g)
        except Exception:
            _s_d, _, _ = find_frenet_coord(path2c, x2c, y2c, samples2c, X0_g)
        dream_s.append(float(_s_d))
        dream_vx.append(float(X0[0]))
        dream_vy.append(float(X0[1]))
        dream_acc.append(float(oa[0]))

        _dream_rect = create_rectangle(X0_g[0], X0_g[1],
                                       mpc_controller.vehicle_length,
                                       mpc_controller.vehicle_width, X0_g[2])
        dream_s_obs.append(float(surroundings.S_obs_calc(_dream_rect)))

        _d2ag = math.sqrt((X0_g[0] - merger.x)**2 +
                          (X0_g[1] - merger.y)**2)
        dream_dist_agent.append(_d2ag)

        # IDEAM — same reference path as DREAM for fair comparison
        if ideam_panel_X0_g is not None:
            try:
                _s_i, _, _ = find_frenet_coord(path1c, x1c, y1c, samples1c,
                                               ideam_panel_X0_g)
            except Exception:
                _s_i, _, _ = find_frenet_coord(path2c, x2c, y2c, samples2c,
                                               ideam_panel_X0_g)
            ideam_s.append(float(_s_i))
            ideam_vx.append(float(ideam_panel_X0[0]))
            ideam_vy.append(float(ideam_panel_X0[1]))
            _oa_i = float(oa_ideam_viz[0] if hasattr(oa_ideam_viz, '__len__')
                          else oa_ideam_viz)
            ideam_acc.append(_oa_i)
            _i_rect = create_rectangle(ideam_panel_X0_g[0], ideam_panel_X0_g[1],
                                       mpc_controller.vehicle_length,
                                       mpc_controller.vehicle_width,
                                       ideam_panel_X0_g[2])
            ideam_s_obs.append(float(surroundings.S_obs_calc(_i_rect)))
            _d2ag_i = math.sqrt((ideam_panel_X0_g[0] - merger.x)**2 +
                                 (ideam_panel_X0_g[1] - merger.y)**2)
            ideam_dist_agent.append(_d2ag_i)
        else:
            ideam_s.append(ideam_s[-1] if ideam_s else 20.0)
            ideam_vx.append(0.0)
            ideam_vy.append(0.0)
            ideam_acc.append(0.0)
            ideam_s_obs.append(100.0)
            ideam_dist_agent.append(float('nan'))

        # ADA arm metrics
        if 'ADA' in RUN_PLANNERS:
            _X0_ada_cur = _arm_ada_state['X0'];  _X0_g_ada_cur = _arm_ada_state['X0_g']
            try:
                _s_a, _, _ = find_frenet_coord(path1c, x1c, y1c, samples1c, _X0_g_ada_cur)
            except Exception:
                _s_a, _, _ = find_frenet_coord(path2c, x2c, y2c, samples2c, _X0_g_ada_cur)
            ada_s.append(float(_s_a))
            ada_vx.append(float(_X0_ada_cur[0]))
            _oa_ada_val = _arm_ada_state['oa']
            ada_acc.append(float(_oa_ada_val[0] if hasattr(_oa_ada_val, '__len__') else _oa_ada_val))
            _a_rect = create_rectangle(_X0_g_ada_cur[0], _X0_g_ada_cur[1],
                                       controller_ada.mpc.vehicle_length,
                                       controller_ada.mpc.vehicle_width, _X0_g_ada_cur[2])
            ada_s_obs.append(float(surroundings.S_obs_calc(_a_rect)))
            ada_dist_agent.append(float(math.hypot(_X0_g_ada_cur[0] - merger.x,
                                                    _X0_g_ada_cur[1] - merger.y)))
        else:
            _X0_ada_cur = [float('nan')]*6; _X0_g_ada_cur = list(X0_g)
            ada_s.append(float('nan')); ada_vx.append(float('nan'))
            ada_acc.append(float('nan')); ada_s_obs.append(float('nan'))
            ada_dist_agent.append(float('nan'))

        # APF arm metrics
        if 'APF' in RUN_PLANNERS:
            _X0_apf_cur = _arm_apf_state['X0'];  _X0_g_apf_cur = _arm_apf_state['X0_g']
            try:
                _s_p, _, _ = find_frenet_coord(path1c, x1c, y1c, samples1c, _X0_g_apf_cur)
            except Exception:
                _s_p, _, _ = find_frenet_coord(path2c, x2c, y2c, samples2c, _X0_g_apf_cur)
            apf_s.append(float(_s_p))
            apf_vx.append(float(_X0_apf_cur[0]))
            _oa_apf_val = _arm_apf_state['oa']
            apf_acc.append(float(_oa_apf_val[0] if hasattr(_oa_apf_val, '__len__') else _oa_apf_val))
            _p_rect = create_rectangle(_X0_g_apf_cur[0], _X0_g_apf_cur[1],
                                       controller_apf.mpc.vehicle_length,
                                       controller_apf.mpc.vehicle_width, _X0_g_apf_cur[2])
            apf_s_obs.append(float(surroundings.S_obs_calc(_p_rect)))
            apf_dist_agent.append(float(math.hypot(_X0_g_apf_cur[0] - merger.x,
                                                    _X0_g_apf_cur[1] - merger.y)))
        else:
            _X0_apf_cur = [float('nan')]*6; _X0_g_apf_cur = list(X0_g)
            apf_s.append(float('nan')); apf_vx.append(float('nan'))
            apf_acc.append(float('nan')); apf_s_obs.append(float('nan'))
            apf_dist_agent.append(float('nan'))

        # OA-CMPC arm metrics
        if 'OA-CMPC' in RUN_PLANNERS:
            _X0_oacmpc_cur = _arm_oacmpc_state['X0'];  _X0_g_oacmpc_cur = _arm_oacmpc_state['X0_g']
            try:
                _s_oc, _, _ = find_frenet_coord(path1c, x1c, y1c, samples1c, _X0_g_oacmpc_cur)
            except Exception:
                _s_oc, _, _ = find_frenet_coord(path2c, x2c, y2c, samples2c, _X0_g_oacmpc_cur)
            oacmpc_s.append(float(_s_oc))
            oacmpc_vx.append(float(_X0_oacmpc_cur[0]))
            _oa_oacmpc_val = _arm_oacmpc_state['oa']
            oacmpc_acc.append(float(_oa_oacmpc_val[0] if hasattr(_oa_oacmpc_val, '__len__') else _oa_oacmpc_val))
            _oc_rect = create_rectangle(_X0_g_oacmpc_cur[0], _X0_g_oacmpc_cur[1],
                                        controller_oacmpc.mpc.vehicle_length,
                                        controller_oacmpc.mpc.vehicle_width, _X0_g_oacmpc_cur[2])
            oacmpc_s_obs.append(float(surroundings.S_obs_calc(_oc_rect)))
            oacmpc_dist_agent.append(float(math.hypot(_X0_g_oacmpc_cur[0] - merger.x,
                                                       _X0_g_oacmpc_cur[1] - merger.y)))
        else:
            _X0_oacmpc_cur = [float('nan')]*6; _X0_g_oacmpc_cur = list(X0_g)
            oacmpc_s.append(float('nan')); oacmpc_vx.append(float('nan'))
            oacmpc_acc.append(float('nan')); oacmpc_s_obs.append(float('nan'))
            oacmpc_dist_agent.append(float('nan'))

        # RL-PPO arm metrics
        if 'RL-PPO' in RUN_PLANNERS:
            _X0_rl_cur = X0_rl;  _X0_g_rl_cur = X0_g_rl
            try:
                _s_rl, _, _ = find_frenet_coord(path1c, x1c, y1c, samples1c, _X0_g_rl_cur)
            except Exception:
                _s_rl, _, _ = find_frenet_coord(path2c, x2c, y2c, samples2c, _X0_g_rl_cur)
            rl_s.append(float(_s_rl))
            rl_vx.append(float(_X0_rl_cur[0]))
            rl_acc.append(float(oa_rl[0] if hasattr(oa_rl, '__len__') else oa_rl))
            _rl_rect = create_rectangle(_X0_g_rl_cur[0], _X0_g_rl_cur[1],
                                        controller_rl.mpc.vehicle_length,
                                        controller_rl.mpc.vehicle_width, _X0_g_rl_cur[2])
            rl_s_obs.append(float(surroundings.S_obs_calc(_rl_rect)))
            rl_dist_agent.append(float(math.hypot(_X0_g_rl_cur[0] - merger.x,
                                                   _X0_g_rl_cur[1] - merger.y)))
        else:
            _X0_rl_cur = [float('nan')]*6; _X0_g_rl_cur = list(X0_g)
            rl_s.append(float('nan')); rl_vx.append(float('nan'))
            rl_acc.append(float('nan')); rl_s_obs.append(float('nan'))
            rl_dist_agent.append(float('nan'))

        # -- TTC and REI per planner  ------------------------------------------
        _m_vx = float(merger.state[6])
        dream_ttc.append(_ttc_to_agent(X0_g,           X0[0],
                                       merger.x, merger.y, _m_vx))
        _ide_pos = ideam_panel_X0_g if ideam_panel_X0_g is not None else X0_g
        _ide_vx  = float(ideam_panel_X0[0]) if ideam_panel_X0_g is not None else X0[0]
        ideam_ttc.append(_ttc_to_agent(_ide_pos,        _ide_vx,
                                       merger.x, merger.y, _m_vx))
        ada_ttc.append(
            _ttc_to_agent(_X0_g_ada_cur, float(_X0_ada_cur[0]), merger.x, merger.y, _m_vx)
            if 'ADA' in RUN_PLANNERS else float('nan'))
        apf_ttc.append(
            _ttc_to_agent(_X0_g_apf_cur, float(_X0_apf_cur[0]), merger.x, merger.y, _m_vx)
            if 'APF' in RUN_PLANNERS else float('nan'))
        oacmpc_ttc.append(
            _ttc_to_agent(_X0_g_oacmpc_cur, float(_X0_oacmpc_cur[0]), merger.x, merger.y, _m_vx)
            if 'OA-CMPC' in RUN_PLANNERS else float('nan'))
        rl_ttc.append(
            _ttc_to_agent(_X0_g_rl_cur, float(_X0_rl_cur[0]), merger.x, merger.y, _m_vx)
            if 'RL-PPO' in RUN_PLANNERS else float('nan'))

        # REI integrand: GVF risk evaluated at each planner's position × speed
        # All arms evaluated on the DREAM field (common evaluation field).
        _r_dream = float(risk_at_ego)
        _r_ideam = float(drift.get_risk_cartesian(float(_ide_pos[0]), float(_ide_pos[1])))
        _r_ada    = (float(drift.get_risk_cartesian(float(_X0_g_ada_cur[0]),    float(_X0_g_ada_cur[1])))
                     if 'ADA' in RUN_PLANNERS else float('nan'))
        _r_apf    = (float(drift.get_risk_cartesian(float(_X0_g_apf_cur[0]),    float(_X0_g_apf_cur[1])))
                     if 'APF' in RUN_PLANNERS else float('nan'))
        _r_oacmpc = (float(drift.get_risk_cartesian(float(_X0_g_oacmpc_cur[0]), float(_X0_g_oacmpc_cur[1])))
                     if 'OA-CMPC' in RUN_PLANNERS else float('nan'))
        dream_rei_integrand.append(_r_dream  * float(X0[0]))
        ideam_rei_integrand.append(_r_ideam  * float(_ide_vx))
        ada_rei_integrand.append(float('nan') if 'ADA' not in RUN_PLANNERS else
                                 _r_ada * float(_X0_ada_cur[0]))
        apf_rei_integrand.append(float('nan') if 'APF' not in RUN_PLANNERS else
                                 _r_apf * float(_X0_apf_cur[0]))
        oacmpc_rei_integrand.append(float('nan') if 'OA-CMPC' not in RUN_PLANNERS else
                                    _r_oacmpc * float(_X0_oacmpc_cur[0]))
        _r_rl = (float(drift.get_risk_cartesian(float(_X0_g_rl_cur[0]), float(_X0_g_rl_cur[1])))
                 if 'RL-PPO' in RUN_PLANNERS else float('nan'))
        rl_rei_integrand.append(float('nan') if 'RL-PPO' not in RUN_PLANNERS else
                                _r_rl * float(_X0_rl_cur[0]))

        # ──────────────────────────────────────────────────────────────────────
        # 11. UPDATE SURROUNDING VEHICLES
        # ──────────────────────────────────────────────────────────────────────
        surroundings.total_update_emergency(i)

        # ──────────────────────────────────────────────────────────────────────
        # 12. VISUALIZATION — every frame, gcf/clf pattern matching reference
        # ──────────────────────────────────────────────────────────────────────
        # Build DREAM horizon rollout for visualization
        X0_vis, X0_g_vis = list(X0), list(X0_g)
        X0_g_vis_list = [list(X0_g_vis)]
        for _k in range(len(oa) - 1):
            _u_vis = [oa[_k + 1], od[_k + 1]]
            X0_vis, X0_g_vis, _, _ = dynamics.propagate(
                X0_vis, _u_vis, dt, X0_g_vis, path_d, sample, x_list, y_list, boundary)
            X0_g_vis_list.append(list(X0_g_vis))
        X0_g_vis_list = np.array(X0_g_vis_list)

        fig = plt.gcf()
        fig.clf()

        # Single world-frame view for all panels — centred on the scenario
        # action area (IDEAM baseline ego) so risk fields are directly
        # comparable across arms regardless of each arm's ego position.
        _ideam_ref_ok = (ideam_panel_X0_g is not None and len(ideam_panel_X0_g) >= 2)
        _ref_x = ideam_panel_X0_g[0] if _ideam_ref_ok else X0_g[0]
        _ref_y = ideam_panel_X0_g[1] if _ideam_ref_ok else X0_g[1]
        x_range_i = y_range_i = None  # assigned below via shared range
        _x_range  = [_ref_x - x_area, _ref_x + x_area]
        _y_range  = [_ref_y - y_area, _ref_y + y_area]

        # Dynamic panel drawing — only selected arms
        _cf_last = None
        _ax_last = None
        for _pi, _pname in enumerate(_active_panels):
            _ax_pan = fig.add_subplot(1, _n_pan, _pi + 1)
            _ax_last = _ax_pan
            if _pname == 'IDEAM':
                draw_panel(_ax_pan,
                           ego_global=ideam_panel_X0_g, ego_state=ideam_panel_X0,
                           vehicle_left=vl_mpc, vehicle_centre=vc_mpc, vehicle_right=vr_mpc,
                           x_range=_x_range, y_range=_y_range,
                           title="IDEAM (Baseline)",
                           horizon=ideam_panel_horizon, risk_f=None, risk_val=None,
                           show_merger=True, ego_color=EGO_IDEAM_COLOR)
            elif _pname == 'DREAM':
                _cf = draw_panel(_ax_pan,
                           ego_global=X0_g, ego_state=X0,
                           vehicle_left=vl_mpc, vehicle_centre=vc_mpc, vehicle_right=vr_mpc,
                           x_range=_x_range, y_range=_y_range,
                           title="DREAM (Ours)",
                           horizon=X0_g_vis_list, risk_f=risk_field, risk_val=risk_at_ego,
                           show_merger=True, ego_color=EGO_DREAM_COLOR)
                if _cf is not None: _cf_last = _cf
            elif _pname == 'OA-CMPC':
                # OA-CMPC: no risk field overlay (pure-MPC, no DRIFT visualisation)
                draw_panel(_ax_pan,
                           ego_global=_arm_oacmpc_state['X0_g'], ego_state=_arm_oacmpc_state['X0'],
                           vehicle_left=vl_mpc, vehicle_centre=vc_mpc, vehicle_right=vr_mpc,
                           x_range=_x_range, y_range=_y_range,
                           title="OA-CMPC (MPC only, no CBF)",
                           horizon=_horiz_oacmpc, risk_f=None, risk_val=None,
                           show_merger=True, ego_color=EGO_OACMPC_COLOR)
            elif _pname == 'ADA':
                _cf = draw_panel(_ax_pan,
                           ego_global=_arm_ada_state['X0_g'], ego_state=_arm_ada_state['X0'],
                           vehicle_left=vl_mpc, vehicle_centre=vc_mpc, vehicle_right=vr_mpc,
                           x_range=_x_range, y_range=_y_range,
                           title="ADA-based planner",
                           horizon=_horiz_ada, risk_f=risk_field_ada, risk_val=_rae_ada,
                           show_merger=True, ego_color=EGO_ADA_COLOR)
                if _cf is not None: _cf_last = _cf
            elif _pname == 'APF':
                _cf = draw_panel(_ax_pan,
                           ego_global=_arm_apf_state['X0_g'], ego_state=_arm_apf_state['X0'],
                           vehicle_left=vl_mpc, vehicle_centre=vc_mpc, vehicle_right=vr_mpc,
                           x_range=_x_range, y_range=_y_range,
                           title="APF-based planner",
                           horizon=_horiz_apf, risk_f=risk_field_apf, risk_val=_rae_apf,
                           show_merger=True, ego_color=EGO_APF_COLOR)
                if _cf is not None: _cf_last = _cf
            elif _pname == 'RL-PPO':
                _cf = draw_panel(_ax_pan,
                           ego_global=X0_g_rl, ego_state=X0_rl,
                           vehicle_left=vl_mpc, vehicle_centre=vc_mpc, vehicle_right=vr_mpc,
                           x_range=_x_range, y_range=_y_range,
                           title="RL-PPO (Decision)",
                           horizon=_horiz_rl, risk_f=risk_field_rl, risk_val=_rae_rl,
                           show_merger=True, ego_color=EGO_RL_COLOR)
                if _cf is not None: _cf_last = _cf

        if _cf_last is not None and _ax_last is not None:
            cbar = fig.colorbar(_cf_last, ax=_ax_last,
                                orientation='vertical', pad=0.02, fraction=0.035)
            cbar.set_label('Risk Level', fontsize=9, weight='bold')
            cbar.ax.tick_params(labelsize=8, colors='black')

        plt.savefig(os.path.join(save_dir, "{}.png".format(i)), dpi=600)

    bar.finish()
    print()
    print("Simulation complete.")
    _fs_parts = []
    if 'DREAM'   in RUN_PLANNERS: _fs_parts.append(f"DREAM={dream_s[-1]:.1f}m")
    if 'IDEAM'   in RUN_PLANNERS: _fs_parts.append(f"IDEAM={ideam_s[-1]:.1f}m")
    if 'OA-CMPC' in RUN_PLANNERS: _fs_parts.append(f"OA-CMPC={oacmpc_s[-1]:.1f}m")
    if 'ADA'     in RUN_PLANNERS: _fs_parts.append(f"ADA={ada_s[-1]:.1f}m")
    if 'APF'     in RUN_PLANNERS: _fs_parts.append(f"APF={apf_s[-1]:.1f}m")
    if 'RL-PPO'  in RUN_PLANNERS: _fs_parts.append(f"RL-PPO={rl_s[-1]:.1f}m")
    print("  Final s: " + "  |  ".join(_fs_parts))

    # Collision / near-collision events
    _nc_dream = [(k, dream_dist_agent[k])
                 for k in range(len(dream_dist_agent))
                 if dream_dist_agent[k] < NEAR_COLLISION_DIST]
    _nc_ideam = [(k, ideam_dist_agent[k])
                 for k in range(len(ideam_dist_agent))
                 if not math.isnan(ideam_dist_agent[k]) and
                 ideam_dist_agent[k] < NEAR_COLLISION_DIST]
    _col_dream = [x for x in _nc_dream if x[1] < COLLISION_DIST]
    _col_ideam = [x for x in _nc_ideam if x[1] < COLLISION_DIST]

    _nc_ada  = [(k, ada_dist_agent[k])  for k in range(len(ada_dist_agent))
                if ada_dist_agent[k] < NEAR_COLLISION_DIST]
    _nc_apf  = [(k, apf_dist_agent[k])  for k in range(len(apf_dist_agent))
                if apf_dist_agent[k] < NEAR_COLLISION_DIST]
    _col_ada = [x for x in _nc_ada  if x[1] < COLLISION_DIST]
    _col_apf = [x for x in _nc_apf  if x[1] < COLLISION_DIST]
    _nc_rl   = [(k, rl_dist_agent[k])  for k in range(len(rl_dist_agent))
                if not math.isnan(rl_dist_agent[k]) and rl_dist_agent[k] < NEAR_COLLISION_DIST]
    _col_rl  = [x for x in _nc_rl   if x[1] < COLLISION_DIST]
    _min_merger_truck = float(np.nanmin(merger_truck_dist)) if merger_truck_dist else float("nan")

    print(f"  DREAM near-collisions (<{NEAR_COLLISION_DIST}m): {len(_nc_dream)}  "
          f"| collisions (<{COLLISION_DIST}m): {len(_col_dream)}")
    print(f"  ADA-based planner   near-collisions (<{NEAR_COLLISION_DIST}m): {len(_nc_ada)}  "
          f"| collisions (<{COLLISION_DIST}m): {len(_col_ada)}")
    print(f"  APF-based planner   near-collisions (<{NEAR_COLLISION_DIST}m): {len(_nc_apf)}  "
          f"| collisions (<{COLLISION_DIST}m): {len(_col_apf)}")
    print(f"  IDEAM       near-collisions (<{NEAR_COLLISION_DIST}m): {len(_nc_ideam)}  "
          f"| collisions (<{COLLISION_DIST}m): {len(_col_ideam)}")
    if 'RL-PPO' in RUN_PLANNERS:
        print(f"  RL-PPO      near-collisions (<{NEAR_COLLISION_DIST}m): {len(_nc_rl)}  "
              f"| collisions (<{COLLISION_DIST}m): {len(_col_rl)}")
    print(f"  Min merger-truck centre distance: {_min_merger_truck:.2f} m")

    # ===========================================================================
    # METRICS PLOT
    # ===========================================================================

    _t_arr   = np.arange(N_t) * dt
    _lc_step = merger.lc_started_step             # step when merger LC triggered
    _reveal_t = (_lc_step * dt) if _lc_step is not None else None

    # ── Scalar metrics (post-loop derivation) ────────────────────────────────
    _tau_c = 3.0    # critical TTC threshold [s] for CMR
    _delta  = 10    # post-reveal offset [steps] for SM_rev (1 s)
    _W1, _W2 = 100, 150   # reveal window [steps] before / after trigger

    def _safe_min(arr, lo, hi):
        sl = [float(v) for v in arr[lo:hi]]   # cast before min() to avoid CasADi comparisons
        return min(sl) if sl else float('nan')

    def _safe_val(arr, idx):
        return float(arr[idx]) if (idx is not None and 0 <= idx < len(arr)) else float('nan')

    if _lc_step is not None:
        _w0, _w1 = max(0, _lc_step - _W1), min(N_t, _lc_step + _W2)
        dream_ttc_min_rev = _safe_min(dream_ttc, _w0, _w1)
        ideam_ttc_min_rev = _safe_min(ideam_ttc, _w0, _w1)
        ada_ttc_min_rev    = _safe_min(ada_ttc,    _w0, _w1)
        apf_ttc_min_rev    = _safe_min(apf_ttc,    _w0, _w1)
        oacmpc_ttc_min_rev = _safe_min(oacmpc_ttc, _w0, _w1)
        rl_ttc_min_rev     = _safe_min(rl_ttc,     _w0, _w1)
        _sm_idx = _lc_step + _delta
        dream_sm_rev  = _safe_val(dream_ttc,  _sm_idx)
        ideam_sm_rev  = _safe_val(ideam_ttc,  _sm_idx)
        ada_sm_rev    = _safe_val(ada_ttc,    _sm_idx)
        apf_sm_rev    = _safe_val(apf_ttc,    _sm_idx)
        oacmpc_sm_rev = _safe_val(oacmpc_ttc, _sm_idx)
        rl_sm_rev     = _safe_val(rl_ttc,     _sm_idx)
        # CMR_occ: fraction of pre-reveal window where TTC < tau_c
        _occ_end_cands = [k for k, occ in enumerate(agent_occluded_record) if not occ]
        _occ_end = _occ_end_cands[0] if _occ_end_cands else N_t
        _f = lambda arr: sum(1 for v in arr[:_occ_end] if float(v) < _tau_c) / max(1, _occ_end)
        dream_cmr, ideam_cmr, ada_cmr, apf_cmr, oacmpc_cmr, rl_cmr = (
            _f(dream_ttc), _f(ideam_ttc), _f(ada_ttc), _f(apf_ttc), _f(oacmpc_ttc), _f(rl_ttc))
    else:
        dream_ttc_min_rev = ideam_ttc_min_rev = ada_ttc_min_rev = apf_ttc_min_rev = oacmpc_ttc_min_rev = rl_ttc_min_rev = float('nan')
        dream_sm_rev = ideam_sm_rev = ada_sm_rev = apf_sm_rev = oacmpc_sm_rev = rl_sm_rev = float('nan')
        dream_cmr = ideam_cmr = ada_cmr = apf_cmr = oacmpc_cmr = rl_cmr = float('nan')

    # REI = ∑ R(x_k,t_k)·v_x,k·Δt
    dream_rei  = float(np.sum(dream_rei_integrand)  * dt)
    ideam_rei  = float(np.sum(ideam_rei_integrand)  * dt)
    ada_rei    = float(np.sum(ada_rei_integrand)    * dt)
    apf_rei    = float(np.sum(apf_rei_integrand)    * dt)
    oacmpc_rei = float(np.sum(oacmpc_rei_integrand) * dt)
    rl_rei     = float(np.sum(rl_rei_integrand)     * dt)

    # Conservatism tax CT_v = mean(v_IDEAM) − mean(v_planner)  [positive = planner slower]
    _mean_v_ideam  = float(np.mean(ideam_vx)) if ideam_vx else 0.0
    dream_ct_v  = _mean_v_ideam - float(np.mean(dream_vx))
    ada_ct_v    = _mean_v_ideam - float(np.mean(ada_vx))
    apf_ct_v    = _mean_v_ideam - float(np.mean(apf_vx))
    oacmpc_ct_v = _mean_v_ideam - float(np.mean(oacmpc_vx))
    rl_ct_v     = _mean_v_ideam - float(np.nanmean(rl_vx)) if rl_vx else 0.0

    # Mean |jerk| = mean(|Δa_x / Δt|) per planner
    def _mean_jerk(acc_arr):
        a = np.array(acc_arr)
        return float(np.mean(np.abs(np.diff(a) / dt))) if len(a) > 1 else 0.0

    dream_mean_jerk  = _mean_jerk(dream_acc)
    ideam_mean_jerk  = _mean_jerk(ideam_acc)
    ada_mean_jerk    = _mean_jerk(ada_acc)
    apf_mean_jerk    = _mean_jerk(apf_acc)
    oacmpc_mean_jerk = _mean_jerk(oacmpc_acc)
    rl_mean_jerk     = _mean_jerk(rl_acc)

    # S_o_min: minimum spacing (uses S_obs track, which is min distance to all surrounding vehicles)
    dream_min_s_obs  = float(np.min(dream_s_obs))  if dream_s_obs  else float('nan')
    ideam_min_s_obs  = float(np.min(ideam_s_obs))  if ideam_s_obs  else float('nan')
    ada_min_s_obs    = float(np.min(ada_s_obs))    if ada_s_obs    else float('nan')
    apf_min_s_obs    = float(np.min(apf_s_obs))    if apf_s_obs    else float('nan')
    oacmpc_min_s_obs = float(np.min(oacmpc_s_obs)) if oacmpc_s_obs else float('nan')
    rl_min_s_obs     = float(np.nanmin(rl_s_obs))  if rl_s_obs     else float('nan')

    # Peak lateral acceleration  ay_max = max(|Δvy / dt|)
    def _ay_max(vy_arr):
        v = np.array([float(x) for x in vy_arr])
        return float(np.max(np.abs(np.diff(v) / dt))) if len(v) > 1 else 0.0

    dream_ay_max = _ay_max(dream_vy)
    ideam_ay_max = _ay_max(ideam_vy)

    # Per-arm planning-time statistics
    def _time_stats(t_arr):
        if not t_arr:
            return float('nan'), float('nan'), float('nan')
        t = np.array(t_arr)
        return float(np.mean(t)), float(np.max(t)), float(np.mean(t > dt))

    dream_tplan_mean,  dream_tplan_max,  dream_rRT  = _time_stats(dream_step_times)
    ideam_tplan_mean,  ideam_tplan_max,  ideam_rRT  = _time_stats(ideam_step_times)
    ada_tplan_mean,    ada_tplan_max,    ada_rRT    = _time_stats(ada_step_times)
    apf_tplan_mean,    apf_tplan_max,    apf_rRT    = _time_stats(apf_step_times)
    oacmpc_tplan_mean, oacmpc_tplan_max, oacmpc_rRT = _time_stats(oacmpc_step_times)
    rl_tplan_mean,     rl_tplan_max,     rl_rRT     = _time_stats(rl_step_times)

    # CT_T: progress-gap conservatism tax — estimated extra time for planner to
    # reach IDEAM's final Frenet position  (s_IDEAM_final - s_planner_final) / mean_v
    _s_ideam_fin = float(ideam_s[-1]) if ideam_s else 0.0
    def _ct_T(s_arr, vx_arr):
        s_f = float(s_arr[-1]) if s_arr else 0.0
        vm  = float(np.mean(vx_arr)) if vx_arr else 1.0
        return (_s_ideam_fin - s_f) / max(vm, 0.5)

    dream_ct_T  = _ct_T(dream_s,  dream_vx)
    ada_ct_T    = _ct_T(ada_s,    ada_vx)
    apf_ct_T    = _ct_T(apf_s,    apf_vx)
    oacmpc_ct_T = _ct_T(oacmpc_s, oacmpc_vx)
    rl_ct_T     = _ct_T(rl_s,     rl_vx)

    # ALT: anticipation lead time = (lc_step - risk_cross_step) * dt  [s]
    # Positive = DREAM detected risk BEFORE the LC event
    if _lc_step is not None and _risk_cross_step is not None:
        dream_alt = (_lc_step - _risk_cross_step) * dt
    else:
        dream_alt = float('nan')

    # MPC failure rates [fraction]
    _n_steps_safe = max(1, N_t)
    dream_fail_rate = dream_mpc_fail / _n_steps_safe
    ideam_fail_rate = ideam_mpc_fail / _n_steps_safe
    ada_fail_rate   = ada_mpc_fail   / _n_steps_safe
    apf_fail_rate   = apf_mpc_fail   / _n_steps_safe

    print("\n── Runtime / Efficiency Summary ──────────────────────────────────────")
    print(f"  DREAM  : t_plan_mean={dream_tplan_mean*1e3:.1f}ms  max={dream_tplan_max*1e3:.1f}ms  "
          f"r_RT={dream_rRT*100:.1f}%  fail={dream_fail_rate*100:.1f}%  "
          f"CT_v={dream_ct_v:+.2f}m/s  CT_T={dream_ct_T:+.1f}s  ALT={dream_alt:.1f}s")
    print(f"  IDEAM  : t_plan_mean={ideam_tplan_mean*1e3:.1f}ms  max={ideam_tplan_max*1e3:.1f}ms  "
          f"r_RT={ideam_rRT*100:.1f}%  fail={ideam_fail_rate*100:.1f}%")
    print(f"  ADA    : t_plan_mean={ada_tplan_mean*1e3:.1f}ms  max={ada_tplan_max*1e3:.1f}ms  "
          f"r_RT={ada_rRT*100:.1f}%  fail={ada_fail_rate*100:.1f}%  CT_v={ada_ct_v:+.2f}m/s")
    print(f"  APF    : t_plan_mean={apf_tplan_mean*1e3:.1f}ms  max={apf_tplan_max*1e3:.1f}ms  "
          f"r_RT={apf_rRT*100:.1f}%  fail={apf_fail_rate*100:.1f}%  CT_v={apf_ct_v:+.2f}m/s")
    print(f"  OA-CMPC: t_plan_mean={oacmpc_tplan_mean*1e3:.1f}ms  max={oacmpc_tplan_max*1e3:.1f}ms  "
          f"r_RT={oacmpc_rRT*100:.1f}%  CT_v={oacmpc_ct_v:+.2f}m/s")
    if 'RL-PPO' in RUN_PLANNERS:
        print(f"  RL-PPO : t_plan_mean={rl_tplan_mean*1e3:.1f}ms  max={rl_tplan_max*1e3:.1f}ms  "
              f"r_RT={rl_rRT*100:.1f}%  CT_v={rl_ct_v:+.2f}m/s")

    _planners = ["DREAM", "ADA-based planner", "APF-based planner", "OA-CMPC", "IDEAM", "RL-PPO"]
    _C  = {"DREAM": EGO_DREAM_COLOR, "IDEAM": EGO_IDEAM_COLOR,
           "ADA-based planner": EGO_ADA_COLOR, "APF-based planner": EGO_APF_COLOR,
           "OA-CMPC": EGO_OACMPC_COLOR, "RL-PPO": EGO_RL_COLOR}
    _LS = {"DREAM": "-", "IDEAM": "--", "ADA-based planner": "-.", "APF-based planner": ":",
           "OA-CMPC": (0, (3, 1, 1, 1)), "RL-PPO": (0, (5, 2))}

    # ── FIGURE A: Mechanism figure — event-aligned time series ───────────────
    # Three panels (risk / speed / TTC) shifted so t=0 = merger LC trigger.
    # Pre-reveal shading highlights the DREAM anticipation window.
    if _lc_step is not None:
        _t0_idx = max(0, _lc_step - _W1)
        _t1_idx = min(N_t, _lc_step + _W2)
        _t_win  = (_t_arr[_t0_idx:_t1_idx] - _lc_step * dt)
        _sl     = slice(_t0_idx, _t1_idx)

        with plt.style.context(["science", "no-latex"]):
            fig_mech, axes_mech = plt.subplots(
                3, 1, figsize=(7, 9), constrained_layout=True)
            fig_mech.suptitle(
                "Mechanism Figure — Event-Aligned Time Series\n"
                r"$(t = 0$ = OccludedMerger LC trigger$)$", fontsize=11)

            def _vline_mech(ax, lbl=True):
                # LC trigger vertical line removed per reporting standard
                if _t0_idx < _lc_step:
                    ax.axvspan(_t_win[0], 0, color='lightyellow',
                               alpha=0.55, label='Occlusion window' if lbl else None)

            # Panel 0: GVF risk at ego (all arms)
            ax0 = axes_mech[0]
            _r_trim = [risk_at_ego_list, risk_at_ego_ada_list, risk_at_ego_apf_list, risk_at_ego_oacmpc_list]
            _r_lbl  = ["DREAM", "ADA-based planner", "APF-based planner", "OA-CMPC"]
            for _rr, _rl in zip(_r_trim, _r_lbl):
                _rs = _rr[_t0_idx:_t1_idx]
                if _rs:
                    ax0.plot(_t_win[:len(_rs)], _rs,
                             color=_C[_rl], ls=_LS[_rl], lw=1.5, label=_rl)
                    if _rl == "DREAM":
                        ax0.fill_between(_t_win[:len(_rs)], _rs,
                                         alpha=0.18, color=_C[_rl])
            _vline_mech(ax0)
            ax0.set_ylabel("$R_{\\mathrm{ego}}$")
            ax0.set_title("(a) GVF Risk at Ego Corridor")
            ax0.legend(fontsize=7, loc='upper left')
            ax0.set_xticklabels([])

            # Panel 1: Speed vx(t)
            ax1 = axes_mech[1]
            for _key, _vxarr in [("DREAM", dream_vx), ("ADA-based planner", ada_vx),
                                   ("APF-based planner", apf_vx), ("OA-CMPC", oacmpc_vx), ("IDEAM", ideam_vx),
                                   ("RL-PPO", rl_vx)]:
                ax1.plot(_t_win, np.array(_vxarr)[_sl],
                         color=_C[_key], ls=_LS[_key], lw=1.5, label=_key)
            _vline_mech(ax1, lbl=False)
            ax1.set_ylabel("$v_x$ [m/s]")
            ax1.set_title("(b) Longitudinal Speed")
            ax1.legend(fontsize=7, loc='upper left')
            ax1.set_xticklabels([])

            # Panel 2: TTC to OccludedMerger
            ax2 = axes_mech[2]
            for _key, _ttcarr in [("DREAM", dream_ttc), ("ADA-based planner", ada_ttc),
                                    ("APF-based planner", apf_ttc), ("OA-CMPC", oacmpc_ttc), ("IDEAM", ideam_ttc),
                                    ("RL-PPO", rl_ttc)]:
                ax2.plot(_t_win, np.array(_ttcarr)[_sl],
                         color=_C[_key], ls=_LS[_key], lw=1.5, label=_key)
            ax2.axhline(_tau_c, color='red', lw=0.9, ls=':', label=f'Critical TTC ({_tau_c}s)')
            _vline_mech(ax2, lbl=False)
            ax2.set_ylabel("TTC [s]")
            ax2.set_xlabel("Time relative to merger LC trigger [s]")
            ax2.set_title("(c) TTC to OccludedMerger")
            ax2.legend(fontsize=7, loc='upper left')
            ax2.set_ylim(0, _TTC_CAP * 1.05)

            plt.savefig(os.path.join(save_dir, "metrics_mechanism.png"),
                        dpi=300, bbox_inches='tight')
            plt.close(fig_mech)

    # ── FIGURE B: Summary scalar metrics bar chart ───────────────────────────
    _bar_labels = ["DREAM", "ADA-based\nplanner", "APF-based\nplanner", "OA-CMPC", "IDEAM", "RL-PPO"]
    _bar_colors = [EGO_DREAM_COLOR, EGO_ADA_COLOR, EGO_APF_COLOR, EGO_OACMPC_COLOR, EGO_IDEAM_COLOR, EGO_RL_COLOR]
    _bar_x      = np.arange(len(_bar_labels))

    def _bar_panel(ax, values, title, ylabel, lower_is_better=True, ymin=None):
        # Guard: convert every element to a plain Python float (blocks CasADi leakage)
        values = [float(v) for v in values]
        bars = ax.bar(_bar_x, values, color=_bar_colors, width=0.55, zorder=3)
        ax.set_xticks(_bar_x);  ax.set_xticklabels(_bar_labels, fontsize=8)
        ax.set_title(title, fontsize=9);  ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis='y', lw=0.5, zorder=0)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        # Highlight best bar
        _valid = [(v, j) for j, v in enumerate(values) if not math.isnan(v)]
        if _valid:
            _best_v, _best_j = (min(_valid) if lower_is_better else max(_valid))
            bars[_best_j].set_edgecolor('gold');  bars[_best_j].set_linewidth(2)
        for bar, val in zip(bars, values):
            if not math.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f" {val:.2f}", ha='center', va='bottom', fontsize=7)

    with plt.style.context(["science", "no-latex"]):
        fig_s, axes_s = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        fig_s.suptitle(
            "DREAM vs Baselines — Scalar Metrics Summary\n"
            "(gold border = best; lower-is-better metrics shaded red for worst)",
            fontsize=10)

        # (0,0) TTC_min_rev — higher is better
        _bar_panel(axes_s[0, 0],
                   [dream_ttc_min_rev, ada_ttc_min_rev, apf_ttc_min_rev, oacmpc_ttc_min_rev, ideam_ttc_min_rev, rl_ttc_min_rev],
                   r"$\mathrm{TTC}^\mathrm{rev}_{\min}$ — Event-conditioned min TTC",
                   "TTC [s]", lower_is_better=False, ymin=0)

        # (0,1) CMR_occ — lower is better
        _bar_panel(axes_s[0, 1],
                   [dream_cmr, ada_cmr, apf_cmr, oacmpc_cmr, ideam_cmr, rl_cmr],
                   r"$\mathrm{CMR}_\mathrm{occ}$ — Critical-moment rate (TTC < "
                   f"{_tau_c:.0f}s)",
                   "Fraction", lower_is_better=True, ymin=0)

        # (0,2) SM_rev — higher is better
        _bar_panel(axes_s[0, 2],
                   [dream_sm_rev, ada_sm_rev, apf_sm_rev, oacmpc_sm_rev, ideam_sm_rev, rl_sm_rev],
                   r"$\mathrm{SM}_\mathrm{rev}$ — Post-reveal TTC (+1 s offset)",
                   "TTC [s]", lower_is_better=False, ymin=0)

        # (1,0) REI — lower is better
        _bar_panel(axes_s[1, 0],
                   [dream_rei, ada_rei, apf_rei, oacmpc_rei, ideam_rei, rl_rei],
                   r"$\mathrm{REI}$ — Risk Exposure Integral",
                   r"$\Sigma\, R \cdot v_x \cdot \Delta t$", lower_is_better=True, ymin=0)

        # (1,1) Conservatism tax CT_v — smaller magnitude is better
        _bar_panel(axes_s[1, 1],
                   [dream_ct_v, ada_ct_v, apf_ct_v, oacmpc_ct_v, 0.0, rl_ct_v],
                   r"$\mathrm{CT}_v$ — Conservatism tax (speed loss vs IDEAM)",
                   r"$\bar{v}_{\mathrm{IDEAM}} - \bar{v}_\mathrm{planner}$ [m/s]",
                   lower_is_better=True)

        # (1,2) Mean |jerk| — lower is better
        _bar_panel(axes_s[1, 2],
                   [dream_mean_jerk, ada_mean_jerk, apf_mean_jerk, oacmpc_mean_jerk, ideam_mean_jerk, rl_mean_jerk],
                   r"Mean $|\dot{a}_x|$ — Motion Stability (avg jerk)",
                   r"m/s$^3$", lower_is_better=True, ymin=0)

        plt.savefig(os.path.join(save_dir, "metrics_summary.png"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig_s)

    # ── FIGURE C: Time-series overview (replaces old 3×2) ────────────────────
    with plt.style.context(["science", "no-latex"]):
        fig_m, axes_m = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
        fig_m.suptitle(
            "DREAM vs ADA-based planner vs APF-based planner vs IDEAM — Two-Threat Scenario",
            fontsize=12)

        def _shade(ax):
            if _reveal_t is not None:
                pass  # LC trigger vertical line removed per reporting standard

        # ── (0,0) Progress s(t)
        ax = axes_m[0, 0]
        if 'DREAM'    in RUN_PLANNERS: ax.plot(_t_arr, dream_s, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
        if 'ADA'      in RUN_PLANNERS: ax.plot(_t_arr, ada_s,   color=_C["ADA-based planner"],   ls=_LS["ADA-based planner"],   label="ADA-based planner")
        if 'APF'      in RUN_PLANNERS: ax.plot(_t_arr, apf_s,   color=_C["APF-based planner"],   ls=_LS["APF-based planner"],   label="APF-based planner")
        if 'OA-CMPC'  in RUN_PLANNERS: ax.plot(_t_arr, oacmpc_s, color=_C["OA-CMPC"],            ls=_LS["OA-CMPC"],             label="OA-CMPC")
        if 'IDEAM'    in RUN_PLANNERS: ax.plot(_t_arr, ideam_s, color=_C["IDEAM"],               ls=_LS["IDEAM"],               label="IDEAM")
        if 'RL-PPO'   in RUN_PLANNERS: ax.plot(_t_arr, rl_s,     color=_C["RL-PPO"],             ls=_LS["RL-PPO"],              label="RL-PPO")
        _shade(ax)
        ax.set_xlabel("t [s]"); ax.set_ylabel("s [m]"); ax.set_title("Progress s(t)")
        ax.legend(fontsize=7)

        # ── (0,1) Speed vx(t)
        ax = axes_m[0, 1]
        if 'DREAM'    in RUN_PLANNERS: ax.plot(_t_arr, dream_vx,  color=_C["DREAM"],              ls=_LS["DREAM"],               label="DREAM")
        if 'ADA'      in RUN_PLANNERS: ax.plot(_t_arr, ada_vx,    color=_C["ADA-based planner"],  ls=_LS["ADA-based planner"],   label="ADA-based planner")
        if 'APF'      in RUN_PLANNERS: ax.plot(_t_arr, apf_vx,    color=_C["APF-based planner"],  ls=_LS["APF-based planner"],   label="APF-based planner")
        if 'OA-CMPC'  in RUN_PLANNERS: ax.plot(_t_arr, oacmpc_vx, color=_C["OA-CMPC"],            ls=_LS["OA-CMPC"],             label="OA-CMPC")
        if 'IDEAM'    in RUN_PLANNERS: ax.plot(_t_arr, ideam_vx,  color=_C["IDEAM"],              ls=_LS["IDEAM"],               label="IDEAM")
        if 'RL-PPO'   in RUN_PLANNERS: ax.plot(_t_arr, rl_vx,     color=_C["RL-PPO"],             ls=_LS["RL-PPO"],              label="RL-PPO")
        _shade(ax)
        ax.set_xlabel("t [s]"); ax.set_ylabel("$v_x$ [m/s]"); ax.set_title("Speed $v_x$(t)")
        ax.legend(fontsize=7)

        # ── (1,0) TTC to OccludedMerger
        ax = axes_m[1, 0]
        if 'DREAM'    in RUN_PLANNERS: ax.plot(_t_arr, dream_ttc,  color=_C["DREAM"],             ls=_LS["DREAM"],               label="DREAM")
        if 'ADA'      in RUN_PLANNERS: ax.plot(_t_arr, ada_ttc,    color=_C["ADA-based planner"], ls=_LS["ADA-based planner"],   label="ADA-based planner")
        if 'APF'      in RUN_PLANNERS: ax.plot(_t_arr, apf_ttc,    color=_C["APF-based planner"], ls=_LS["APF-based planner"],   label="APF-based planner")
        if 'OA-CMPC'  in RUN_PLANNERS: ax.plot(_t_arr, oacmpc_ttc, color=_C["OA-CMPC"],           ls=_LS["OA-CMPC"],             label="OA-CMPC")
        if 'IDEAM'    in RUN_PLANNERS: ax.plot(_t_arr, ideam_ttc,  color=_C["IDEAM"],             ls=_LS["IDEAM"],               label="IDEAM")
        if 'RL-PPO'   in RUN_PLANNERS: ax.plot(_t_arr, rl_ttc,     color=_C["RL-PPO"],            ls=_LS["RL-PPO"],              label="RL-PPO")
        ax.axhline(_tau_c, color='red', lw=0.9, ls=':', label=f'TTC_crit = {_tau_c}s')
        _shade(ax)
        ax.set_xlabel("t [s]"); ax.set_ylabel("TTC [s]")
        ax.set_title("TTC to OccludedMerger")
        ax.legend(fontsize=7); ax.set_ylim(0, min(30, _TTC_CAP))

        # ── (1,1) DRIFT risk at ego
        ax = axes_m[1, 1]
        if 'DREAM' in RUN_PLANNERS:
            ax.plot(_t_arr[:len(risk_at_ego_list)], risk_at_ego_list,
                    color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
            ax.fill_between(_t_arr[:len(risk_at_ego_list)], risk_at_ego_list,
                            alpha=0.20, color=_C["DREAM"])
        if 'ADA' in RUN_PLANNERS:
            ax.plot(_t_arr[:len(risk_at_ego_ada_list)], risk_at_ego_ada_list,
                    color=_C["ADA-based planner"], ls=_LS["ADA-based planner"], label="ADA-based planner")
        if 'APF' in RUN_PLANNERS:
            ax.plot(_t_arr[:len(risk_at_ego_apf_list)], risk_at_ego_apf_list,
                    color=_C["APF-based planner"], ls=_LS["APF-based planner"], label="APF-based planner")
        if 'OA-CMPC' in RUN_PLANNERS:
            ax.plot(_t_arr[:len(risk_at_ego_oacmpc_list)], risk_at_ego_oacmpc_list,
                    color=_C["OA-CMPC"], ls=_LS["OA-CMPC"], label="OA-CMPC")
        _shade(ax)
        ax.set_xlabel("t [s]"); ax.set_ylabel("R(ego)")
        ax.set_title("GVF Risk at Ego Corridor")
        ax.legend(fontsize=7)

        # ── (2,0) Longitudinal acceleration
        ax = axes_m[2, 0]
        if 'DREAM'   in RUN_PLANNERS: ax.plot(_t_arr, dream_acc,  color=_C["DREAM"],             ls=_LS["DREAM"],             label="DREAM")
        if 'ADA'     in RUN_PLANNERS: ax.plot(_t_arr, ada_acc,    color=_C["ADA-based planner"], ls=_LS["ADA-based planner"],   label="ADA-based planner")
        if 'APF'     in RUN_PLANNERS: ax.plot(_t_arr, apf_acc,    color=_C["APF-based planner"], ls=_LS["APF-based planner"],   label="APF-based planner")
        if 'OA-CMPC' in RUN_PLANNERS: ax.plot(_t_arr, oacmpc_acc, color=_C["OA-CMPC"],           ls=_LS["OA-CMPC"],             label="OA-CMPC")
        if 'IDEAM'   in RUN_PLANNERS: ax.plot(_t_arr, ideam_acc,  color=_C["IDEAM"],             ls=_LS["IDEAM"],               label="IDEAM")
        if 'RL-PPO'  in RUN_PLANNERS: ax.plot(_t_arr, rl_acc,    color=_C["RL-PPO"],            ls=_LS["RL-PPO"],              label="RL-PPO")
        _shade(ax); ax.axhline(0, color='black', lw=0.5)
        ax.set_xlabel("t [s]"); ax.set_ylabel("$a_x$ [m/s²]")
        ax.set_title("Longitudinal Acceleration")
        ax.legend(fontsize=7)

        # ── (2,1) Min spacing S_o(t)
        ax = axes_m[2, 1]
        if 'DREAM'   in RUN_PLANNERS: ax.plot(_t_arr, dream_s_obs,  color=_C["DREAM"],             ls=_LS["DREAM"],             label="DREAM")
        if 'ADA'     in RUN_PLANNERS: ax.plot(_t_arr, ada_s_obs,    color=_C["ADA-based planner"], ls=_LS["ADA-based planner"],   label="ADA-based planner")
        if 'APF'     in RUN_PLANNERS: ax.plot(_t_arr, apf_s_obs,    color=_C["APF-based planner"], ls=_LS["APF-based planner"],   label="APF-based planner")
        if 'OA-CMPC' in RUN_PLANNERS: ax.plot(_t_arr, oacmpc_s_obs, color=_C["OA-CMPC"],           ls=_LS["OA-CMPC"],             label="OA-CMPC")
        if 'IDEAM'   in RUN_PLANNERS: ax.plot(_t_arr, ideam_s_obs,  color=_C["IDEAM"],             ls=_LS["IDEAM"],               label="IDEAM")
        if 'RL-PPO'  in RUN_PLANNERS: ax.plot(_t_arr, rl_s_obs,    color=_C["RL-PPO"],            ls=_LS["RL-PPO"],              label="RL-PPO")
        ax.axhline(2.0, color='red', lw=0.8, ls=':', label="Safety threshold")
        _shade(ax)
        ax.set_xlabel("t [s]"); ax.set_ylabel("$S_o$ [m]")
        ax.set_title("Min Spacing to Surrounding Vehicles")
        ax.legend(fontsize=7); ax.set_ylim(bottom=0)

        plt.savefig(os.path.join(save_dir, "metrics_uncertainty.png"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig_m)

    # Save numeric metrics
    np.save(os.path.join(save_dir, "metrics_uncertainty.npy"), {
        # ── Time series ──────────────────────────────────────────────────────
        "dream_s":               dream_s,
        "dream_vx":              dream_vx,
        "dream_acc":             dream_acc,
        "dream_s_obs":           dream_s_obs,
        "dream_dist_agent":      dream_dist_agent,
        "dream_ttc":             dream_ttc,
        "dream_rei_integrand":   dream_rei_integrand,
        "risk_at_ego":           risk_at_ego_list,
        "ada_s":                 ada_s,
        "ada_vx":                ada_vx,
        "ada_acc":               ada_acc,
        "ada_s_obs":             ada_s_obs,
        "ada_dist_agent":        ada_dist_agent,
        "ada_ttc":               ada_ttc,
        "ada_rei_integrand":     ada_rei_integrand,
        "risk_at_ego_ada":       risk_at_ego_ada_list,
        "apf_s":                 apf_s,
        "apf_vx":                apf_vx,
        "apf_acc":               apf_acc,
        "apf_s_obs":             apf_s_obs,
        "apf_dist_agent":        apf_dist_agent,
        "apf_ttc":               apf_ttc,
        "apf_rei_integrand":     apf_rei_integrand,
        "risk_at_ego_apf":       risk_at_ego_apf_list,
        "oacmpc_s":              oacmpc_s,
        "oacmpc_vx":             oacmpc_vx,
        "oacmpc_acc":            oacmpc_acc,
        "oacmpc_s_obs":          oacmpc_s_obs,
        "oacmpc_dist_agent":     oacmpc_dist_agent,
        "oacmpc_ttc":            oacmpc_ttc,
        "oacmpc_rei_integrand":  oacmpc_rei_integrand,
        "risk_at_ego_oacmpc":    risk_at_ego_oacmpc_list,
        "rl_s":                  rl_s,
        "rl_vx":                 rl_vx,
        "rl_acc":                rl_acc,
        "rl_s_obs":              rl_s_obs,
        "rl_dist_agent":         rl_dist_agent,
        "rl_ttc":                rl_ttc,
        "rl_rei_integrand":      rl_rei_integrand,
        "ideam_s":               ideam_s,
        "ideam_vx":              ideam_vx,
        "ideam_acc":             ideam_acc,
        "ideam_s_obs":           ideam_s_obs,
        "ideam_dist_agent":      ideam_dist_agent,
        "ideam_ttc":             ideam_ttc,
        "ideam_rei_integrand":   ideam_rei_integrand,
        # ── Scalar metrics (paper tables) ────────────────────────────────────
        "ttc_min_rev":  {"DREAM": dream_ttc_min_rev, "ADA": ada_ttc_min_rev,
                         "APF":   apf_ttc_min_rev,   "OA-CMPC": oacmpc_ttc_min_rev,
                         "IDEAM": ideam_ttc_min_rev,  "RL-PPO": rl_ttc_min_rev},
        "cmr_occ":      {"DREAM": dream_cmr,          "ADA":   ada_cmr,
                         "APF":   apf_cmr,            "OA-CMPC": oacmpc_cmr,
                         "IDEAM": ideam_cmr,          "RL-PPO": rl_cmr},
        "sm_rev":       {"DREAM": dream_sm_rev,        "ADA":   ada_sm_rev,
                         "APF":   apf_sm_rev,          "OA-CMPC": oacmpc_sm_rev,
                         "IDEAM": ideam_sm_rev,        "RL-PPO": rl_sm_rev},
        "rei":          {"DREAM": dream_rei,           "ADA":   ada_rei,
                         "APF":   apf_rei,             "OA-CMPC": oacmpc_rei,
                         "IDEAM": ideam_rei,           "RL-PPO": rl_rei},
        "ct_v":         {"DREAM": dream_ct_v,          "ADA":   ada_ct_v,
                         "APF":   apf_ct_v,            "OA-CMPC": oacmpc_ct_v,
                         "IDEAM": 0.0,                 "RL-PPO": rl_ct_v},
        "mean_jerk":    {"DREAM": dream_mean_jerk,     "ADA":   ada_mean_jerk,
                         "APF":   apf_mean_jerk,       "OA-CMPC": oacmpc_mean_jerk,
                         "IDEAM": ideam_mean_jerk,     "RL-PPO": rl_mean_jerk},
        "min_spacing":  {"DREAM": dream_min_s_obs,     "ADA":   ada_min_s_obs,
                         "APF":   apf_min_s_obs,       "OA-CMPC": oacmpc_min_s_obs,
                         "IDEAM": ideam_min_s_obs,     "RL-PPO": rl_min_s_obs},
        # ── Scenario metadata ────────────────────────────────────────────────
        "agent_occluded":        agent_occluded_record,
        "merger_lc_step":        merger.lc_started_step,
        "merger_truck_dist":     merger_truck_dist,
        "merger_ttc_at_trigger": _merger_ttc_at_trigger,
        "near_collision_dist":   NEAR_COLLISION_DIST,
        "collision_dist":        COLLISION_DIST,
        "tau_c":                 _tau_c,
    })

    print(f"\nFrames saved to: {save_dir}")
    print(f"Time-series plot:  {save_dir}/metrics_uncertainty.png")
    print(f"Mechanism figure:  {save_dir}/metrics_mechanism.png"
          f"{'  (skipped — no trigger)' if _lc_step is None else ''}")
    print(f"Summary metrics:   {save_dir}/metrics_summary.png")
    print(f"\nScalar metrics:")
    print(f"  TTC_min_rev : DREAM={dream_ttc_min_rev:.2f}  ADA={ada_ttc_min_rev:.2f}"
          f"  APF={apf_ttc_min_rev:.2f}  OA-CMPC={oacmpc_ttc_min_rev:.2f}"
          f"  IDEAM={ideam_ttc_min_rev:.2f}  RL-PPO={rl_ttc_min_rev:.2f}  [s, ↑better]")
    print(f"  CMR_occ     : DREAM={dream_cmr:.3f}  ADA={ada_cmr:.3f}"
          f"  APF={apf_cmr:.3f}  OA-CMPC={oacmpc_cmr:.3f}"
          f"  IDEAM={ideam_cmr:.3f}  RL-PPO={rl_cmr:.3f}  [frac, ↓better]")
    print(f"  SM_rev      : DREAM={dream_sm_rev:.2f}  ADA={ada_sm_rev:.2f}"
          f"  APF={apf_sm_rev:.2f}  OA-CMPC={oacmpc_sm_rev:.2f}"
          f"  IDEAM={ideam_sm_rev:.2f}  RL-PPO={rl_sm_rev:.2f}  [s, ↑better]")
    print(f"  REI         : DREAM={dream_rei:.2f}  ADA={ada_rei:.2f}"
          f"  APF={apf_rei:.2f}  OA-CMPC={oacmpc_rei:.2f}"
          f"  IDEAM={ideam_rei:.2f}  RL-PPO={rl_rei:.2f}  [↓better]")
    print(f"  CT_v        : DREAM={dream_ct_v:.2f}  ADA={ada_ct_v:.2f}"
          f"  APF={apf_ct_v:.2f}  OA-CMPC={oacmpc_ct_v:.2f}"
          f"  IDEAM=0.00  RL-PPO={rl_ct_v:.2f}  [m/s vs IDEAM, ↓better]")
    print(f"  Mean |jerk| : DREAM={dream_mean_jerk:.3f}  ADA={ada_mean_jerk:.3f}"
          f"  APF={apf_mean_jerk:.3f}  OA-CMPC={oacmpc_mean_jerk:.3f}"
          f"  IDEAM={ideam_mean_jerk:.3f}  RL-PPO={rl_mean_jerk:.3f}  [m/s³, ↓better]")

# ===========================================================================
# BATCH MODE RUNNER
# ===========================================================================
# Activated when RUN_MODE == "batch" (set near the top of this file).
# Iterates over all BATCH_SUFFIX files in BATCH_DIR (200 unique scenarios),
# resets surroundings + DRIFT per episode, runs a compact simulation loop
# (no figure generation), and prints a statistics summary at the end.
# ===========================================================================

if RUN_MODE == "batch":
    import glob as _glob
    os.makedirs(BATCH_OUT, exist_ok=True)

    _batch_files = sorted(
        [f for f in _glob.glob(os.path.join(BATCH_DIR, f"*{BATCH_SUFFIX}"))
         if os.path.isfile(f)],
        key=lambda p: int(os.path.basename(p).split("_")[0])
    )
    print(f"\n{'='*70}")
    print(f"BATCH MODE  |  {len(_batch_files)} scenarios  |  {BATCH_N_T} steps each")
    print(f"{'='*70}\n")

    _all_ep = []   # list of per-episode metric dicts

    for _bi, _bf in enumerate(_batch_files):
        print(f"[{_bi+1:3d}/{len(_batch_files)}] {os.path.basename(_bf)}", end=" ... ", flush=True)
        _ep_t0 = time.time()

        # ── surroundings ──────────────────────────────────────────────────
        _bs = Surrounding_Vehicles(steer_range, dt, boundary, _bf)
        _bs.vd_center_all[1] = 30.0
        _bs.vd_right_all[1]  = 30.0
        for (_tl, _tv), _tvd in TRUCK_DESIGNATIONS.items():
            if   _tl == 0: _bs.vd_left_all[_tv]   = _tvd
            elif _tl == 1: _bs.vd_center_all[_tv]  = _tvd
            elif _tl == 2: _bs.vd_right_all[_tv]   = _tvd
        for _pp in range(_N_PREADVANCE):
            _bs.total_update_emergency(_pp)

        _vl0b, _vc0b, _vr0b = _bs.get_vehicles_states()
        _tk_b = [_vl0b, _vc0b, _vr0b][_AGENT_TRUCK_LANE][_AGENT_TRUCK_IDX]

        # ── merger + fast car ─────────────────────────────────────────────
        _ms_b = find_safe_merger_spawn_s(
            float(_tk_b[0]) - MERGER_SPAWN_TRUCK_BACKOFF, _vr0b)
        _mg_b = OccludedMerger(paths=[path1c, path2c, path3c],
                               path_data=_path_data,
                               s_init=float(_ms_b["s"]), vd=MERGER_VD)
        _fc_b = LeftLaneFastCar(paths=[path1c, path2c, path3c],
                                path_data=_path_data,
                                steer_range=steer_range,
                                s_init=LEFT_FAST_CAR_S_INIT,
                                vd=LEFT_FAST_CAR_VD, lane_idx=0)

        # ── DRIFT reset + warmup ──────────────────────────────────────────
        _vd_b = convert_to_drift(_vl0b, _vc0b, _vr0b,
                                 truck_set=set(TRUCK_DESIGNATIONS.keys()))
        _vd_b.append(_mg_b.to_drift_vehicle(vid=998))
        _vd_b.append(_fc_b.to_drift_vehicle(vid=997))
        _bX0  = [8.0, 0.0, 0.0, 20.0, 0.0, 0.0]
        _bX0g = [path1c(_bX0[3])[0], path1c(_bX0[3])[1],
                 path1c.get_theta_r(_bX0[3])]
        _ego_iv = drift_create_vehicle(
            vid=0, x=_bX0g[0], y=_bX0g[1],
            vx=_bX0[0]*math.cos(_bX0g[2]),
            vy=_bX0[0]*math.sin(_bX0g[2]), vclass='car')
        _ego_iv['heading'] = _bX0g[2]
        controller.drift.reset()
        controller.drift.warmup(_vd_b, _ego_iv, dt=dt, duration=5.0, substeps=3)
        controller_ada.drift.reset()
        controller_ada.drift.warmup(_vd_b, _ego_iv, dt=dt, duration=5.0,
                                    substeps=3, source_fn=compute_Q_ADA)
        controller_apf.drift.reset()
        controller_apf.drift.warmup(_vd_b, _ego_iv, dt=dt, duration=5.0,
                                    substeps=3, source_fn=compute_Q_APF)
        controller_oacmpc.drift.reset()
        controller_oacmpc.drift.warmup(_vd_b, _ego_iv, dt=dt, duration=5.0,
                                       substeps=3, source_fn=compute_Q_OACMPC)

        # ── arm states ────────────────────────────────────────────────────
        _barm_gvf = dict(
            X0=list(_bX0), X0_g=list(_bX0g), oa=0.0, od=0.0, last_X=None,
            ovx=0.0, ovy=0.0, owz=0.0, oS=float(_bX0[3]), oey=0.0, oepsi=0.0,
            path_changed=0, path_d=path1c,
            controller=controller, drift_obj=controller.drift, source_fn=None,
            utils_obj=utils, decision_obj=decision_maker,
            d0_base=utils.d0, Th_base=utils.Th,
            al_base=controller.mpc.a_l, bl_base=controller.mpc.b_l,
            P_base=controller.mpc.P.copy(), proactive_cooldown=0,
        )
        _barm_ada = dict(
            X0=list(_bX0), X0_g=list(_bX0g), oa=0.0, od=0.0, last_X=None,
            ovx=0.0, ovy=0.0, owz=0.0, oS=float(_bX0[3]), oey=0.0, oepsi=0.0,
            path_changed=0, path_d=path1c,
            controller=controller_ada, drift_obj=controller_ada.drift,
            source_fn=compute_Q_ADA, utils_obj=utils_ada,
            decision_obj=decision_maker_ada,
            d0_base=utils_ada.d0, Th_base=utils_ada.Th,
            al_base=controller_ada.mpc.a_l, bl_base=controller_ada.mpc.b_l,
            P_base=controller_ada.mpc.P.copy(), proactive_cooldown=0,
        )
        _barm_apf = dict(
            X0=list(_bX0), X0_g=list(_bX0g), oa=0.0, od=0.0, last_X=None,
            ovx=0.0, ovy=0.0, owz=0.0, oS=float(_bX0[3]), oey=0.0, oepsi=0.0,
            path_changed=0, path_d=path1c,
            controller=controller_apf, drift_obj=controller_apf.drift,
            source_fn=compute_Q_APF, utils_obj=utils_apf,
            decision_obj=decision_maker_apf,
            d0_base=utils_apf.d0, Th_base=utils_apf.Th,
            al_base=controller_apf.mpc.a_l, bl_base=controller_apf.mpc.b_l,
            P_base=controller_apf.mpc.P.copy(), proactive_cooldown=0,
        )
        _barm_oacmpc = dict(
            X0=list(_bX0), X0_g=list(_bX0g), oa=0.0, od=0.0, last_X=None,
            ovx=0.0, ovy=0.0, owz=0.0, oS=float(_bX0[3]), oey=0.0, oepsi=0.0,
            path_changed=0, path_d=path1c,
            controller=controller_oacmpc, drift_obj=controller_oacmpc.drift,
            source_fn=compute_Q_OACMPC, utils_obj=utils_oacmpc,
            decision_obj=decision_maker_oacmpc,
            d0_base=utils_oacmpc.d0, Th_base=utils_oacmpc.Th,
            al_base=controller_oacmpc.mpc.a_l, bl_base=controller_oacmpc.mpc.b_l,
            P_base=controller_oacmpc.mpc.P.copy(), proactive_cooldown=0,
        )
        # IDEAM state
        _bX0i  = list(_bX0);  _bX0gi  = list(_bX0g)
        _boa_i = 0.0;  _bod_i = 0.0;  _blX_i = None;  _bpc_i = 0
        utils_ideam_viz.d0 = utils_ideam_viz.__class__(**util_params()).d0
        utils_ideam_viz.Th = utils_ideam_viz.__class__(**util_params()).Th

        # ── per-episode accumulators ──────────────────────────────────────
        _bm = {k: [] for k in (
            'gvf_vx', 'gvf_acc', 'gvf_vy', 'gvf_sp',
            'ideam_vx', 'ideam_acc', 'ideam_vy', 'ideam_sp',
            'ada_vx', 'ada_acc', 'apf_vx', 'apf_acc',
            'oacmpc_vx', 'oacmpc_acc',
            'gvf_ttc', 'ideam_ttc', 'ada_ttc', 'apf_ttc', 'oacmpc_ttc',
            'gvf_rei', 'ideam_rei', 'ada_rei', 'apf_rei', 'oacmpc_rei',
            'gvf_t', 'ideam_t', 'ada_t', 'apf_t', 'oacmpc_t',
            'occ',
        )}
        _bfail = {'gvf': 0, 'ideam': 0, 'ada': 0, 'apf': 0, 'oacmpc': 0}
        _b_lc_step = None
        _b_risk_cross = None
        _b_ideam_lcc = None

        # ── mini main loop ────────────────────────────────────────────────
        for _si in range(BATCH_N_T):
            _vl_b, _vc_b, _vr_b = _bs.get_vehicles_states()
            for (_tl, _tv), _tvd in TRUCK_DESIGNATIONS.items():
                _ta = [_vl_b, _vc_b, _vr_b]
                if len(_ta[_tl]) > _tv:
                    _ta[_tl][_tv][6] = min(float(_ta[_tl][_tv][6]), float(_tvd))
            _tk_cur = [_vl_b, _vc_b, _vr_b][_AGENT_TRUCK_LANE][_AGENT_TRUCK_IDX]

            # Update synthetic agents
            _fc_b.update(_vl_b)
            _fc_b.check_occlusion(_bX0gi[0], _bX0gi[1], _tk_cur)
            if not _fc_b.revealed and (not _fc_b.occluded or _si >= LEFT_FAST_CAR_REVEAL_FALLBACK):
                _fc_b.revealed = True; _fc_b.reveal_step = _si
            _mg_b.update(
                _vl_b, _vc_b, _vr_b,
                truck_state=_tk_cur,
                ego_state=_bX0i, ego_global=_bX0gi,
                left_fast_row=(_fc_b.to_ideam_row() if _fc_b.revealed else None),
                step_idx=_si,
            )
            if _b_lc_step is None and _mg_b.lc_started_step is not None:
                _b_lc_step = _mg_b.lc_started_step

            # DRIFT vehicle list
            _vdr_b = convert_to_drift(_vl_b, _vc_b, _vr_b,
                                      truck_set=set(TRUCK_DESIGNATIONS.keys()))
            _vdr_b.append(_mg_b.to_drift_vehicle(vid=998))
            _vdr_b.append(_fc_b.to_drift_vehicle(vid=997))

            # MPC lanes (merger visible only after 50% LC)
            _vl_m = sort_lane(np.asarray(_vl_b, dtype=float).copy())
            _vc_m = sort_lane(np.asarray(_vc_b, dtype=float).copy())
            _vr_m = sort_lane(np.asarray(_vr_b, dtype=float).copy())
            if _fc_b.revealed:
                _vl_m = append_lane_row(_vl_m, _fc_b.to_ideam_row())
                _vl_m = sort_lane(_vl_m)
            if _mg_b.in_centre_lane():
                _vc_m = append_lane_row(_vc_m, _mg_b.to_center_lane_row())
                _vc_m = sort_lane(_vc_m)

            # ─ DREAM-GVF ─
            _t0g = time.time()
            try:
                _barm_gvf, _, _r_gvf, _ = _run_dream_arm(
                    _barm_gvf, _vdr_b, _vl_m, _vc_m, _vr_m)
            except Exception:
                _bfail['gvf'] += 1
                _r_gvf = 0.0
            _bm['gvf_t'].append(time.time() - _t0g)
            if _b_risk_cross is None and _r_gvf > config_integration.decision_risk_threshold:
                _b_risk_cross = _si

            # ─ IDEAM ─
            _t0i = time.time()
            try:
                _res_i = ideam_agent_step(
                    _bX0i, _bX0gi, _boa_i, _bod_i, _blX_i, _bpc_i,
                    baseline_mpc_viz, utils_ideam_viz, decision_maker, dynamics,
                    _vl_m, _vc_m, _vr_m)
                if _res_i['ok']:
                    _bX0i = list(_res_i['X0']); _bX0gi = list(_res_i['X0_g'])
                    _boa_i = _res_i['oa']; _bod_i = _res_i['od']
                    _blX_i = _res_i['last_X']; _bpc_i = _res_i['path_changed']
                    if (_b_ideam_lcc is None and
                            _res_i['path_now'] == 0 and _res_i['path_dindex'] == 1):
                        _b_ideam_lcc = _si
                else:
                    _bfail['ideam'] += 1
            except Exception:
                _bfail['ideam'] += 1
            _bm['ideam_t'].append(time.time() - _t0i)

            # ─ ADA ─
            _t0a = time.time()
            try:
                _barm_ada, _, _, _ = _run_dream_arm(
                    _barm_ada, _vdr_b, _vl_m, _vc_m, _vr_m)
            except Exception:
                _bfail['ada'] += 1
            _bm['ada_t'].append(time.time() - _t0a)

            # ─ APF ─
            _t0p = time.time()
            try:
                _barm_apf, _, _, _ = _run_dream_arm(
                    _barm_apf, _vdr_b, _vl_m, _vc_m, _vr_m)
            except Exception:
                _bfail['apf'] += 1
            _bm['apf_t'].append(time.time() - _t0p)

            # ─ OA-CMPC ─
            _t0o = time.time()
            try:
                _barm_oacmpc, _, _, _ = _run_dream_arm(
                    _barm_oacmpc, _vdr_b, _vl_m, _vc_m, _vr_m)
            except Exception:
                _bfail['oacmpc'] += 1
            _bm['oacmpc_t'].append(time.time() - _t0o)

            # Advance surroundings
            _bs.total_update_emergency(_si)

            # ── per-step metric collection ────────────────────────────────
            _gvf_X0 = _barm_gvf['X0']; _gvf_X0g = _barm_gvf['X0_g']
            _ada_X0 = _barm_ada['X0']; _apf_X0  = _barm_apf['X0']
            _oacmpc_X0 = _barm_oacmpc['X0']
            _mv = float(_mg_b.state[6])
            _mg_x, _mg_y = _mg_b.x, _mg_b.y

            _bm['gvf_vx'].append(float(_gvf_X0[0]));  _bm['gvf_acc'].append(float(_barm_gvf['oa'][0] if hasattr(_barm_gvf['oa'], '__len__') else _barm_gvf['oa']))
            _bm['gvf_vy'].append(float(_gvf_X0[1]))
            _bm['ideam_vx'].append(float(_bX0i[0])); _bm['ideam_acc'].append(float(_boa_i[0] if hasattr(_boa_i, '__len__') else _boa_i))
            _bm['ideam_vy'].append(float(_bX0i[1]))
            _bm['ada_vx'].append(float(_ada_X0[0]));  _bm['ada_acc'].append(float(_barm_ada['oa'][0] if hasattr(_barm_ada['oa'], '__len__') else _barm_ada['oa']))
            _bm['apf_vx'].append(float(_apf_X0[0]));  _bm['apf_acc'].append(float(_barm_apf['oa'][0] if hasattr(_barm_apf['oa'], '__len__') else _barm_apf['oa']))
            _bm['oacmpc_vx'].append(float(_oacmpc_X0[0]));  _bm['oacmpc_acc'].append(float(_barm_oacmpc['oa'][0] if hasattr(_barm_oacmpc['oa'], '__len__') else _barm_oacmpc['oa']))

            # Min spacing: min dist to all surrounding vehicles + merger
            _nb = [(float(r[3]), float(r[4])) for _la in [_vl_b, _vc_b, _vr_b] for r in _la]
            _nb.append((_mg_x, _mg_y))
            _min_sp = lambda pos: min((math.hypot(float(pos[0])-nx, float(pos[1])-ny) for nx,ny in _nb), default=float('inf'))
            _bm['gvf_sp'].append(_min_sp(_gvf_X0g)); _bm['ideam_sp'].append(_min_sp(_bX0gi))

            # TTC to merger
            _bm['gvf_ttc'].append(_ttc_to_agent(_gvf_X0g, _gvf_X0[0], _mg_x, _mg_y, _mv))
            _bm['ideam_ttc'].append(_ttc_to_agent(_bX0gi,  _bX0i[0],  _mg_x, _mg_y, _mv))
            _bm['ada_ttc'].append(_ttc_to_agent(_barm_ada['X0_g'], float(_ada_X0[0]), _mg_x, _mg_y, _mv))
            _bm['apf_ttc'].append(_ttc_to_agent(_barm_apf['X0_g'], float(_apf_X0[0]), _mg_x, _mg_y, _mv))
            _bm['oacmpc_ttc'].append(_ttc_to_agent(_barm_oacmpc['X0_g'], float(_oacmpc_X0[0]), _mg_x, _mg_y, _mv))

            # REI (all evaluated on GVF field)
            _r_g  = float(controller.drift.get_risk_cartesian(float(_gvf_X0g[0]),          float(_gvf_X0g[1])))
            _r_i  = float(controller.drift.get_risk_cartesian(float(_bX0gi[0]),             float(_bX0gi[1])))
            _r_a  = float(controller.drift.get_risk_cartesian(float(_barm_ada['X0_g'][0]),  float(_barm_ada['X0_g'][1])))
            _r_p  = float(controller.drift.get_risk_cartesian(float(_barm_apf['X0_g'][0]),  float(_barm_apf['X0_g'][1])))
            _r_o  = float(controller.drift.get_risk_cartesian(float(_barm_oacmpc['X0_g'][0]), float(_barm_oacmpc['X0_g'][1])))
            _bm['gvf_rei'].append(_r_g * float(_gvf_X0[0]))
            _bm['ideam_rei'].append(_r_i * float(_bX0i[0]))
            _bm['ada_rei'].append(_r_a * float(_ada_X0[0]))
            _bm['apf_rei'].append(_r_p * float(_apf_X0[0]))
            _bm['oacmpc_rei'].append(_r_o * float(_oacmpc_X0[0]))
            _bm['occ'].append(_mg_b.is_occluded())

        # ── scalar metrics for this episode ──────────────────────────────
        _ep_n = BATCH_N_T
        _lcs  = _b_lc_step
        _W1b, _W2b, _db = 50, 75, 5
        _tau_cb = 3.0

        def _bs_min(arr, lo, hi):
            sl = [float(v) for v in arr[lo:hi]]
            return min(sl) if sl else float('nan')
        def _bs_val(arr, idx):
            return float(arr[idx]) if (idx is not None and 0 <= idx < len(arr)) else float('nan')
        def _bs_rei(arr): return float(np.sum(arr) * dt)
        def _bs_ctv(arr): return float(np.mean(_bm['ideam_vx'])) - float(np.mean(arr)) if arr else float('nan')
        def _bs_jerk(arr):
            a = np.array(arr); return float(np.mean(np.abs(np.diff(a)/dt))) if len(a)>1 else 0.0
        def _bs_aymax(arr):
            v = np.array(arr); return float(np.max(np.abs(np.diff(v)/dt))) if len(v)>1 else 0.0
        def _bs_tplan(arr):
            if not arr: return float('nan'), float('nan'), float('nan')
            t = np.array(arr)
            return float(np.mean(t)), float(np.max(t)), float(np.mean(t > dt))

        if _lcs is not None:
            _w0b = max(0, _lcs - _W1b); _w1b = min(_ep_n, _lcs + _W2b)
            _ttcmin = {k: _bs_min(_bm[f'{k}_ttc'], _w0b, _w1b) for k in ('gvf','ideam','ada','apf','oacmpc')}
            _smrev  = {k: _bs_val(_bm[f'{k}_ttc'], _lcs + _db)  for k in ('gvf','ideam','ada','apf','oacmpc')}
            _occ_end_b = next((k for k,o in enumerate(_bm['occ']) if not o), _ep_n)
            _cmr = {k: sum(1 for v in _bm[f'{k}_ttc'][:_occ_end_b] if float(v)<_tau_cb) / max(1,_occ_end_b) for k in ('gvf','ideam','ada','apf','oacmpc')}
        else:
            _ttcmin = _smrev = _cmr = {k: float('nan') for k in ('gvf','ideam','ada','apf','oacmpc')}

        _rei   = {k: _bs_rei(_bm[f'{k}_rei'])  for k in ('gvf','ideam','ada','apf','oacmpc')}
        _ctv   = {k: _bs_ctv(_bm[f'{k}_vx'])   for k in ('gvf','ada','apf','oacmpc')}
        _jerk  = {k: _bs_jerk(_bm[f'{k}_acc']) for k in ('gvf','ideam','ada','apf','oacmpc')}
        _aymax = {k: _bs_aymax(_bm[f'{k}_vy']) for k in ('gvf','ideam')}
        _somin = {k: float(np.min(_bm[f'{k}_sp'])) if _bm[f'{k}_sp'] else float('nan') for k in ('gvf','ideam')}
        _gvf_tm, _gvf_tmax, _gvf_rRT   = _bs_tplan(_bm['gvf_t'])
        _ideam_tm, _ideam_tmax, _ideam_rRT = _bs_tplan(_bm['ideam_t'])
        _ada_tm, _ada_tmax, _ada_rRT    = _bs_tplan(_bm['ada_t'])
        _apf_tm, _apf_tmax, _apf_rRT    = _bs_tplan(_bm['apf_t'])
        _oacmpc_tm, _oacmpc_tmax, _oacmpc_rRT = _bs_tplan(_bm['oacmpc_t'])
        _alt_b = (_b_lc_step - _b_risk_cross) * dt if (_b_lc_step is not None and _b_risk_cross is not None) else float('nan')

        _all_ep.append({
            'ttcmin_gvf': _ttcmin['gvf'],  'ttcmin_ideam': _ttcmin['ideam'],
            'ttcmin_ada': _ttcmin['ada'],   'ttcmin_apf':   _ttcmin['apf'],
            'ttcmin_oacmpc': _ttcmin['oacmpc'],
            'smrev_gvf':  _smrev['gvf'],    'smrev_ideam':  _smrev['ideam'],
            'smrev_ada':  _smrev['ada'],     'smrev_apf':    _smrev['apf'],
            'smrev_oacmpc': _smrev['oacmpc'],
            'cmr_gvf':    _cmr['gvf'],      'cmr_ideam':    _cmr['ideam'],
            'cmr_ada':    _cmr['ada'],       'cmr_apf':      _cmr['apf'],
            'cmr_oacmpc': _cmr['oacmpc'],
            'rei_gvf':    _rei['gvf'],       'rei_ideam':    _rei['ideam'],
            'rei_ada':    _rei['ada'],        'rei_apf':      _rei['apf'],
            'rei_oacmpc': _rei['oacmpc'],
            'ctv_gvf':    _ctv['gvf'],       'ctv_ada':      _ctv['ada'],
            'ctv_apf':    _ctv['apf'],        'ctv_oacmpc':   _ctv['oacmpc'],
            'jerk_gvf':   _jerk['gvf'],      'jerk_ideam':   _jerk['ideam'],
            'jerk_ada':   _jerk['ada'],       'jerk_apf':     _jerk['apf'],
            'jerk_oacmpc': _jerk['oacmpc'],
            'aymax_gvf':  _aymax['gvf'],     'aymax_ideam':  _aymax['ideam'],
            'somin_gvf':  _somin['gvf'],     'somin_ideam':  _somin['ideam'],
            'tplan_gvf':  _gvf_tm,           'tplan_ideam':  _ideam_tm,
            'tplan_ada':  _ada_tm,            'tplan_apf':    _apf_tm,
            'tplan_oacmpc': _oacmpc_tm,
            'tmax_gvf':   _gvf_tmax,         'tmax_ideam':   _ideam_tmax,
            'tmax_ada':   _ada_tmax,          'tmax_apf':     _apf_tmax,
            'tmax_oacmpc': _oacmpc_tmax,
            'rRT_gvf':    _gvf_rRT,          'rRT_ideam':    _ideam_rRT,
            'rRT_ada':    _ada_rRT,           'rRT_apf':      _apf_rRT,
            'rRT_oacmpc': _oacmpc_rRT,
            'fail_gvf':   _bfail['gvf']/_ep_n, 'fail_ideam': _bfail['ideam']/_ep_n,
            'fail_ada':   _bfail['ada']/_ep_n,  'fail_apf':   _bfail['apf']/_ep_n,
            'fail_oacmpc': _bfail['oacmpc']/_ep_n,
            'alt_gvf':    _alt_b,
            'lc_fired':   _lcs is not None,
        })
        print(f"done  ({time.time()-_ep_t0:.1f}s)  TTC_min DREAM={_ttcmin['gvf']:.2f}s IDEAM={_ttcmin['ideam']:.2f}s  "
              f"fail: GVF={_bfail['gvf']} IDEAM={_bfail['ideam']} ADA={_bfail['ada']} APF={_bfail['apf']} OACMPC={_bfail['oacmpc']}")

    # ── Save raw episode data ─────────────────────────────────────────────
    _batch_npy = os.path.join(BATCH_OUT, "batch_metrics.npy")
    np.save(_batch_npy, _all_ep)
    print(f"\n[BATCH] Raw episode metrics saved → {_batch_npy}")

    # ── Summary statistics ────────────────────────────────────────────────
    def _stat(key):
        vals = [float(ep[key]) for ep in _all_ep if not math.isnan(float(ep[key]))]
        if not vals: return float('nan'), float('nan')
        return float(np.mean(vals)), float(np.std(vals))

    def _fmt(key, scale=1.0, unit=""):
        mu, sd = _stat(key)
        return f"{mu*scale:.3f} ± {sd*scale:.3f}{unit}" if not math.isnan(mu) else "  N/A"

    _lc_frac = 100 * sum(1 for ep in _all_ep if ep['lc_fired']) / max(1, len(_all_ep))
    print(f"\n{'='*78}")
    print(f"BATCH STATISTICS  ({len(_all_ep)} episodes, {_lc_frac:.0f}% with LC event)")
    print(f"{'='*78}")
    _H = f"{'Metric':<20} {'DREAM':>16} {'IDEAM':>16} {'ADA-based':>16} {'APF-based':>16} {'OA-CMPC':>16}"
    print(_H)
    print("-" * len(_H))

    def _row(label, gk, ik, ak, pk, ok, scale=1.0, unit=""):
        g = _fmt(gk, scale, unit)
        i = _fmt(ik, scale, unit)
        a = _fmt(ak, scale, unit)
        p = _fmt(pk, scale, unit)
        o = _fmt(ok, scale, unit)
        print(f"  {label:<18} {g:>16} {i:>16} {a:>16} {p:>16} {o:>16}")

    _row("TTC_min_rev [s]",   "ttcmin_gvf","ttcmin_ideam","ttcmin_ada","ttcmin_apf","ttcmin_oacmpc")
    _row("SM_rev [s]",        "smrev_gvf", "smrev_ideam", "smrev_ada", "smrev_apf", "smrev_oacmpc")
    _row("CMR_occ [frac]",    "cmr_gvf",   "cmr_ideam",   "cmr_ada",   "cmr_apf",   "cmr_oacmpc")
    _row("REI",               "rei_gvf",   "rei_ideam",   "rei_ada",   "rei_apf",   "rei_oacmpc")
    _row("S_o_min [m]",       "somin_gvf", "somin_ideam", "somin_gvf", "somin_gvf", "somin_gvf")
    _row("CT_v [m/s]",        "ctv_gvf",   "ctv_gvf",     "ctv_ada",   "ctv_apf",   "ctv_oacmpc")
    _row("Mean jerk [m/s³]",  "jerk_gvf",  "jerk_ideam",  "jerk_ada",  "jerk_apf",  "jerk_oacmpc")
    _row("ay_max [m/s²]",     "aymax_gvf", "aymax_ideam", "aymax_gvf", "aymax_gvf", "aymax_gvf")
    _row("t_plan mean [ms]",  "tplan_gvf", "tplan_ideam", "tplan_ada", "tplan_apf", "tplan_oacmpc", scale=1e3)
    _row("t_plan max [ms]",   "tmax_gvf",  "tmax_ideam",  "tmax_ada",  "tmax_apf",  "tmax_oacmpc",  scale=1e3)
    _row("r_RT [%]",          "rRT_gvf",   "rRT_ideam",   "rRT_ada",   "rRT_apf",   "rRT_oacmpc",   scale=100)
    _row("Fail rate [%]",     "fail_gvf",  "fail_ideam",  "fail_ada",  "fail_apf",  "fail_oacmpc",  scale=100)

    _alt_mu, _alt_sd = _stat("alt_gvf")
    print(f"  {'ALT [s]':<18} {_alt_mu:.2f} ± {_alt_sd:.2f}  (DREAM only)")

    print(f"{'='*78}")
    print(f"[BATCH] Results saved to {BATCH_OUT}")

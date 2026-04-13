"""
PINN Risk Field — Physics-Informed Neural Network for DRIFT Risk Propagation
=============================================================================
Implements the advection-diffusion-telegrapher PDE in a learned surrogate:

    τ ∂²R/∂t²  +  ∂R/∂t  +  ∇·(vR)  =  ∇·(D∇R)  +  Q(x,t)  −  λR

Phases
------
1. Data loading  : exiD recording → DRIFT vehicle dicts → numerical PDE snapshots
2. PINN training : R_θ(x,y,t) with physics residual + data loss + IC/BC losses
3. Validation    : L2 / relative error vs numerical solver, timing comparison

Usage
-----
    conda run -n base python pinn_risk_field.py
or
    python pinn_risk_field.py --recording 00 --epochs 2000 --device cpu

"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # avoid OMP clash on Windows
import sys
import time
import argparse
import textwrap
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scienceplots  # noqa: F401 — registers science/bright/etc. styles with matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Path setup — find DREAM root from this file's location
# ---------------------------------------------------------------------------
DREAM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DREAM_ROOT)

from tracks_import import read_from_csv
from config import Config as cfg
from pde_solver import (
    PDESolver,
    compute_total_Q,
    compute_velocity_field,
    compute_diffusion_field,
    create_vehicle as drift_create_vehicle,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Module-level constant so Normalizer and ExiDLoader share the same default
_PERCEPTION_RANGE_DEFAULT = 80.0    # metres

# ===========================================================================
# PHASE 1 — DATA LOADING
# ===========================================================================

class ExiDLoader:
    """
    Load one exiD recording, subsample to dt=0.1 s, convert each frame to
    DRIFT vehicle dicts, run the numerical PDESolver, and collect risk-field
    snapshots for PINN training.

    Attributes
    ----------
    snapshots : list of dict
        Each entry: {'t': float, 'R': ndarray(ny,nx), 'Q': ..., 'vx': ...,
                     'vy': ..., 'D': ..., 'occ_mask': ...}
    x_grid, y_grid : 1-D arrays — grid axes
    """

    FRAME_RATE   = 25          # exiD native frame rate [Hz]
    DT_TARGET    = 0.1         # desired simulation step [s]
    FRAME_STRIDE = max(1, int(FRAME_RATE * DT_TARGET))   # = 2 or 3 frames

    def __init__(self, data_dir: str, recording_id: str = "00",
                 max_seconds: float = 40.0, warmup_seconds: float = 4.0,
                 substeps: int = 3,
                 perception_range: float = _PERCEPTION_RANGE_DEFAULT):
        self.data_dir        = data_dir
        self.recording_id    = recording_id
        self.max_seconds     = max_seconds
        self.warmup_seconds  = warmup_seconds
        self.substeps        = substeps
        self.perception_range = perception_range  # instance override of class default

        self.snapshots  = []
        self.x_grid     = cfg.x
        self.y_grid     = cfg.y
        self._X, self._Y = cfg.X, cfg.Y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list:
        """Full pipeline: read CSV → vehicles → PDE → snapshots."""
        tracks, tracks_meta, rec_meta = self._read_csv()
        print(f"[Phase1] Recording {self.recording_id}: "
              f"{len(tracks)} tracks, {rec_meta['frameRate']:.0f} Hz, "
              f"duration ≈ {rec_meta['duration']:.1f} s")

        frame_rate  = rec_meta['frameRate']
        stride      = max(1, round(frame_rate * self.DT_TARGET))
        dt_actual   = stride / frame_rate
        max_frames  = int(self.max_seconds * frame_rate)
        warmup_frames = int(self.warmup_seconds * frame_rate)
        print(f"[Phase1] stride={stride}, dt={dt_actual:.3f}s, "
              f"warmup={self.warmup_seconds}s, run={self.max_seconds}s")

        # Build frame→vehicle lookup
        frame_vehicles = self._build_frame_lookup(tracks, tracks_meta)
        all_frames = sorted(frame_vehicles.keys())
        if not all_frames:
            raise RuntimeError("No frames found in recording.")

        solver = PDESolver()

        # ------ coordinate normalisation factors ------
        # exiD uses UTM metres; cfg uses a RELATIVE grid.
        # We map exiD world coords onto the cfg grid via a simple
        # translation: put the median traffic position at grid centre.
        mid_frame = all_frames[len(all_frames)//2]
        self._compute_coord_offset(frame_vehicles[mid_frame])

        # ------ warm-up (don't store) ------
        print(f"[Phase1] Warming up PDE for {self.warmup_seconds}s ...", end="", flush=True)
        wf_start = all_frames[0]
        wf_end   = min(all_frames[-1], wf_start + warmup_frames)
        t_sim    = 0.0
        for f in all_frames:
            if f < wf_start or f > wf_end:
                continue
            if (f - wf_start) % stride != 0:
                continue
            vehicles, ego = self._frame_to_drift(frame_vehicles[f])
            if ego is None:
                continue
            self._pde_step(solver, vehicles, ego, dt_actual)
            t_sim += dt_actual
        print(" done")

        # ------ main recording window ------
        run_start = wf_end
        run_end   = min(all_frames[-1], wf_start + max_frames)
        t_sim     = 0.0
        stored    = 0
        for f in all_frames:
            if f < run_start or f > run_end:
                continue
            if (f - run_start) % stride != 0:
                continue
            vehicles, ego = self._frame_to_drift(frame_vehicles[f])
            if ego is None:
                continue

            Q_total, _, _, occ_mask = compute_total_Q(
                vehicles, ego, self._X, self._Y)
            vx, vy, *_ = compute_velocity_field(
                vehicles, ego, self._X, self._Y)
            D = compute_diffusion_field(occ_mask, self._X, self._Y, vehicles, ego)

            R = self._pde_step(solver, vehicles, ego, dt_actual,
                               Q_total=Q_total, vx=vx, vy=vy, D=D)

            # Context features: agent count + distance-to-nearest-agent field
            N_agents = len(vehicles)
            if N_agents > 0:
                axy = np.array([[v['x'], v['y']] for v in vehicles],
                               dtype=np.float32)   # (N, 2)
                dx  = self._X[:, :, np.newaxis] - axy[:, 0]  # (ny, nx, N)
                dy  = self._Y[:, :, np.newaxis] - axy[:, 1]
                dist_nearest = np.sqrt(dx**2 + dy**2).min(axis=2).astype(np.float32)
            else:
                dist_nearest = np.full(self._X.shape, 1000.0, dtype=np.float32)

            self.snapshots.append({
                't'          : t_sim,
                'R'          : R.copy(),
                'Q'          : Q_total.copy(),
                'vx'         : vx.copy(),
                'vy'         : vy.copy(),
                'D'          : D.copy(),
                'occ_mask'   : occ_mask.copy(),
                'N_agents'   : float(N_agents),
                'dist_nearest': dist_nearest,
            })
            t_sim += dt_actual
            stored += 1

        print(f"[Phase1] Stored {stored} snapshots over "
              f"{t_sim:.1f}s (dt={dt_actual:.3f}s)")
        print(f"[Phase1] R stats — max={max(s['R'].max() for s in self.snapshots):.3f}, "
              f"mean={np.mean([s['R'].mean() for s in self.snapshots]):.4f}")
        return self.snapshots

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_csv(self):
        tf  = os.path.join(self.data_dir, f"{self.recording_id}_tracks.csv")
        tmf = os.path.join(self.data_dir, f"{self.recording_id}_tracksMeta.csv")
        rmf = os.path.join(self.data_dir, f"{self.recording_id}_recordingMeta.csv")
        return read_from_csv(tf, tmf, rmf)

    def _build_frame_lookup(self, tracks, tracks_meta):
        """Return dict: frame_id → list of (x,y,vx,vy,ax,ay,heading,vclass)."""
        meta_by_id = {m['trackId']: m for m in tracks_meta}
        lookup = {}
        for t in tracks:
            tid    = t['trackId']
            vclass = meta_by_id[tid].get('class', 'car').lower()
            if vclass not in ('car', 'truck', 'van', 'bus', 'motorcycle'):
                continue
            drift_class = 'truck' if vclass in ('truck', 'bus', 'van') else 'car'
            frames = t['frame']
            for i, f in enumerate(frames):
                f = int(f)
                entry = {
                    'trackId': tid,
                    'x'      : float(t['xCenter'][i]),
                    'y'      : float(t['yCenter'][i]),
                    'vx'     : float(t['xVelocity'][i]),
                    'vy'     : float(t['yVelocity'][i]),
                    'ax'     : float(t['xAcceleration'][i]),
                    'ay'     : float(t['yAcceleration'][i]),
                    'heading': float(t['heading'][i]) * np.pi / 180.0,
                    'class'  : drift_class,
                }
                lookup.setdefault(f, []).append(entry)
        return lookup

    def _compute_coord_offset(self, entries):
        """Store offset to map exiD UTM → cfg-grid local coords."""
        if not entries:
            self._ox = 0.0
            self._oy = 0.0
            return
        xs = [e['x'] for e in entries]
        ys = [e['y'] for e in entries]
        # Map median exiD position to cfg grid centre
        x_grid_mid = (cfg.x_min + cfg.x_max) / 2
        y_grid_mid = (cfg.y_min + cfg.y_max) / 2
        self._ox = np.median(xs) - x_grid_mid
        self._oy = np.median(ys) - y_grid_mid
        print(f"[Phase1] Coord offset: ox={self._ox:.1f}, oy={self._oy:.1f} m")

    PERCEPTION_RANGE = _PERCEPTION_RANGE_DEFAULT   # metres — vehicles beyond this are excluded from Q/field

    def _frame_to_drift(self, entries):
        """
        Convert one frame's entries to (vehicles_list, ego_dict).

        Only vehicles within PERCEPTION_RANGE of ego are included.  This
        prevents distant Gaussian blobs from creating spatial comb artefacts
        in Q that the PINN then tries to fit, causing oscillatory R fields.
        """
        # First pass: build all vehicles within grid + buffer
        raw = []
        for i, e in enumerate(entries):
            x  = e['x'] - self._ox
            y  = e['y'] - self._oy
            if not (cfg.x_min - 20 < x < cfg.x_max + 20 and
                    cfg.y_min - 20 < y < cfg.y_max + 20):
                continue
            vx = e['vx']
            vy = e['vy']
            v = drift_create_vehicle(vid=i+1, x=x, y=y, vx=vx, vy=vy,
                                     vclass=e['class'])
            v['heading'] = e['heading']
            v['a']       = np.sqrt(e['ax']**2 + e['ay']**2) * (
                -1 if e['ax'] * vx + e['ay'] * vy < 0 else 1)
            raw.append(v)

        if not raw:
            return raw, None

        # Select ego: first moving car-class vehicle
        ego = None
        for v in raw:
            if v['class'] == 'car' and np.hypot(v['vx'], v['vy']) > 1.0:
                ego = v
                break
        if ego is None:
            ego = raw[0]

        # Perception range filter — keep only vehicles within perception_range of ego.
        # Uses self.perception_range (instance), falling back to class default.
        # Set perception_range=float('inf') at construction to disable.
        perc = self.perception_range
        ex, ey = ego['x'], ego['y']
        vehicles = [v for v in raw
                    if np.hypot(v['x'] - ex, v['y'] - ey) <= perc]

        return vehicles, ego

    def _pde_step(self, solver, vehicles, ego, dt,
                  Q_total=None, vx=None, vy=None, D=None):
        if Q_total is None:
            Q_total, _, _, occ_mask = compute_total_Q(
                vehicles, ego, self._X, self._Y)
            vx, vy, *_ = compute_velocity_field(
                vehicles, ego, self._X, self._Y)
            D = compute_diffusion_field(occ_mask, self._X, self._Y, vehicles, ego)
        sub_dt = dt / self.substeps
        for _ in range(self.substeps):
            R = solver.step(Q_total, D, vx, vy, dt=sub_dt)
        return R


# ===========================================================================
# PHASE 2 — PINN MODEL & TRAINER
# ===========================================================================

class RandomFourierFeatures(nn.Module):
    """
    Fixed random projection: (N, D) → (N, 2*n_features) via [sin, cos] encoding.

    Addresses spectral bias — plain normalised inputs cause MLPs to learn low-
    frequency functions preferentially.  RFF lifts inputs into a richer basis
    so the network can represent sharp risk gradients near vehicles.

    The projection matrix B is fixed (not learned) and stored as a buffer.
    """

    def __init__(self, input_dim: int = 7, n_features: int = 64, scale: float = 10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, n_features) * scale)
        self.out_dim = 2 * n_features

    def forward(self, x):
        proj = x @ self.B                           # (N, n_features)
        return torch.cat([torch.sin(2 * np.pi * proj),
                          torch.cos(2 * np.pi * proj)], dim=-1)   # (N, 2*n_features)


class RiskFieldNet(nn.Module):
    """
    PINN:  (x̂, ŷ, t̂, Q̂[, N̂_agents, d̂_nearest,] v̂x, v̂y, D̂) → R ≥ 0

    Optional Random Fourier Feature (RFF) front-end maps the normalised
    input to a 2*rff_features-dim sinusoidal embedding before the MLP.
    Skip connection re-injects the embedding at the midpoint layer.

    use_context=True  → 9-dim input (adds N_agents, dist_nearest columns)
    use_rff=True      → RFF embedding front-end (fixes spectral bias)
    """

    def __init__(self, hidden: int = 128, depth: int = 6,
                 use_rff: bool = False, rff_features: int = 64, rff_scale: float = 10.0,
                 use_context: bool = False):
        super().__init__()
        base_dim = 9 if use_context else 7   # +N_agents, +dist_nearest when context enabled

        self.use_context = use_context

        self.use_rff     = use_rff
        self.skip_at     = depth // 2
        self.depth       = depth
        self.hidden      = hidden

        if use_rff:
            self.rff  = RandomFourierFeatures(base_dim, rff_features, rff_scale)
            in_dim    = self.rff.out_dim      # 2 * rff_features
        else:
            self.rff  = None
            in_dim    = base_dim

        self._skip_in_dim = in_dim   # what gets concatenated at the skip layer

        layers = []
        prev = in_dim
        for i in range(depth):
            n_in = prev + in_dim if i == self.skip_at else prev
            layers.append(nn.Linear(n_in, hidden))
            prev = hidden

        self.layers = nn.ModuleList(layers)
        self.out    = nn.Linear(hidden, 1)
        self.act    = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inp):
        """inp: (N, 7) normalised inputs."""
        if self.use_rff:
            enc = self.rff(inp)   # (N, 2*rff_features)
        else:
            enc = inp             # (N, 7)

        h = enc
        for i, layer in enumerate(self.layers):
            if i == self.skip_at:
                h = torch.cat([h, enc], dim=-1)
            h = self.act(layer(h))
        return torch.nn.functional.softplus(self.out(h))   # R ≥ 0


class FieldInterpolator:
    """
    Wraps a sequence of 2-D snapshots and provides interpolation to arbitrary
    (x, y, t) query points via a single 3-D RegularGridInterpolator per field.

    Memory: O(T × ny × nx × 5 fields) — suitable for small datasets (< ~500 snaps).
    For larger datasets use FlatSampleCache instead.
    """

    def __init__(self, snapshots, x_grid, y_grid):
        self.times  = np.array([s['t'] for s in snapshots])
        self.x_grid = x_grid
        self.y_grid = y_grid

        self._interp3d = {}
        for key in ('R', 'Q', 'vx', 'vy', 'D', 'dist_nearest'):
            if key not in snapshots[0]:
                continue
            arrs = np.stack([s[key] for s in snapshots], axis=0)  # (T, ny, nx)
            self._interp3d[key] = RegularGridInterpolator(
                (self.times, y_grid, x_grid), arrs,
                method='linear', bounds_error=False, fill_value=0.0)

        # N_agents is a scalar per snapshot — broadcast to 2-D for interpolation
        if 'N_agents' in snapshots[0]:
            ny, nx = len(y_grid), len(x_grid)
            Na_arrs = np.stack([
                np.full((ny, nx), float(s['N_agents'])) for s in snapshots
            ], axis=0).astype(np.float32)
            self._interp3d['N_agents'] = RegularGridInterpolator(
                (self.times, y_grid, x_grid), Na_arrs,
                method='linear', bounds_error=False, fill_value=0.0)

    def query(self, x_np, y_np, t_np):
        """Vectorised 3-D interpolation. Returns dict of (N,) float32 arrays."""
        t_c = np.clip(t_np, self.times[0], self.times[-1])
        pts = np.column_stack([t_c, y_np, x_np]).astype(np.float64)
        return {k: interp(pts).astype(np.float32)
                for k, interp in self._interp3d.items()}

    # --- compatibility shims so PINNTrainer can use either cache type ---
    def sample_data(self, n, rng=None):
        """Draw n random grid-aligned samples; returns dict of (n,) arrays."""
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, len(self.times), n)
        ny, nx = len(self.y_grid), len(self.x_grid)
        yi  = rng.integers(0, ny, n)
        xi  = rng.integers(0, nx, n)
        out = {
            'x'  : self.x_grid[xi].astype(np.float32),
            'y'  : self.y_grid[yi].astype(np.float32),
            't'  : self.times[idx].astype(np.float32),
        }
        for key, interp3d in self._interp3d.items():
            pts = np.column_stack([out['t'], out['y'], out['x']]).astype(np.float64)
            out[key] = interp3d(pts).astype(np.float32)
        return out

    def sample_colloc(self, n, rng=None):
        """Same as sample_data but returns only the physics input columns."""
        return self.sample_data(n, rng)


class FlatSampleCache:
    """
    Memory-efficient alternative to FieldInterpolator for large datasets.

    Pre-samples `pts_per_snap` random grid points from every snapshot at
    construction time and stores them as a flat float32 array of shape
    (N_total, 8) with columns [x, y, t, Q, vx, vy, D, R].

    Memory: N_snaps × pts_per_snap × 8 × 4 bytes
    Example: 4500 snaps × 100 pts = 14 MB (vs ~1.8 GB for 3-D interpolator)

    Query time: O(1) — random indexing into the pre-built table.
    """

    # Columns: x, y, t, Q, N_agents, dist_nearest, vx, vy, D, R
    KEYS = ('x', 'y', 't', 'Q', 'N_agents', 'dist_nearest', 'vx', 'vy', 'D', 'R')

    def __init__(self, snapshots, x_grid, y_grid,
                 pts_per_snap: int = 100, seed: int = 0):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.times  = np.array([s['t'] for s in snapshots], dtype=np.float32)

        ny, nx  = len(y_grid), len(x_grid)
        rng     = np.random.default_rng(seed)
        N       = len(snapshots) * pts_per_snap
        n_cols  = len(self.KEYS)

        buf = np.empty((N, n_cols), dtype=np.float32)
        row = 0
        for s in snapshots:
            yi = rng.integers(0, ny, pts_per_snap)
            xi = rng.integers(0, nx, pts_per_snap)
            buf[row:row + pts_per_snap, 0] = x_grid[xi]
            buf[row:row + pts_per_snap, 1] = y_grid[yi]
            buf[row:row + pts_per_snap, 2] = s['t']
            buf[row:row + pts_per_snap, 3] = s['Q'][yi, xi]
            buf[row:row + pts_per_snap, 4] = float(s.get('N_agents', 0))
            dn = s.get('dist_nearest', None)
            buf[row:row + pts_per_snap, 5] = (
                dn[yi, xi] if dn is not None else 1000.0)
            buf[row:row + pts_per_snap, 6] = s['vx'][yi, xi]
            buf[row:row + pts_per_snap, 7] = s['vy'][yi, xi]
            buf[row:row + pts_per_snap, 8] = s['D'][yi, xi]
            buf[row:row + pts_per_snap, 9] = s['R'][yi, xi]
            row += pts_per_snap

        self._buf = buf   # (N, 10)
        self._N   = N
        mb = N * n_cols * 4 / 1e6
        print(f"[FlatSampleCache] {N:,} samples from {len(snapshots)} snaps "
              f"({pts_per_snap} pts/snap, {n_cols} cols) — {mb:.1f} MB")

    def _draw(self, n, rng=None):
        idx = rng.integers(0, self._N, n) if rng is not None \
              else np.random.randint(0, self._N, n)
        return self._buf[idx]   # (n, 8)

    def sample_data(self, n, rng=None):
        """Draw n samples; returns dict with keys x,y,t,Q,vx,vy,D,R."""
        rows = self._draw(n, rng)
        return {k: rows[:, i] for i, k in enumerate(self.KEYS)}

    def sample_colloc(self, n, rng=None):
        """Same interface — physics collocation uses same columns."""
        return self.sample_data(n, rng)

    def save(self, path: str):
        np.save(path, self._buf)
        print(f"[FlatSampleCache] saved → {path}")

    @classmethod
    def load(cls, path: str, x_grid, y_grid, times):
        obj = object.__new__(cls)
        obj.x_grid = x_grid
        obj.y_grid = y_grid
        obj.times  = times
        obj._buf   = np.load(path)
        obj._N     = len(obj._buf)
        print(f"[FlatSampleCache] loaded ← {path}  ({obj._N:,} samples)")
        return obj


def build_cache(snapshots, x_grid, y_grid,
                pts_per_snap: int = 100, threshold: int = 400):
    """
    Auto-select cache type based on dataset size.

    ≤ threshold snapshots → FieldInterpolator (supports true arbitrary-t queries)
    >  threshold snapshots → FlatSampleCache  (memory-efficient, grid-aligned)
    """
    if len(snapshots) <= threshold:
        print(f"[cache] {len(snapshots)} snaps → FieldInterpolator (3-D)")
        return FieldInterpolator(snapshots, x_grid, y_grid)
    else:
        print(f"[cache] {len(snapshots)} snaps → FlatSampleCache "
              f"({pts_per_snap} pts/snap)")
        return FlatSampleCache(snapshots, x_grid, y_grid,
                               pts_per_snap=pts_per_snap)


class Normalizer:
    """
    Stores min/max for each channel and maps to/from [−1, 1].
    Channels: x, y, t, Q, vx, vy, D

    Stats are computed incrementally (one snapshot at a time) so no full
    field stack is held in memory — safe for large multi-recording datasets.
    """

    def __init__(self, snapshots, x_grid, y_grid):
        # Scalar accumulators — never allocate a full (T, ny, nx) array
        t_min = t_max = None
        Q_max = vx_min = vx_max = vy_min = vy_max = D_min = D_max = R_max = None
        Na_max = dn_max = None

        for s in snapshots:
            t     = float(s['t'])
            t_min = t if t_min is None else min(t_min, t)
            t_max = t if t_max is None else max(t_max, t)

            Q_max  = max(Q_max,  float(s['Q'].max()))  if Q_max  is not None else float(s['Q'].max())
            vx_min = min(vx_min, float(s['vx'].min())) if vx_min is not None else float(s['vx'].min())
            vx_max = max(vx_max, float(s['vx'].max())) if vx_max is not None else float(s['vx'].max())
            vy_min = min(vy_min, float(s['vy'].min())) if vy_min is not None else float(s['vy'].min())
            vy_max = max(vy_max, float(s['vy'].max())) if vy_max is not None else float(s['vy'].max())
            D_min  = min(D_min,  float(s['D'].min()))  if D_min  is not None else float(s['D'].min())
            D_max  = max(D_max,  float(s['D'].max()))  if D_max  is not None else float(s['D'].max())
            R_max  = max(R_max,  float(s['R'].max()))  if R_max  is not None else float(s['R'].max())

            # Context features (optional — present only in new snapshots)
            if 'N_agents' in s:
                v = float(s['N_agents'])
                Na_max = max(Na_max, v) if Na_max is not None else v
            if 'dist_nearest' in s:
                v = float(s['dist_nearest'].max())
                dn_max = max(dn_max, v) if dn_max is not None else v

        self.ranges = {
            'x'          : (float(x_grid.min()), float(x_grid.max())),
            'y'          : (float(y_grid.min()), float(y_grid.max())),
            't'          : (t_min,               max(t_max,   1e-3)),
            'Q'          : (0.0,                 max(Q_max,   1e-3)),
            'vx'         : (vx_min,              max(vx_max,  1e-3)),
            'vy'         : (vy_min,              max(vy_max,  1e-3)),
            'D'          : (D_min,               max(D_max,   1e-3)),
            'R'          : (0.0,                 max(R_max,   1e-3)),
            'N_agents'   : (0.0,                 max(Na_max or 1.0, 1.0)),
            'dist_nearest': (0.0,                max(dn_max or _PERCEPTION_RANGE_DEFAULT,
                                                     _PERCEPTION_RANGE_DEFAULT)),
        }
        # PDE physical parameters (not normalised, just stored)
        self.lambda_decay = cfg.lambda_decay
        self.tau          = cfg.tau

    def norm(self, vals, key):
        lo, hi = self.ranges[key]
        return 2.0 * (vals - lo) / max(hi - lo, 1e-8) - 1.0

    def denorm(self, vals, key):
        lo, hi = self.ranges[key]
        return (vals + 1.0) / 2.0 * (hi - lo) + lo

    def build_input(self, x, y, t, Q, vx, vy, D,
                    N_agents=None, dist_nearest=None, device='cpu'):
        """
        Assemble normalised tensor.

        Returns (N, 7) when N_agents/dist_nearest are None,
        or     (N, 9) when both context arrays are provided.
        Order: [x, y, t, Q, (N_agents, dist_nearest,) vx, vy, D]
        """
        def _t(arr, key):
            return torch.tensor(
                self.norm(np.asarray(arr, dtype=np.float32), key),
                dtype=torch.float32)

        cols = [_t(x, 'x'), _t(y, 'y'), _t(t, 't'), _t(Q, 'Q')]
        if N_agents is not None and dist_nearest is not None:
            # Broadcast scalar N_agents to the same length as x if needed
            Na = (np.full_like(np.asarray(x, dtype=np.float32), float(N_agents))
                  if np.ndim(N_agents) == 0 else np.asarray(N_agents, dtype=np.float32))
            cols.append(_t(Na,                              'N_agents'))
            cols.append(_t(np.asarray(dist_nearest, dtype=np.float32), 'dist_nearest'))
        cols += [_t(vx, 'vx'), _t(vy, 'vy'), _t(D, 'D')]
        return torch.stack(cols, dim=-1).to(device)


class PINNTrainer:
    """
    Train RiskFieldNet on:
      - L_data : MSE between R_θ and numerical solver R at snapshot grid points
      - L_phys : PDE residual at random collocation points
      - L_ic   : R(x,y,0) = 0 (initial condition)
      - L_bc   : R = 0 at grid boundary rows/cols

    Total loss = w_data*L_data + w_phys*L_phys + w_ic*L_ic + w_bc*L_bc
    """

    def __init__(self, snapshots, norm: Normalizer,
                 interp,          # FieldInterpolator or FlatSampleCache
                 hidden=128, depth=6,
                 use_rff=False, rff_features=64, rff_scale=10.0,
                 use_context=False,
                 device='cpu',
                 w_data=1.0, w_phys=1.0, w_ic=0.5, w_bc=0.2, w_smooth=0.0,
                 n_colloc=2048, n_data=1024):

        self.norm        = norm
        self.cache       = interp
        self.interp      = interp
        self.device      = torch.device(device)
        self.snaps       = snapshots
        self.use_context = use_context

        self.w_data   = w_data
        self.w_phys   = w_phys
        self.w_ic     = w_ic
        self.w_bc     = w_bc
        self.w_smooth = w_smooth
        self.n_co     = n_colloc
        self.n_da     = n_data

        self.model   = RiskFieldNet(
            hidden=hidden, depth=depth,
            use_rff=use_rff, rff_features=rff_features, rff_scale=rff_scale,
            use_context=use_context,
        ).to(self.device)
        self.history = {'loss': [], 'L_data': [], 'L_phys': [],
                        'L_ic': [], 'L_bc': [], 'L_smooth': []}

        # (data sampled dynamically each epoch — no cache needed)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, epochs=2000, lr=1e-3, print_every=200):
        opt  = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        t0   = time.time()

        rff_tag = (f"RFF(feat={self.model.rff.B.shape[1]}, "
                   f"scale={self.model.rff.B.std().item()/1.0:.1f})"
                   if self.model.use_rff else "no-RFF")
        print(f"\n[Phase2] Training PINN for {epochs} epochs on {self.device} ...")
        print(f"         arch={self.model.hidden}×{self.model.depth}  {rff_tag}")
        print(f"         n_data={self.n_da}, n_colloc={self.n_co}, "
              f"weights=(data={self.w_data}, phys={self.w_phys}, "
              f"ic={self.w_ic}, bc={self.w_bc}, smooth={self.w_smooth})")

        for ep in range(1, epochs + 1):
            self.model.train()
            opt.zero_grad()

            L_data   = self._data_loss()
            L_phys   = self._physics_loss()
            L_ic     = self._ic_loss()
            L_bc     = self._bc_loss()
            L_smooth = self._smooth_loss() if self.w_smooth > 0.0 else torch.tensor(0.0, device=self.device)

            loss = (self.w_data   * L_data  + self.w_phys * L_phys +
                    self.w_ic     * L_ic    + self.w_bc   * L_bc   +
                    self.w_smooth * L_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()

            for key, val in zip(
                    ('loss', 'L_data', 'L_phys', 'L_ic', 'L_bc', 'L_smooth'),
                    (loss, L_data, L_phys, L_ic, L_bc, L_smooth)):
                self.history[key].append(val.item())

            if ep % print_every == 0 or ep == 1:
                elapsed = time.time() - t0
                smooth_str = (f"  smooth={L_smooth.item():.4e}"
                              if self.w_smooth > 0.0 else "")
                print(f"  ep {ep:5d}/{epochs}  loss={loss.item():.4e}  "
                      f"data={L_data.item():.4e}  phys={L_phys.item():.4e}  "
                      f"ic={L_ic.item():.4e}  bc={L_bc.item():.4e}"
                      f"{smooth_str}  t={elapsed:.0f}s")

        print(f"[Phase2] Training done. Final loss = {self.history['loss'][-1]:.4e}")
        return self.history

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _build_input_from_samp(self, samp):
        """Build normalised input tensor from a cache sample dict.
        Includes context features (N_agents, dist_nearest) when use_context=True."""
        if self.use_context and 'N_agents' in samp:
            return self.norm.build_input(
                samp['x'], samp['y'], samp['t'],
                samp['Q'], samp['vx'], samp['vy'], samp['D'],
                N_agents=samp['N_agents'], dist_nearest=samp['dist_nearest'],
                device=str(self.device))
        return self.norm.build_input(
            samp['x'], samp['y'], samp['t'],
            samp['Q'], samp['vx'], samp['vy'], samp['D'],
            device=str(self.device))

    def _data_loss(self):
        """MSE between PINN output and numerical R, using cache.sample_data()."""
        samp   = self.cache.sample_data(self.n_da)
        inp    = self._build_input_from_samp(samp)
        R_pred = self.model(inp).squeeze(-1)
        R_true = torch.tensor(samp['R'], dtype=torch.float32, device=self.device)

        R_scale = self.norm.ranges['R'][1]
        return torch.mean((R_pred - R_true / max(R_scale, 1e-3))**2)

    def _physics_loss(self):
        """
        Compute PDE residual at random (x,y,t) collocation points.

        Residual = τ R_tt + R_t + ∇·(v R) − ∇·(D∇R) + λR − Q = 0

        For efficiency we use forward-mode AD (torch.autograd.grad with
        create_graph=True) on the 7-input network.
        """
        # Sample collocation points from the cache
        # (grid-aligned for FlatSampleCache; arbitrary-t for FieldInterpolator)
        samp  = self.cache.sample_colloc(self.n_co)
        x_np  = samp['x']
        y_np  = samp['y']
        t_np  = samp['t']
        Q_np  = samp['Q']
        vx_np = samp['vx']
        vy_np = samp['vy']
        D_np  = samp['D']

        # Build normalised input WITH requires_grad on raw coords
        x_t  = torch.tensor(x_np,  requires_grad=True, device=self.device)
        y_t  = torch.tensor(y_np,  requires_grad=True, device=self.device)
        t_t  = torch.tensor(t_np,  requires_grad=True, device=self.device)

        # Normalise (differentiable linear ops)
        def _norm(v, key):
            lo, hi = self.norm.ranges[key]
            return 2.0 * (v - lo) / max(hi - lo, 1e-8) - 1.0

        xn = _norm(x_t, 'x')
        yn = _norm(y_t, 'y')
        tn = _norm(t_t, 't')

        Q_t  = torch.tensor(Q_np,  dtype=torch.float32, device=self.device)
        vx_t = torch.tensor(vx_np, dtype=torch.float32, device=self.device)
        vy_t = torch.tensor(vy_np, dtype=torch.float32, device=self.device)
        D_t  = torch.tensor(D_np,  dtype=torch.float32, device=self.device)

        Qn  = _norm(Q_t,  'Q')
        vxn = _norm(vx_t, 'vx')
        vyn = _norm(vy_t, 'vy')
        Dn  = _norm(D_t,  'D')

        if self.use_context and 'N_agents' in samp:
            Na_t = torch.tensor(samp['N_agents'], dtype=torch.float32, device=self.device)
            dn_t = torch.tensor(samp['dist_nearest'], dtype=torch.float32, device=self.device)
            Nan  = _norm(Na_t, 'N_agents')
            dnn  = _norm(dn_t, 'dist_nearest')
            inp  = torch.stack([xn, yn, tn, Qn, Nan, dnn, vxn, vyn, Dn], dim=-1)
        else:
            inp  = torch.stack([xn, yn, tn, Qn, vxn, vyn, Dn], dim=-1)
        R    = self.model(inp).squeeze(-1)

        # First derivatives via autograd
        ones = torch.ones_like(R)
        R_x, = torch.autograd.grad(R, x_t, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)
        R_y, = torch.autograd.grad(R, y_t, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)
        R_t, = torch.autograd.grad(R, t_t, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)

        # Second time derivative
        R_tt, = torch.autograd.grad(R_t, t_t, grad_outputs=ones,
                                     create_graph=True, retain_graph=True)

        # Second spatial derivatives for Laplacian
        R_xx, = torch.autograd.grad(R_x, x_t, grad_outputs=ones,
                                     create_graph=True, retain_graph=True)
        R_yy, = torch.autograd.grad(R_y, y_t, grad_outputs=ones,
                                     create_graph=True, retain_graph=True)

        # Spatial derivatives of D (treated as constant at each point — no grad)
        # Approximate ∇D numerically from precomputed field
        # Use D_t directly (constant at collocation points)
        D_val = D_t.detach()

        # PDE terms (dimensional, not normalised — we convert back)
        # Note: network output R is in [0,1] normalised scale;
        #       gradients with respect to raw x,y,t are in that scale / metres or seconds
        # We compute residual in network-output space / raw-coord space
        lam  = self.norm.lambda_decay
        tau  = self.norm.tau

        # ∇·(D∇R) ≈ D*(R_xx + R_yy)  (∇D·∇R neglected at collocation level)
        diffusion = D_val * (R_xx + R_yy)

        # ∇·(vR) = vx*R_x + vy*R_y  (incompressible approximation)
        advection = vx_t * R_x + vy_t * R_y

        # Normalise Q to network output scale
        Q_net = Q_t / max(self.norm.ranges['R'][1], 1e-3)
        lam_R = lam * R

        # Telegrapher residual
        if tau > 0:
            residual = tau * R_tt + R_t + advection - diffusion + lam_R - Q_net
        else:
            residual = R_t + advection - diffusion + lam_R - Q_net

        return torch.mean(residual**2)

    def _context_zeros(self, n):
        """Return (Na, dn) default arrays for IC/BC/smooth losses (no agents present)."""
        if not self.use_context:
            return None, None
        return (np.zeros(n, dtype=np.float32),
                np.full(n, _PERCEPTION_RANGE_DEFAULT, dtype=np.float32))

    def _ic_loss(self):
        """R(x, y, t=0) = 0 (zero initial risk field)."""
        n    = max(256, self.n_co // 4)
        x_np = np.random.uniform(cfg.x_min, cfg.x_max, n).astype(np.float32)
        y_np = np.random.uniform(cfg.y_min, cfg.y_max, n).astype(np.float32)
        t_np = np.zeros(n, dtype=np.float32)
        Q_np = np.zeros(n, dtype=np.float32)
        vx_np= np.zeros(n, dtype=np.float32)
        vy_np= np.zeros(n, dtype=np.float32)
        D_np = np.full(n, cfg.D0, dtype=np.float32)
        Na, dn = self._context_zeros(n)

        inp  = self.norm.build_input(x_np, y_np, t_np, Q_np, vx_np, vy_np, D_np,
                                     N_agents=Na, dist_nearest=dn,
                                     device=str(self.device))
        R    = self.model(inp).squeeze(-1)
        return torch.mean(R**2)

    def _bc_loss(self):
        """R = 0 at the four grid edges."""
        n    = max(128, self.n_co // 8)
        t_np = np.random.uniform(self.interp.times[0], self.interp.times[-1], n
                                  ).astype(np.float32)
        Q_np = np.zeros(n, dtype=np.float32)
        vx_np= np.zeros(n, dtype=np.float32)
        vy_np= np.zeros(n, dtype=np.float32)
        D_np = np.full(n, cfg.D0, dtype=np.float32)
        Na, dn = self._context_zeros(n)

        # Left edge (x = x_min)
        xl = np.full(n, cfg.x_min, dtype=np.float32)
        yl = np.random.uniform(cfg.y_min, cfg.y_max, n).astype(np.float32)
        inp_l = self.norm.build_input(xl, yl, t_np, Q_np, vx_np, vy_np, D_np,
                                      N_agents=Na, dist_nearest=dn,
                                      device=str(self.device))
        R_l = self.model(inp_l).squeeze(-1)

        # Right edge (x = x_max)
        xr = np.full(n, cfg.x_max, dtype=np.float32)
        inp_r = self.norm.build_input(xr, yl, t_np, Q_np, vx_np, vy_np, D_np,
                                      N_agents=Na, dist_nearest=dn,
                                      device=str(self.device))
        R_r = self.model(inp_r).squeeze(-1)

        return torch.mean(R_l**2) + torch.mean(R_r**2)

    def _smooth_loss(self):
        """
        Laplacian smoothness penalty: penalise ||∇²R||² = (R_xx + R_yy)².

        Motivation: the diffusion term D∇²R in the PDE naturally wants smooth
        fields, but the data loss alone can drive the network to produce
        sharp local peaks at vehicle positions.  Penalising the spatial
        Laplacian directly suppresses those oscillations.

        Uses autograd (same approach as _physics_loss) on a small batch of
        random collocation points — no extra forward passes needed.
        """
        n = max(128, self.n_co // 8)
        x_np  = np.random.uniform(cfg.x_min, cfg.x_max, n).astype(np.float32)
        y_np  = np.random.uniform(cfg.y_min, cfg.y_max, n).astype(np.float32)
        t_np  = np.random.uniform(self.interp.times[0], self.interp.times[-1],
                                   n).astype(np.float32)
        Q_np  = np.zeros(n, dtype=np.float32)
        vx_np = np.zeros(n, dtype=np.float32)
        vy_np = np.zeros(n, dtype=np.float32)
        D_np  = np.full(n, cfg.D0, dtype=np.float32)

        x_t = torch.tensor(x_np, requires_grad=True, device=self.device)
        y_t = torch.tensor(y_np, requires_grad=True, device=self.device)
        t_t = torch.tensor(t_np,  device=self.device)

        def _norm(v, key):
            lo, hi = self.norm.ranges[key]
            return 2.0 * (v - lo) / max(hi - lo, 1e-8) - 1.0

        cols = [
            _norm(x_t, 'x'), _norm(y_t, 'y'), _norm(t_t, 't'),
            torch.tensor(_norm(Q_np,  'Q'),  device=self.device),
        ]
        if self.use_context:
            Na_np = np.zeros(n, dtype=np.float32)
            dn_np = np.full(n, _PERCEPTION_RANGE_DEFAULT, dtype=np.float32)
            cols.append(torch.tensor(_norm(Na_np, 'N_agents'),    device=self.device))
            cols.append(torch.tensor(_norm(dn_np, 'dist_nearest'),device=self.device))
        cols += [
            torch.tensor(_norm(vx_np, 'vx'), device=self.device),
            torch.tensor(_norm(vy_np, 'vy'), device=self.device),
            torch.tensor(_norm(D_np,  'D'),  device=self.device),
        ]
        inp = torch.stack(cols, dim=-1)

        R = self.model(inp).squeeze(-1)
        ones = torch.ones_like(R)

        R_x,  = torch.autograd.grad(R, x_t, grad_outputs=ones,
                                     create_graph=True, retain_graph=True)
        R_xx, = torch.autograd.grad(R_x, x_t, grad_outputs=ones,
                                     create_graph=True, retain_graph=True)
        R_y,  = torch.autograd.grad(R, y_t, grad_outputs=ones,
                                     create_graph=True, retain_graph=True)
        R_yy, = torch.autograd.grad(R_y, y_t, grad_outputs=ones,
                                     create_graph=True)

        laplacian = R_xx + R_yy
        return torch.mean(laplacian ** 2)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_field(self, snap_idx: int):
        """
        Predict full risk field for snapshot snap_idx.
        Returns ndarray (ny, nx).
        """
        snap  = self.snaps[snap_idx]
        ny, nx = snap['R'].shape
        t_val = snap['t']

        yy, xx = np.meshgrid(self.interp.y_grid, self.interp.x_grid, indexing='ij')
        x_flat  = xx.ravel().astype(np.float32)
        y_flat  = yy.ravel().astype(np.float32)
        t_flat  = np.full_like(x_flat, t_val)
        Q_flat  = snap['Q'].ravel().astype(np.float32)
        vx_flat = snap['vx'].ravel().astype(np.float32)
        vy_flat = snap['vy'].ravel().astype(np.float32)
        D_flat  = snap['D'].ravel().astype(np.float32)

        Na_flat = dn_flat = None
        if self.use_context:
            Na_flat = np.full_like(x_flat, float(snap.get('N_agents', 0)))
            dn      = snap.get('dist_nearest', None)
            dn_flat = (dn.ravel().astype(np.float32)
                       if dn is not None
                       else np.full_like(x_flat, _PERCEPTION_RANGE_DEFAULT))

        inp = self.norm.build_input(
            x_flat, y_flat, t_flat, Q_flat, vx_flat, vy_flat, D_flat,
            N_agents=Na_flat, dist_nearest=dn_flat,
            device=str(self.device))

        self.model.eval()
        with torch.no_grad():
            R_norm = self.model(inp).squeeze(-1).cpu().numpy()

        R_scale = self.norm.ranges['R'][1]
        return (R_norm * R_scale).reshape(ny, nx)

    def predict_field_from_arrays(self, X, Y, t_val, Q, vx, vy, D,
                                  N_agents=None, dist_nearest=None):
        """
        Predict risk field at arbitrary grid arrays.

        Parameters
        ----------
        X, Y   : 2-D ndarray (ny, nx) — grid coordinates in PINN training space
        t_val  : float — simulation time
        Q, vx, vy, D : 2-D ndarray (ny, nx) — source / transport fields
        N_agents     : int or None — agent count (used when model was trained with
                       --use_context; ignored otherwise)
        dist_nearest : 2-D ndarray (ny, nx) or None — distance-to-nearest-agent field

        Returns
        -------
        R_pred : 2-D ndarray (ny, nx)
        """
        ny, nx  = X.shape
        x_flat  = X.ravel().astype(np.float32)
        y_flat  = Y.ravel().astype(np.float32)
        t_flat  = np.full(x_flat.shape, t_val, dtype=np.float32)
        Q_flat  = Q.ravel().astype(np.float32)
        vx_flat = vx.ravel().astype(np.float32)
        vy_flat = vy.ravel().astype(np.float32)
        D_flat  = D.ravel().astype(np.float32)

        Na_flat = dn_flat = None
        if self.use_context:
            Na_flat = (np.full_like(x_flat, float(N_agents))
                       if N_agents is not None else np.zeros_like(x_flat))
            dn_flat = (dist_nearest.ravel().astype(np.float32)
                       if dist_nearest is not None
                       else np.full_like(x_flat, _PERCEPTION_RANGE_DEFAULT))

        inp = self.norm.build_input(
            x_flat, y_flat, t_flat, Q_flat, vx_flat, vy_flat, D_flat,
            N_agents=Na_flat, dist_nearest=dn_flat,
            device=str(self.device))

        self.model.eval()
        with torch.no_grad():
            R_norm = self.model(inp).squeeze(-1).cpu().numpy()

        R_scale = self.norm.ranges['R'][1]
        return (R_norm * R_scale).reshape(ny, nx)

    def save(self, path: str):
        torch.save({
            'model_state' : self.model.state_dict(),
            'history'     : self.history,
            'norm_ranges' : self.norm.ranges,
            'hidden'      : self.model.hidden,
            'depth'       : self.model.depth,
            'use_rff'     : self.model.use_rff,
            'rff_features': (self.model.rff.B.shape[1]
                             if self.model.use_rff else 64),
            'rff_scale'   : (float(self.model.rff.B.std())
                             if self.model.use_rff else 10.0),
            'use_context'      : self.model.use_context,
            'perception_range' : ExiDLoader.PERCEPTION_RANGE,
        }, path)
        print(f"[Phase2] Model saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        if 'history' in ckpt:
            self.history = ckpt['history']
        print(f"[Phase2] Model loaded ← {path}")


# ===========================================================================
# PHASE 3 — VALIDATION
# ===========================================================================

class PINNValidator:
    """
    Compare PINN predictions against numerical PDESolver on held-out snapshots.

    Metrics
    -------
    - Pointwise L2 error
    - Relative L2 error (||R_pred - R_num||_2 / ||R_num||_2)
    - Max absolute error
    - Pearson correlation
    - Inference time ratio (PINN vs PDE step)
    """

    def __init__(self, trainer: PINNTrainer, test_snaps: list):
        self.trainer     = trainer
        self.test_snaps  = test_snaps

    def run(self, plot: bool = True, save_dir: str = "pinn_validation") -> dict:
        os.makedirs(save_dir, exist_ok=True)
        metrics = {
            'l2_abs': [], 'l2_rel': [], 'max_err': [], 'corr': [],
            'pinn_time_ms': [], 'pde_step_time_ms': [],
        }

        print(f"\n[Phase3] Validating on {len(self.test_snaps)} held-out snapshots ...")

        for i, snap in enumerate(self.test_snaps):
            R_num = snap['R']   # ground-truth from numerical solver

            # PINN prediction
            t0 = time.time()
            snap_idx = self.trainer.snaps.index(snap) if snap in self.trainer.snaps else 0
            R_pinn = self.trainer.predict_field(snap_idx)
            pinn_ms = (time.time() - t0) * 1000

            # Numerical solver step time (benchmark)
            pde_solver_tmp = PDESolver()
            pde_solver_tmp.R = snap['R'].copy()
            t0 = time.time()
            _ = pde_solver_tmp.step(
                snap['Q'], snap['D'], snap['vx'], snap['vy'], dt=0.1)
            pde_ms = (time.time() - t0) * 1000

            # Metrics
            diff   = R_pinn - R_num
            l2_abs = float(np.sqrt(np.mean(diff**2)))
            l2_rel = float(np.linalg.norm(diff) /
                           (np.linalg.norm(R_num) + 1e-8))
            max_err = float(np.max(np.abs(diff)))
            corr   = float(np.corrcoef(R_pinn.ravel(), R_num.ravel())[0, 1])

            metrics['l2_abs'].append(l2_abs)
            metrics['l2_rel'].append(l2_rel)
            metrics['max_err'].append(max_err)
            metrics['corr'].append(corr)
            metrics['pinn_time_ms'].append(pinn_ms)
            metrics['pde_step_time_ms'].append(pde_ms)

            if i % max(1, len(self.test_snaps)//5) == 0:
                print(f"  snap {i:3d}  t={snap['t']:.1f}s  "
                      f"L2_rel={l2_rel:.3f}  corr={corr:.3f}  "
                      f"PINN={pinn_ms:.1f}ms  PDE={pde_ms:.1f}ms")

            if plot and i < 6:
                self._plot_comparison(snap, R_num, R_pinn, i, save_dir)

        # Summary
        def _s(arr): return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"
        print("\n[Phase3] ======= VALIDATION SUMMARY =======")
        print(f"  L2 absolute error  : {_s(metrics['l2_abs'])}")
        print(f"  L2 relative error  : {_s(metrics['l2_rel'])}")
        print(f"  Max absolute error : {_s(metrics['max_err'])}")
        print(f"  Pearson correlation: {_s(metrics['corr'])}")
        print(f"  PINN inference     : {_s(metrics['pinn_time_ms'])} ms")
        print(f"  PDE solver step    : {_s(metrics['pde_step_time_ms'])} ms")
        print(f"  Speedup (approx)   : "
              f"{np.mean(metrics['pde_step_time_ms'])/max(np.mean(metrics['pinn_time_ms']),1e-3):.1f}x")

        if plot:
            self._plot_loss_curve(save_dir)
            self._plot_error_over_time(metrics, save_dir)

        return metrics

    # ------------------------------------------------------------------
    # SciencePlots styling helper
    # ------------------------------------------------------------------

    @staticmethod
    def _sci_style():
        """Return the SciencePlots style context to use for all plots."""
        import matplotlib.pyplot as plt
        # 'science' base + 'bright' colour palette; no-latex fallback if needed
        try:
            return plt.style.context(['science', 'bright'])
        except Exception:
            return plt.style.context(['science'])

    # ------------------------------------------------------------------
    # Plot 1 — Side-by-side risk-field comparison
    # ------------------------------------------------------------------

    def _plot_comparison(self, snap, R_num, R_pinn, idx, save_dir):
        try:
            x_km = self.trainer.interp.x_grid / 1e3 if \
                self.trainer.interp.x_grid.max() > 500 else self.trainer.interp.x_grid
            y_km = self.trainer.interp.y_grid / 1e3 if \
                self.trainer.interp.y_grid.max() > 500 else self.trainer.interp.y_grid
            extent = [x_km[0], x_km[-1], y_km[0], y_km[-1]]
            unit   = 'km' if x_km.max() > 500 else 'm'

            diff = R_pinn - R_num
            vmax = max(float(R_num.max()), float(R_pinn.max()), 0.1)
            emax = float(np.abs(diff).max())

            with plt.style.context(['science', 'bright']):
                fig = plt.figure(figsize=(13, 3.6))
                gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

                # ---- panel 1: Numerical PDE ----
                ax0 = fig.add_subplot(gs[0])
                im0 = ax0.imshow(R_num, origin='lower', aspect='auto',
                                 extent=extent, vmin=0, vmax=vmax, cmap='inferno')
                ax0.set_title(r'Numerical PDE  ($t={:.1f}$ s)'.format(snap['t']))
                ax0.set_xlabel(f'$x$ [{unit}]')
                ax0.set_ylabel(f'$y$ [{unit}]')
                div0 = make_axes_locatable(ax0)
                cb0  = plt.colorbar(im0, cax=div0.append_axes('right', size='5%', pad=0.05))
                cb0.set_label(r'Risk $\mathcal{R}$')

                # ---- panel 2: PINN ----
                ax1 = fig.add_subplot(gs[1])
                im1 = ax1.imshow(R_pinn, origin='lower', aspect='auto',
                                 extent=extent, vmin=0, vmax=vmax, cmap='inferno')
                ax1.set_title(r'PINN Surrogate')
                ax1.set_xlabel(f'$x$ [{unit}]')
                ax1.set_ylabel(f'$y$ [{unit}]')
                div1 = make_axes_locatable(ax1)
                cb1  = plt.colorbar(im1, cax=div1.append_axes('right', size='5%', pad=0.05))
                cb1.set_label(r'Risk $\mathcal{R}$')

                # ---- panel 3: Absolute error ----
                ax2 = fig.add_subplot(gs[2])
                im2 = ax2.imshow(np.abs(diff), origin='lower', aspect='auto',
                                 extent=extent, vmin=0, vmax=emax, cmap='hot')
                ax2.set_title(r'$|\hat{\mathcal{R}} - \mathcal{R}|$'
                              f'  (max={emax:.2f})')
                ax2.set_xlabel(f'$x$ [{unit}]')
                ax2.set_ylabel(f'$y$ [{unit}]')
                div2 = make_axes_locatable(ax2)
                cb2  = plt.colorbar(im2, cax=div2.append_axes('right', size='5%', pad=0.05))
                cb2.set_label('Abs. Error')

                fig.suptitle(
                    r'PINN vs. Numerical Solver — exiD Recording'
                    f'  |  snap {idx}',
                    y=1.01, fontsize=9)
                plt.savefig(os.path.join(save_dir, f"comparison_{idx:03d}.pdf"),
                            dpi=150, bbox_inches='tight')
                plt.savefig(os.path.join(save_dir, f"comparison_{idx:03d}.png"),
                            dpi=150, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"  [plot] skipped: {e}")

    # ------------------------------------------------------------------
    # Plot 2 — Training loss curves
    # ------------------------------------------------------------------

    def _plot_loss_curve(self, save_dir):
        try:
            h      = self.trainer.history
            if not h.get('loss'):
                print("  [loss plot] no training history — skipping loss curve")
                return
            epochs = np.arange(1, len(h['loss']) + 1)

            with plt.style.context(['science', 'bright']):
                fig, ax = plt.subplots(figsize=(6, 3.5))

                ax.semilogy(epochs, h['loss'],   lw=1.8, label=r'$\mathcal{L}_\mathrm{total}$')
                ax.semilogy(epochs, h['L_data'], lw=1.2, ls='--',
                            label=r'$\mathcal{L}_\mathrm{data}$')
                ax.semilogy(epochs, h['L_phys'], lw=1.2, ls='-.',
                            label=r'$\mathcal{L}_\mathrm{phys}$')
                ax.semilogy(epochs, h['L_ic'],   lw=1.0, ls=':',
                            label=r'$\mathcal{L}_\mathrm{IC}$')
                ax.semilogy(epochs, h['L_bc'],   lw=1.0, ls=(0, (3, 1, 1, 1)),
                            label=r'$\mathcal{L}_\mathrm{BC}$')

                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(r'PINN Training Loss — $\tau\partial_{tt}R + \partial_t R'
                             r'+ \nabla\cdot(vR) = \nabla\cdot(D\nabla R) + Q - \lambda R$')
                ax.legend(frameon=True, fontsize=7, loc='upper right')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "loss_curve.pdf"),
                            dpi=150, bbox_inches='tight')
                plt.savefig(os.path.join(save_dir, "loss_curve.png"),
                            dpi=150, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"  [loss plot] skipped: {e}")

    # ------------------------------------------------------------------
    # Plot 3 — Validation metrics over time
    # ------------------------------------------------------------------

    def _plot_error_over_time(self, metrics, save_dir):
        try:
            times    = np.array([s['t'] for s in self.test_snaps])
            l2_rel   = np.array(metrics['l2_rel'])
            corr     = np.array(metrics['corr'])
            max_err  = np.array(metrics['max_err'])

            with plt.style.context(['science', 'bright']):
                fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

                # L2 relative error
                axes[0].plot(times, l2_rel, lw=1.4, marker='o', ms=2.5)
                axes[0].axhline(l2_rel.mean(), ls='--', lw=0.9, color='grey',
                                label=f'mean = {l2_rel.mean():.3f}')
                axes[0].set_ylabel(r'$\|\hat{R}-R\|_2/\|R\|_2$')
                axes[0].legend(fontsize=7)

                # Pearson correlation
                axes[1].plot(times, corr, lw=1.4, marker='s', ms=2.5, color='C1')
                axes[1].axhline(corr.mean(), ls='--', lw=0.9, color='grey',
                                label=f'mean = {corr.mean():.3f}')
                axes[1].set_ylabel(r'Pearson $\rho$')
                axes[1].set_ylim(max(0, corr.min() - 0.05), 1.02)
                axes[1].legend(fontsize=7)

                # Max absolute error
                axes[2].plot(times, max_err, lw=1.4, marker='^', ms=2.5, color='C2')
                axes[2].axhline(max_err.mean(), ls='--', lw=0.9, color='grey',
                                label=f'mean = {max_err.mean():.2f}')
                axes[2].set_ylabel(r'$\max|\hat{R}-R|$')
                axes[2].set_xlabel(r'Time $t$ [s]')
                axes[2].legend(fontsize=7)

                fig.suptitle(r'PINN Surrogate vs.\ Numerical Solver (held-out)',
                             fontsize=10)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "error_over_time.pdf"),
                            dpi=150, bbox_inches='tight')
                plt.savefig(os.path.join(save_dir, "error_over_time.png"),
                            dpi=150, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"  [error plot] skipped: {e}")


# ===========================================================================
# DATASET HELPERS
# ===========================================================================

KNOWN_DATASETS = ('exiD', 'inD', 'rounD', 'uniD')


def parse_datasets(dataset_arg: str) -> list:
    """
    Resolve --dataset argument to a list of dataset names.

    Accepted forms:
      all             → all four known datasets
      inD             → single dataset
      inD,rounD       → comma-separated list
    """
    s = dataset_arg.strip()
    if s.lower() == 'all':
        return list(KNOWN_DATASETS)
    parts = [p.strip() for p in s.split(',') if p.strip()]
    for p in parts:
        if p not in KNOWN_DATASETS:
            raise ValueError(f"Unknown dataset {p!r}. Known: {KNOWN_DATASETS}")
    return parts


def load_multi_dataset(dataset_names: list, data_root: str,
                        recording_arg: str, max_sec: float,
                        warmup_sec: float,
                        perception_range: float = _PERCEPTION_RANGE_DEFAULT) -> tuple:
    """
    Load recordings from one or more datasets, concatenate all snapshots,
    and return (all_snapshots, last_loader).

    When multiple datasets are given, recording_arg is applied to each
    dataset independently ("all" means all recordings of that dataset).
    """
    all_snaps   = []
    t_offset    = 0.0
    last_loader = None

    for ds in dataset_names:
        data_dir = os.path.join(data_root, ds)
        if not os.path.isdir(data_dir):
            print(f"  [skip] dataset {ds}: directory not found at {data_dir}")
            continue
        try:
            rec_ids = parse_recording_ids(recording_arg, data_dir)
        except FileNotFoundError as e:
            print(f"  [skip] dataset {ds}: {e}")
            continue

        print(f"\n[Phase1] ══ Dataset: {ds}  ({len(rec_ids)} recordings) ══")
        snaps, loader = load_all_recordings(
            recording_ids=rec_ids,
            data_dir=data_dir,
            max_sec=max_sec,
            warmup_sec=warmup_sec,
            perception_range=perception_range,
        )
        # Re-stamp t so snapshots from different datasets don't overlap
        for s in snaps:
            s['t'] = s['t'] + t_offset
        if snaps:
            t_offset += snaps[-1]['t'] - snaps[0]['t'] + (
                snaps[1]['t'] - snaps[0]['t'] if len(snaps) > 1 else 0.1)
        all_snaps  += snaps
        last_loader = loader

    if not all_snaps:
        raise RuntimeError("No snapshots loaded from any dataset.")

    print(f"\n[Phase1] Combined total: {len(all_snaps)} snapshots "
          f"from {len(dataset_names)} dataset(s)  ({t_offset:.1f}s)")
    return all_snaps, last_loader


def smooth_Q_temporal(snapshots: list, cutoff_hz: float = 2.0) -> list:
    """
    Low-pass filter the Q field along the time axis to remove high-frequency
    comb artefacts caused by overlapping Gaussian vehicle kernels.

    Applies a 3rd-order Butterworth zero-phase filter (filtfilt) to every
    grid point independently along the snapshot sequence.  Keeps Q ≥ 0.

    Parameters
    ----------
    snapshots  : list of snapshot dicts (must have 't' and 'Q' keys)
    cutoff_hz  : low-pass cutoff frequency in Hz (default 2.0 Hz)
                 True physics (diffusion) operates at <1 Hz; vehicles move
                 at ~10 Hz (dt=0.1 s), so 2 Hz removes noise cleanly.

    Returns
    -------
    Same list, Q fields replaced in-place.
    """
    from scipy.signal import butter, filtfilt

    if len(snapshots) < 12:
        print("[Q smooth] skipped — fewer than 12 snapshots")
        return snapshots

    # Estimate sample rate from time axis
    times  = np.array([s['t'] for s in snapshots])
    dt_arr = np.diff(times)
    dt_med = float(np.median(dt_arr[dt_arr > 0]))
    fs     = 1.0 / max(dt_med, 0.01)

    wn = min(cutoff_hz / (fs / 2.0), 0.99)
    b, a = butter(3, wn, btype='low')

    Q_seq    = np.stack([s['Q'] for s in snapshots], axis=0)   # (T, ny, nx)
    Q_smooth = filtfilt(b, a, Q_seq, axis=0)
    Q_smooth = np.maximum(Q_smooth, 0.0)

    for i, s in enumerate(snapshots):
        s['Q'] = Q_smooth[i].astype(np.float32)

    print(f"[Q smooth] Butterworth LP {cutoff_hz:.1f} Hz applied to "
          f"{len(snapshots)} snapshots  (fs≈{fs:.1f} Hz)")
    return snapshots


def discover_recordings(data_dir: str) -> list:
    """Return sorted list of zero-padded recording IDs found in data_dir."""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(data_dir, "*_tracks.csv")))
    ids   = [os.path.basename(f).replace("_tracks.csv", "") for f in files]
    if not ids:
        raise FileNotFoundError(
            f"No *_tracks.csv files found in {data_dir!r}. "
            "Check --dataset and --data_root.")
    return ids


def parse_recording_ids(recording_arg: str, data_dir: str) -> list:
    """
    Resolve --recording argument to a list of IDs.

    Accepted forms:
      all          → every recording in data_dir
      00           → single recording "00"
      00,02,07     → explicit comma-separated list
    """
    s = recording_arg.strip()
    if s.lower() == 'all':
        return discover_recordings(data_dir)
    return [r.strip() for r in s.split(',') if r.strip()]


def load_all_recordings(recording_ids: list, data_dir: str,
                         max_sec: float, warmup_sec: float,
                         perception_range: float = _PERCEPTION_RANGE_DEFAULT) -> tuple:
    """
    Load multiple recordings, offset their time axes so they form one
    continuous sequence, and return (all_snapshots, last_loader).

    Time is reset to 0 at the start of each recording's *run* window
    (after warm-up), then offset by the cumulative end time of all
    previous recordings so the FieldInterpolator sees strictly increasing t.

    Parameters
    ----------
    perception_range : float
        Passed to ExiDLoader.  Agents beyond this distance from ego are
        excluded from Q computation.  Pass float('inf') to disable (e.g.
        when evaluating an old model that was trained without the filter).

    Returns
    -------
    all_snapshots : list of snapshot dicts (t axis is globally monotone)
    last_loader   : the ExiDLoader instance for the last recording
                    (used to extract grid info)
    """
    all_snaps  = []
    t_offset   = 0.0
    last_loader = None

    for rec_id in recording_ids:
        print(f"\n[Phase1] ── Recording {rec_id} ──")
        try:
            loader = ExiDLoader(
                data_dir=data_dir,
                recording_id=rec_id,
                max_seconds=max_sec,
                warmup_seconds=warmup_sec,
                perception_range=perception_range,
            )
            snaps = loader.load()
        except Exception as exc:
            print(f"  [skip] recording {rec_id}: {exc}")
            continue

        if not snaps:
            print(f"  [skip] recording {rec_id}: no snapshots produced")
            continue

        # Re-stamp t with global offset so times are unique across recordings
        for s in snaps:
            s['t'] = s['t'] + t_offset

        t_offset   += snaps[-1]['t'] - snaps[0]['t'] + (snaps[1]['t'] - snaps[0]['t'])
        all_snaps  += snaps
        last_loader = loader

    if not all_snaps:
        raise RuntimeError("No snapshots loaded from any recording.")

    print(f"\n[Phase1] Total snapshots across all recordings: {len(all_snaps)} "
          f"({t_offset:.1f}s cumulative)")
    return all_snaps, last_loader


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PINN Risk Field — DREAM project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
          Dataset / recording examples
          ----------------------------
            --dataset inD  --recording all            # all inD recordings
            --dataset exiD --recording 00             # single exiD recording
            --dataset rounD --recording 00,02,05      # three rounD recordings
            --dataset inD,rounD --recording all       # inD + rounD combined
            --dataset all   --recording all           # all four datasets
            --dataset inD,exiD,rounD --recording all  # three datasets
        """))
    parser.add_argument('--dataset',    default='inD',
                        help=('Dataset name(s): single (inD), comma list (inD,rounD), '
                              'or "all" for all four datasets. '
                              f'Known: {KNOWN_DATASETS}'))
    parser.add_argument('--data_root',  default=os.path.join(DREAM_ROOT, 'data'),
                        help='Root folder that contains the dataset sub-directories')
    parser.add_argument('--recording',  default='00',
                        help='Recording ID ("00"), comma list ("00,02"), or "all"')
    parser.add_argument('--epochs',     type=int,   default=2000)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--hidden',       type=int,   default=128)
    parser.add_argument('--depth',        type=int,   default=6)
    parser.add_argument('--use_rff',      action='store_true',
                        help='Enable Random Fourier Features front-end (fixes spectral bias)')
    parser.add_argument('--rff_features', type=int,   default=64,
                        help='Number of RFF frequencies (output dim = 2×this)')
    parser.add_argument('--rff_scale',    type=float, default=10.0,
                        help='RFF frequency scale (1–50; higher = finer spatial detail)')
    parser.add_argument('--use_context',  action='store_true',
                        help='Add N_agents + dist_nearest context features (9-D input instead of 7-D)')
    parser.add_argument('--n_colloc',   type=int,   default=2048)
    parser.add_argument('--n_data',     type=int,   default=1024)
    parser.add_argument('--pts_per_snap', type=int, default=100,
                        help='Pre-sampled pts per snapshot (FlatSampleCache only)')
    parser.add_argument('--max_sec',    type=float, default=40.0,
                        help='Max seconds of recording to use')
    parser.add_argument('--warmup_sec', type=float, default=4.0)
    parser.add_argument('--train_frac', type=float, default=0.8,
                        help='Fraction of snapshots for training')
    parser.add_argument('--device',     default='cuda' if
                        torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_model', default='pinn_risk_field.pt')
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--w_data',       type=float, default=1.0)
    parser.add_argument('--w_phys',       type=float, default=1.0,
                        help='PDE residual weight (was 0.1; 1.0–3.0 recommended)')
    parser.add_argument('--w_ic',         type=float, default=0.5)
    parser.add_argument('--w_bc',         type=float, default=0.2)
    parser.add_argument('--w_smooth',     type=float, default=0.2,
                        help='Laplacian smoothness penalty weight (0 = disabled)')
    parser.add_argument('--q_smooth',     action='store_true',
                        help='Pre-filter Q snapshots with 2 Hz Butterworth LP (reduces comb artefacts)')
    parser.add_argument('--q_smooth_hz',  type=float, default=2.0,
                        help='Cutoff frequency for --q_smooth (default 2.0 Hz)')
    parser.add_argument('--perception_range', type=float, default=80.0,
                        help='Only vehicles within this range of ego contribute to Q (m). '
                             '0 = disabled (use all vehicles). '
                             'Saved in checkpoint so inference automatically matches.')
    parser.add_argument('--no_plot',    action='store_true')
    parser.add_argument('--phase1_only',action='store_true',
                        help='Only run Phase 1 (data loading + PDE smoke-test)')
    args = parser.parse_args()

    # Resolve dataset list
    dataset_names = parse_datasets(args.dataset)
    multi = len(dataset_names) > 1

    # Override class attribute so save() stores the correct runtime value.
    # perception_range == 0 means "disabled" → use float('inf').
    _eff_perc = float('inf') if args.perception_range <= 0 else args.perception_range
    ExiDLoader.PERCEPTION_RANGE = _eff_perc

    # Auto-name outputs
    ds_tag  = 'multi' if multi else dataset_names[0]
    rec_tag = args.recording.replace(',', '+') if args.recording.lower() != 'all' else 'all'
    save_dir   = os.path.join(DREAM_ROOT, f"pinn_validation_{ds_tag}_{rec_tag}")
    model_path = (args.save_model if args.save_model != 'pinn_risk_field.pt'
                  else f"pinn_{ds_tag}_{rec_tag}.pt")

    print("=" * 60)
    print("  PINN Risk Field — DREAM project")
    print(f"  Dataset(s) : {dataset_names}")
    print(f"  Recordings : {args.recording}")
    print(f"  data_root  : {args.data_root}")
    print(f"  Device     : {args.device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Outputs    : {save_dir}/  |  {model_path}")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # PHASE 1 — Data loading (single or multi-dataset)
    # ----------------------------------------------------------------
    print("\n--- PHASE 1: Data Loading ---")
    if multi:
        snapshots, last_loader = load_multi_dataset(
            dataset_names=dataset_names,
            data_root=args.data_root,
            recording_arg=args.recording,
            max_sec=args.max_sec,
            warmup_sec=args.warmup_sec,
            perception_range=_eff_perc,
        )
    else:
        data_dir = os.path.join(args.data_root, dataset_names[0])
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {data_dir!r}\n"
                f"Check --data_root ({args.data_root}) and --dataset ({args.dataset})")
        rec_ids = parse_recording_ids(args.recording, data_dir)
        snapshots, last_loader = load_all_recordings(
            recording_ids=rec_ids,
            data_dir=data_dir,
            max_sec=args.max_sec,
            warmup_sec=args.warmup_sec,
            perception_range=_eff_perc,
        )

    if args.phase1_only:
        print("\n[Phase1] Smoke-test complete. Exiting (--phase1_only).")
        return

    if len(snapshots) < 10:
        raise RuntimeError(
            f"Too few snapshots ({len(snapshots)}). "
            "Increase --max_sec, add more recordings, or check the recording ID.")

    # Optional Q temporal smoothing (reduces comb artefacts from overlapping Gaussians)
    if args.q_smooth:
        snapshots = smooth_Q_temporal(snapshots, cutoff_hz=args.q_smooth_hz)

    # Train / test split (chronological)
    split = int(len(snapshots) * args.train_frac)
    train_snaps = snapshots[:split]
    test_snaps  = snapshots[split:]
    print(f"\n  Train snapshots: {len(train_snaps)}, Test: {len(test_snaps)}")

    # ----------------------------------------------------------------
    # PHASE 2 — PINN
    # ----------------------------------------------------------------
    print("\n--- PHASE 2: PINN Training ---")
    norm  = Normalizer(train_snaps, last_loader.x_grid, last_loader.y_grid)
    cache = build_cache(train_snaps, last_loader.x_grid, last_loader.y_grid,
                        pts_per_snap=args.pts_per_snap)

    trainer = PINNTrainer(
        snapshots    = train_snaps,
        norm         = norm,
        interp       = cache,
        hidden       = args.hidden,
        depth        = args.depth,
        use_rff      = args.use_rff,
        rff_features = args.rff_features,
        rff_scale    = args.rff_scale,
        use_context  = args.use_context,
        device       = args.device,
        w_data       = args.w_data,
        w_phys       = args.w_phys,
        w_ic         = args.w_ic,
        w_bc         = args.w_bc,
        w_smooth     = args.w_smooth,
        n_colloc     = args.n_colloc,
        n_data       = args.n_data,
    )

    if args.load_model and os.path.isfile(args.load_model):
        trainer.load(args.load_model)
    else:
        trainer.train(epochs=args.epochs, lr=args.lr)
        trainer.save(model_path)

    # ----------------------------------------------------------------
    # PHASE 3 — Validation
    # ----------------------------------------------------------------
    print("\n--- PHASE 3: Validation ---")
    all_snaps_backup = trainer.snaps
    trainer.snaps    = train_snaps + test_snaps   # extend for predict_field indexing

    validator = PINNValidator(trainer, test_snaps)
    metrics   = validator.run(
        plot     = not args.no_plot,
        save_dir = save_dir,
    )

    trainer.snaps = all_snaps_backup
    print("\nDone.")
    return metrics


if __name__ == "__main__":
    main()

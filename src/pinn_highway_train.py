"""
Highway-domain PINN trainer
============================
Generates synthetic traffic using IDM vehicles on the config_highway.py grid,
runs the DRIFT PDE solver to build ground-truth risk-field samples, and trains
a RiskFieldNet on those samples.

Why synthetic, not real-dataset:
  The RL env (dream_env_pinn.py) uses config_highway.py physics with 9 IDM
  vehicles on a straight road.  The existing pinn_multi_all.pt was trained on
  a curved-road scene with Q up to 162, D up to 76.8, t up to 1645 s — none
  of those ranges overlap the highway RL env.  Training on synthetic IDM
  rollouts on config_highway.py produces a PINN whose normalizer stats match
  what the RL env actually feeds in at query time.

Output:  pinn_highway.pt  (pinn_adapter.py prefers this over pinn_multi_all.pt)

Usage
-----
  # quick smoke check (CPU, ~5 min):
  python pinn_highway_train.py --episodes 5 --steps 200 --epochs 500 --device cpu

  # recommended first run (~30-60 min on GPU):
  python pinn_highway_train.py --episodes 25 --steps 400 --epochs 2000 --device cuda

  # longer / higher quality:
  python pinn_highway_train.py --episodes 50 --steps 400 --epochs 3000 --device cuda
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------------------------
# Monkey-patch config.Config → config_highway.Config BEFORE importing
# pinn_risk_field, so Normalizer / PINNTrainer._ic_loss etc. use highway domain.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config as _cfg_module
from config_highway import Config as hwy_cfg
_cfg_module.Config = hwy_cfg    # must happen before `from pinn_risk_field import ...`

from pinn_risk_field import RiskFieldNet        # only import the model class
from pde_solver import (
    PDESolver, compute_total_Q, compute_velocity_field,
    compute_diffusion_field, create_vehicle as drift_create_vehicle,
)
from IDM_general import IDM

# ---------------------------------------------------------------------------
# Scenario catalogue  (mirrors dream_env_pinn._SCENARIOS)
# ---------------------------------------------------------------------------
_SCENARIOS = {
    "dangerous": {
        "position": [
            18.0, 5.6, 7.0, 0.0,   100.0, 5.2, 7.0, 0.0,   130.0, 5.0, 7.0, 0.0,
            35.0, 9.0, 7.0, 0.0,    75.0, 9.0, 7.0, 0.0,   120.0, 9.0, 7.0, 0.0,
            34.0, 1.8, 7.0, 0.0,    70.0, 2.0, 7.0, 0.0,   140.0, 1.6, 7.0, 0.0,
            60.0, 5.3, 10.0,
        ],
        "initial_V": {
            "U1": 12.0, "U2": 10.0, "U3":  9.0,
            "D1":  9.0, "D2": 12.0, "D3":  9.0,
            "E0":  8.5, "E1": 11.5, "E2": 12.0,
        },
    },
    "faster": {
        "position": [
            16.8, 5.4, 6.9, 0.0,    71.6, 4.9, 6.7, 0.0,  146.5, 5.0, 7.3, 0.0,
            36.5, 8.8, 6.4, 0.0,    70.4, 9.0, 6.8, 0.0,  148.9, 9.0, 7.6, 0.0,
            37.5, 1.9, 7.9, 0.0,    73.0, 1.6, 7.0, 0.0,  153.9, 1.7, 6.5, 0.0,
            45.0, 5.1, 6.5,
        ],
        "initial_V": {
            "U1": 16.6, "U2": 16.1, "U3":  8.3,
            "D1": 12.8, "D2": 16.7, "D3": 12.1,
            "E0": 16.0, "E1":  8.6, "E2":  8.4,
        },
    },
    "dense": {
        "position": [
            10.0, 5.4, 6.0, 0.0,    50.0, 5.1, 5.5, 0.0,   90.0, 5.0, 6.0, 0.0,
            15.0, 8.8, 7.0, 0.0,    55.0, 9.0, 6.5, 0.0,   95.0, 9.2, 6.8, 0.0,
            12.0, 2.0, 7.5, 0.0,    52.0, 1.8, 7.0, 0.0,   92.0, 1.6, 6.5, 0.0,
            30.0, 5.25, 8.5,
        ],
        "initial_V": {
            "U1": 8.0, "U2": 7.5, "U3": 8.5,
            "D1": 9.0, "D2": 8.5, "D3": 7.0,
            "E0": 6.0, "E1": 5.5, "E2": 6.5,
        },
    },
}

_LANE_HI   = 7.0
_LANE_LO   = 3.5
_WHEELBASE = 2.7
_DT        = 0.1


# ---------------------------------------------------------------------------
# IDM helpers
# ---------------------------------------------------------------------------

def _idm_to_drift_vehicles(idm: IDM) -> list:
    pairs = [
        ('E0', idm.E0_X, idm.E0_Y, getattr(idm, 'E0_V', 7.0)),
        ('E1', idm.E1_X, idm.E1_Y, getattr(idm, 'E1_V', 7.0)),
        ('E2', idm.E2_X, idm.E2_Y, getattr(idm, 'E2_V', 7.0)),
        ('U1', idm.U1_X, idm.U1_Y, getattr(idm, 'U1_V', 7.0)),
        ('U2', idm.U2_X, idm.U2_Y, getattr(idm, 'U2_V', 7.0)),
        ('U3', idm.U3_X, idm.U3_Y, getattr(idm, 'U3_V', 7.0)),
        ('D1', idm.D1_X, idm.D1_Y, getattr(idm, 'D1_V', 7.0)),
        ('D2', idm.D2_X, idm.D2_Y, getattr(idm, 'D2_V', 7.0)),
        ('D3', idm.D3_X, idm.D3_Y, getattr(idm, 'D3_V', 7.0)),
    ]
    vehicles = []
    for vid, x, y, v in pairs:
        vd = drift_create_vehicle(vid=vid, x=x, y=y, vx=v, vy=0.0)
        vd['heading'] = 0.0
        vehicles.append(vd)
    return vehicles


def _idm_step(idm: IDM):
    ude = idm.Judge_Location(_LANE_HI, _LANE_LO)
    kw = dict(L=_WHEELBASE, S0=2.0, T=1.5, a=5.0, b=1.67,
              V0={"U1": 12.0, "U2": 10.0, "U3":  9.0,
                  "D1":  9.0, "D2": 12.0, "D3":  9.0,
                  "E0":  8.5, "E1": 11.5, "E2": 12.0})
    # IDM API uses positional args: L, S0, T_head, a_max, b, V0
    args6 = (_WHEELBASE, 2.0, 1.5, 5.0, 1.67,
             {"U1": 12.0, "U2": 10.0, "U3":  9.0,
              "D1":  9.0, "D2": 12.0, "D3":  9.0,
              "E0":  8.5, "E1": 11.5, "E2": 12.0})
    if ude == "U":
        idm.update_state_onlane(_LANE_HI, _LANE_LO, *args6)
        idm.update_state_E(*args6)
        idm.update_state_D(*args6)
    elif ude == "D":
        idm.update_state_onlane(_LANE_HI, _LANE_LO, *args6)
        idm.update_state_E(*args6)
        idm.update_state_U(*args6)
    else:
        idm.update_state_onlane(_LANE_HI, _LANE_LO, *args6)
        idm.update_state_U(*args6)
        idm.update_state_D(*args6)


# ---------------------------------------------------------------------------
# Streaming data generator
# ---------------------------------------------------------------------------

# FlatSampleCache column order (must match pinn_risk_field.FlatSampleCache.KEYS)
_KEYS = ('x', 'y', 't', 'Q', 'N_agents', 'dist_nearest', 'vx', 'vy', 'D', 'R')
_N_COLS = len(_KEYS)   # 10


class SyntheticHighwayStream:
    """
    Generate training data for the highway PINN using IDM traffic + DRIFT PDE.

    Memory strategy: run the full 500×60 PDE grid each step (required for
    correct advection), but sample only `pts_per_snap` random points per
    timestep.  The flat sample buffer stays at O(n_steps × pts_per_snap × 40 bytes)
    regardless of grid size.

    Attributes
    ----------
    buf : np.ndarray (N_total, 10)  — pre-built flat sample buffer
    norm_ranges : dict               — min/max stats for each channel
    times : np.ndarray               — t values of all stored snapshots
    """

    def __init__(self, n_episodes: int = 20, steps_per_episode: int = 400,
                 warmup_steps: int = 40, pts_per_snap: int = 150, seed: int = 0):
        self.n_episodes        = n_episodes
        self.steps_per_episode = steps_per_episode
        self.warmup_steps      = warmup_steps
        self.pts_per_snap      = pts_per_snap
        self.rng               = np.random.default_rng(seed)

        self.X = hwy_cfg.X    # (60, 500) meshgrid
        self.Y = hwy_cfg.Y
        self.x_grid = hwy_cfg.x   # (500,)
        self.y_grid = hwy_cfg.y   # (60,)
        self._ny, self._nx = self.X.shape

        # Accumulators — built during generate()
        self.buf         = None          # (N_total, 10) float32
        self.norm_ranges = None
        self.times       = None
        self._n_snaps    = 0

        # Running min/max for each channel
        self._stats = {k: [np.inf, -np.inf] for k in _KEYS}

    # ------------------------------------------------------------------

    def generate(self) -> None:
        """Run all episodes and populate self.buf, self.norm_ranges, self.times."""
        keys = list(_SCENARIOS.keys())
        rows = []
        time_list = []
        t0 = time.time()

        for ep in range(self.n_episodes):
            key = keys[ep % len(keys)]
            ep_rows, ep_times = self._run_episode(key)
            rows.extend(ep_rows)
            time_list.extend(ep_times)
            elapsed = time.time() - t0
            print(f"[SyntheticStream] ep {ep+1:3d}/{self.n_episodes}  "
                  f"key={key:10s}  snaps_this={len(ep_rows)}  "
                  f"total={len(rows)}  elapsed={elapsed:.0f}s")

        self.buf   = np.concatenate(rows, axis=0).astype(np.float32)   # (N, 10)
        self.times = np.array(time_list, dtype=np.float32)
        self._n_snaps = len(rows)

        # Finalise norm_ranges from accumulated stats
        self.norm_ranges = {}
        for i, k in enumerate(_KEYS):
            lo, hi = self._stats[k]
            self.norm_ranges[k] = (float(lo), max(float(hi), float(lo) + 1e-3))
        # x, y override with full grid extents (ensures boundary coverage)
        self.norm_ranges['x'] = (float(self.x_grid.min()), float(self.x_grid.max()))
        self.norm_ranges['y'] = (float(self.y_grid.min()), float(self.y_grid.max()))

        elapsed = time.time() - t0
        print(f"\n[SyntheticStream] {len(self.buf):,} samples from "
              f"{self._n_snaps} snapshots in {elapsed:.0f}s")
        print("Normalizer ranges:")
        for k, (lo, hi) in self.norm_ranges.items():
            print(f"  {k:12s}: [{lo:.4f}, {hi:.4f}]")

    # ------------------------------------------------------------------

    def _run_episode(self, scenario_key: str):
        scenario = _SCENARIOS[scenario_key]
        pos = list(scenario["position"])
        V0  = dict(scenario["initial_V"])

        # Randomise: ego ±8 m x, ±1.5 m/s; surrounding speeds ±2 m/s
        pos[-3] += float(self.rng.uniform(-8.0, 8.0))
        pos[-2] += float(self.rng.uniform(-0.3, 0.3))
        pos[-1]  = max(3.0, pos[-1] + float(self.rng.uniform(-2.0, 2.0)))
        for k in V0:
            V0[k] = max(3.0, V0[k] + float(self.rng.uniform(-2.0, 2.0)))

        idm = IDM(pos, _DT, _WHEELBASE)
        for name, v in V0.items():
            setattr(idm, f"{name}_V", v)

        ego_x = pos[-3]
        ego_y = pos[-2]
        ego_v = pos[-1]
        solver = PDESolver()

        # ---- PDE warmup (not recorded) ----
        for _ in range(self.warmup_steps):
            vehicles = _idm_to_drift_vehicles(idm)
            ego_dict = drift_create_vehicle(vid=0, x=ego_x, y=ego_y, vx=ego_v, vy=0.0)
            ego_dict['heading'] = 0.0
            Q_total, _, _, occ_mask = compute_total_Q(
                vehicles, ego_dict, self.X, self.Y)
            vx_g, vy_g, *_ = compute_velocity_field(
                vehicles, ego_dict, self.X, self.Y)
            D_g = compute_diffusion_field(occ_mask, self.X, self.Y, vehicles, ego_dict)
            solver.step(Q_total, D_g, vx_g, vy_g, dt=_DT)
            ego_x += ego_v * _DT
            _idm_step(idm)

        # ---- recording window ----
        ep_rows  = []
        ep_times = []
        t_sim    = 0.0

        for step in range(self.steps_per_episode):
            vehicles = _idm_to_drift_vehicles(idm)
            ego_dict = drift_create_vehicle(vid=0, x=ego_x, y=ego_y, vx=ego_v, vy=0.0)
            ego_dict['heading'] = 0.0

            Q_total, _, _, occ_mask = compute_total_Q(
                vehicles, ego_dict, self.X, self.Y)
            vx_g, vy_g, *_ = compute_velocity_field(
                vehicles, ego_dict, self.X, self.Y)
            D_g = compute_diffusion_field(
                occ_mask, self.X, self.Y, vehicles, ego_dict)
            R = solver.step(Q_total, D_g, vx_g, vy_g, dt=_DT)

            # dist_nearest field (needed for FlatSampleCache columns)
            N_ag = len(vehicles)
            if N_ag > 0:
                axy = np.array([[v['x'], v['y']] for v in vehicles], dtype=np.float32)
                dist_nearest = np.sqrt(
                    ((self.X[:, :, np.newaxis] - axy[:, 0])**2 +
                     (self.Y[:, :, np.newaxis] - axy[:, 1])**2)
                ).min(axis=2).astype(np.float32)
            else:
                dist_nearest = np.full(self.X.shape, 1000.0, dtype=np.float32)

            # Sample pts_per_snap random grid points
            yi = self.rng.integers(0, self._ny, self.pts_per_snap)
            xi = self.rng.integers(0, self._nx, self.pts_per_snap)

            row = np.empty((self.pts_per_snap, _N_COLS), dtype=np.float32)
            row[:, 0] = self.x_grid[xi]          # x
            row[:, 1] = self.y_grid[yi]           # y
            row[:, 2] = t_sim                     # t
            row[:, 3] = Q_total[yi, xi]           # Q
            row[:, 4] = float(N_ag)               # N_agents
            row[:, 5] = dist_nearest[yi, xi]      # dist_nearest
            row[:, 6] = vx_g[yi, xi]              # vx
            row[:, 7] = vy_g[yi, xi]              # vy
            row[:, 8] = D_g[yi, xi]               # D
            row[:, 9] = R[yi, xi]                 # R

            ep_rows.append(row)
            ep_times.append(t_sim)

            # Update running min/max stats
            for i, k in enumerate(_KEYS):
                col = row[:, i]
                self._stats[k][0] = min(self._stats[k][0], float(col.min()))
                self._stats[k][1] = max(self._stats[k][1], float(col.max()))

            t_sim  += _DT
            ego_x  += ego_v * _DT
            _idm_step(idm)

        return ep_rows, ep_times

    # ------------------------------------------------------------------
    # FlatSampleCache-compatible interface for PINNHighwayTrainer
    # ------------------------------------------------------------------

    def sample(self, n: int, rng=None):
        """Draw n random rows from the flat buffer."""
        idx = (rng.integers(0, len(self.buf), n) if rng is not None
               else self.rng.integers(0, len(self.buf), n))
        rows = self.buf[idx]
        return {k: rows[:, i] for i, k in enumerate(_KEYS)}


# ---------------------------------------------------------------------------
# Highway PINN trainer (self-contained, no ExiDLoader/Normalizer inheritance)
# ---------------------------------------------------------------------------

def _norm1(val, lo, hi):
    """Normalise val (numpy or tensor) to [-1, 1]."""
    return 2.0 * (val - lo) / max(hi - lo, 1e-8) - 1.0


class HighwayPINNTrainer:
    """
    Trains RiskFieldNet on data from SyntheticHighwayStream.

    Loss terms:
      L_data   : MSE between R_θ and numerical R at sampled grid points
      L_phys   : PDE residual (advection-diffusion) at collocation points
      L_ic     : R(x, y, t=0) = 0
      L_bc     : R = 0 at x=x_min and x=x_max boundaries
    """

    def __init__(self, stream: SyntheticHighwayStream,
                 hidden: int = 256, depth: int = 8,
                 use_rff: bool = True, rff_features: int = 64, rff_scale: float = 10.0,
                 device: str = 'cpu',
                 w_data: float = 1.0, w_phys: float = 0.5,
                 w_ic: float = 0.3,   w_bc: float = 0.1,
                 n_data: int = 1024,  n_colloc: int = 2048):

        self.stream   = stream
        self.nr       = stream.norm_ranges      # shorthand
        self.device   = torch.device(device)
        self.w_data   = w_data
        self.w_phys   = w_phys
        self.w_ic     = w_ic
        self.w_bc     = w_bc
        self.n_data   = n_data
        self.n_co     = n_colloc
        self.rng      = np.random.default_rng(42)

        # PDE constants from highway config
        self.lambda_d = hwy_cfg.lambda_decay   # 0.15 [1/s]
        self.tau      = hwy_cfg.tau            # 0.0  [s]

        self.model = RiskFieldNet(
            hidden=hidden, depth=depth,
            use_rff=use_rff, rff_features=rff_features, rff_scale=rff_scale,
            use_context=False,
        ).to(self.device)

        self.history = {'loss': [], 'L_data': [], 'L_phys': [], 'L_ic': [], 'L_bc': []}

    # ------------------------------------------------------------------
    # Build normalised input tensor from a sample dict
    # ------------------------------------------------------------------

    def _inp(self, samp):
        """Build (N, 7) normalised float32 tensor from a sample dict."""
        def _t(arr, key):
            lo, hi = self.nr[key]
            return torch.tensor(_norm1(np.asarray(arr, dtype=np.float32), lo, hi),
                                dtype=torch.float32, device=self.device)
        return torch.stack([
            _t(samp['x'], 'x'), _t(samp['y'], 'y'), _t(samp['t'], 't'),
            _t(samp['Q'], 'Q'), _t(samp['vx'], 'vx'),
            _t(samp['vy'], 'vy'), _t(samp['D'], 'D'),
        ], dim=-1)

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _data_loss(self):
        samp   = self.stream.sample(self.n_data, self.rng)
        inp    = self._inp(samp)
        R_pred = self.model(inp).squeeze(-1)
        R_true = torch.tensor(samp['R'], dtype=torch.float32, device=self.device)
        R_scale = max(self.nr['R'][1], 1e-3)
        return torch.mean((R_pred - R_true / R_scale) ** 2)

    def _physics_loss(self):
        """PDE residual: R_t + ∇·(vR) − D∇²R + λR − Q = 0"""
        samp  = self.stream.sample(self.n_co, self.rng)

        # Raw coordinates with requires_grad
        x_t = torch.tensor(samp['x'], requires_grad=True, device=self.device)
        y_t = torch.tensor(samp['y'], requires_grad=True, device=self.device)
        t_t = torch.tensor(samp['t'], requires_grad=True, device=self.device)

        # Normalise (differentiable)
        def _nrm(v, key):
            lo, hi = self.nr[key]
            return 2.0 * (v - lo) / max(hi - lo, 1e-8) - 1.0

        xn = _nrm(x_t, 'x'); yn = _nrm(y_t, 'y'); tn = _nrm(t_t, 't')

        Q_t  = torch.tensor(samp['Q'],  dtype=torch.float32, device=self.device)
        vx_t = torch.tensor(samp['vx'], dtype=torch.float32, device=self.device)
        vy_t = torch.tensor(samp['vy'], dtype=torch.float32, device=self.device)
        D_t  = torch.tensor(samp['D'],  dtype=torch.float32, device=self.device)

        Qn  = _nrm(Q_t,  'Q')
        vxn = _nrm(vx_t, 'vx')
        vyn = _nrm(vy_t, 'vy')
        Dn  = _nrm(D_t,  'D')

        inp = torch.stack([xn, yn, tn, Qn, vxn, vyn, Dn], dim=-1)
        R   = self.model(inp).squeeze(-1)

        ones = torch.ones_like(R)
        R_x,  = torch.autograd.grad(R, x_t, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)
        R_y,  = torch.autograd.grad(R, y_t, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)
        R_t,  = torch.autograd.grad(R, t_t, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)
        R_xx, = torch.autograd.grad(R_x, x_t, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)
        R_yy, = torch.autograd.grad(R_y, y_t, grad_outputs=ones,
                                    create_graph=True, retain_graph=True)
        if self.tau > 0:
            R_tt, = torch.autograd.grad(R_t, t_t, grad_outputs=ones,
                                        create_graph=True, retain_graph=True)
        else:
            R_tt = torch.zeros_like(R_t)

        R_scale = max(self.nr['R'][1], 1e-3)
        Q_net   = Q_t / R_scale          # Q in network output scale
        diffusion  = D_t * (R_xx + R_yy)
        advection  = vx_t * R_x + vy_t * R_y
        lam_R      = self.lambda_d * R

        residual = self.tau * R_tt + R_t + advection - diffusion + lam_R - Q_net
        return torch.mean(residual ** 2)

    def _ic_loss(self):
        """R(x, y, t=0) = 0"""
        n    = max(256, self.n_co // 4)
        rng2 = np.random.default_rng(99)
        x_np = rng2.uniform(hwy_cfg.x_min, hwy_cfg.x_max, n).astype(np.float32)
        y_np = rng2.uniform(hwy_cfg.y_min, hwy_cfg.y_max, n).astype(np.float32)
        samp = {
            'x': x_np, 'y': y_np, 't': np.zeros(n, dtype=np.float32),
            'Q': np.zeros(n, dtype=np.float32),
            'vx': np.zeros(n, dtype=np.float32),
            'vy': np.zeros(n, dtype=np.float32),
            'D':  np.full(n, hwy_cfg.D0, dtype=np.float32),
        }
        inp = self._inp(samp)
        R   = self.model(inp).squeeze(-1)
        return torch.mean(R ** 2)

    def _bc_loss(self):
        """R = 0 at left (x=x_min) and right (x=x_max) boundaries."""
        n     = max(128, self.n_co // 8)
        rng2  = np.random.default_rng(77)
        t_np  = rng2.uniform(0.0, max(self.stream.times), n).astype(np.float32)
        y_np  = rng2.uniform(hwy_cfg.y_min, hwy_cfg.y_max, n).astype(np.float32)
        base  = {
            't': t_np, 'y': y_np,
            'Q': np.zeros(n, dtype=np.float32),
            'vx': np.zeros(n, dtype=np.float32),
            'vy': np.zeros(n, dtype=np.float32),
            'D':  np.full(n, hwy_cfg.D0, dtype=np.float32),
        }

        def _bc_side(x_val):
            samp = dict(base, x=np.full(n, x_val, dtype=np.float32))
            return self.model(self._inp(samp)).squeeze(-1)

        R_l = _bc_side(hwy_cfg.x_min)
        R_r = _bc_side(hwy_cfg.x_max)
        return torch.mean(R_l ** 2) + torch.mean(R_r ** 2)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, epochs: int = 2000, lr: float = 1e-3, print_every: int = 200):
        opt   = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        rff_tag = (f"RFF(feat={self.model.rff.B.shape[1]})"
                   if self.model.use_rff else "no-RFF")
        print(f"\n[HighwayPINN] Training {epochs} epochs on {self.device}  "
              f"arch={self.model.hidden}x{self.model.depth}  {rff_tag}")
        t0 = time.time()

        for ep in range(1, epochs + 1):
            self.model.train()
            opt.zero_grad()

            L_data = self._data_loss()
            L_phys = self._physics_loss()
            L_ic   = self._ic_loss()
            L_bc   = self._bc_loss()
            loss   = (self.w_data * L_data + self.w_phys * L_phys
                      + self.w_ic * L_ic   + self.w_bc   * L_bc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()

            for key, val in zip(
                    ('loss', 'L_data', 'L_phys', 'L_ic', 'L_bc'),
                    (loss, L_data, L_phys, L_ic, L_bc)):
                self.history[key].append(val.item())

            if ep % print_every == 0 or ep == 1:
                print(f"  ep {ep:5d}/{epochs}  loss={loss.item():.4e}  "
                      f"data={L_data.item():.4e}  phys={L_phys.item():.4e}  "
                      f"ic={L_ic.item():.4e}  bc={L_bc.item():.4e}  "
                      f"t={time.time()-t0:.0f}s")

        print(f"[HighwayPINN] Done. Final loss = {self.history['loss'][-1]:.4e}")

    # ------------------------------------------------------------------
    # Save — same checkpoint format as PINNTrainer.save()
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            'model_state' : self.model.state_dict(),
            'history'     : self.history,
            'norm_ranges' : self.stream.norm_ranges,
            'hidden'      : self.model.hidden,
            'depth'       : self.model.depth,
            'use_rff'     : self.model.use_rff,
            'rff_features': (self.model.rff.B.shape[1]
                             if self.model.use_rff else 64),
            'rff_scale'   : (float(self.model.rff.B.std())
                             if self.model.use_rff else 10.0),
            'use_context' : self.model.use_context,
            'perception_range': 80.0,
            'source'      : 'synthetic_highway',
        }, path)
        print(f"[HighwayPINN] Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Post-training probe: checks spatial variation with the newly saved model
# ---------------------------------------------------------------------------

def _probe(path: str, device: str, stream: SyntheticHighwayStream):
    from rl.risk.pinn_adapter import PINNRiskAdapter
    adapter = PINNRiskAdapter(checkpoint_path=path, device=device,
                              inference_x_range=(hwy_cfg.x_min, hwy_cfg.x_max),
                              inference_y_range=(hwy_cfg.y_min, hwy_cfg.y_max))
    if not adapter.available:
        print("[probe] Adapter unavailable — skipping.")
        return

    zero  = np.zeros_like(stream.X)
    D_flt = np.full_like(stream.X, hwy_cfg.D0)

    # (a) spatial variation: ego alone, varying x along centre lane
    print("\n--- Spatial variation probe (no traffic, varying x) ---")
    r_vals = []
    for dx in [0, 50, 100, 200, 400, 800]:
        feat = adapter.query_risk_features(
            ego_x=50.0 + dx, ego_y=5.25, t=10.0,
            Q_grid=zero, vx_grid=zero, vy_grid=zero, D_grid=D_flt,
            sim_cfg=hwy_cfg, lane_centers=hwy_cfg.lane_centers, current_lane=1)
        r_vals.append(feat.get('r_ego', 0.0))
        print(f"  x={50+dx:4d}  r_ego={feat.get('r_ego',0):.6f}  "
              f"r_20m={feat.get('r_20m',0):.6f}  "
              f"grad_x={feat.get('grad_x',0):.6f}")
    var_empty = max(r_vals) - min(r_vals)
    print(f"  Variation (no traffic): {var_empty:.6f}")

    # (b) near-vehicle probe: single vehicle 10 m ahead of ego
    print("\n--- Near-vehicle probe (vehicle at ego+10m, same lane) ---")
    single_Q = np.zeros_like(stream.X)
    vy_est   = np.zeros_like(stream.X)
    vx_est   = np.full_like(stream.X, 10.0)  # approx vehicle velocity
    # Place a single Q source at (80, 5.25)
    from pde_solver import compute_total_Q
    veh_ahead = [drift_create_vehicle(vid=1, x=80.0, y=5.25, vx=10.0, vy=0.0)]
    ego_dict  = drift_create_vehicle(vid=0, x=70.0, y=5.25, vx=10.0, vy=0.0)
    single_Q, _, _, _ = compute_total_Q(veh_ahead, ego_dict, stream.X, stream.Y)

    feat_near = adapter.query_risk_features(
        ego_x=70.0, ego_y=5.25, t=5.0,
        Q_grid=single_Q, vx_grid=vx_est, vy_grid=vy_est, D_grid=D_flt,
        sim_cfg=hwy_cfg, lane_centers=hwy_cfg.lane_centers, current_lane=1)
    feat_far  = adapter.query_risk_features(
        ego_x=200.0, ego_y=5.25, t=5.0,
        Q_grid=single_Q, vx_grid=vx_est, vy_grid=vy_est, D_grid=D_flt,
        sim_cfg=hwy_cfg, lane_centers=hwy_cfg.lane_centers, current_lane=1)

    print(f"  Near (x=70, veh at 80):  r_ego={feat_near.get('r_ego',0):.6f}  "
          f"r_5m={feat_near.get('r_5m',0):.6f}  "
          f"r_20m={feat_near.get('r_20m',0):.6f}")
    print(f"  Far  (x=200, no traffic): r_ego={feat_far.get('r_ego',0):.6f}")
    ratio = feat_near.get('r_ego', 0) / max(feat_far.get('r_ego', 1e-6), 1e-6)
    print(f"  Near/Far ratio: {ratio:.2f}  "
          f"({'PASS: near-vehicle > far' if ratio > 2.0 else 'WARN: ratio < 2 — PINN still flat'})")

    # (c) left/right lane separation
    print("\n--- Lane separation probe (adjacent traffic in left lane) ---")
    veh_left = [drift_create_vehicle(vid=1, x=70.0, y=8.75, vx=10.0, vy=0.0)]
    ego_c    = drift_create_vehicle(vid=0, x=70.0, y=5.25, vx=10.0, vy=0.0)
    Q_left, _, _, _ = compute_total_Q(veh_left, ego_c, stream.X, stream.Y)
    feat_c   = adapter.query_risk_features(
        ego_x=70.0, ego_y=5.25, t=5.0,
        Q_grid=Q_left, vx_grid=vx_est, vy_grid=vy_est, D_grid=D_flt,
        sim_cfg=hwy_cfg, lane_centers=hwy_cfg.lane_centers, current_lane=1)
    print(f"  r_ego={feat_c.get('r_ego',0):.6f}  "
          f"r_left={feat_c.get('r_left',0):.6f}  "
          f"r_right={feat_c.get('r_right',0):.6f}")
    lr_sep = abs(feat_c.get('r_left',0) - feat_c.get('r_right',0))
    print(f"  Left/right separation: {lr_sep:.6f}  "
          f"({'PASS' if lr_sep > 0.05 else 'WARN: no lane separation'})")

    print(f"\n[probe] Summary:")
    print(f"  spatial variation (no traffic): {var_empty:.6f}")
    print(f"  near/far ratio:                 {ratio:.2f}")
    print(f"  left/right lane separation:     {lr_sep:.6f}")
    ok = (var_empty > 0.05) and (ratio > 2.0) and (lr_sep > 0.05)
    print(f"  Overall: {'PASS — PINN ready for PPO smoke run' if ok else 'WARN — consider more epochs or episodes'}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument('--episodes',  type=int,   default=20,
                   help='Number of synthetic IDM episodes (default 20)')
    p.add_argument('--steps',     type=int,   default=400,
                   help='Steps to record per episode  (default 400 = 40 s)')
    p.add_argument('--warmup',    type=int,   default=40,
                   help='PDE warm-up steps before recording (default 40)')
    p.add_argument('--pts',       type=int,   default=150,
                   help='Sampled grid points per snapshot (default 150)')
    p.add_argument('--epochs',    type=int,   default=2000,
                   help='Training epochs (default 2000)')
    p.add_argument('--lr',        type=float, default=1e-3,
                   help='Learning rate (default 1e-3)')
    p.add_argument('--hidden',    type=int,   default=256,
                   help='Hidden units per layer (default 256)')
    p.add_argument('--depth',     type=int,   default=8,
                   help='Number of hidden layers (default 8)')
    p.add_argument('--no-rff',    action='store_true',
                   help='Disable Random Fourier Features')
    p.add_argument('--device',    type=str,   default='cpu',
                   help='torch device (default cpu; use cuda for GPU)')
    p.add_argument('--out',       type=str,   default='pinn_highway.pt',
                   help='Output checkpoint filename (default pinn_highway.pt)')
    p.add_argument('--seed',      type=int,   default=42)
    p.add_argument('--no-probe',  action='store_true',
                   help='Skip post-training spatial probe')
    args = p.parse_args()

    print("=" * 60)
    print("Highway-domain PINN training")
    print(f"  Grid     : {hwy_cfg.nx}x{hwy_cfg.ny}  "
          f"x=[{hwy_cfg.x_min},{hwy_cfg.x_max}]  "
          f"y=[{hwy_cfg.y_min},{hwy_cfg.y_max}]")
    print(f"  Episodes : {args.episodes} x {args.steps} steps  "
          f"({args.episodes*args.steps:,} snapshots, "
          f"{args.episodes*args.steps*args.pts:,} sampled pts)")
    print(f"  Arch     : {args.hidden}x{args.depth}  "
          f"RFF={'off' if args.no_rff else 'on'}  ctx=off")
    print(f"  Epochs   : {args.epochs}  lr={args.lr}  device={args.device}")
    print("=" * 60)

    # Phase 1: generate synthetic data
    stream = SyntheticHighwayStream(
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        warmup_steps=args.warmup,
        pts_per_snap=args.pts,
        seed=args.seed,
    )
    stream.generate()

    # Phase 2: train
    trainer = HighwayPINNTrainer(
        stream,
        hidden=args.hidden, depth=args.depth,
        use_rff=not args.no_rff, device=args.device,
        w_data=1.0, w_phys=0.5, w_ic=0.3, w_bc=0.1,
        n_data=1024, n_colloc=2048,
    )
    trainer.train(epochs=args.epochs, lr=args.lr, print_every=200)

    # Phase 3: save
    out_path = os.path.join(_REPO_ROOT, args.out)
    trainer.save(out_path)

    # Phase 4: spatial variation probe
    if not args.no_probe:
        _probe(out_path, args.device, stream)


if __name__ == '__main__':
    main()

# RL Debug — Current Pipeline (Actual Implementation)

> Evidence-based audit produced 2026-04-13.
> Source files: `rl/env/dream_env_pinn.py`, `rl/risk/pinn_adapter.py`,
> `rl/reward/reward_fn.py`, `rl/train.py`, `rl/safety/cbf_filter.py`.

---

## Full Runtime Flow

```
Step t
  │
  ├─ IDM.update()        — 9 surrounding vehicles advance by dt=0.1s
  │
  ├─ KinematicModel      — ego bicycle model state:
  │                         (ego_x, ego_y, ego_v, ego_yaw)
  │
  ├─ _build_obs()        ─────────────────────────────────────────────┐
  │     │                                                              │
  │     ├─ pde_solver.compute_total_Q(vehicles, ego, X, Y)            │
  │     │     → Q_grid (ny×nx)  [highway range: 0–10 vs train: 0–162]│
  │     ├─ pde_solver.compute_velocity_field(...)                      │
  │     │     → vx_grid, vy_grid                                       │
  │     ├─ pde_solver.compute_diffusion_field(...)                     │
  │     │     → D_grid  [highway: constant 0.3 = training minimum]    │
  │     │                                                              │
  │     ├─ pinn._adapter.query_risk_features(                         │
  │     │     ..., N_agents=9, ...)    ← BUG: N_agents not a param   │
  │     │     → TypeError caught silently → _pinn_features = {}       │
  │     │     → obs[12:19] = 0.0 (always)                             │
  │     │                                                              │
  │     └─ obs = 22-D vector (float32)                                │
  │           slots 0–11:  vehicle kinematics  (working correctly)    │
  │           slots 12–19: PINN risk features  (always ZERO)         │
  │           slot  20:    in_merge flag                               │
  │           slot  21:    cbf_active flag                             │
  │                                                                    └──
  ├─ PPO policy forward pass
  │     → (a_raw, δ_raw) = mean of Gaussian  [deterministic at eval]
  │
  ├─ CBFSafetyFilter.project(a_raw, δ_raw, ...)
  │     → (a_safe, δ_safe)  [clips action when gap or boundary CBF violated]
  │     → cbf_active ∈ {0, 1}
  │
  ├─ KinematicModel.update_state(a_safe, δ_safe)
  │     → new (ego_x, ego_y, ego_v, ego_yaw)
  │
  ├─ compute_reward(...)
  │     → r_progress, r_speed, r_comfort, r_lane, r_near_miss
  │
  ├─ _compute_pinn_cost()
  │     → (r_ego + r_20m) / 10  but _pinn_features = {} → cost = 0.0 (always)
  │
  ├─ total_reward = reward - 0.5 * 0.0 = reward   [risk term is always zero]
  │
  └─ PPO rollout buffer ← (obs, a_raw, log_prob, reward, value, done)
         ↓
         (after 2048 steps)
         GAE advantage estimation
         PPO clipped surrogate update
```

---

## What is NOT happening

| Expected | Actual |
|----------|--------|
| PINN queried every step | PINN call raises TypeError, caught silently → always skipped |
| obs[12–19] carry risk gradient | obs[12–19] = [0, 0, 0, 0, 0, 0, 0, 0] every step |
| PPO learns risk-avoidance | PPO receives zero risk signal; learns only from progress + terminals |
| c_pinn modulates reward | c_pinn = 0.0 for all 300,000 training steps |
| Temporal risk propagation used | t normalises to −1.0 (extreme of training range); temporal info absent |

---

## PINN Loading vs PINN Execution

The PINN checkpoint **loads successfully** — architecture is inferred from the state dict,
weights are loaded, and forward passes can be called manually (as shown in the probe).
The failure is at the **call site** in `_build_obs()`.

The call in `rl/env/dream_env_pinn.py` (line ~716):
```python
feat = self._pinn.query_risk_features(
    ego_x=ego_x, ego_y=ego_y,
    t=self._sim_t,
    Q_grid=Q_total, vx_grid=vx_g, vy_grid=vy_g, D_grid=D_g,
    sim_cfg=_cfg,
    N_agents=9,           # ← not a parameter of query_risk_features
    lane_centers=cfg.LANE_CENTERS,
    current_lane=curr_lane,
)
```

`query_risk_features` signature has no `N_agents` parameter.
Python raises `TypeError: got an unexpected keyword argument 'N_agents'`.
The surrounding `except Exception: pass` block absorbs it silently.
`_pinn_features` stays `{}` and risk obs stays 0.

---

## Evidence

From diagnostic probe `docs/rl_diagnostics_probe.py`:

```
obs slots 12-21 (PINN risk + context): [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
PINN features at reset: {}
```

From training log (`rl/logs/train_log.csv`, all 13305 episodes):
```
ep_cost = 0.0  (every single episode from step 1 to step 301044)
```

From manual PINN probe (calling correctly, without N_agents):
```
r_ego=0.228461  r_5m=0.228457  r_10m=0.228457  r_20m=0.228458
grad_x=0.0  grad_y=-0.000046
r_left=0.228621  r_right=0.228298
```
The PINN does produce output — but the output is flat (all values ≈ 0.228).

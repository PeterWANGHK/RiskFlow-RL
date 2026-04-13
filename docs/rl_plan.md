# Stage 1 — RL Integration Plan (PINN-centric architecture)

> Produced: 2026-04-12  (revised: PINN as risk source, RL as primary controller)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DREAM PINN-RL Stack                             │
│                                                                         │
│   Traffic state (IDM vehicles)                                          │
│           │                                                             │
│   pde_solver helpers ──► Q(x,y,t), v(x,y,t), D(x,y,t)   ← CHEAP       │
│   (no time-integration; source/velocity/diffusion fields only)         │
│           │                                                             │
│   PINNRiskAdapter ────► R̂(x,y,t)  ∂R̂/∂x  ∂R̂/∂y         ← ~1 ms      │
│   (trained RiskFieldNet, forward pass only)                             │
│           │                                                             │
│   22-D observation vector                                               │
│           │                                                             │
│   PPO Policy (MLP 256×256) ──► (a_raw, δ_raw) ∈ Box(2)                 │
│           │                                                             │
│   CBFSafetyFilter ────────────────────────────────────────► safe        │
│   (analytical gap + lane-boundary CBF, no QP solver)                   │
│           │                                                             │
│   KinematicModel.update_state(a_safe, δ_safe)                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**What is new vs existing DREAM:**
- The PINN *replaces* the numerical PDE time-integration as the risk source at runtime.
- A PPO agent *replaces* the IDEAM MPC as the primary longitudinal + lateral controller.
- The CBF filter *replaces* the LMPC-CBF safety layer with an analytical projection
  (no optimisation solver, ~0.1 ms).

---

## Design Choices

### Choice 1: PINN as risk source (not numerical solver)
The trained `RiskFieldNet` (6-layer MLP, hidden=128–256) maps
(x̂, ŷ, t̂, Q̂, v̂x, v̂y, D̂) → R̂ in a single forward pass.
This is ~100× faster than the PDESolver operator-split time-stepping.
No warm-up period is needed; the network can be queried from t=0.

Trade-off: The PINN is an approximation. For Stage 6, the network should be
retrained on the highway scenario geometry to reduce approximation error.

### Choice 2: Continuous Box(2) action space
(a ∈ [-4.0, 1.5] m/s², δ ∈ [-0.35, 0.35] rad) gives the PPO agent fine-grained
control over both speed and lateral position.  Lane changes emerge from steering
rather than being a discrete action.  This is a more realistic control interface.

### Choice 3: Analytical CBF instead of MPC
Three independent CBF constraints, each projecting one action dimension:
1. **Longitudinal**: a ≤ a_max_cbf derived from gap-to-leader and headway
2. **Road-boundary**: δ clipped to prevent crossing road edges
3. **Lateral car-to-car**: δ clipped when adjacent vehicle is within 1.5 m laterally

No QP solver is needed. The projection is O(n_vehicles) and runs in <0.1 ms.

### Choice 4: PPO with GAE
Standard on-policy PPO with Generalised Advantage Estimation.  The PINN risk
cost is logged separately so the reward signal and safety cost can be decoupled
for a future Lagrangian PPO or CPO extension.

---

## Observation Vector (22-D)

```
Slots  0– 4  : ego kinematics (v_x, e_y, e_psi, last_a, last_δ)
Slots  5– 7  : current-lane leader (ds, dv, a_lead)
Slots  8– 9  : left-lane nearest vehicle (ds_left, dv_left)
Slots 10–11  : right-lane nearest vehicle (ds_right, dv_right)
Slots 12–15  : PINN risk lookahead (r_ego, r_5m, r_10m, r_20m)
Slots 16–17  : PINN risk gradient (∂R̂/∂x, ∂R̂/∂y)
Slots 18–19  : PINN lane risk (r_left, r_right)
Slot  20     : merge zone flag
Slot  21     : CBF-active flag (1 if last action was modified by CBF)
```

---

## Reward Function

```
total_reward = r_progress + r_speed + r_comfort + r_lane + r_near_miss
             − W_RISK * c_pinn
```

| Term | Formula | Weight |
|------|---------|--------|
| r_progress | Δx / (v_target · dt), clip [−1, 2] | 1.0 |
| r_speed | −(v − v_target)² / v_target² | 0.5 |
| r_comfort | −(a/4)² − (δ/0.4)² | 0.2 |
| r_lane | −(e_y / OFFROAD_LATERAL)² | 0.3 |
| r_near_miss | −W_NEAR_MISS · (1 − gap / NEAR_MISS_DIST) when gap < 8 m | 10.0 |

Safety cost (logged separately, folded in with W_RISK):

```
c_pinn = (r_ego + r_20m) / (2 · NORM_RISK_PINN)
```

Terminal rewards: collision −100, off-road −50, stall −20, timeout 0.

---

## Files

| File | Role |
|------|------|
| `rl/config/rl_config.py` | `RLConfig` dataclass — all hyperparameters |
| `rl/obs/observation_builder.py` | 20-D obs builder for the legacy Discrete env |
| `rl/reward/reward_fn.py` | `compute_reward`, `compute_safety_cost`, `terminal_reward` |
| `rl/risk/pinn_adapter.py` | `PINNRiskAdapter` — PINN runtime inference wrapper |
| `rl/safety/cbf_filter.py` | `CBFSafetyFilter` — analytical CBF projection |
| `rl/env/dream_env.py` | Legacy Discrete(9) environment (Stage 2 baseline) |
| `rl/env/dream_env_pinn.py` | **PINN-RL environment** — Box(2) + PINN + CBF |
| `rl/train.py` | PPO training script (Stage 4) |
| `rl/eval.py` | Evaluation and benchmark script (Stage 5) |

---

## Stages

| Stage | Status | Deliverable |
|-------|--------|-------------|
| 0 | done | Repo audit (`docs/stage0_repo_understanding.md`) |
| 1 | done | This plan |
| 2 | done | `DREAMHighwayEnv` (Discrete) + smoke test |
| 3 | done | Reward and safety cost design |
| 4 | done | `DREAMPINNEnv` (Box) + CBF filter + PINN adapter |
| 5 | next  | PPO training script (`rl/train.py`) |
| 6 | planned | Benchmark PINN-RL vs DREAM conservative baseline |

---

## Known Limitations and Future Work

1. **PINN domain mismatch**: The trained PINN used `config.py` coordinates
   (y ∈ [−225, −45] m).  The RL env uses `config_highway.py` (y ∈ [0, 14] m).
   The `PINNRiskAdapter` remaps coordinates at inference time — this is an
   approximation.  For publication quality, retrain the PINN on the highway
   scenario (Stage 6 goal).

2. **CBF completeness**: The current filter handles longitudinal gap, road
   boundaries, and lateral car proximity.  It does not handle the case where
   ego is simultaneously constrained from both sides (e.g., between two close
   vehicles).  For this case the filter reverts to a hard steer-to-centre.

3. **No multi-episode scenario diversity**: The three predefined scenarios
   rotate.  Adding random-density scenarios would improve generalisation.

4. **PPO hyperparameters not tuned**: Stage 5 requires a sweep over learning
   rate, clip ε, GAE λ, value function coefficient, etc.

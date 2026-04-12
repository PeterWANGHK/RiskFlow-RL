# Stage 4 — PINN-RL Environment and PPO Training Script

> Produced: 2026-04-12

---

## What was built

### New files

| File | Role |
|------|------|
| `rl/risk/pinn_adapter.py` | `PINNRiskAdapter` — loads trained PINN checkpoint, wraps inference |
| `rl/safety/cbf_filter.py` | `CBFSafetyFilter` — analytical CBF safety projection |
| `rl/env/dream_env_pinn.py` | `DREAMPINNEnv` — PINN-RL gym env with Box(2) actions |
| `rl/train.py` | Self-contained PPO training loop + evaluation CLI |

### Modified files

| File | Change |
|------|--------|
| `docs/stage1_rl_plan.md` | Revised with PINN-centric architecture |

---

## Architecture

```
IDM vehicles → pde_solver.compute_total_Q/velocity/diffusion
                     │  (Q, vx, vy, D grids — no PDE time integration)
                     ▼
           PINNRiskAdapter.query_risk_features()
                     │  (r_ego, r_5m, r_10m, r_20m, ∂R̂/∂x, ∂R̂/∂y, r_left, r_right)
                     ▼
           22-D observation  →  PPO policy  →  (a_raw, δ_raw)
                                                     │
                                          CBFSafetyFilter.project()
                                                     │
                                              (a_safe, δ_safe)
                                                     │
                                        KinematicModel.update_state()
```

---

## DREAMPINNEnv

### Action space
`Box(2, float32)`:
- `a ∈ [-4.0, 1.5]` m/s² (acceleration)
- `δ ∈ [-0.35, 0.35]` rad (steering)

### Observation space (22-D, float32, clipped to [-3, 3])

| Slot | Name | Description |
|------|------|-------------|
| 0 | v_x | ego speed / 15 |
| 1 | e_y | lateral error from lane centre / 2 |
| 2 | e_psi | heading error / 0.4 |
| 3 | last_a | last safe acceleration / 4 |
| 4 | last_δ | last safe steering / 0.4 |
| 5 | ds_curr | gap to lane leader / 30 − 1 |
| 6 | dv_curr | relative speed / 10 |
| 7 | a_lead | leader acceleration / 4 (placeholder: 0) |
| 8 | ds_left | gap to left-lane vehicle / 30 − 1 |
| 9 | dv_left | relative speed / 10 |
| 10 | ds_right | gap to right-lane vehicle / 30 − 1 |
| 11 | dv_right | relative speed / 10 |
| 12 | r_ego | PINN risk at ego / 5 |
| 13 | r_5m | PINN risk 5 m ahead / 5 |
| 14 | r_10m | PINN risk 10 m ahead / 5 |
| 15 | r_20m | PINN risk 20 m ahead / 5 |
| 16 | ∂R̂/∂x | risk gradient x / 2 |
| 17 | ∂R̂/∂y | risk gradient y / 2 |
| 18 | r_left | PINN risk, left lane corridor / 5 |
| 19 | r_right | PINN risk, right lane corridor / 5 |
| 20 | in_merge | 1 if ego x ∈ [30, 70] m |
| 21 | cbf_active | 1 if CBF clipped last action |

### Episode termination
- **Collision**: bumper gap < 4.5 m → reward −100
- **Off-road**: lateral deviation from lane centre > 2.8 m → reward −50
- **Stall**: v < 0.8 m/s for 30 steps → reward −20
- **Timeout**: 400 steps reached → truncated, reward 0

### Scenarios
Three predefined initial conditions cycle across episodes:
- `dangerous`: close traffic in all lanes, mix of fast/slow vehicles
- `faster`: higher-speed scenario, larger gaps
- `dense`: tight following, slow traffic

---

## CBFSafetyFilter

Three independent analytical projections (all O(n) in number of vehicles):

1. **Longitudinal**: `a_safe = min(a_raw, a_max_cbf)`
   where `a_max_cbf = (v_lead − v_ego + γ_lon · h_lon) / dt`
   and `h_lon = gap − (D_MIN + T_HEAD · v_ego)`

2. **Lane boundaries**: `δ_safe = clip(δ, δ_min_cbf, δ_max_cbf)`
   where `δ_min = −γ_lat · h_low / v`,  `δ_max = γ_lat · h_high / v`
   (`h_low = y − LANE_LEFT_LIMIT`,  `h_high = LANE_RIGHT_LIMIT − y`)

3. **Adjacent vehicle lateral**: same form, activated when lateral gap < 1.5 m

Default parameters: `D_MIN=4.5`, `T_HEAD=0.8 s`, `γ_lon=1.5`, `γ_lat=2.0`.

---

## PINNRiskAdapter

- Loads trained `RiskFieldNet` checkpoint (preferred: `pinn_multi_all.pt`, h=256 d=8)
- Remaps highway coordinates → PINN training domain at inference time
- Computes `∂R̂/∂x, ∂R̂/∂y` via PyTorch autograd (~0.5 ms)
- `query_risk_features()` returns 8-value risk feature dict for obs slots 12–19
- Falls back to zero risk if no checkpoint is available

---

## PPO Training Script (rl/train.py)

Self-contained PPO with GAE:

```bash
# Minimal test (500 steps, no saves):
python rl/train.py --steps 500 --no-save

# Full training (300 k steps, saves checkpoints):
python rl/train.py

# Resume from checkpoint:
python rl/train.py --resume rl/checkpoints/ppo_best.pt

# Evaluate saved policy:
python rl/train.py --eval rl/checkpoints/ppo_final.pt --eval-eps 20
```

**Outputs**: `rl/checkpoints/ppo_step_N.pt`, `ppo_best.pt`, `ppo_final.pt`, `rl/logs/train_log.csv`.

**Observed throughput**: ~126 steps/s on CPU. 300 k steps ≈ 40 min.

### PPO hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| Total steps | 300 000 |
| Rollout steps | 2 048 |
| Epochs per rollout | 10 |
| Minibatch size | 256 |
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| Clip ε | 0.2 |
| Learning rate | 3 × 10⁻⁴ |
| Entropy coeff | 0.01 |
| Hidden size | 256 × 256 |
| Log std init | −0.5 |

---

## Smoke Test Results

```
=== DREAMPINNEnv smoke test ===
[PINNAdapter] Loaded pinn_multi_all.pt  (h=256 d=8 rff=False ctx=False)
  inference domain: x(-10.0, 1000.0) y(-3.0, 14.0) t_clip=1645s  R_scale=10.0
[DREAMPINNEnv] PINN adapter: loaded
Initial obs shape : (22,), dtype=float32
Initial obs range : [-0.758, 1.000]
observation_space.contains(obs) OK
Cumulative reward (20 steps): 12.973
CBF activations: 0.0
PINN available : True
=== smoke test passed ===
```

---

## Known Limitations

1. **PINN cost = 0.00** during the mini-training run: the normalised cost
   `(r_ego + r_20m) / (2 · 5.0)` is near zero because PINN risk values are
   small on this straight-highway scenario.  W_RISK penalty has negligible effect
   until the agent is in a genuinely risky zone.  For Stage 5, verify that risk
   values increase meaningfully near vehicles.

2. **a_lead slot (obs[7]) is always 0**: leader acceleration is not tracked
   by the IDM model's public interface.  Low priority — the gap/dv features
   already encode relative dynamics.

3. **Rollout size (2048) > quick-test budget**: `--steps 500` still collects
   2048 steps because the rollout buffer must fill before each PPO update.
   Pass `--steps 2048` for the minimal one-update test.

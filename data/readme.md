# Social-Friendliness Analysis in `rl/data/`

A concise technical recap of how the dataset-extraction package quantifies
*tactically socially-friendly* human driving — what the labels actually
measure, the thresholds that govern them, and the panels the visualiser
emits.  All paths below are relative to the repo root.

## 1. Operational definition

The package does **not** model politeness as a soft attribute (e.g.
yielding, eye contact).  It treats "social-friendliness" as four
auditable, outcome-aware properties of a driver's tactical decisions:

| Property                                 | Captured by                                     |
| ---------------------------------------- | ----------------------------------------------- |
| Doesn't raise corridor risk for the scene | `future_risk_change`, `near_miss_future`, `collision_future` |
| Takes lane changes only when advantageous | `lane_change_advantage_flag`, `best_adv`, `lane_change_success` |
| Doesn't sit forever behind a slow leader  | `blocked_by_leader_flag`, `escape_success_flag`  |
| Acts coherently with lane utility         | `utility[3]` vs `action_9way`                    |

Each is computed per (ego, frame) sample over a configurable look-ahead.

## 2. Pipeline

```
data/<dataset>/*.csv     →  tracks_import.read_from_dataset()
                         →  rl.data.historical_extractor.extract_many()  (schema v3)
                         →  rl.data.plot_behavior_summary.render_figure()
```

Inputs supported by `tracks_import`: highD / inD / rounD / uniD / exiD
(`*_tracks.csv` schema) and the special datasets SQM-N-4 / YTDJ-3 /
XAM-N-5 / XAM-N-6 (Excel/CSV + pixel CSV, normalised in
[`tracks_import.py`](tracks_import.py:501)).

Schema-v3 output is one record per timestep per moving-car ego with
`MIN_TRACK_DURATION_S=4 s`, `MIN_EGO_SPEED=2 m/s` filters
([historical_extractor.py:137](rl/data/historical_extractor.py:137)).

## 3. Per-frame tactical features

Computed in `_per_frame_features`
([historical_extractor.py:309](rl/data/historical_extractor.py:309)):

* **Slotted neighbours** in the (front/rear) × (same/left/right) ego
  frame, nearest wins per slot.  Slot definition uses
  `LANE_WIDTH_ASSUMED=3.5 m` and `PERCEPTION_RANGE=60 m`.
* **Per-lane scalars** (3-vectors over `[curr, left, right]`):
  * `gap_fwd[j]`         — distance to that lane's front leader (m)
  * `rel_speed[j]`       — closing speed to that leader (m/s)
  * `lane_risk[j]`       — mean of `risk_corridor_tau` for τ ∈ {1,2,3} s
* **Lane utility** (matches `rl/reward/reward_fn.py::_lane_advantage`):
  $$
  U_j = \frac{\min(\text{gap}_j,\,80)}{D_0} - \frac{\Delta v_j}{V_0} - \frac{\text{risk}_j}{R_0}
  $$
  with `D0=30 m`, `V0=5 m/s`, `R0=2`
  ([historical_extractor.py:148](rl/data/historical_extractor.py:148)).
* **Advantages**: `adv_left = U_left − U_curr`, `adv_right = U_right − U_curr`,
  `best_adv = max(adv_left, adv_right)`.

Because the `(D0, V0, R0)` constants are *literally the same* in the
extractor and the online reward, offline-derived statistics on
`best_adv` directly transfer to the online policy's reward landscape.

## 4. Risk proxy (DRIFT-calibrated, no-PDE)

Online datasets contain no DRIFT field, so the extractor uses an
analytic surrogate ([rl/data/risk_proxy.py](rl/data/risk_proxy.py:1)):

$$
R(x,y) = \sum_i \exp\!\left(-\frac{(x-x_i)^2 + (y-y_i)^2}{2\sigma^2}\right) \cdot \big(1 + \alpha\, \max(0, \text{closing}_i)\big)
$$

with `SIGMA_KERNEL=2.0 m`, `V0_KERNEL=5.0 m/s`.  The amplitude is
calibrated to match `DRIFTInterface.get_risk_cartesian` so that the
analytic risk and the on-line PDE risk are on the same scale.

Time-parametrised corridor risk integrates this field along
`[ego_vx · τ, lateral_offset]` for τ ∈ {1, 2, 3, 4} s using
`CORRIDOR_N_SAMPLES=6` samples per corridor.

## 5. Outcome labels (3-s look-ahead)

Action-label horizon `1.5 s`; outcome-label horizon `3.0 s`
(both adjustable via CLI).  Definitions live at
[historical_extractor.py:151](rl/data/historical_extractor.py:151) and
[historical_extractor.py:620](rl/data/historical_extractor.py:620).

| Label                          | Definition                                                                                                                                                                                | Threshold(s)                       |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `future_risk_change`           | `corridor_curr[τ=2 s]` at i+H − at i                                                                                                                                                      | continuous                         |
| `lane_change_success`          | `future_lane_delta ≠ 0` ∧ lane stable for `SETTLING_FRAMES=10` ∧ no collision                                                                                                              | settling window 10 frames          |
| `near_miss_future`             | `min‖p_ego − p_other‖` over [i, i+H] < `NEAR_MISS_THR`                                                                                                                                    | 6.0 m                              |
| `collision_future`             | same min-distance < `COLLISION_THR`                                                                                                                                                       | 2.0 m                              |
| `blocked_by_leader_flag`       | `gap_curr < BLOCKED_GAP_THR` ∧ `ego_speed > 2 m/s` ∧ `v_leader < BLOCKED_SPEED_FRAC · v_ego`                                                                                              | 25 m, 0.7×                         |
| `escape_success_flag`          | blocked ∧ `future_lane_delta ≠ 0` ∧ `future_risk_change < 0` ∧ (`future_gap_gain > 0` ∨ `future_speed_delta > 0`)                                                                          | derived                            |
| `lane_change_advantage_flag`   | `future_gap_gain > TAU_D` ∧ `future_risk_change < −TAU_R`                                                                                                                                 | `TAU_D=5 m`, `TAU_R=0.2`           |
| `short_horizon_return_proxy`   | `1.0·progress − 0.2·∫risk − 0.1·comfort − 3.0·near_miss`                                                                                                                                   | weights frozen at the top of file  |

`escape_success_flag` is the canonical "social escape" measure:
non-zero only when the human (a) was stuck, (b) changed lane, (c) the
corridor risk *dropped*, and (d) they didn't just slow further.  This
is exactly the rule we want the policy to imitate when it overtakes a
slow leader without forcing a bad cut.

## 6. Visual diagnostics

`rl/data/plot_behavior_summary.py` renders a 6-panel paper-style figure
on top of these labels:

| Panel | Tests                                                                                                  | Reads                                                                                  |
| ----- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| (a)   | 9-way action distribution                                                                              | `action_9way`                                                                          |
| (b)   | lane-delta and speed-mode marginals                                                                    | derived from `action_9way`                                                             |
| (c)   | outcome rates: LC success/adv \| LC, blocked, escape \| blocked, near-miss, collision                 | `lane_change_success`, `lane_change_advantage_flag`, `blocked_…`, `escape_…`, `near_…`, `collision_…` |
| (d)   | `future_risk_change` distribution **conditioned on the human's lane action** (keep / right / left)     | `future_risk_change` × lane-block of `action_9way`                                     |
| (e)   | calibration: P(human takes LC) vs binned `best_adv`                                                    | `best_adv`, `action_9way`                                                              |
| (f)   | dataset-comparison heatmap across the seven panel-(c) rates                                            | aggregated row from `summary_row()`                                                    |

Panels (d) and (e) are the diagnostic most tightly coupled to the
expert's "social-friendliness" question:

* (d) shows **whether the human's chosen lane action makes the corridor
  safer or more dangerous** — a socially-friendly population concentrates
  the LC violins below zero.
* (e) shows **whether human LC propensity tracks the lane utility our
  reward also uses** — a socially-friendly population's curve rises
  monotonically with `best_adv`.

## 7. Calibration to the policy reward

Constants shared with `rl/reward/reward_fn.py` and
`rl/config/rl_config.py`:

| Symbol                        | Extractor name             | Reward name                         |
| ----------------------------- | -------------------------- | ----------------------------------- |
| Gap normaliser `D0`           | `LANE_ADV_D0   = 30 m`     | `LANE_ADV_GAP_D0`                   |
| Closing-speed normaliser `V0` | `LANE_ADV_V0   = 5 m/s`    | `LANE_ADV_DV_V0`                    |
| Risk normaliser `R0`          | `LANE_ADV_R0   = 2`        | `LANE_ADV_RISK_R0`                  |
| Blocked gap                   | `BLOCKED_GAP_THR = 25 m`   | `INACTION_LEADER_DIST`              |
| Blocked speed fraction        | `BLOCKED_SPEED_FRAC = 0.7` | `INACTION_SPEED_FRAC`               |

Therefore, behaviour-summary statistics computed from the human data
are directly comparable to those produced by a policy rollout — the
same `_lane_advantage` formula evaluates both, and the blocked / escape
masks use the same thresholds.  Drift here would be a silent
miscalibration; bumping `SCHEMA_VERSION` (currently 3) is the contract
to re-extract.

## 8. Limitations and explicit gaps

1. **No yielding/merge cooperation features.**  Cooperative gaps offered
   to merging traffic are not labelled.  Adding a `gap_offered_to_merger`
   signal would require a second pass over the frame index keyed on
   merger candidates.
2. **Risk proxy is amplitude-calibrated, not shape-calibrated.**  The
   shadow term used by DRIFT for occlusions is not modelled offline, so
   datasets without ground-truth occluders (everywhere here) will
   under-weight occluded-threat scenarios.
3. **Outcome horizon fixed at 3 s.**  Long-horizon social effects (e.g.
   chain-reaction braking of followers) need a separate label pass.
4. **Schema v2 datasets** in `rl/checkpoints/` (`bc_combined.npz`,
   `bc_highd_full.npz`, etc.) lack panels (c)–(f); re-extraction with
   the v3 extractor is required to use them in the visualiser.
5. **`actions` semantics.**  In v3 `action_9way` is "what the human will
   do over the next `horizon_sec`", not the instantaneous action.  This
   biases panel (a) toward "keep / maintain" because most 1.5-s windows
   contain no lane change — interpret outcome rates per-LC, not
   per-frame.

## 9. How an expert can audit the package quickly

1. Re-extract one recording with `--limit-tracks 50` to keep wall-time
   under two minutes:
   ```bash
   python -m rl.data.historical_extractor \
       --dataset-format highD --data-dir data/highD \
       --recordings 01 --limit-tracks 50 \
       --out-path /tmp/audit.npz
   ```
2. Inspect every label distribution end-to-end with one command:
   ```bash
   python -m rl.data.plot_behavior_summary \
       --inputs /tmp/audit.npz --out figures/audit
   ```
3. Cross-check that `lc_advantage_frac_lc` in panel (f) lies in the
   expected `0.05–0.30` band (printed by `summarize_dataset`); values
   outside that band indicate a labelling drift.
4. Verify `Δrisk(adv=1) < Δrisk(adv=0)` in the extractor's sanity
   report — this is the headline calibration check between the
   advantage label and the risk proxy.

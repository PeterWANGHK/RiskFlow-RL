The recommended social-RL path is:

1. train on `highway-fast-v0`
2. evaluate on `highway-v0` and `merge-v0`
3. optionally train/evaluate on `roundabout-v0`

## 1. Train Our Social-RL on HighwayEnv

### PPO: highway-fast-v0 -> highway-v0 + merge-v0

```powershell
python -m rl.train_highwayenv_social_sb3 `
  --algo ppo `
  --env-id highway-fast-v0 `
  --eval-env-id highway-v0 merge-v0 `
  --reward-config rl/config/social_reward_v1.json `
  --ablation A5 `
  --traffic-preset medium `
  --total-steps 20000 `
  --eval-freq 2000 `
  --eval-episodes 5 `
  --n-envs 4 `
  --run-dir rl/logs/social_ppo_a5
```

### DQN: highway-fast-v0 -> highway-v0 + merge-v0

```powershell
python -m rl.train_highwayenv_social_sb3 `
  --algo dqn `
  --env-id highway-fast-v0 `
  --eval-env-id highway-v0 merge-v0 `
  --reward-config rl/config/social_reward_v1.json `
  --ablation A5 `
  --traffic-preset medium `
  --total-steps 20000 `
  --eval-freq 2000 `
  --eval-episodes 5 `
  --run-dir rl/logs/social_dqn_a5
```

## 2. Evaluate Our Social-RL

### Evaluate the best checkpoint

```powershell
python -m rl.eval_highwayenv_social_sb3 `
  --algo ppo `
  --checkpoint rl/logs/social_ppo_a5/checkpoints/best_model.zip `
  --env-id highway-v0 merge-v0 `
  --reward-config rl/config/social_reward_v1.json `
  --ablation A5 `
  --episodes 20 `
  --use-drift true `
  --traffic-preset medium `
  --out rl/logs/social_ppo_a5/eval_best_20ep.json
```

### Evaluate the final checkpoint

```powershell
python -m rl.eval_highwayenv_social_sb3 `
  --algo ppo `
  --checkpoint rl/logs/social_ppo_a5/checkpoints/final_model.zip `
  --env-id highway-v0 merge-v0 `
  --reward-config rl/config/social_reward_v1.json `
  --ablation A5 `
  --episodes 20 `
  --use-drift true `
  --traffic-preset medium `
  --out rl/logs/social_ppo_a5/eval_final_20ep.json
```

### Drift-ablation check on the same policy

```powershell
python -m rl.eval_highwayenv_social_sb3 `
  --algo ppo `
  --checkpoint rl/logs/social_ppo_a5/checkpoints/best_model.zip `
  --env-id highway-v0 merge-v0 `
  --reward-config rl/config/social_reward_v1.json `
  --ablation A5 `
  --episodes 20 `
  --use-drift false `
  --traffic-preset medium `
  --out rl/logs/social_ppo_a5/eval_best_20ep_nodrift.json
```

## 3. Plot Training Curves

```powershell
python -m rl.plot_highwayenv_social_training --run-dir rl/logs/social_ppo_a5
```

This writes SciencePlots figures next to the run:

- `learning_curves.png/.pdf`
- `reward_components.png/.pdf`
- `safety_courtesy.png/.pdf`
- `final_eval_bars.png/.pdf`

The plotter supports both the standard `highway-v0` / `merge-v0` setup and single-env runs such as `roundabout-v0`.

## 4. Train Our Social-RL on Roundabout

### PPO: roundabout-v0

```powershell
python -m rl.train_highwayenv_social_sb3 `
  --algo ppo `
  --env-id roundabout-v0 `
  --eval-env-id roundabout-v0 `
  --reward-config rl/config/social_reward_v1.json `
  --ablation A5 `
  --traffic-preset medium `
  --duration 20 `
  --total-steps 20000 `
  --eval-freq 2000 `
  --eval-episodes 5 `
  --n-envs 4 `
  --run-dir rl/logs/social_ppo_a5_roundabout
```

### Evaluate the roundabout policy

```powershell
python -m rl.eval_highwayenv_social_sb3 `
  --algo ppo `
  --checkpoint rl/logs/social_ppo_a5_roundabout/checkpoints/best_model.zip `
  --env-id roundabout-v0 `
  --reward-config rl/config/social_reward_v1.json `
  --ablation A5 `
  --episodes 20 `
  --use-drift true `
  --traffic-preset medium `
  --duration 20 `
  --out rl/logs/social_ppo_a5_roundabout/eval_best_20ep.json
```

### Plot the roundabout training curves

```powershell
python -m rl.plot_highwayenv_social_training --run-dir rl/logs/social_ppo_a5_roundabout
```

## 5. Train Stock Baselines First, Then Evaluate With Social Metrics

### Stock PPO vs IDM/MOBIL on roundabout-v0

This trains a stock PPO baseline if the checkpoint is missing, then evaluates it with the online social/risk metric suite.

```powershell
python -m rl.compare_stock_policy_vs_idm_social `
  --algo ppo `
  --checkpoint rl/checkpoints/sb3_roundabout_ppo `
  --train-env-id roundabout-v0 `
  --eval-env-id roundabout-v0 `
  --train-if-missing true `
  --train-steps 20000 `
  --episodes 20 `
  --duration 20 `
  --n-envs 4 `
  --save-dir rl/logs/roundabout_stock_ppo_vs_idm_social
```

### Stock DQN vs IDM/MOBIL on roundabout-v0

```powershell
python -m rl.compare_stock_policy_vs_idm_social `
  --algo dqn `
  --checkpoint rl/checkpoints/sb3_roundabout_dqn `
  --train-env-id roundabout-v0 `
  --eval-env-id roundabout-v0 `
  --train-if-missing true `
  --train-steps 20000 `
  --episodes 20 `
  --duration 20 `
  --save-dir rl/logs/roundabout_stock_dqn_vs_idm_social
```

The output folders include:

- `summary.json`
- `summary.md`
- `summary_table.csv`
- `summary_table.md`
- `metrics_summary.png`
- `social_metrics_summary.png`

## 6. Save Side-by-Side Animation Frames

### Stock PPO vs IDM/MOBIL

```powershell
python -m rl.compare_sb3_highway_ppo_vs_idm `
  --ppo-checkpoint rl/logs/roundabout_stock_curve_compare/ppo_highway_v0 `
  --eval-env-id roundabout-v0 `
  --episodes 10 `
  --duration 20 `
  --save-dir rl/logs/roundabout_stock_ppo_vs_idm_anim `
  --ppo-label stock-ppo `
  --baseline-label IDM/MOBIL `
  --save-frames true `
  --frame-seed 0
```

### Stock PPO vs IDM/MOBIL with DRIFT field overlay

```powershell
python -m rl.compare_sb3_highway_ppo_vs_idm `
  --ppo-checkpoint rl/logs/roundabout_stock_curve_compare/ppo_highway_v0 `
  --eval-env-id roundabout-v0 `
  --episodes 10 `
  --duration 20 `
  --save-dir rl/logs/roundabout_stock_ppo_vs_idm_anim_risk `
  --ppo-label stock-ppo `
  --baseline-label IDM/MOBIL `
  --save-frames-with-risk true `
  --frame-seed 0 `
  --drift-warmup-s 1.0
```

This writes `frames/step_0000.png`, `step_0001.png`, ... plus summary plots.

## 7. Notes on Roundabout vs Highway

- `roundabout-v0` is more scripted than `highway-v0` or `highway-fast-v0`.
- `duration` is important for roundabout runs.
- `vehicles_count` and `vehicles_density` matter less on `roundabout-v0`, because the native scenario hardcodes several spawned vehicles.
- `sv_speed_min`, `sv_speed_max`, and `sv_speed_noise` still affect the surrounding-vehicle initialization through the project traffic wrapper.

# RiskFlow-RL
Physics-Informed Field Propagation and Reinforcement Learning for Socially-Aware Autonomous Driving

### proposed framework:
![Methodology graph](assests/RiskFlow-RL.png)


### PINN training
```
python pinn_risk_field.py --dataset inD --recording all --epochs 3000 --q_smooth --w_data 1.0 --w_phys 0.5 --w_ic 0.2 --w_bc 0.2 --w_smooth 0.3 --n_data 4096 --n_colloc 4096 --pts_per_snap 400 --save_model pinn_inD_all.pt
```
### demonstrations of the numerically solved risk field and PINN generated risk field:
![PINN_examples](assests/DRIFT_PINN_1.gif)

### Dataset processing
```
# Load the recorded trajectories:
python run_track_visualization.py --dataset [name of the dataset (e.g., highD; SQM-N-4)] --recording 00
# Example: load the behaviors from the SQM-N-4 dataset and store into .npz file 
python -m rl.data.historical_extractor --dataset SQM-N-4 --data-dir data/SQM-N-4   --out-path rl/checkpoints/bc_sqm_v3.npz
```
### RL training and evaluation in heterogeneous traffic (PPO only)
```
# 1. Extract ALL recordings into one dataset
python -m rl.data.historical_extractor --data-dir data/exiD --recordings all --out-path rl/checkpoints/bc_dataset_full.npz --horizon-sec 1.5

# 2. BC pretrain on the full dataset
python -m rl.train_bc --dataset rl/checkpoints/bc_dataset_full.npz --out rl/checkpoints/decision_policy_bc.pt

# 3. PPO fine-tune (with the new opportunity-aware reward)
python -m rl.train_decision_ppo --bc-checkpoint rl/checkpoints/decision_policy_bc.pt --out rl/checkpoints/decision_policy_ppo.pt --total-steps 200000

# 4. Evaluate (on both pure car traffic or heterogeneous traffic)
# in heterogenous traffic with truck-trailer occlusion and merging
python highway_test.py --models RL-PPO IDEAM DREAM --rl-decision-checkpoint rl/checkpoints/decision_policy_ppo.pt --steps 250
# in pure car traffic
python highway_test.py --scenario-mode purecar --ego-start-lane center --rl-policy-mode decision --rl-decision-checkpoint rl/checkpoints/decision_policy_ppo.pt --models all --mode single
# in suddent merging scenario: (compare against baseline MPC-CBF)
python uncertainty_merger.py --models "RL-PPO" "IDEAM" --steps 100 --rl-policy-mode ppo --rl-checkpoint rl/checkpoints/ppo_best.pt --save-dir figsave_merger_rl_vs_ideam --save-frames false

```
### Complete Implementation (updated on 25 Apr 2026)
```bash
cd C:/RiskFlow\_RL

# Train BC (if not already trained)
python -m rl.train\_bc --out rl/checkpoints/decision\_policy\_bc.pt

# Train PPO v3
python -m rl.train\_decision\_ppo \\
  --bc-checkpoint rl/checkpoints/decision\_policy\_bc.pt \\
  --out rl/checkpoints/decision\_policy\_ppo\_v3.pt \\
  --total-steps 200000 --rollout-steps 2048 \\
  --entropy-coef 0.05 --lr 1e-4 \\
  --log-path rl/logs/decision\_ppo\_v3\_log.json

# Main paper figure
python -m rl.plot\_training\_curves \\
  --logs rl/logs/decision\_ppo\_v3\_log.json \\
  --labels "DREAM-RL (ours)" \\
  --out figures/ppo\_training.pdf --diagnostic

# Diagnostic breakdown (reveals the reward-dominance issue)
python -m rl.diagnose\_reward\_drop \\
  --logs rl/logs/decision\_ppo\_v3\_log.json \\
  --out figures/reward\_diagnosis\_v3.pdf

# Evaluation — merger scenario
python uncertainty\_merger.py \\
  --rl-policy-mode decision \\
  --rl-decision-checkpoint rl/checkpoints/decision\_policy\_ppo\_v3.pt \\
  --steps 100 --models all --save-dir figsave\_merger\_v3\_rl

# Evaluation — 3-lane dense highway
python highway\_test.py \\
  --rl-policy-mode decision \\
  --rl-decision-checkpoint rl/checkpoints/decision\_policy\_ppo\_v3.pt \\
  --steps 400 --save-dir figsave\_test\_v3\_rl
```

## Datasets used in this project (download links):
[Ubiquitous Traffic Eyes](http://www.seutraffic.com/#/download)

[leveLXData](https://levelxdata.com/)

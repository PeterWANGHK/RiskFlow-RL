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
python uncertainty_merger_DREAM.py --models "RL-PPO" "IDEAM" --steps 100 --rl-policy-mode ppo --rl-checkpoint rl/checkpoints/ppo_best.pt --save-dir figsave_merger_rl_vs_ideam --save-frames false

```
## Datasets used in this project (download links):
[Ubiquitous Traffic Eyes](http://www.seutraffic.com/#/download)

[leveLXData](https://levelxdata.com/)

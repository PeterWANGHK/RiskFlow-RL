# RiskFlow-RL
RiskFlow-RL: Learning Risk and Opportunity-Aware Policies for Social-friendly Autonomous Driving via Continuous Field Propagation

### proposed framework:
![Methodology graph](assests/RiskFlow-RL.png)


### PINN training:
```
python pinn_risk_field.py --dataset inD --recording all --epochs 3000 --q_smooth --w_data 1.0 --w_phys 0.5 --w_ic 0.2 --w_bc 0.2 --w_smooth 0.3 --n_data 4096 --n_colloc 4096 --pts_per_snap 400 --save_model pinn_inD_all.pt
```

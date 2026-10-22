import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.scratch_evaluate_k2_800ep import all_keys, all_c, all_m, all_N0, data
from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ode_func = AdaptivePureConsumerResourceODE(N=48, K=2)
model = HierarchicalPhysicalNeuralODE(ode_func, num_conditions=3, K=2).to(device)
model.load_state_dict(torch.load('Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth', weights_only=False))
model.eval()

with torch.no_grad():
    N_curr = all_N0.clone()
    preds_N = [N_curr.cpu().numpy()]
    for day in range(1, 11):
        N_pred = model.step_forward(N_curr, all_c, all_m, delta_t=0.10)
        preds_N.append(N_pred.cpu().numpy())
        N_curr = N_pred
    preds_N = np.stack(preds_N, axis=1)

all_y_true = []
all_y_pred = []

for i, (nutrient, S, comm) in enumerate(all_keys):
    mask_np = all_m[i].cpu().numpy().astype(bool)
    days = data['timepoints'][nutrient][(S, comm)]
    day_nums = np.array([int(d.replace('Day', '')) for d in days])
    true_rel = data['relative_abundance'][(nutrient, S, comm)].numpy()
    true_od = data['biomass_od'][(nutrient, S, comm)].mean(dim=0).numpy()[day_nums]
    true_abs = true_rel * true_od[:, None]
    pred_abs = preds_N[i, day_nums]
    
    all_y_true.append(true_abs[1:, mask_np].flatten())
    all_y_pred.append(pred_abs[1:, mask_np].flatten())

vec_true = np.concatenate(all_y_true)
vec_pred = np.concatenate(all_y_pred)

r_global, _ = pearsonr(vec_true, vec_pred)
r2_global = r_global**2
cod_global = 1.0 - np.sum((vec_true - vec_pred)**2) / np.sum((vec_true - np.mean(vec_true))**2)

print(f"POOLED ACROSS ALL DATA POINTS (N={len(vec_true)} observations):")
print(f"  Global Pooled Pearson R^2 : {r2_global:.4f} (r = {r_global:.4f})")
print(f"  Global Pooled Nash-Sutcliffe R^2 : {cod_global:.4f}")


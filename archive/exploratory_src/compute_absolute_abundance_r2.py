import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.scratch_evaluate_k2_800ep import all_keys, all_c, all_m, all_N0, data
from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the pure K=2 800-epoch model
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
        
    preds_N = np.stack(preds_N, axis=1) # (99, 11, 48)

# Compute Absolute Abundance R2 for every community
results = []

for i, (nutrient, S, comm) in enumerate(all_keys):
    mask_np = all_m[i].cpu().numpy().astype(bool)
    days = data['timepoints'][nutrient][(S, comm)]
    day_nums = np.array([int(d.replace('Day', '')) for d in days])
    
    true_rel = data['relative_abundance'][(nutrient, S, comm)].numpy() # (T_obs, 48)
    true_od = data['biomass_od'][(nutrient, S, comm)].mean(dim=0).numpy() # (11,)
    true_od_obs = true_od[day_nums] # (T_obs,)
    
    # Absolute Abundance ground truth: true_rel * true_od
    true_abs = true_rel * true_od_obs[:, None] # (T_obs, 48)
    
    # Predicted Absolute Abundance from ODE: preds_N
    pred_abs = preds_N[i, day_nums] # (T_obs, 48)
    
    # Filter to member species across unrolled steps (t >= 1)
    y_true_abs = true_abs[1:, mask_np].flatten()
    y_pred_abs = pred_abs[1:, mask_np].flatten()
    
    # 1. Pearson r and r^2 on absolute abundance
    r_abs, _ = pearsonr(y_true_abs, y_pred_abs)
    r2_abs_pearson = float(r_abs**2) if not np.isnan(r_abs) else 0.0
    
    # 2. Nash-Sutcliffe Coefficient of Determination (1 - SS_res / SS_tot) on absolute abundance
    ss_res = np.sum((y_true_abs - y_pred_abs)**2)
    ss_tot = np.sum((y_true_abs - np.mean(y_true_abs))**2)
    cod_abs = float(1.0 - ss_res / (ss_tot + 1e-9))
    
    # 3. Log-scale Pearson r^2 on absolute abundance
    log_true = np.log10(np.clip(y_true_abs, 1e-6, None))
    log_pred = np.log10(np.clip(y_pred_abs, 1e-6, None))
    r_log, _ = pearsonr(log_true, log_pred)
    r2_log_abs = float(r_log**2) if not np.isnan(r_log) else 0.0
    
    # 4. Day 10 Endpoint Absolute Abundance R2
    y_true_end = true_abs[-1, mask_np]
    y_pred_end = pred_abs[-1, mask_np]
    r_end, _ = pearsonr(y_true_end, y_pred_end)
    r2_end_abs = float(r_end**2) if not np.isnan(r_end) else 0.0
    
    # Also relative abundance metrics for direct comparison
    true_rel_flat = true_rel[1:, mask_np].flatten()
    pred_bio = preds_N[i, day_nums].sum(axis=-1, keepdims=True)
    pred_rel = pred_abs / np.where(pred_bio > 0, pred_bio, 1.0)
    pred_rel_flat = pred_rel[1:, mask_np].flatten()
    
    r_rel, _ = pearsonr(true_rel_flat, pred_rel_flat)
    r2_rel_pearson = float(r_rel**2) if not np.isnan(r_rel) else 0.0
    cod_rel = float(1.0 - np.sum((true_rel_flat - pred_rel_flat)**2) / (np.sum((true_rel_flat - np.mean(true_rel_flat))**2) + 1e-9))
    
    results.append({
        'nutrient': nutrient,
        'S': S,
        'comm': comm,
        'r2_abs_pearson': r2_abs_pearson,
        'cod_abs_nash': cod_abs,
        'r2_abs_log': r2_log_abs,
        'r2_abs_end': r2_end_abs,
        'r2_rel_pearson': r2_rel_pearson,
        'cod_rel_nash': cod_rel
    })

df = pd.DataFrame(results)

print("="*85)
print("ABSOLUTE ABUNDANCE (N_i = Rel * OD) PREDICTABILITY REPORT FOR K=2 PHYS-ODE")
print("="*85)

print("\n--- GLOBAL PERFORMANCE ACROSS ALL 99 COMMUNITIES ---")
print(f"1. Absolute Abundance Pearson R^2 (Trajectory)  : Mean = {df['r2_abs_pearson'].mean():.4f} | Median = {df['r2_abs_pearson'].median():.4f}")
print(f"2. Absolute Abundance Nash-Sutcliffe R^2 (CoD)  : Mean = {df['cod_abs_nash'].mean():.4f} | Median = {df['cod_abs_nash'].median():.4f}")
print(f"3. Absolute Abundance Log10 Pearson R^2         : Mean = {df['r2_abs_log'].mean():.4f} | Median = {df['r2_abs_log'].median():.4f}")
print(f"4. Absolute Abundance Day 10 Endpoint R^2       : Mean = {df['r2_abs_end'].mean():.4f} | Median = {df['r2_abs_end'].median():.4f}")

print("\n--- COMPARISON WITH RELATIVE ABUNDANCE ONLY ---")
print(f"Relative Abundance Pearson R^2                  : Mean = {df['r2_rel_pearson'].mean():.4f} | Median = {df['r2_rel_pearson'].median():.4f}")
print(f"Relative Abundance Nash-Sutcliffe R^2 (CoD)     : Mean = {df['cod_rel_nash'].mean():.4f} | Median = {df['cod_rel_nash'].median():.4f}")

print("\n--- BREAKDOWN BY NUTRIENT MEDIA (ABSOLUTE ABUNDANCE PEARSON R^2) ---")
print(df.groupby('nutrient')[['r2_abs_pearson', 'cod_abs_nash', 'r2_abs_log', 'r2_abs_end']].agg(['mean', 'median']))

print("\n--- BREAKDOWN BY SPECIES RICHNESS S (ABSOLUTE ABUNDANCE PEARSON R^2) ---")
print(df.groupby('S')[['r2_abs_pearson', 'cod_abs_nash', 'r2_abs_log', 'r2_abs_end']].agg(['mean', 'median']))


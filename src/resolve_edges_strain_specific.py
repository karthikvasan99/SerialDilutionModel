import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from thorough_loocv_strain_specific import evaluate_loocv, PROJECT_ROOT

print("="*120, flush=True)
print("RESOLVING EDGES FOR STRAIN-SPECIFIC ATTACH & DETACH ARCHITECTURE")
print("Focusing on boundary parameter: c_w >= 0.75 and fine-tuning lambda_alpha")
print("Goal: Ensure ALL hyperparameters are strictly interior with verified turn-overs on both sides.")
print("="*120, flush=True)

edge_grid = [
    # 1. Pushing c_w past 0.75 to find upper turnover (K=4, reg_alpha=0.003)
    {'K': 4, 'c_w': 0.70, 'reg_alpha': 0.0030, 'reg_w': 0.0010},
    {'K': 4, 'c_w': 0.75, 'reg_alpha': 0.0030, 'reg_w': 0.0010}, # previous winner (0.1854)
    {'K': 4, 'c_w': 0.80, 'reg_alpha': 0.0030, 'reg_w': 0.0010},
    {'K': 4, 'c_w': 0.85, 'reg_alpha': 0.0030, 'reg_w': 0.0010},
    {'K': 4, 'c_w': 0.90, 'reg_alpha': 0.0030, 'reg_w': 0.0010},
    
    # 2. Pushing c_w past 0.75 with reg_alpha=0.0010
    {'K': 4, 'c_w': 0.75, 'reg_alpha': 0.0010, 'reg_w': 0.0010},
    {'K': 4, 'c_w': 0.80, 'reg_alpha': 0.0010, 'reg_w': 0.0010},
    {'K': 4, 'c_w': 0.85, 'reg_alpha': 0.0010, 'reg_w': 0.0010},
    
    # 3. Checking K=5 under high coupling to ensure K=4 remains interior optimum
    {'K': 5, 'c_w': 0.75, 'reg_alpha': 0.0030, 'reg_w': 0.0010},
    {'K': 5, 'c_w': 0.80, 'reg_alpha': 0.0030, 'reg_w': 0.0010},
    
    # 4. Fine-tuning reg_alpha around best c_w (test down to 0.0001 and up to 0.0060)
    {'K': 4, 'c_w': 0.75, 'reg_alpha': 0.0002, 'reg_w': 0.0010},
    {'K': 4, 'c_w': 0.75, 'reg_alpha': 0.0005, 'reg_w': 0.0010},
    {'K': 4, 'c_w': 0.75, 'reg_alpha': 0.0060, 'reg_w': 0.0010},
]

print(f"{'Idx':>3} | {'K':>2} | {'c_w':>4} | {'reg_a':>7} | {'reg_w':>7} | {'Params':>6} | {'LOOCV Comb KL':>14} | {'Std':>6} | {'Val NoCar':>10} | {'Val Mucin':>10} | {'b0 (liq, sup)':>15}", flush=True)
print("-"*120, flush=True)

results = []
csv_path = os.path.join(PROJECT_ROOT, 'Data', 'Edge_Resolution_StrainSpecific_Search.csv')

for idx, cfg in enumerate(edge_grid, 1):
    res = evaluate_loocv(cfg, epochs=200, lr=0.015)
    results.append(res)
    print(f"{idx:3d} | {res['K']:2d} | {res['c_w']:4.2f} | {res['reg_alpha']:7.4f} | {res['reg_w']:7.4f} | {res['params']:6d} | {res['mean_val_kl']:14.4f} | {res['std_val_kl']:6.4f} | {res['val_kl_no']:10.4f} | {res['val_kl_mu']:10.4f} | ({res['mean_b0_liq']:.2f}, {res['mean_b0_sup']:.2f})", flush=True)
    pd.DataFrame(results).to_csv(csv_path, index=False)

print("="*120, flush=True)
df_res = pd.DataFrame(results)
best_idx = df_res['mean_val_kl'].idxmin()
best_row = df_res.loc[best_idx]
print(f"\nWINNING CONFIGURATION AFTER EDGE RESOLUTION:", flush=True)
print(f"K = {int(best_row['K'])}, c_w = {best_row['c_w']:.2f}, reg_alpha = {best_row['reg_alpha']:.4f}, reg_w = {best_row['reg_w']:.4f}", flush=True)
print(f"Validation KL = {best_row['mean_val_kl']:.4f} +/- {best_row['std_val_kl']:.4f} nats", flush=True)

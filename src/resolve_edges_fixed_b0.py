import os
import sys
import torch
import numpy as np
import pandas as pd
from thorough_loocv_fixed_b0 import evaluate_loocv, PROJECT_ROOT

print("="*120, flush=True)
print("RESOLVING EDGES FOR STRAIN-SPECIFIC FIXED-B0 MODEL")
print("Exploring: c_w > 0.75, b0 > 2.2, reg_alpha < 0.003, and K in {3,4,5,6}")
print("="*120, flush=True)

edge_grid = [
    # 1. Pushing c_w past 0.75 (0.80, 0.85) on K=4, b0=1.6
    {'K': 4, 'c_w': 0.70, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
    {'K': 4, 'c_w': 0.75, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001}, # previous winner (0.1906)
    {'K': 4, 'c_w': 0.80, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
    {'K': 4, 'c_w': 0.85, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
    
    # 2. Pushing b0 past 2.2 (2.5, 2.8, 3.2) on K=4, c_w=0.75
    {'K': 4, 'c_w': 0.75, 'b0_val': 2.00, 'reg_alpha': 0.010, 'reg_w': 0.001},
    {'K': 4, 'c_w': 0.75, 'b0_val': 2.40, 'reg_alpha': 0.010, 'reg_w': 0.001},
    {'K': 4, 'c_w': 0.75, 'b0_val': 2.80, 'reg_alpha': 0.010, 'reg_w': 0.001},
    {'K': 4, 'c_w': 0.75, 'b0_val': 3.20, 'reg_alpha': 0.010, 'reg_w': 0.001},
    
    # 3. Pushing reg_alpha below 0.003 (0.001, 0.0005) on K=4, c_w=0.75, b0=2.0
    {'K': 4, 'c_w': 0.75, 'b0_val': 2.00, 'reg_alpha': 0.0005, 'reg_w': 0.001},
    {'K': 4, 'c_w': 0.75, 'b0_val': 2.00, 'reg_alpha': 0.0010, 'reg_w': 0.001},
    {'K': 4, 'c_w': 0.75, 'b0_val': 2.00, 'reg_alpha': 0.0030, 'reg_w': 0.001},
    
    # 4. Checking K in {3, 5, 6} under high coupling c_w=0.75
    {'K': 3, 'c_w': 0.75, 'b0_val': 2.00, 'reg_alpha': 0.0010, 'reg_w': 0.001},
    {'K': 5, 'c_w': 0.75, 'b0_val': 2.00, 'reg_alpha': 0.0010, 'reg_w': 0.001},
    {'K': 6, 'c_w': 0.75, 'b0_val': 2.00, 'reg_alpha': 0.0010, 'reg_w': 0.001},
]

print(f"{'Idx':>3} | {'K':>2} | {'c_w':>4} | {'b0':>4} | {'reg_a':>7} | {'reg_w':>6} | {'Params':>6} | {'LOOCV Comb KL':>14} | {'Std':>6} | {'Val NoCar':>10} | {'Val Mucin':>10}", flush=True)
print("-"*120, flush=True)

results = []
csv_path = os.path.join(PROJECT_ROOT, 'Data', 'Edge_Resolution_StrainSpecific_FixedB0.csv')

for idx, cfg in enumerate(edge_grid, 1):
    res = evaluate_loocv(cfg, epochs=200, lr=0.015)
    results.append(res)
    print(f"{idx:3d} | {res['K']:2d} | {res['c_w']:4.2f} | {res['b0_val']:4.1f} | {res['reg_alpha']:7.4f} | {res['reg_w']:6.4f} | {res['params']:6d} | {res['mean_val_kl']:14.4f} | {res['std_val_kl']:6.4f} | {res['val_kl_no']:10.4f} | {res['val_kl_mu']:10.4f}", flush=True)
    pd.DataFrame(results).to_csv(csv_path, index=False)

print("="*120, flush=True)
df_res = pd.DataFrame(results)
best_idx = df_res['mean_val_kl'].idxmin()
best_row = df_res.loc[best_idx]
print(f"\nWINNING CONFIGURATION AFTER EDGE RESOLUTION:", flush=True)
print(f"K = {int(best_row['K'])}, c_w = {best_row['c_w']:.2f}, b0 = {best_row['b0_val']:.2f}, reg_alpha = {best_row['reg_alpha']:.4f}, reg_w = {best_row['reg_w']:.4f}", flush=True)
print(f"Validation KL = {best_row['mean_val_kl']:.4f} +/- {best_row['std_val_kl']:.4f} nats", flush=True)

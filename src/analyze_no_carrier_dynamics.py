import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location='cpu', weights_only=False)

no_c_tensor = data['no_c_tensor'].numpy()       # (3, 6, 50)
plain_c_tensor = data['plain_c_tensor'].numpy() # (3, 6, 50)
plain_s_tensor = data['plain_s_tensor'].numpy() # (3, 6, 50)
mucin_c_tensor = data['mucin_c_tensor'].numpy() # (3, 6, 50)
mucin_s_tensor = data['mucin_s_tensor'].numpy() # (3, 6, 50)
strains = data['strains']

print("--- 1. Why is persistence in No Carrier only 0.1098? ---")
# Check how much community changes between P1, P2, P3, P4, P5, P6
for name, tensor in [("No Carrier", no_c_tensor), ("Plain Supernatant", plain_s_tensor), ("Mucin Supernatant", mucin_s_tensor)]:
    print(f"\nCondition: {name}")
    # Mean abundance across replicates at each passage
    mean_p = tensor.mean(axis=0) # (6, 50)
    for p in range(5):
        r, _ = pearsonr(mean_p[p], mean_p[p+1])
        l1 = np.abs(mean_p[p] - mean_p[p+1]).sum() / 2.0 # Bray-Curtis / total variation
        print(f"  P{p+1} -> P{p+2}: Pearson r = {r:.4f}, Total Variation Distance = {l1:.4f}")
    
    # Also compare P2 to P6 directly
    r_2_6, _ = pearsonr(mean_p[1], mean_p[5])
    l1_2_6 = np.abs(mean_p[1] - mean_p[5]).sum() / 2.0
    print(f"  Overall P2 -> P6 shift: Pearson r = {r_2_6:.4f}, Total Variation Distance = {l1_2_6:.4f}")

    # Variance across passages P3..P6 (total sum of squares)
    t_eval = tensor[:, 2:, :].flatten()
    print(f"  Eval Target (P3..P6) Total Variance (SS_tot / N): {np.var(t_eval):.6f}")

print("\n--- 2. Dominant Strains in No Carrier vs Carrier ---")
top_no_c = np.argsort(no_c_tensor[:, -1, :].mean(axis=0))[::-1][:5]
top_plain = np.argsort(plain_c_tensor[:, -1, :].mean(axis=0))[::-1][:5]
top_mucin = np.argsort(mucin_c_tensor[:, -1, :].mean(axis=0))[::-1][:5]

print("Top 5 strains at P6 in No Carrier:")
for idx in top_no_c:
    print(f"  {strains[idx]}: {no_c_tensor[:, -1, idx].mean():.3f} (Plain: {plain_c_tensor[:, -1, idx].mean():.3f}, Mucin: {mucin_c_tensor[:, -1, idx].mean():.3f})")

print("\nTop 5 strains at P6 in Plain Biofilm:")
for idx in top_plain:
    print(f"  {strains[idx]}: {plain_c_tensor[:, -1, idx].mean():.3f} (No Carrier: {no_c_tensor[:, -1, idx].mean():.3f})")

import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the raw 3-vessel tensors (vessels R1, R2, R3)
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location='cpu', weights_only=False)

no_c_tensor = data['no_c_tensor'].numpy()       # shape (3, 6, 50) -> (vessel, passage, strain)
plain_c_tensor = data['plain_c_tensor'].numpy() # shape (3, 6, 50)
plain_s_tensor = data['plain_s_tensor'].numpy() # shape (3, 6, 50)
mucin_c_tensor = data['mucin_c_tensor'].numpy() # shape (3, 6, 50)
mucin_s_tensor = data['mucin_s_tensor'].numpy() # shape (3, 6, 50)

def compute_r2(true_vals, pred_vals):
    diff_sq = np.sum((true_vals - pred_vals)**2)
    tot_sq = np.sum((true_vals - np.mean(true_vals))**2)
    return 1.0 - diff_sq / tot_sq

# Combine compartments for carrier conditions
plain_full = np.concatenate([plain_c_tensor, plain_s_tensor], axis=-1) # (3, 6, 100)
mucin_full = np.concatenate([mucin_c_tensor, mucin_s_tensor], axis=-1) # (3, 6, 100)
no_c_full = no_c_tensor # (3, 6, 50)

conditions = {
    'No Carrier': no_c_full,
    'Plain Agar': plain_full,
    'Mucin Agar': mucin_full
}

print("="*85)
print("EMPIRICAL COMPARISON: REPLICATE-TO-REPLICATE R2 vs HELD-OUT VESSEL LOOCV")
print("="*85)

for cond_name, tensor in conditions.items():
    print(f"\nCondition: {cond_name}")
    # Pairwise Replicate R2
    r2_12 = compute_r2(tensor[1].flatten(), tensor[0].flatten())
    r2_13 = compute_r2(tensor[2].flatten(), tensor[0].flatten())
    r2_23 = compute_r2(tensor[2].flatten(), tensor[1].flatten())
    r2_pairwise = [r2_12, r2_13, r2_23]
    
    r_12, _ = pearsonr(tensor[1].flatten(), tensor[0].flatten())
    r_13, _ = pearsonr(tensor[2].flatten(), tensor[0].flatten())
    r_23, _ = pearsonr(tensor[2].flatten(), tensor[1].flatten())
    r_pairwise = [r_12, r_13, r_23]
    
    print(f"  Pairwise Replicate R2 (R1 vs R2, R1 vs R3, R2 vs R3):")
    print(f"    R1 vs R2: R2 = {r2_12:.4f} (r = {r_12:.4f})")
    print(f"    R1 vs R3: R2 = {r2_13:.4f} (r = {r_13:.4f})")
    print(f"    R2 vs R3: R2 = {r2_23:.4f} (r = {r_23:.4f})")
    print(f"    Mean Pairwise Replicate R2 = {np.mean(r2_pairwise):.4f} +/- {np.std(r2_pairwise):.4f}")
    
    # Leave-One-Vessel-Out: Predict held-out vessel using the MEAN of the other 2 vessels
    # Fold 1: Predict R3 using mean(R1, R2)
    pred_r3 = (tensor[0] + tensor[1]) / 2.0
    r2_loo_r3 = compute_r2(tensor[2].flatten(), pred_r3.flatten())
    r_loo_r3, _ = pearsonr(tensor[2].flatten(), pred_r3.flatten())
    
    # Fold 2: Predict R2 using mean(R1, R3)
    pred_r2 = (tensor[0] + tensor[2]) / 2.0
    r2_loo_r2 = compute_r2(tensor[1].flatten(), pred_r2.flatten())
    r_loo_r2, _ = pearsonr(tensor[1].flatten(), pred_r2.flatten())
    
    # Fold 3: Predict R1 using mean(R2, R3)
    pred_r1 = (tensor[1] + tensor[2]) / 2.0
    r2_loo_r1 = compute_r2(tensor[0].flatten(), pred_r1.flatten())
    r_loo_r1, _ = pearsonr(tensor[0].flatten(), pred_r1.flatten())
    
    loo_emp_r2 = [r2_loo_r1, r2_loo_r2, r2_loo_r3]
    loo_emp_r = [r_loo_r1, r_loo_r2, r_loo_r3]
    
    print(f"  Predicting Held-Out Vessel using Mean of Training Vessels:")
    print(f"    Held-out R1: R2 = {r2_loo_r1:.4f} (r = {r_loo_r1:.4f})")
    print(f"    Held-out R2: R2 = {r2_loo_r2:.4f} (r = {r_loo_r2:.4f})")
    print(f"    Held-out R3: R2 = {r2_loo_r3:.4f} (r = {r_loo_r3:.4f})")
    print(f"    Mean Empirical Replicate Baseline R2 = {np.mean(loo_emp_r2):.4f} +/- {np.std(loo_emp_r2):.4f} (r = {np.mean(loo_emp_r):.4f})")


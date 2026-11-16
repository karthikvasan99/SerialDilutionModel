import os
import sys
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from test_start_from_p2 import (
    plain_c_tensor, plain_s_tensor, mucin_c_tensor, mucin_s_tensor,
    p_c_p2, p_s_p2, p_c_auto, p_s_auto, compute_kl
)

print("="*85)
print("PASSAGE-SPECIFIC KL DIVERGENCE: CRM vs PERSISTENCE (FROM P2)")
print("="*85)

for cond_name, t_c, t_s, p_c_m, p_s_m, p_c_a, p_s_a in [
    ("Plain Agar", plain_c_tensor, plain_s_tensor, p_c_p2, p_s_p2, p_c_auto, p_s_auto),
    ("Mucin Agar", mucin_c_tensor, mucin_s_tensor, p_c_p2, p_s_p2, p_c_auto, p_s_auto)
]:
    print(f"\nCondition: {cond_name}")
    t_c_np = t_c.cpu().numpy()
    t_s_np = t_s.cpu().numpy()
    p_c_np = p_c_m.cpu().numpy()
    p_s_np = p_s_m.cpu().numpy()
    p_c_a_np = p_c_a.cpu().numpy()
    p_s_a_np = p_s_a.cpu().numpy()
    
    # Persistence from P2
    pers_c_np = np.repeat(t_c_np[:, 1:2, :], 6, axis=1)
    pers_s_np = np.repeat(t_s_np[:, 1:2, :], 6, axis=1)
    
    for p_idx, p_name in enumerate(["P3 (Day 9)", "P4 (Day 12)", "P5 (Day 15)", "P6 (Day 18)"], start=2):
        # KL for Carrier Biofilm
        kl_c_m = compute_kl(t_c_np[:, p_idx, :], p_c_np[:, p_idx, :])
        kl_c_p = compute_kl(t_c_np[:, p_idx, :], pers_c_np[:, p_idx, :])
        kl_c_a = compute_kl(t_c_np[:, p_idx, :], p_c_a_np[:, p_idx, :])
        
        # KL for Supernatant
        kl_s_m = compute_kl(t_s_np[:, p_idx, :], p_s_np[:, p_idx, :])
        kl_s_p = compute_kl(t_s_np[:, p_idx, :], pers_s_np[:, p_idx, :])
        kl_s_a = compute_kl(t_s_np[:, p_idx, :], p_s_a_np[:, p_idx, :])
        
        # Mean across compartments
        kl_mean_from_p2 = (kl_c_m + kl_s_m) / 2.0
        kl_mean_pers = (kl_c_p + kl_s_p) / 2.0
        kl_mean_auto = (kl_c_a + kl_s_a) / 2.0
        
        win_p2 = "CRM WINS!" if kl_mean_from_p2 < kl_mean_pers else "Persistence wins"
        win_auto = "CRM WINS!" if kl_mean_auto < kl_mean_pers else "Persistence wins"
        
        print(f"  {p_name}:")
        print(f"    Persistence KL:          {kl_mean_pers:.4f}")
        print(f"    CRM Autonomous from P0:  {kl_mean_auto:.4f}  --> {win_auto}")
        print(f"    CRM Forecast from P2:    {kl_mean_from_p2:.4f}  --> {win_p2}")

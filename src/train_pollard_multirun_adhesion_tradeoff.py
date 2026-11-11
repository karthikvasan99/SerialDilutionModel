"""
Train Multiple Random Initializations of Dual-Theta Unified CRM (K=3)
to obtain Robust Trait Distributions and Error Bars for Microbes:
- Physical Adhesion Ratio (k_attach^mucin / k_shed) [X-axis]
- Biofilm Metabolic Shift (||Theta^(c)|| / ||Theta^(s)||) [Y-axis]

Generates:
1. Pollard_Coupled_Biofilm_Adhesion_vs_Metabolic_Tradeoff_ErrorBars.png (Dedicated focus on Plot D with 2D error bars, phylum coloring, marginals, and regression CI)
2. Updates Plot D in Pollard_Coupled_Biofilm_Adhesion_Traits.png (4-panel figure)
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr, linregress

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from poster_fig2_dual_theta import (
    DualThetaUnifiedCRM,
    train_dual_theta_model,
    evaluate_3_conditions,
    strains_keep,
    no_c_tensor,
    plain_c_tensor,
    plain_s_tensor,
    mucin_c_tensor,
    mucin_s_tensor,
    device
)

def run_multi_seed_fits(seeds=[42, 101, 102, 103, 104, 105, 106, 107, 108, 109], epochs=200):
    cache_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_DualTheta_MultiRun_10Seeds.pt')
    
    if os.path.exists(cache_path):
        print(f"Loading cached multi-run results from {cache_path}...")
        results = torch.load(cache_path, map_location='cpu', weights_only=False)
        return results
        
    print(f"Starting {len(seeds)} independent training runs (epochs={epochs})...")
    runs_data = []
    
    t_start = time.time()
    for s_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"\n--- Run {s_idx + 1}/{len(seeds)} (Seed {seed}) ---")
        model = DualThetaUnifiedCRM(N=len(strains_keep), K=3).to(device)
        t0 = time.time()
        model, best_loss = train_dual_theta_model(model, epochs=epochs, lr=0.015, verbose=False)
        elapsed = time.time() - t0
        
        model.eval()
        with torch.no_grad():
            Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params()
            r2 = evaluate_3_conditions(model, Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta)
            
            k_shed_np = k_shed.cpu().numpy()
            k_att_plain_np = k_attach[0].cpu().numpy()
            k_att_mucin_np = k_attach[1].cpu().numpy()
            Theta_c_np = Theta_c.cpu().numpy()
            Theta_s_np = Theta_s.cpu().numpy()
            
            ratio_mucin = k_att_mucin_np / (k_shed_np + 1e-4)
            ratio_plain = k_att_plain_np / (k_shed_np + 1e-4)
            th_c_norm = np.linalg.norm(Theta_c_np, axis=1)
            th_s_norm = np.linalg.norm(Theta_s_np, axis=1)
            theta_ratio = th_c_norm / (th_s_norm + 1e-4)
            
            r_val, p_val = pearsonr(np.log10(ratio_mucin), theta_ratio)
            rho_val, _ = spearmanr(ratio_mucin, theta_ratio)
            
            print(f"Seed {seed:3d} finished in {elapsed:.1f}s | R2: No-C={r2[0]:.2f}, Plain={r2[1]:.2f}, Mucin={r2[2]:.2f} | Trade-off r={r_val:.2f} (p={p_val:.3e}), rho={rho_val:.2f}")
            
            runs_data.append({
                'seed': seed,
                'r2': r2,
                'best_loss': best_loss,
                'k_shed': k_shed_np,
                'k_att_plain': k_att_plain_np,
                'k_att_mucin': k_att_mucin_np,
                'ratio_mucin': ratio_mucin,
                'ratio_plain': ratio_plain,
                'th_c_norm': th_c_norm,
                'th_s_norm': th_s_norm,
                'theta_ratio': theta_ratio,
                'pearson_r': r_val,
                'p_val': p_val,
                'spearman_rho': rho_val
            })
            
    total_time = time.time() - t_start
    print(f"\nAll {len(seeds)} runs completed in {total_time/60.0:.2f} minutes!")
    
    results = {
        'seeds': seeds,
        'runs_data': runs_data,
        'strains': strains_keep
    }
    torch.save(results, cache_path)
    print(f"Saved multi-run results to {cache_path}")
    return results

if __name__ == '__main__':
    run_multi_seed_fits()

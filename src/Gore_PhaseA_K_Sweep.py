"""
Phase A: Systematic Latent Dimension (K) Sweep for Gore Emergent Phases
Searches K in increments of 1: K in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].

Evaluation Metric: Multi-step autoregressive Linear Pearson R2 (on relative abundance fractions x_i in [0, 1]).
Loss Function: Pure KL Divergence (nn.KLDivLoss).
"""

import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'Figures')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results')

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(pt_path):
    data = torch.load(pt_path, weights_only=False)
    rel_dict = data['relative_abundance']
    masks_dict = data['presence_masks']
    metadata_df = data['metadata']
    species_list = data['species']
    
    cond_map = {'LN': 0, 'MN': 1, 'HN': 2}
    
    batch_11_seqs, batch_11_conds, batch_11_keys = [], [], []
    batch_4_seqs, batch_4_conds, batch_4_keys = [], [], []
    
    for (cond, s_val, c_name), tensor in rel_dict.items():
        c_idx = cond_map[cond]
        key = (cond, s_val, c_name)
        if tensor.size(0) == 11:
            batch_11_seqs.append(tensor)
            batch_11_conds.append(c_idx)
            batch_11_keys.append(key)
        else:
            batch_4_seqs.append(tensor)
            batch_4_conds.append(c_idx)
            batch_4_keys.append(key)
            
    b11_seqs = torch.stack(batch_11_seqs).to(device)
    b11_conds = torch.tensor(batch_11_conds, dtype=torch.long).to(device)
    
    b4_seqs = torch.stack(batch_4_seqs).to(device)
    b4_conds = torch.tensor(batch_4_conds, dtype=torch.long).to(device)
    
    return {
        'b11_seqs': b11_seqs,
        'b11_conds': b11_conds,
        'b11_keys': batch_11_keys,
        'b4_seqs': b4_seqs,
        'b4_conds': b4_conds,
        'b4_keys': batch_4_keys,
        'masks': masks_dict,
        'metadata': metadata_df,
        'species': species_list,
        'cond_map': cond_map,
        'rel_dict': rel_dict
    }


def train_single_k(dataset, K, epochs=350, lr=0.008, l2_lambda=0.01, ode_step_size=0.1, seed=42):
    set_seed(seed)
    num_species = len(dataset['species'])
    num_conditions = len(dataset['cond_map'])
    
    model = LatentAssemblyNeuralODE(
        num_species=num_species,
        num_latent_dims=K,
        num_conditions=num_conditions,
        default_dilution_factor=30.0
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    b11_s = dataset['b11_seqs']
    b11_c = dataset['b11_conds']
    b4_s = dataset['b4_seqs']
    b4_c = dataset['b4_conds']
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Scheduled sampling decay (1.0 -> 0.0)
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        # Batch 11
        x_curr = b11_s[:, 0, :]
        batch11_loss = 0.0
        for t in range(1, b11_s.size(1)):
            log_pred, x_pred, _, _ = model(x_curr, b11_c, ode_method='euler', ode_step_size=ode_step_size)
            target = b11_s[:, t, :]
            batch11_loss += criterion(log_pred, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred
            
        # Batch 4
        x_curr_4 = b4_s[:, 0, :]
        batch4_loss = 0.0
        for t in range(1, b4_s.size(1)):
            log_pred_4, x_pred_4, _, _ = model(x_curr_4, b4_c, ode_method='euler', ode_step_size=ode_step_size)
            target_4 = b4_s[:, t, :]
            batch4_loss += criterion(log_pred_4, target_4)
            x_curr_4 = tf_ratio * target_4 + (1.0 - tf_ratio) * x_pred_4
            
        # L2 Regularization on biological parameters Theta and W
        l2_reg = sum(p.pow(2.0).sum() for name, p in model.named_parameters() if 'Theta' in name or 'W' in name)
        
        loss = (batch11_loss + batch4_loss) / (b11_s.size(1) - 1 + b4_s.size(1) - 1) + l2_lambda * l2_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
    train_duration = time.time() - start_time
    return model, train_duration


def evaluate_model_rollouts(model, dataset, K, ode_step_size=0.1):
    """
    Evaluates pure autoregressive multi-step rollouts for all 99 communities.
    Computes both Linear Pearson R2 and Log Pearson R2.
    """
    model.eval()
    results = []
    
    rel_dict = dataset['rel_dict']
    masks_dict = dataset['masks']
    cond_map = dataset['cond_map']
    meta = dataset['metadata']
    
    with torch.no_grad():
        for (cond, s_val, c_name), true_traj in rel_dict.items():
            c_idx = torch.tensor([cond_map[cond]], dtype=torch.long, device=device)
            mask = masks_dict[(s_val, c_name)].numpy()
            
            # Autoregressive forward rollout
            curr_state = true_traj[0:1, :].to(device)
            pred_traj = [curr_state.cpu().squeeze(0).numpy()]
            
            for t in range(1, true_traj.size(0)):
                _, next_state, _, _ = model(curr_state, c_idx, ode_method='euler', ode_step_size=ode_step_size)
                pred_traj.append(next_state.squeeze(0).cpu().numpy())
                curr_state = next_state
                
            pred_mat = np.array(pred_traj)
            true_mat = true_traj.numpy()
            
            t_un = true_mat[1:, mask].flatten()
            p_un = pred_mat[1:, mask].flatten()
            
            # Linear Pearson R2
            r_lin, _ = pearsonr(t_un, p_un)
            r2_lin = float(r_lin**2) if not np.isnan(r_lin) else 0.0
            
            # Log Pearson R2
            t_log = np.log10(np.clip(t_un, 1e-6, None))
            p_log = np.log10(np.clip(p_un, 1e-6, None))
            r_log, _ = pearsonr(t_log, p_log)
            r2_log = float(r_log**2) if not np.isnan(r_log) else 0.0
            
            # Day 10 Endpoint Linear R2
            t_end = true_mat[-1, mask]
            p_end = pred_mat[-1, mask]
            r_end, _ = pearsonr(t_end, p_end)
            r2_end_lin = float(r_end**2) if not np.isnan(r_end) else 0.0
            
            # Stability classification from metadata
            sub = meta[(meta['S'] == s_val) & (meta['community_name'] == c_name)]
            is_fluc = bool(sub[f'{cond}_fluctuating'].values[0]) if len(sub) > 0 else False
            emp_cv = float(sub[f'{cond}_CV'].values[0]) if len(sub) > 0 else 0.0
            
            results.append({
                'K': K,
                'condition': cond,
                'S': s_val,
                'community': c_name,
                'r2_linear': r2_lin,
                'r2_log': r2_log,
                'r2_endpoint_linear': r2_end_lin,
                'emp_cv': emp_cv,
                'is_fluctuating': is_fluc,
                'state': 'Chaotic/Fluctuating' if is_fluc else 'Stable'
            })
            
    return pd.DataFrame(results)


def plot_k_sweep_results(summary_df, detailed_df):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    k_vals = summary_df['K'].values
    
    # --- PANEL 1: Overall Mean and Median Linear R2 vs K ---
    ax1 = axes[0, 0]
    ax1.plot(k_vals, summary_df['mean_r2_linear'], 'o-', color='#2c3e50', lw=2.5, ms=8, label='Mean Linear $R^2$')
    ax1.plot(k_vals, summary_df['median_r2_linear'], 's--', color='#3498db', lw=2.0, ms=7, label='Median Linear $R^2$')
    best_k = summary_df.loc[summary_df['mean_r2_linear'].idxmax()]['K']
    best_r2 = summary_df['mean_r2_linear'].max()
    ax1.axvline(best_k, color='red', linestyle=':', lw=1.5, label=f'Optimal K={int(best_k)} ($R^2={best_r2:.3f}$)')
    ax1.set_xlabel('Number of Latent Resource Pools ($K$)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Autoregressive Linear $R^2$', fontsize=13, fontweight='bold')
    ax1.set_title('A) Global Trajectory Predictability vs. Capacity $K$', fontsize=15, fontweight='bold')
    ax1.set_xticks(k_vals)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=11)
    
    # --- PANEL 2: Linear R2 by Nutrient Regime vs K ---
    ax2 = axes[0, 1]
    ax2.plot(k_vals, summary_df['ln_r2_linear'], 'o-', color='#2ecc71', lw=2.2, ms=7, label='Low Nutrient (LN)')
    ax2.plot(k_vals, summary_df['mn_r2_linear'], 's-', color='#f39c12', lw=2.2, ms=7, label='Medium Nutrient (MN)')
    ax2.plot(k_vals, summary_df['hn_r2_linear'], '^-', color='#e74c3c', lw=2.2, ms=7, label='High Nutrient (HN)')
    ax2.set_xlabel('Number of Latent Resource Pools ($K$)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Linear $R^2$', fontsize=13, fontweight='bold')
    ax2.set_title('B) Predictability Across Nutrient Regimes vs. $K$', fontsize=15, fontweight='bold')
    ax2.set_xticks(k_vals)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=11)
    
    # --- PANEL 3: Linear R2 by Dynamic State (Stable vs Chaotic) ---
    ax3 = axes[1, 0]
    ax3.plot(k_vals, summary_df['stable_r2_linear'], 'o-', color='#16a085', lw=2.2, ms=7, label='Stable Communities (n=50)')
    ax3.plot(k_vals, summary_df['chaotic_r2_linear'], 's--', color='#c0392b', lw=2.2, ms=7, label='Chaotic/Fluctuating (n=49)')
    ax3.set_xlabel('Number of Latent Resource Pools ($K$)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Linear $R^2$', fontsize=13, fontweight='bold')
    ax3.set_title('C) Stable vs. Chaotic Dynamics Fit vs. $K$', fontsize=15, fontweight='bold')
    ax3.set_xticks(k_vals)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(fontsize=11)
    
    # --- PANEL 4: Linear R2 vs S=12 Bifurcation Communities in HN ---
    ax4 = axes[1, 1]
    ax4.plot(k_vals, summary_df['s12_hn_stable_r2'], 'o-', color='#2980b9', lw=2.2, ms=7, label='S=12 HN Stable (3 comms)')
    ax4.plot(k_vals, summary_df['s12_hn_chaotic_r2'], 's--', color='#d35400', lw=2.2, ms=7, label='S=12 HN Chaotic (5 comms)')
    ax4.set_xlabel('Number of Latent Resource Pools ($K$)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Linear $R^2$', fontsize=13, fontweight='bold')
    ax4.set_title('D) The S=12 Bifurcation Point (HN) vs. $K$', fontsize=15, fontweight='bold')
    ax4.set_xticks(k_vals)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(fontsize=11)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'Gore_PhaseA_K_Sweep_Comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[FIGURE] Saved K sweep comparison figure to:\n  {fig_path}")


def main():
    pt_path = os.path.join(DATA_DIR, 'Gore_processed_data.pt')
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Missing {pt_path}")
        
    dataset = load_dataset(pt_path)
    k_range = list(range(2, 13))  # [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    print("="*75)
    print("PHASE A: SYSTEMATIC LATENT CAPACITY (K) SWEEP: K in [2..12]")
    print(f"Device: {device} | Metric: Multi-step Autoregressive Linear R2")
    print("="*75)
    
    all_eval_dfs = []
    summary_records = []
    
    best_overall_r2 = -1.0
    best_model_weights = None
    best_k = None
    
    total_start = time.time()
    
    for K in k_range:
        print(f"\n>>> Running K = {K:2d} (Parameters: {48*K + K*K + 48 + 3*K})...")
        model, train_time = train_single_k(dataset, K, epochs=350, lr=0.008, l2_lambda=0.01, ode_step_size=0.1)
        
        eval_df = evaluate_model_rollouts(model, dataset, K, ode_step_size=0.1)
        all_eval_dfs.append(eval_df)
        
        # Summary metrics
        mean_lin = eval_df['r2_linear'].mean()
        med_lin = eval_df['r2_linear'].median()
        mean_log = eval_df['r2_log'].mean()
        
        ln_lin = eval_df[eval_df['condition'] == 'LN']['r2_linear'].mean()
        mn_lin = eval_df[eval_df['condition'] == 'MN']['r2_linear'].mean()
        hn_lin = eval_df[eval_df['condition'] == 'HN']['r2_linear'].mean()
        
        stable_lin = eval_df[eval_df['state'] == 'Stable']['r2_linear'].mean()
        chaotic_lin = eval_df[eval_df['state'] == 'Chaotic/Fluctuating']['r2_linear'].mean()
        
        # S=12 HN specific
        s12_hn = eval_df[(eval_df['condition'] == 'HN') & (eval_df['S'] == 12)]
        s12_hn_stable = s12_hn[s12_hn['state'] == 'Stable']['r2_linear'].mean()
        s12_hn_chaotic = s12_hn[s12_hn['state'] == 'Chaotic/Fluctuating']['r2_linear'].mean()
        
        print(f"    Done in {train_time:.1f}s | Mean Linear R2: {mean_lin:.3f} (Med: {med_lin:.3f}) | LN: {ln_lin:.3f} | MN: {mn_lin:.3f} | HN: {hn_lin:.3f}")
        print(f"    Stable Comms: {stable_lin:.3f} | Chaotic Comms: {chaotic_lin:.3f} | S=12 HN (Stb: {s12_hn_stable:.3f}, Ch: {s12_hn_chaotic:.3f})")
        
        if mean_lin > best_overall_r2:
            best_overall_r2 = mean_lin
            best_k = K
            best_model_weights = copy.deepcopy(model.state_dict())
            
        summary_records.append({
            'K': K,
            'num_params': 48*K + K*K + 48 + 3*K,
            'train_time_sec': train_time,
            'mean_r2_linear': mean_lin,
            'median_r2_linear': med_lin,
            'mean_r2_log': mean_log,
            'ln_r2_linear': ln_lin,
            'mn_r2_linear': mn_lin,
            'hn_r2_linear': hn_lin,
            'stable_r2_linear': stable_lin,
            'chaotic_r2_linear': chaotic_lin,
            's12_hn_stable_r2': s12_hn_stable,
            's12_hn_chaotic_r2': s12_hn_chaotic
        })
        
    total_time = time.time() - total_start
    print("\n" + "="*75)
    print(f"SWEEP COMPLETE in {total_time/60:.1f} minutes! Optimal K = {best_k} (Mean Linear R2 = {best_overall_r2:.3f})")
    print("="*75)
    
    # Save Best Model
    best_model_path = os.path.join(DATA_DIR, f'Gore_Global_ODE_Model_Best_K{best_k}.pth')
    torch.save(best_model_weights, best_model_path)
    print(f"[MODEL] Saved best model (K={best_k}) to:\n  {best_model_path}")
    
    # Save Detailed and Summary DataFrames
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(RESULTS_DIR, 'Gore_PhaseA_K_Sweep_Summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    detailed_df = pd.concat(all_eval_dfs, ignore_index=True)
    detailed_csv_path = os.path.join(RESULTS_DIR, 'Gore_PhaseA_K_Sweep_Detailed.csv')
    detailed_df.to_csv(detailed_csv_path, index=False)
    
    print(f"[RESULTS] Saved summary table to:\n  {summary_csv_path}")
    print(f"[RESULTS] Saved detailed table to:\n  {detailed_csv_path}")
    
    # Print formatted table
    print("\nSummary Table:")
    print(summary_df[['K', 'num_params', 'mean_r2_linear', 'median_r2_linear', 'ln_r2_linear', 'mn_r2_linear', 'hn_r2_linear', 'stable_r2_linear', 'chaotic_r2_linear']].to_string(index=False))
    
    # Plot Comparison Figures
    plot_k_sweep_results(summary_df, detailed_df)


if __name__ == '__main__':
    main()

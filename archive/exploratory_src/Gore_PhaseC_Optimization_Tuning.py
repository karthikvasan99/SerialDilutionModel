"""
Phase C: Optimization & Regularization Fine-Tuning for Gore Global ODE
Explores:
  1. L2 Regularization Penalty: lambda in [0.01, 0.005, 0.001, 0.0005, 0.0]
  2. Decoupled L2 Penalties: lambda_Theta vs. lambda_W (allowing sharp traits with bounded cross-feeding)
  3. Learning Rate Schedulers: Flat LR vs. CosineAnnealingLR (decaying to 1e-4)
  4. Training Duration: 350 vs. 500 epochs

Fixed Foundation (from Phases A & B):
  - K = 7 latent dimensions
  - Euler solver with Delta t = 0.10
  - Scheduled sampling floor TF_min = 0.10
  - Pure KLDivLoss, 30-fold dilution factor
  - Evaluated on autonomous multi-step Linear Pearson R2 (relative abundance x_i in [0, 1])
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


def train_opt_model(dataset, K=7, epochs=350, lr_init=0.008, use_cosine_lr=False,
                    l2_theta=0.01, l2_w=0.01, ode_step_size=0.1, tf_min=0.10, seed=42):
    set_seed(seed)
    num_species = len(dataset['species'])
    num_conditions = len(dataset['cond_map'])
    
    model = LatentAssemblyNeuralODE(
        num_species=num_species,
        num_latent_dims=K,
        num_conditions=num_conditions,
        default_dilution_factor=30.0
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4) if use_cosine_lr else None
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    b11_s = dataset['b11_seqs']
    b11_c = dataset['b11_conds']
    b4_s = dataset['b4_seqs']
    b4_c = dataset['b4_conds']
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Scheduled sampling decay to tf_min
        raw_tf = 1.0 - (epoch / (epochs * 0.75))
        tf_ratio = max(tf_min, min(1.0, raw_tf))
        
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
            
        # Decoupled L2 regularization
        theta_reg = sum(p.pow(2.0).sum() for name, p in model.named_parameters() if 'Theta' in name)
        w_reg = sum(p.pow(2.0).sum() for name, p in model.named_parameters() if 'W' in name)
        reg_loss = (l2_theta * theta_reg) + (l2_w * w_reg)
        
        loss = (batch11_loss + batch4_loss) / (b11_s.size(1) - 1 + b4_s.size(1) - 1) + reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
    train_duration = time.time() - start_time
    return model, train_duration


def evaluate_rollouts(model, dataset, config_name, ode_step_size=0.1):
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
            
            r_lin, _ = pearsonr(t_un, p_un)
            r2_lin = float(r_lin**2) if not np.isnan(r_lin) else 0.0
            
            t_log = np.log10(np.clip(t_un, 1e-6, None))
            p_log = np.log10(np.clip(p_un, 1e-6, None))
            r_log, _ = pearsonr(t_log, p_log)
            r2_log = float(r_log**2) if not np.isnan(r_log) else 0.0
            
            t_end = true_mat[-1, mask]
            p_end = pred_mat[-1, mask]
            r_end, _ = pearsonr(t_end, p_end)
            r2_end_lin = float(r_end**2) if not np.isnan(r_end) else 0.0
            
            sub = meta[(meta['S'] == s_val) & (meta['community_name'] == c_name)]
            is_fluc = bool(sub[f'{cond}_fluctuating'].values[0]) if len(sub) > 0 else False
            
            results.append({
                'config': config_name,
                'condition': cond,
                'S': s_val,
                'community': c_name,
                'r2_linear': r2_lin,
                'r2_log': r2_log,
                'r2_endpoint_linear': r2_end_lin,
                'state': 'Chaotic/Fluctuating' if is_fluc else 'Stable'
            })
            
    # Calculate parameter statistics
    with torch.no_grad():
        theta_norm = float(torch.norm(model.ode_func.Theta).cpu().item())
        w_norm = float(torch.norm(model.ode_func.W).cpu().item())
        r0_sum_hn = float(torch.nn.functional.softplus(model.a[2]).sum().cpu().item())
        
    df_res = pd.DataFrame(results)
    return df_res, theta_norm, w_norm, r0_sum_hn


def plot_phase_c_results(summary_df):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    configs = summary_df['config'].values
    x = np.arange(len(configs))
    
    # Panel 1: Bar chart of overall Linear R2 and Stable R2
    ax1 = axes[0]
    b1 = ax1.bar(x - 0.2, summary_df['mean_r2_linear'], width=0.4, label='Overall Mean Linear $R^2$', color='#2c3e50', edgecolor='black')
    b2 = ax1.bar(x + 0.2, summary_df['stable_r2_linear'], width=0.4, label='Stable Communities ($n=50$)', color='#27ae60', edgecolor='black')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=35, ha='right', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Autoregressive Linear $R^2$', fontsize=13, fontweight='bold')
    ax1.set_title('A) Predictive Accuracy across Regularization & LR Schemes', fontsize=14, fontweight='bold')
    ax1.set_ylim([0.45, 0.70])
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax1.legend(fontsize=11, loc='upper left')
    
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.004, f'{h:.3f}', ha='center', va='bottom', fontsize=8.5)
        
    # Panel 2: Biological Parameters (Theta Norm & W Norm)
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    l1 = ax2.plot(x, summary_df['theta_norm'], 'o-', color='#e74c3c', lw=2.5, ms=7, label=r'Trait Matrix $\|\Theta\|_F$')
    l2 = ax2_twin.plot(x, summary_df['w_norm'], 's--', color='#8e44ad', lw=2.5, ms=7, label=r'Cross-Feeding $\|W\|_F$')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=35, ha='right', fontsize=9, fontweight='bold')
    ax2.set_ylabel(r'Trait Matrix Norm $\|\Theta\|_F$', fontsize=13, fontweight='bold', color='#e74c3c')
    ax2_twin.set_ylabel(r'Cross-Feeding Norm $\|W\|_F$', fontsize=13, fontweight='bold', color='#8e44ad')
    ax2.set_title('B) Parameter Scaling & Cross-Feeding Engagement', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'Gore_PhaseC_Optimization_Comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[FIGURE] Saved Phase C comparison plot to:\n  {fig_path}")


def main():
    pt_path = os.path.join(DATA_DIR, 'Gore_processed_data.pt')
    dataset = load_dataset(pt_path)
    
    configs = [
        {'name': 'L2_0.010-FlatLR (Base)',     'l2_t': 0.01,   'l2_w': 0.01,  'lr': 0.008, 'cosine': False, 'epochs': 350},
        {'name': 'L2_0.005-FlatLR',            'l2_t': 0.005,  'l2_w': 0.005, 'lr': 0.008, 'cosine': False, 'epochs': 350},
        {'name': 'L2_0.001-FlatLR',            'l2_t': 0.001,  'l2_w': 0.001, 'lr': 0.008, 'cosine': False, 'epochs': 350},
        {'name': 'L2_0.0005-FlatLR',           'l2_t': 0.0005, 'l2_w': 0.0005,'lr': 0.008, 'cosine': False, 'epochs': 350},
        {'name': 'L2_0.000-FlatLR (NoReg)',    'l2_t': 0.0,    'l2_w': 0.0,   'lr': 0.008, 'cosine': False, 'epochs': 350},
        {'name': 'L2_0.001-CosineLR',          'l2_t': 0.001,  'l2_w': 0.001, 'lr': 0.010, 'cosine': True,  'epochs': 350},
        {'name': 'L2_0.001-CosineLR-500ep',    'l2_t': 0.001,  'l2_w': 0.001, 'lr': 0.010, 'cosine': True,  'epochs': 500},
        {'name': 'Decoupled-CosineLR',         'l2_t': 0.0005, 'l2_w': 0.005, 'lr': 0.010, 'cosine': True,  'epochs': 350},
        {'name': 'Decoupled-CosineLR-500ep',   'l2_t': 0.0005, 'l2_w': 0.005, 'lr': 0.010, 'cosine': True,  'epochs': 500},
    ]
    
    print("="*85)
    print("PHASE C: OPTIMIZATION & REGULARIZATION FINE-TUNING")
    print(f"Device: {device} | Model: K=7 (454 params) | Solver: Euler (dt=0.10, TF_min=0.10)")
    print("="*85)
    
    all_eval_dfs = []
    summary_records = []
    
    best_r2 = -1.0
    best_config = None
    best_weights = None
    
    total_start = time.time()
    
    for cfg in configs:
        c_name = cfg['name']
        print(f"\n>>> Running Config: {c_name} (L2_theta={cfg['l2_t']}, L2_w={cfg['l2_w']}, Cosine={cfg['cosine']}, Epochs={cfg['epochs']})...")
        model, train_time = train_opt_model(
            dataset, K=7, epochs=cfg['epochs'], lr_init=cfg['lr'],
            use_cosine_lr=cfg['cosine'], l2_theta=cfg['l2_t'], l2_w=cfg['l2_w'],
            ode_step_size=0.10, tf_min=0.10
        )
        
        df_res, th_norm, w_norm, r0_hn = evaluate_rollouts(model, dataset, c_name, ode_step_size=0.10)
        all_eval_dfs.append(df_res)
        
        mean_lin = df_res['r2_linear'].mean()
        med_lin = df_res['r2_linear'].median()
        ln_lin = df_res[df_res['condition'] == 'LN']['r2_linear'].mean()
        mn_lin = df_res[df_res['condition'] == 'MN']['r2_linear'].mean()
        hn_lin = df_res[df_res['condition'] == 'HN']['r2_linear'].mean()
        stb_lin = df_res[df_res['state'] == 'Stable']['r2_linear'].mean()
        ch_lin = df_res[df_res['state'] == 'Chaotic/Fluctuating']['r2_linear'].mean()
        day10_lin = df_res['r2_endpoint_linear'].mean()
        
        print(f"    Done in {train_time:.1f}s | Mean Linear R2: {mean_lin:.3f} (Med: {med_lin:.3f}) | LN: {ln_lin:.3f} | MN: {mn_lin:.3f} | HN: {hn_lin:.3f}")
        print(f"    Stable: {stb_lin:.3f} | Chaotic: {ch_lin:.3f} | Day 10 Endpoint R2: {day10_lin:.3f} | Theta Norm: {th_norm:.2f} | W Norm: {w_norm:.4f}")
        
        if mean_lin > best_r2:
            best_r2 = mean_lin
            best_config = c_name
            best_weights = copy.deepcopy(model.state_dict())
            
        summary_records.append({
            'config': c_name,
            'l2_theta': cfg['l2_t'],
            'l2_w': cfg['l2_w'],
            'lr_init': cfg['lr'],
            'cosine_lr': cfg['cosine'],
            'epochs': cfg['epochs'],
            'train_time_sec': train_time,
            'mean_r2_linear': mean_lin,
            'median_r2_linear': med_lin,
            'ln_r2_linear': ln_lin,
            'mn_r2_linear': mn_lin,
            'hn_r2_linear': hn_lin,
            'stable_r2_linear': stb_lin,
            'chaotic_r2_linear': ch_lin,
            'day10_endpoint_r2': day10_lin,
            'theta_norm': th_norm,
            'w_norm': w_norm,
            'r0_hn_sum': r0_hn
        })
        
    total_time = time.time() - total_start
    print("\n" + "="*85)
    print(f"PHASE C COMPLETE in {total_time/60:.1f} minutes! Best: {best_config} (Mean Linear R2 = {best_r2:.3f})")
    print("="*85)
    
    # Save Best Model Checkpoint
    best_model_path = os.path.join(DATA_DIR, 'Gore_Global_ODE_Model_PhaseC_Best.pth')
    torch.save(best_weights, best_model_path)
    print(f"[MODEL] Saved best model checkpoint to:\n  {best_model_path}")
    
    # Save Summary and Detailed CSVs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(RESULTS_DIR, 'Gore_PhaseC_Optimization_Summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    detailed_df = pd.concat(all_eval_dfs, ignore_index=True)
    detailed_csv_path = os.path.join(RESULTS_DIR, 'Gore_PhaseC_Optimization_Detailed.csv')
    detailed_df.to_csv(detailed_csv_path, index=False)
    
    print(f"[RESULTS] Summary table saved to:\n  {summary_csv_path}")
    print(f"[RESULTS] Detailed table saved to:\n  {detailed_csv_path}")
    
    print("\nSummary Table:")
    print(summary_df[['config', 'mean_r2_linear', 'median_r2_linear', 'ln_r2_linear', 'mn_r2_linear', 'hn_r2_linear', 'stable_r2_linear', 'chaotic_r2_linear', 'theta_norm', 'w_norm']].to_string(index=False))
    
    # Plot Phase C Comparison Figures
    plot_phase_c_results(summary_df)


if __name__ == '__main__':
    main()

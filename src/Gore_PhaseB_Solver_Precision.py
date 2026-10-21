"""
Phase B: Solver Precision & Numerical Step Size (Delta t) Exploration
Investigates:
  1. Forward Euler with Delta t in [0.10, 0.05, 0.02]
  2. 2nd-order Midpoint with Delta t = 0.05
  3. 4th-order Runge-Kutta (RK4) with Delta t in [0.10, 0.05]
  4. Scheduled sampling floor (TF_min in [0.0, 0.10, 0.20]) to anchor multi-step drift.

Evaluation: Autoregressive Linear Pearson R2 across all 99 communities (overall, by regime, by stability state).
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


def train_solver_model(dataset, K=7, epochs=350, lr=0.008, l2_lambda=0.01,
                       ode_method='euler', ode_step_size=0.1, tf_min=0.0, seed=42):
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
        
        # Scheduled sampling decay down to tf_min
        raw_tf = 1.0 - (epoch / (epochs * 0.75))
        tf_ratio = max(tf_min, min(1.0, raw_tf))
        
        # Batch 11
        x_curr = b11_s[:, 0, :]
        batch11_loss = 0.0
        for t in range(1, b11_s.size(1)):
            log_pred, x_pred, _, _ = model(x_curr, b11_c, ode_method=ode_method, ode_step_size=ode_step_size)
            target = b11_s[:, t, :]
            batch11_loss += criterion(log_pred, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred
            
        # Batch 4
        x_curr_4 = b4_s[:, 0, :]
        batch4_loss = 0.0
        for t in range(1, b4_s.size(1)):
            log_pred_4, x_pred_4, _, _ = model(x_curr_4, b4_c, ode_method=ode_method, ode_step_size=ode_step_size)
            target_4 = b4_s[:, t, :]
            batch4_loss += criterion(log_pred_4, target_4)
            x_curr_4 = tf_ratio * target_4 + (1.0 - tf_ratio) * x_pred_4
            
        l2_reg = sum(p.pow(2.0).sum() for name, p in model.named_parameters() if 'Theta' in name or 'W' in name)
        loss = (batch11_loss + batch4_loss) / (b11_s.size(1) - 1 + b4_s.size(1) - 1) + l2_lambda * l2_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
    duration = time.time() - start_time
    return model, duration


def evaluate_rollouts(model, dataset, ode_method, ode_step_size, config_name):
    model.eval()
    results = []
    passage_errors = {p: [] for p in range(1, 11)}
    
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
                _, next_state, _, _ = model(curr_state, c_idx, ode_method=ode_method, ode_step_size=ode_step_size)
                pred_traj.append(next_state.squeeze(0).cpu().numpy())
                curr_state = next_state
                
                # Record passage-by-passage R2
                if true_traj.size(0) == 11:
                    t_pt = true_traj[t, mask].numpy()
                    p_pt = next_state.squeeze(0).cpu().numpy()[mask]
                    r_pt, _ = pearsonr(t_pt, p_pt)
                    passage_errors[t].append(r_pt**2 if not np.isnan(r_pt) else 0.0)
                
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
            
            sub = meta[(meta['S'] == s_val) & (meta['community_name'] == c_name)]
            is_fluc = bool(sub[f'{cond}_fluctuating'].values[0]) if len(sub) > 0 else False
            
            results.append({
                'config': config_name,
                'ode_method': ode_method,
                'ode_step_size': ode_step_size,
                'condition': cond,
                'S': s_val,
                'community': c_name,
                'r2_linear': r2_lin,
                'r2_log': r2_log,
                'state': 'Chaotic/Fluctuating' if is_fluc else 'Stable'
            })
            
    df_res = pd.DataFrame(results)
    mean_passage_r2 = {p: np.mean(vals) for p, vals in passage_errors.items()}
    return df_res, mean_passage_r2


def plot_phase_b_results(summary_df, passage_dict):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Panel 1: Bar chart of overall Linear R2 across solver configurations
    ax1 = axes[0]
    configs = summary_df['config'].values
    x = np.arange(len(configs))
    
    bars1 = ax1.bar(x - 0.2, summary_df['mean_r2_linear'], width=0.4, label='Overall Mean Linear $R^2$', color='#34495e', edgecolor='black')
    bars2 = ax1.bar(x + 0.2, summary_df['stable_r2_linear'], width=0.4, label='Stable Communities ($n=50$)', color='#27ae60', edgecolor='black')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=35, ha='right', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Autoregressive Linear $R^2$', fontsize=13, fontweight='bold')
    ax1.set_title('A) Predictive Performance by Solver Precision & TF Floor', fontsize=14, fontweight='bold')
    ax1.set_ylim([0.45, 0.70])
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax1.legend(fontsize=11, loc='upper left')
    
    # Annotate values on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=9)
        
    # Panel 2: Passage-by-Passage Rollout R2 (Days 1 to 10)
    ax2 = axes[1]
    days = np.arange(1, 11)
    
    color_map = {
        'Euler-dt0.10 (Base)': '#95a5a6',
        'Euler-dt0.05': '#7f8c8d',
        'Euler-dt0.02': '#34495e',
        'Midpoint-dt0.05': '#f39c12',
        'RK4-dt0.10': '#3498db',
        'RK4-dt0.05': '#2980b9',
        'RK4-dt0.05 (TF10%)': '#9b59b6',
        'RK4-dt0.05 (TF20%)': '#e74c3c'
    }
    
    for cfg_name, day_vals in passage_dict.items():
        curve = [day_vals[d] for d in days]
        c = color_map.get(cfg_name, '#000000')
        lw = 2.8 if 'TF' in cfg_name or 'RK4-dt0.05' in cfg_name else 1.6
        ls = '-' if 'RK4' in cfg_name else '--'
        ax2.plot(days, curve, marker='o', ms=5, lw=lw, ls=ls, color=c, label=cfg_name)
        
    ax2.set_xlabel('Serial Dilution Passage (Day 1 to 10)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Autoregressive Linear $R^2$', fontsize=13, fontweight='bold')
    ax2.set_title('B) Multi-Passage Trajectory Drift: Day 1 to Day 10', fontsize=14, fontweight='bold')
    ax2.set_xticks(days)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9, loc='lower left')
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'Gore_PhaseB_Solver_Comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[FIGURE] Saved Phase B figure to:\n  {fig_path}")


def main():
    pt_path = os.path.join(DATA_DIR, 'Gore_processed_data.pt')
    dataset = load_dataset(pt_path)
    
    configs = [
        {'name': 'Euler-dt0.10 (Base)', 'method': 'euler', 'step': 0.10, 'tf_min': 0.0},
        {'name': 'Euler-dt0.05',        'method': 'euler', 'step': 0.05, 'tf_min': 0.0},
        {'name': 'Euler-dt0.02',        'method': 'euler', 'step': 0.02, 'tf_min': 0.0},
        {'name': 'Midpoint-dt0.05',     'method': 'midpoint', 'step': 0.05, 'tf_min': 0.0},
        {'name': 'RK4-dt0.10',          'method': 'rk4',   'step': 0.10, 'tf_min': 0.0},
        {'name': 'RK4-dt0.05',          'method': 'rk4',   'step': 0.05, 'tf_min': 0.0},
        {'name': 'RK4-dt0.05 (TF10%)',  'method': 'rk4',   'step': 0.05, 'tf_min': 0.10},
        {'name': 'RK4-dt0.05 (TF20%)',  'method': 'rk4',   'step': 0.05, 'tf_min': 0.20},
    ]
    
    print("="*80)
    print("PHASE B: SOLVER PRECISION, STEP SIZE (Delta t), & TF ANCHORING EXPLORATION")
    print(f"Device: {device} | Model: K=7 (454 params) | Epochs: 350")
    print("="*80)
    
    all_detailed = []
    summary_records = []
    passage_dict = {}
    
    best_r2 = -1.0
    best_config_name = None
    best_weights = None
    
    total_start = time.time()
    
    for cfg in configs:
        cfg_name = cfg['name']
        print(f"\n>>> Running Config: {cfg_name} (Method: {cfg['method']}, Step: {cfg['step']}, TF_min: {cfg['tf_min']})...")
        model, train_time = train_solver_model(
            dataset, K=7, epochs=350, lr=0.008, l2_lambda=0.01,
            ode_method=cfg['method'], ode_step_size=cfg['step'], tf_min=cfg['tf_min']
        )
        
        df_res, p_errs = evaluate_rollouts(model, dataset, cfg['method'], cfg['step'], cfg_name)
        all_detailed.append(df_res)
        passage_dict[cfg_name] = p_errs
        
        mean_lin = df_res['r2_linear'].mean()
        med_lin = df_res['r2_linear'].median()
        ln_lin = df_res[df_res['condition'] == 'LN']['r2_linear'].mean()
        mn_lin = df_res[df_res['condition'] == 'MN']['r2_linear'].mean()
        hn_lin = df_res[df_res['condition'] == 'HN']['r2_linear'].mean()
        stb_lin = df_res[df_res['state'] == 'Stable']['r2_linear'].mean()
        ch_lin = df_res[df_res['state'] == 'Chaotic/Fluctuating']['r2_linear'].mean()
        
        print(f"    Done in {train_time:.1f}s | Mean Linear R2: {mean_lin:.3f} (Med: {med_lin:.3f}) | LN: {ln_lin:.3f} | MN: {mn_lin:.3f} | HN: {hn_lin:.3f}")
        print(f"    Stable: {stb_lin:.3f} | Chaotic: {ch_lin:.3f} | Day 10 R2: {p_errs[10]:.3f}")
        
        if mean_lin > best_r2:
            best_r2 = mean_lin
            best_config_name = cfg_name
            best_weights = copy.deepcopy(model.state_dict())
            
        summary_records.append({
            'config': cfg_name,
            'method': cfg['method'],
            'step_size': cfg['step'],
            'tf_min': cfg['tf_min'],
            'train_time_sec': train_time,
            'mean_r2_linear': mean_lin,
            'median_r2_linear': med_lin,
            'ln_r2_linear': ln_lin,
            'mn_r2_linear': mn_lin,
            'hn_r2_linear': hn_lin,
            'stable_r2_linear': stb_lin,
            'chaotic_r2_linear': ch_lin,
            'day1_r2': p_errs[1],
            'day5_r2': p_errs[5],
            'day10_r2': p_errs[10]
        })
        
    total_time = time.time() - total_start
    print("\n" + "="*80)
    print(f"PHASE B COMPLETE in {total_time/60:.1f} minutes! Best: {best_config_name} (Mean Linear R2 = {best_r2:.3f})")
    print("="*80)
    
    # Save Best Checkpoint
    best_model_path = os.path.join(DATA_DIR, 'Gore_Global_ODE_Model_PhaseB_Best.pth')
    torch.save(best_weights, best_model_path)
    print(f"[MODEL] Saved best model to:\n  {best_model_path}")
    
    # Save CSVs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(RESULTS_DIR, 'Gore_PhaseB_Solver_Summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    detailed_df = pd.concat(all_detailed, ignore_index=True)
    detailed_csv_path = os.path.join(RESULTS_DIR, 'Gore_PhaseB_Solver_Detailed.csv')
    detailed_df.to_csv(detailed_csv_path, index=False)
    
    print(f"[RESULTS] Summary table saved to:\n  {summary_csv_path}")
    print(f"[RESULTS] Detailed table saved to:\n  {detailed_csv_path}")
    
    print("\nSummary Table:")
    print(summary_df[['config', 'train_time_sec', 'mean_r2_linear', 'median_r2_linear', 'ln_r2_linear', 'mn_r2_linear', 'hn_r2_linear', 'stable_r2_linear', 'chaotic_r2_linear', 'day10_r2']].to_string(index=False))
    
    # Plot figures
    plot_phase_b_results(summary_df, passage_dict)


if __name__ == '__main__':
    main()

"""
Publication-Quality Visualization of the Optimal Gore Global Neural ODE Model
Generates a comprehensive 6-panel evaluation figure:
  Panel A: Global 1:1 Parity Scatter Plot across all 99 communities (Linear scale).
  Panel B: Trajectory Predictability (Linear R2) across Pool Sizes S in {3, 6, 12, 24, 48}.
  Panel C: Longitudinal Trajectory Rollout: Low Nutrient (LN) Stable Damping.
  Panel D: Longitudinal Trajectory Rollout: Medium Nutrient (MN) Equilibrium Coexistence.
  Panel E: Longitudinal Trajectory Rollout: High Nutrient (HN) Persistent Oscillations & Blooms.
  Panel F: Learned Latent Resource Coordinates R(0) along the Chemical Nutrient Loading Axis.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'Figures')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results')

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE
from Gore_PhaseC_Optimization_Tuning import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_full_evaluation(model, dataset):
    model.eval()
    rel_dict = dataset['rel_dict']
    masks_dict = dataset['masks']
    cond_map = dataset['cond_map']
    meta = dataset['metadata']
    species_list = dataset['species']
    
    records = []
    traj_store = {}
    
    with torch.no_grad():
        for (cond, s_val, c_name), true_traj in rel_dict.items():
            c_idx = torch.tensor([cond_map[cond]], dtype=torch.long, device=device)
            mask = masks_dict[(s_val, c_name)].numpy()
            
            curr_state = true_traj[0:1, :].to(device)
            pred_traj = [curr_state.cpu().squeeze(0).numpy()]
            
            for t in range(1, true_traj.size(0)):
                _, next_state, _, _ = model(curr_state, c_idx, ode_method='euler', ode_step_size=0.10)
                pred_traj.append(next_state.squeeze(0).cpu().numpy())
                curr_state = next_state
                
            pred_mat = np.array(pred_traj) # (T, 48)
            true_mat = true_traj.numpy()   # (T, 48)
            
            t_un = true_mat[1:, mask].flatten()
            p_un = pred_mat[1:, mask].flatten()
            
            r_lin, _ = pearsonr(t_un, p_un)
            r2_lin = float(r_lin**2) if not np.isnan(r_lin) else 0.0
            
            sub = meta[(meta['S'] == s_val) & (meta['community_name'] == c_name)]
            is_fluc = bool(sub[f'{cond}_fluctuating'].values[0]) if len(sub) > 0 else False
            
            key = (cond, s_val, c_name)
            traj_store[key] = {
                'true': true_mat,
                'pred': pred_mat,
                'mask': mask,
                'days': np.arange(true_mat.shape[0]) if cond != 'MN' or s_val >= 24 else np.arange(7, 11)
            }
            
            records.append({
                'condition': cond,
                'S': s_val,
                'community': c_name,
                'r2_linear': r2_lin,
                'state': 'Chaotic/Fluctuating' if is_fluc else 'Stable'
            })
            
    df_eval = pd.DataFrame(records)
    return df_eval, traj_store


def make_6panel_plot(model, dataset, df_eval, traj_store):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 3x2 Publication Figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.22)
    
    cond_colors = {'LN': '#27ae60', 'MN': '#f39c12', 'HN': '#e74c3c'}
    
    # =========================================================================
    # PANEL A: Global Parity Scatter Plot (1:1 Fit)
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    all_t, all_p, all_c = [], [], []
    for (cond, s_val, c_name), data in traj_store.items():
        mask = data['mask']
        t_vals = data['true'][1:, mask].flatten()
        p_vals = data['pred'][1:, mask].flatten()
        all_t.extend(t_vals)
        all_p.extend(p_vals)
        all_c.extend([cond_colors[cond]] * len(t_vals))
        
    all_t = np.array(all_t)
    all_p = np.array(all_p)
    overall_r, _ = pearsonr(all_t, all_p)
    overall_r2 = overall_r**2
    median_r2 = df_eval['r2_linear'].median()
    
    for c_name, c_hex in cond_colors.items():
        sub_idx = [i for i, row in df_eval.iterrows() if row['condition'] == c_name]
        t_sub = np.concatenate([traj_store[(row['condition'], row['S'], row['community'])]['true'][1:, traj_store[(row['condition'], row['S'], row['community'])]['mask']].flatten() for _, row in df_eval.loc[sub_idx].iterrows()])
        p_sub = np.concatenate([traj_store[(row['condition'], row['S'], row['community'])]['pred'][1:, traj_store[(row['condition'], row['S'], row['community'])]['mask']].flatten() for _, row in df_eval.loc[sub_idx].iterrows()])
        r_c, _ = pearsonr(t_sub, p_sub)
        ax_a.scatter(t_sub, p_sub, alpha=0.32, s=22, c=c_hex, label=f'{c_name} (Regime $R^2={r_c**2:.2f}$)')
        
    ax_a.plot([0.0, 1.0], [0.0, 1.0], 'k--', lw=2.5, label='1:1 Parity Line')
    ax_a.set_xlim([-0.02, 1.02])
    ax_a.set_ylim([-0.02, 1.02])
    ax_a.set_xlabel('Empirical Relative Abundance ($x_i$)', fontsize=13, fontweight='bold')
    ax_a.set_ylabel(r'Model Autoregressive Prediction ($\hat{x}_i$)', fontsize=13, fontweight='bold')
    ax_a.set_title(f'A) Global Autoregressive Trajectory Fit ($R^2 = {overall_r2:.3f}$, Med $R^2 = {median_r2:.3f}$)', fontsize=14, fontweight='bold')
    ax_a.legend(fontsize=10.5, loc='upper left')
    ax_a.grid(True, linestyle=':', alpha=0.6)
    
    # =========================================================================
    # PANEL B: Predictability across Pool Sizes S & Regimes
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    s_groups = [3, 6, 12, 24, 48]
    x = np.arange(len(s_groups))
    width = 0.26
    
    for i, cond in enumerate(['LN', 'MN', 'HN']):
        means = []
        medians = []
        for s in s_groups:
            sub = df_eval[(df_eval['condition'] == cond) & (df_eval['S'] == s)]
            means.append(sub['r2_linear'].mean())
            medians.append(sub['r2_linear'].median())
            
        ax_b.bar(x + (i - 1) * width, medians, width, label=f'{cond}',
                 color=cond_colors[cond], edgecolor='black', linewidth=1.2, zorder=3)
                 
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f'S={s}' for s in s_groups], fontsize=12, fontweight='bold')
    ax_b.set_ylim([0, 1.05])
    ax_b.set_ylabel('Median Autoregressive Linear $R^2$', fontsize=13, fontweight='bold')
    ax_b.set_title('B) Predictability across Initial Pool Sizes ($S$) by Regime', fontsize=14, fontweight='bold')
    ax_b.legend(fontsize=11, loc='upper right')
    ax_b.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)
    
    # Palette for individual bacterial species in time-courses
    species_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#339966', '#cc3333'
    ]
    
    # =========================================================================
    # PANEL C: LN Stable Coexistence Rollout (S=12 Community 4)
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    key_ln = ('LN', 12, 'Community4')
    ln_data = traj_store[key_ln]
    mask_ln = np.where(ln_data['mask'])[0]
    days_ln = ln_data['days']
    r2_ln = df_eval[(df_eval['condition']=='LN') & (df_eval['S']==12) & (df_eval['community']=='Community4')]['r2_linear'].values[0]
    
    # Sort species by maximum abundance
    top_sp_ln = sorted(mask_ln, key=lambda idx: np.max(ln_data['true'][:, idx]), reverse=True)
    
    for i, sp_idx in enumerate(top_sp_ln[:6]):
        sp_name = dataset['species'][sp_idx]
        c = species_palette[i % len(species_palette)]
        ax_c.plot(days_ln, ln_data['true'][:, sp_idx], '-', color=c, lw=2.2, label=f'{sp_name}')
        ax_c.plot(days_ln, ln_data['pred'][:, sp_idx], '--o', color=c, lw=1.8, ms=4.5, alpha=0.85)
        
    ax_c.set_xlabel('Passage Day (0 to 10)', fontsize=13, fontweight='bold')
    ax_c.set_ylabel('Relative Abundance ($x_i$)', fontsize=13, fontweight='bold')
    ax_c.set_title(f'C) Low Nutrient (LN) Damped Coexistence: S=12 Comm 4 ($R^2 = {r2_ln:.3f}$)', fontsize=14, fontweight='bold')
    ax_c.grid(True, linestyle=':', alpha=0.6)
    
    # Custom legend for solid vs dashed
    legend_elements_c = [
        Line2D([0], [0], color='black', lw=2.2, label='Empirical Data (Solid)'),
        Line2D([0], [0], color='black', lw=1.8, ls='--', marker='o', ms=4, label='Model Rollout (Dashed)')
    ]
    ax_c.legend(handles=legend_elements_c, fontsize=10, loc='upper right')
    
    # =========================================================================
    # PANEL D: MN Equilibrium Coexistence Rollout (S=12 Community 6)
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    key_mn = ('MN', 12, 'Community6')
    mn_data = traj_store[key_mn]
    mask_mn = np.where(mn_data['mask'])[0]
    days_mn = mn_data['days']
    r2_mn = df_eval[(df_eval['condition']=='MN') & (df_eval['S']==12) & (df_eval['community']=='Community6')]['r2_linear'].values[0]
    
    top_sp_mn = sorted(mask_mn, key=lambda idx: np.max(mn_data['true'][:, idx]), reverse=True)
    for i, sp_idx in enumerate(top_sp_mn[:6]):
        sp_name = dataset['species'][sp_idx]
        c = species_palette[i % len(species_palette)]
        ax_d.plot(days_mn, mn_data['true'][:, sp_idx], '-', color=c, lw=2.2, label=f'{sp_name}')
        ax_d.plot(days_mn, mn_data['pred'][:, sp_idx], '--o', color=c, lw=1.8, ms=4.5, alpha=0.85)
        
    ax_d.set_xlabel('Passage Day (7 to 10)', fontsize=13, fontweight='bold')
    ax_d.set_ylabel('Relative Abundance ($x_i$)', fontsize=13, fontweight='bold')
    ax_d.set_title(f'D) Medium Nutrient (MN) Equilibrium: S=12 Comm 6 ($R^2 = {r2_mn:.3f}$)', fontsize=14, fontweight='bold')
    ax_d.grid(True, linestyle=':', alpha=0.6)
    ax_d.legend(handles=legend_elements_c, fontsize=10, loc='upper right')
    
    # =========================================================================
    # PANEL E: HN Oscillations & Blooms Rollout (S=12 Community 4)
    # =========================================================================
    ax_e = fig.add_subplot(gs[2, 0])
    key_hn = ('HN', 12, 'Community4')
    hn_data = traj_store[key_hn]
    mask_hn = np.where(hn_data['mask'])[0]
    days_hn = hn_data['days']
    r2_hn = df_eval[(df_eval['condition']=='HN') & (df_eval['S']==12) & (df_eval['community']=='Community4')]['r2_linear'].values[0]
    
    top_sp_hn = sorted(mask_hn, key=lambda idx: np.max(hn_data['true'][:, idx]), reverse=True)
    for i, sp_idx in enumerate(top_sp_hn[:6]):
        sp_name = dataset['species'][sp_idx]
        c = species_palette[i % len(species_palette)]
        ax_e.plot(days_hn, hn_data['true'][:, sp_idx], '-', color=c, lw=2.2, label=f'{sp_name}')
        ax_e.plot(days_hn, hn_data['pred'][:, sp_idx], '--o', color=c, lw=1.8, ms=4.5, alpha=0.85)
        
    ax_e.set_xlabel('Passage Day (0 to 10)', fontsize=13, fontweight='bold')
    ax_e.set_ylabel('Relative Abundance ($x_i$)', fontsize=13, fontweight='bold')
    ax_e.set_title(f'E) High Nutrient (HN) Oscillations & Blooms: S=12 Comm 4 ($R^2 = {r2_hn:.3f}$)', fontsize=14, fontweight='bold')
    ax_e.grid(True, linestyle=':', alpha=0.6)
    ax_e.legend(handles=legend_elements_c, fontsize=10, loc='upper right')
    
    # =========================================================================
    # PANEL F: Latent Resource Geometry R(0) along Chemical Axis
    # =========================================================================
    ax_f = fig.add_subplot(gs[2, 1])
    with torch.no_grad():
        R_0 = torch.nn.functional.softplus(model.a).cpu().numpy() # (3, 7)
        
    pca = PCA(n_components=2)
    R_pca = pca.fit_transform(R_0)
    var_exp = pca.explained_variance_ratio_ * 100
    
    labels = [
        f'LN: Base Media (0g/L Gluc, 0g/L Urea) — Total $R_0={R_0[0].sum():.2f}$',
        f'MN: Base + 5g/L Gluc, 4g/L Urea — Total $R_0={R_0[1].sum():.2f}$',
        f'HN: Base + 20g/L Gluc, 16g/L Urea — Total $R_0={R_0[2].sum():.2f}$'
    ]
    
    # Trajectory line
    ax_f.plot(R_pca[:, 0], R_pca[:, 1], 'k--', lw=2.2, zorder=2)
    
    # Arrowheads along the trajectory
    ax_f.annotate('', xy=(R_pca[1, 0], R_pca[1, 1]), xytext=(R_pca[0, 0], R_pca[0, 1]),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0))
    ax_f.annotate('', xy=(R_pca[2, 0], R_pca[2, 1]), xytext=(R_pca[1, 0], R_pca[1, 1]),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0))
                 
    text_positions = [
        (-1.25, -0.40),  # LN: above-right of LN point
        (0.12, 1.05),    # MN: to the right of MN point
        (1.20, -0.30)    # HN: above HN point
    ]
                 
    for i, c_name in enumerate(['LN', 'MN', 'HN']):
        ax_f.scatter(R_pca[i, 0], R_pca[i, 1], s=520, c=cond_colors[c_name],
                     edgecolors='black', linewidth=2.2, zorder=4, label=labels[i])
        ax_f.text(text_positions[i][0], text_positions[i][1], c_name,
                  fontsize=14, fontweight='bold', ha='center', va='center', zorder=5)
                  
    ax_f.set_xlim([R_pca[:, 0].min() - 0.35, R_pca[:, 0].max() + 0.35])
    ax_f.set_ylim([R_pca[:, 1].min() - 0.35, R_pca[:, 1].max() + 0.35])
                  
    ax_f.set_xlabel(f'Latent Resource PC1 ({var_exp[0]:.1f}% Variance)', fontsize=13, fontweight='bold')
    ax_f.set_ylabel(f'Latent Resource PC2 ({var_exp[1]:.1f}% Variance)', fontsize=13, fontweight='bold')
    ax_f.set_title('F) Learned Latent Resource Coordinates $R(0)$ along Nutrient Axis', fontsize=14, fontweight='bold')
    ax_f.legend(fontsize=10.5, loc='upper left')
    ax_f.grid(True, linestyle=':', alpha=0.6)
    
    # Save figure
    out_path = os.path.join(FIGURES_DIR, 'Gore_Optimal_Model_Fits.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[FIGURE] Saved 6-panel optimal model fit figure to:\n  {out_path}")


def main():
    pt_path = os.path.join(DATA_DIR, 'Gore_processed_data.pt')
    dataset = load_dataset(pt_path)
    
    # Load optimal model
    model_path = os.path.join(DATA_DIR, 'Gore_Global_ODE_Model_PhaseC_Best.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}")
        
    weights = torch.load(model_path, map_location=device)
    model = LatentAssemblyNeuralODE(
        num_species=len(dataset['species']),
        num_latent_dims=7,
        num_conditions=len(dataset['cond_map']),
        default_dilution_factor=30.0
    ).to(device)
    model.load_state_dict(weights)
    
    print("Evaluating optimal model rollouts...")
    df_eval, traj_store = run_full_evaluation(model, dataset)
    
    print("\nGenerating 6-panel publication figure...")
    make_6panel_plot(model, dataset, df_eval, traj_store)
    print("Completed successfully!")


if __name__ == '__main__':
    main()

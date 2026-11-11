"""
Generate Updated Companion Figures for Dual-Theta Unified CRM (K=3):
1. Pollard_Coupled_Biofilm_Adhesion_Traits.png:
   - Panel A: Adhesion Partitioning Ratio (k_attach / k_shed) for Mucin Agar
   - Panel B: Substrate Affinity (k_attach Mucin vs Plain Agar)
   - Panel C: Attached Biofilm vs Planktonic Traits (||Theta^(c)|| vs ||Theta^(s)||)
   - Panel D: Self-Limitation alpha vs Biofilm Retention
2. Pollard_Coupled_Biofilm_Simultaneous_Fit.png:
   - 4-Panel Parity Plots: No Carrier, Plain Agar, Mucin Agar, and Global Multi-Condition Combined
3. Pollard_Coupled_Biofilm_Longitudinal_Trajectories.png:
   - P1..P6 Trajectories for 4 representative functional phenotypes across compartments
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from poster_fig2_dual_theta import (
    DualThetaUnifiedCRM,
    strains_keep,
    no_c_tensor,
    plain_c_tensor,
    plain_s_tensor,
    mucin_c_tensor,
    mucin_s_tensor,
    device,
    passages
)

def main():
    print("="*70)
    print("GENERATING UPDATED FIGURES FOR DUAL-THETA CRM (K=3)")
    print("="*70)
    
    ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_DualTheta_K3_CRM.pth')
    model = DualThetaUnifiedCRM(N=len(strains_keep), K=3).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params()
        
        # Rollouts
        p_no_c = model.rollout_no_carrier(Theta_s, W, alpha, R0[0], b0_liq, tf_ratio=0.0).cpu().numpy()
        t_no_c = no_c_tensor[:, 1:, :].cpu().numpy()
        
        p_pl_c, p_pl_s = model.rollout_carrier(plain_c_tensor, plain_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta, tf_ratio=0.0)
        p_pl_c = p_pl_c.cpu().numpy()
        p_pl_s = p_pl_s.cpu().numpy()
        t_pl_c = plain_c_tensor[:, 1:, :].cpu().numpy()
        t_pl_s = plain_s_tensor[:, 1:, :].cpu().numpy()
        
        p_mu_c, p_mu_s = model.rollout_carrier(mucin_c_tensor, mucin_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta, tf_ratio=0.0)
        p_mu_c = p_mu_c.cpu().numpy()
        p_mu_s = p_mu_s.cpu().numpy()
        t_mu_c = mucin_c_tensor[:, 1:, :].cpu().numpy()
        t_mu_s = mucin_s_tensor[:, 1:, :].cpu().numpy()
        
        # Parameters to numpy
        Theta_c_np = Theta_c.cpu().numpy()
        Theta_s_np = Theta_s.cpu().numpy()
        alpha_np = alpha.cpu().numpy()
        k_shed_np = k_shed.cpu().numpy()
        k_att_plain_np = k_attach[0].cpu().numpy()
        k_att_mucin_np = k_attach[1].cpu().numpy()
        
    # Helper for stats
    def calc_metrics(obs, pred):
        o_fl = obs.flatten()
        p_fl = pred.flatten()
        r2 = 1.0 - np.sum((o_fl - p_fl)**2) / np.sum((o_fl - np.mean(o_fl))**2)
        r, _ = pearsonr(o_fl, p_fl)
        rho, _ = spearmanr(o_fl, p_fl)
        mae = np.mean(np.abs(o_fl - p_fl))
        bc = np.mean(np.sum(np.abs(obs - pred), axis=-1) / (np.sum(obs + pred, axis=-1) + 1e-8))
        return {'r2': r2, 'r': r, 'rho': rho, 'mae': mae, 'bc': bc}

    m_no_c = calc_metrics(t_no_c, p_no_c)
    m_pl_c = calc_metrics(t_pl_c, p_pl_c)
    m_pl_s = calc_metrics(t_pl_s, p_pl_s)
    m_pl_comb = calc_metrics(np.concatenate([t_pl_c, t_pl_s], axis=1), np.concatenate([p_pl_c, p_pl_s], axis=1))
    m_mu_c = calc_metrics(t_mu_c, p_mu_c)
    m_mu_s = calc_metrics(t_mu_s, p_mu_s)
    m_mu_comb = calc_metrics(np.concatenate([t_mu_c, t_mu_s], axis=1), np.concatenate([p_mu_c, p_mu_s], axis=1))
    
    all_t = np.concatenate([t_no_c.flatten(), t_pl_c.flatten(), t_pl_s.flatten(), t_mu_c.flatten(), t_mu_s.flatten()])
    all_p = np.concatenate([p_no_c.flatten(), p_pl_c.flatten(), p_pl_s.flatten(), p_mu_c.flatten(), p_mu_s.flatten()])
    m_global = calc_metrics(all_t, all_p)
    
    plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
    
    # =========================================================================
    # FIGURE 1: ADHESION & NICHE SPECIALIZATION TRAITS
    # =========================================================================
    print("Generating Figure 1: Adhesion Traits & Biofilm Specialization...")
    fig = plt.figure(figsize=(15, 12), dpi=300)
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28, left=0.08, right=0.96, top=0.93, bottom=0.07)
    
    adhesion_ratio_mucin = k_att_mucin_np / (k_shed_np + 1e-4)
    adhesion_ratio_plain = k_att_plain_np / (k_shed_np + 1e-4)
    theta_c_mag = np.linalg.norm(Theta_c_np, axis=1)
    theta_s_mag = np.linalg.norm(Theta_s_np, axis=1)
    
    clean_strains = [s.replace('-', ' ') for s in strains_keep]
    df_traits = pd.DataFrame({
        'Strain': clean_strains,
        'k_shed': k_shed_np,
        'k_att_plain': k_att_plain_np,
        'k_att_mucin': k_att_mucin_np,
        'ratio_plain': adhesion_ratio_plain,
        'ratio_mucin': adhesion_ratio_mucin,
        'theta_c_mag': theta_c_mag,
        'theta_s_mag': theta_s_mag,
        'theta_ratio': theta_c_mag / (theta_s_mag + 1e-4),
        'alpha': alpha_np
    })
    
    # -------------------------------------------------------------
    # Panel A: Adhesion Preference Ratio (Mucin Agar)
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    df_sorted = df_traits.sort_values(by='ratio_mucin', ascending=True)
    extremes = pd.concat([df_sorted.head(10), df_sorted.tail(10)])
    colors = ['#d95f02']*10 + ['#00356B']*10
    y_pos = np.arange(len(extremes))
    
    ax_a.barh(y_pos, extremes['ratio_mucin'], color=colors, alpha=0.88, edgecolor='black', lw=0.6)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([s if len(s) < 28 else s[:25]+'...' for s in extremes['Strain']], fontsize=8.5)
    ax_a.axvline(1.0, color='gray', linestyle='--', lw=1.2, alpha=0.8)
    ax_a.set_xlabel("Adhesion Partitioning Ratio ($k_{\\mathrm{attach}}^{\\mathrm{mucin}} / k_{\\mathrm{shed}}$)", fontsize=11, fontweight='bold')
    ax_a.set_title("A. Learned Adhesion Trait Divergence (Mucin Agar)\n(Orange: Liquid Shedders | Navy: Biofilm Colonizers)", 
                   fontsize=11.5, fontweight='bold', pad=10)
    ax_a.grid(True, axis='x', ls=':', alpha=0.5)
    
    # -------------------------------------------------------------
    # Panel B: Substrate Affinity: Mucin vs Plain Agar Attachment
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(df_traits['k_att_plain'], df_traits['k_att_mucin'], 
                 c='#0072B2', alpha=0.75, edgecolors='k', lw=0.6, s=65)
    max_k = max(df_traits['k_att_plain'].max(), df_traits['k_att_mucin'].max()) * 1.08
    ax_b.plot([0, max_k], [0, max_k], 'k--', lw=1.2, alpha=0.7, label='Equal Affinity ($y = x$)')
    
    # Annotate top mucin-specific strains
    df_traits['mucin_vs_plain'] = df_traits['k_att_mucin'] / (df_traits['k_att_plain'] + 1e-4)
    mucin_preferring = df_traits.sort_values(by='mucin_vs_plain', ascending=False).head(5)
    for _, row in mucin_preferring.iterrows():
        short_name = " ".join(row['Strain'].split()[:2])
        ax_b.annotate(short_name, (row['k_att_plain'], row['k_att_mucin']),
                      xytext=(6, 3), textcoords='offset points', fontsize=8, fontweight='bold', color='#D55E00')
                      
    ax_b.set_title(r"B. Hydrogel Substrate Affinity" "\n" r"($k_{\mathrm{attach}}^{\mathrm{mucin}}$ vs $k_{\mathrm{attach}}^{\mathrm{plain}}$, in $\mathrm{day}^{-1}$)", 
                   fontsize=11.5, fontweight='bold', pad=10)
    ax_b.set_xlabel(r"Plain Agar Attachment Rate, $k_{\mathrm{attach}}^{\mathrm{plain}}$ ($\mathrm{day}^{-1}$)", fontsize=11, fontweight='bold')
    ax_b.set_ylabel(r"Mucin Agar Attachment Rate, $k_{\mathrm{attach}}^{\mathrm{mucin}}$ ($\mathrm{day}^{-1}$)", fontsize=11, fontweight='bold')
    ax_b.set_xlim(-0.02, max_k)
    ax_b.set_ylim(-0.02, max_k)
    ax_b.legend(frameon=True, fontsize=9.5, loc='lower right')
    ax_b.grid(True, ls=':', alpha=0.5)
    
    # -------------------------------------------------------------
    # Panel C: Attached Biofilm vs Planktonic Metabolic Shift (||Theta_c|| vs ||Theta_s||)
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    sc = ax_c.scatter(df_traits['theta_s_mag'], df_traits['theta_c_mag'], 
                      c=np.log10(df_traits['ratio_mucin']), cmap='coolwarm',
                      edgecolors='k', lw=0.6, s=75, alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax_c, pad=0.03)
    cbar.set_label("$\\log_{10}(k_{\\mathrm{attach}} / k_{\\mathrm{shed}})$", fontsize=10)
    
    max_th = max(df_traits['theta_s_mag'].max(), df_traits['theta_c_mag'].max()) * 1.08
    ax_c.plot([0, max_th], [0, max_th], 'k--', lw=1.2, alpha=0.7, label='Equal Affinity ($y = x$)')
    
    # Highlight top biofilm metabolic upregulators
    biofilm_metab = df_traits.sort_values(by='theta_ratio', ascending=False).head(4)
    for _, row in biofilm_metab.iterrows():
        short_name = " ".join(row['Strain'].split()[:2])
        ax_c.annotate(short_name, (row['theta_s_mag'], row['theta_c_mag']),
                      xytext=(5, 3), textcoords='offset points', fontsize=8, fontweight='bold', color='#00356B')
                      
    ax_c.set_title("C. Phenotypic Shift: Biofilm vs Planktonic Traits\n($\\|\\mathbf{\\Theta}^{(c)}\\|$ vs $\\|\\mathbf{\\Theta}^{(s)}\\|$)", 
                   fontsize=11.5, fontweight='bold', pad=10)
    ax_c.set_xlabel("Floating Planktonic Trait Magnitude $\\|\\mathbf{\\Theta}^{(s)}\\|$", fontsize=11, fontweight='bold')
    ax_c.set_ylabel("Attached Biofilm Trait Magnitude $\\|\\mathbf{\\Theta}^{(c)}\\|$", fontsize=11, fontweight='bold')
    ax_c.set_xlim(0.4, max_th)
    ax_c.set_ylim(0.4, max_th)
    ax_c.legend(frameon=True, fontsize=9.5, loc='upper left')
    ax_c.grid(True, ls=':', alpha=0.5)
    
    # -------------------------------------------------------------
    # Panel D: Adhesion Ratio vs Biofilm Metabolic Ratio
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.scatter(df_traits['ratio_mucin'], df_traits['theta_ratio'], 
                 c='#2CA02C', alpha=0.75, edgecolors='k', lw=0.6, s=65)
    ax_d.axvline(1.0, color='gray', linestyle='--', lw=1, alpha=0.7)
    ax_d.axhline(1.0, color='gray', linestyle='--', lw=1, alpha=0.7)
    
    # Annotate top coupled colonizers
    top_colonizers = df_traits[(df_traits['ratio_mucin'] > 2.5) & (df_traits['theta_ratio'] > 1.0)]
    for _, row in top_colonizers.head(4).iterrows():
        short_name = " ".join(row['Strain'].split()[:2])
        ax_d.annotate(short_name, (row['ratio_mucin'], row['theta_ratio']),
                      xytext=(5, 3), textcoords='offset points', fontsize=8, fontweight='bold', color='#1B7837')
                      
    r_corr, _ = pearsonr(np.log10(df_traits['ratio_mucin']), df_traits['theta_ratio'])
    ax_d.set_title(f"D. Physical Adhesion vs Metabolic Specialization\n(Pearson r = {r_corr:+.2f})", 
                   fontsize=11.5, fontweight='bold', pad=10)
    ax_d.set_xlabel(r"Physical Adhesion Ratio ($k_{\mathrm{attach}}^{\mathrm{mucin}} / k_{\mathrm{shed}}$)", fontsize=11, fontweight='bold')
    ax_d.set_ylabel(r"Biofilm Metabolic Shift ($\|\mathbf{\Theta}^{(c)}\| / \|\mathbf{\Theta}^{(s)}\|$)", fontsize=11, fontweight='bold')
    ax_d.grid(True, ls=':', alpha=0.5)
    
    fig1_path = os.path.join(PROJECT_ROOT, 'Pollard_Coupled_Biofilm_Adhesion_Traits.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # =========================================================================
    # FIGURE 2: SIMULTANEOUS PARITY / ACCURACY ACROSS ALL 3 CONDITIONS
    # =========================================================================
    print("Generating Figure 2: Simultaneous Parity Across 3 Real Conditions...")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), dpi=300)
    
    # Panel A: No Carrier
    ax = axes[0]
    ax.scatter(t_no_c.flatten(), p_no_c.flatten(), color='#00356B', alpha=0.35, edgecolors='none', s=24)
    max_no_c = max(t_no_c.max(), p_no_c.max()) * 1.05
    ax.plot([0, max_no_c], [0, max_no_c], 'k--', lw=1.5, alpha=0.7)
    ax.set_title(f"A. No Carrier (Liquid Broth)\n$R^2 = {m_no_c['r2']:.2f}$ | $r = {m_no_c['r']:.2f}$ | $\\rho = {m_no_c['rho']:.2f}$", 
                 fontsize=11.5, fontweight='bold', pad=10)
    ax.set_xlabel("Observed Relative Abundance", fontsize=10.5, fontweight='bold')
    ax.set_ylabel("Coupled ODE Predicted Abundance", fontsize=10.5, fontweight='bold')
    ax.set_xlim(-0.01, max_no_c)
    ax.set_ylim(-0.01, max_no_c)
    ax.grid(True, ls=':', alpha=0.4)
    
    # Panel B: Plain Agar
    ax = axes[1]
    ax.scatter(t_pl_c.flatten(), p_pl_c.flatten(), color='#2b5c8f', alpha=0.35, label='Carrier Biofilm', s=20)
    ax.scatter(t_pl_s.flatten(), p_pl_s.flatten(), color='#d95f02', alpha=0.35, label='Liquid Supernatant', s=20)
    max_pl = max(t_pl_c.max(), p_pl_c.max(), t_pl_s.max(), p_pl_s.max()) * 1.05
    ax.plot([0, max_pl], [0, max_pl], 'k--', lw=1.5, alpha=0.7)
    ax.set_title(f"B. Plain Agar Microcosm\n$R^2 = {m_pl_comb['r2']:.2f}$ | $r = {m_pl_comb['r']:.2f}$ | $\\rho = {m_pl_comb['rho']:.2f}$", 
                 fontsize=11.5, fontweight='bold', pad=10)
    ax.set_xlabel("Observed Relative Abundance", fontsize=10.5, fontweight='bold')
    ax.set_ylabel("Coupled ODE Predicted Abundance", fontsize=10.5, fontweight='bold')
    ax.set_xlim(-0.01, max_pl)
    ax.set_ylim(-0.01, max_pl)
    ax.legend(frameon=True, fontsize=9.5, loc='upper left')
    ax.grid(True, ls=':', alpha=0.4)
    
    # Panel C: Mucin Agar
    ax = axes[2]
    ax.scatter(t_mu_c.flatten(), p_mu_c.flatten(), color='#2b5c8f', alpha=0.35, label='Carrier Biofilm', s=20)
    ax.scatter(t_mu_s.flatten(), p_mu_s.flatten(), color='#d95f02', alpha=0.35, label='Liquid Supernatant', s=20)
    max_mu = max(t_mu_c.max(), p_mu_c.max(), t_mu_s.max(), p_mu_s.max()) * 1.05
    ax.plot([0, max_mu], [0, max_mu], 'k--', lw=1.5, alpha=0.7)
    ax.set_title(f"C. Mucin Agar Microcosm\n$R^2 = {m_mu_comb['r2']:.2f}$ | $r = {m_mu_comb['r']:.2f}$ | $\\rho = {m_mu_comb['rho']:.2f}$", 
                 fontsize=11.5, fontweight='bold', pad=10)
    ax.set_xlabel("Observed Relative Abundance", fontsize=10.5, fontweight='bold')
    ax.set_ylabel("Coupled ODE Predicted Abundance", fontsize=10.5, fontweight='bold')
    ax.set_xlim(-0.01, max_mu)
    ax.set_ylim(-0.01, max_mu)
    ax.legend(frameon=True, fontsize=9.5, loc='upper left')
    ax.grid(True, ls=':', alpha=0.4)
    
    # Panel D: Global Multi-Condition Combined
    ax = axes[3]
    ax.scatter(t_no_c.flatten(), p_no_c.flatten(), color='#7570b3', alpha=0.25, label=f"Liquid Broth ($R^2={m_no_c['r2']:.2f}$)", s=18)
    ax.scatter(t_pl_c.flatten(), p_pl_c.flatten(), color='#2b5c8f', alpha=0.25, label=f"Plain Carrier ($R^2={m_pl_c['r2']:.2f}$)", s=18)
    ax.scatter(t_pl_s.flatten(), p_pl_s.flatten(), color='#e6ab02', alpha=0.25, label=f"Plain Supern ($R^2={m_pl_s['r2']:.2f}$)", s=18)
    ax.scatter(t_mu_c.flatten(), p_mu_c.flatten(), color='#1b9e77', alpha=0.25, label=f"Mucin Carrier ($R^2={m_mu_c['r2']:.2f}$)", s=18)
    ax.scatter(t_mu_s.flatten(), p_mu_s.flatten(), color='#d95f02', alpha=0.25, label=f"Mucin Supern ($R^2={m_mu_s['r2']:.2f}$)", s=18)
    overall_max = max(max_no_c, max_pl, max_mu)
    ax.plot([0, overall_max], [0, overall_max], 'k--', lw=1.5, alpha=0.7)
    ax.set_title(f"D. Global Multi-Condition Parity\n$R^2 = {m_global['r2']:.2f}$ | $r = {m_global['r']:.2f}$ | $\\rho = {m_global['rho']:.2f}$ | MAE = {m_global['mae']:.3f}", 
                 fontsize=11.5, fontweight='bold', pad=10)
    ax.set_xlabel("Observed Relative Abundance", fontsize=10.5, fontweight='bold')
    ax.set_ylabel("Coupled ODE Predicted Abundance", fontsize=10.5, fontweight='bold')
    ax.set_xlim(-0.01, overall_max)
    ax.set_ylim(-0.01, overall_max)
    ax.legend(frameon=True, fontsize=8.5, loc='upper left')
    ax.grid(True, ls=':', alpha=0.4)
    
    plt.tight_layout()
    fig2_path = os.path.join(PROJECT_ROOT, 'Pollard_Coupled_Biofilm_Simultaneous_Fit.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # =========================================================================
    # FIGURE 3: LONGITUDINAL PASSAGE TRAJECTORIES (P1..P6)
    # =========================================================================
    print("Generating Figure 3: Longitudinal Passage Dynamics...")
    # Select 4 representative strains across functional niches:
    # 1. Top Biofilm Colonizer (Clostridium sporogenes)
    # 2. Top Liquid Shedder (Bacteroides ovatus)
    # 3. Top Mucin-Specific Colonizer (Bacteroides sp 2-1-22)
    # 4. Abundant Prevalent Generalist (Phocaeicola vulgatus / Bacteroides fragilis)
    
    strains_chosen = [
        'Clostridium-sporogenes-ATCC-15579',
        'Bacteroides-ovatus-ATCC-8483',
        'Bacteroides-sp-2-1-22',
        'Bacteroides-fragilis-2-1-16'
    ]
    strains_indices = [strains_keep.index(s) for s in strains_chosen]
    strain_labels = [
        'C. sporogenes\n(Biofilm Specialist: High Adhesion)',
        'B. ovatus\n(Planktonic Shedder: High Shedding)',
        'Bacteroides sp 2-1-22\n(Mucin-Specific Colonizer)',
        'B. fragilis 2-1-16\n(Abundant Gut Commensal)'
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 11), sharex=True, dpi=300)
    passages_num = [1, 2, 3, 4, 5, 6]
    rep_idx = 0 # first replicate
    
    for col_idx, (s_idx, s_label) in enumerate(zip(strains_indices, strain_labels)):
        # Row 0: Carrier Biofilm (Plain vs Mucin)
        ax_c = axes[0, col_idx]
        obs_mu_c = mucin_c_tensor[rep_idx, :, s_idx].cpu().numpy()
        pred_mu_c = np.concatenate([[obs_mu_c[0]], p_mu_c[rep_idx, :, s_idx]])
        obs_pl_c = plain_c_tensor[rep_idx, :, s_idx].cpu().numpy()
        pred_pl_c = np.concatenate([[obs_pl_c[0]], p_pl_c[rep_idx, :, s_idx]])
        
        ax_c.plot(passages_num, obs_mu_c, 'o-', color='#1b9e77', lw=2.2, ms=6, label='Observed Mucin Carrier')
        ax_c.plot(passages_num, pred_mu_c, 's--', color='#66c2a5', lw=1.8, ms=5, label='Coupled ODE Mucin Carrier')
        ax_c.plot(passages_num, obs_pl_c, '^-', color='#2b5c8f', lw=1.8, ms=5, alpha=0.7, label='Observed Plain Carrier')
        ax_c.plot(passages_num, pred_pl_c, 'v--', color='#8da0cb', lw=1.5, ms=4, alpha=0.7, label='Coupled ODE Plain Carrier')
        
        ax_c.set_title(f"{s_label}\n[Carrier Biofilm Compartment]", fontsize=10.5, fontweight='bold')
        ax_c.set_ylabel("Relative Abundance", fontsize=10)
        ax_c.set_xticks(passages_num)
        ax_c.set_xticklabels([f'P{i}' for i in passages_num])
        ax_c.grid(True, ls=':', alpha=0.4)
        if col_idx == 0:
            ax_c.legend(frameon=True, fontsize=8.2, loc='best')
            
        # Row 1: Liquid Supernatant (Plain vs Mucin)
        ax_s = axes[1, col_idx]
        obs_mu_s = mucin_s_tensor[rep_idx, :, s_idx].cpu().numpy()
        pred_mu_s = np.concatenate([[obs_mu_s[0]], p_mu_s[rep_idx, :, s_idx]])
        obs_pl_s = plain_s_tensor[rep_idx, :, s_idx].cpu().numpy()
        pred_pl_s = np.concatenate([[obs_pl_s[0]], p_pl_s[rep_idx, :, s_idx]])
        
        ax_s.plot(passages_num, obs_mu_s, 'o-', color='#d95f02', lw=2.2, ms=6, label='Observed Mucin Supernatant')
        ax_s.plot(passages_num, pred_mu_s, 's--', color='#fc8d59', lw=1.8, ms=5, label='Coupled ODE Mucin Supernatant')
        ax_s.plot(passages_num, obs_pl_s, '^-', color='#e6ab02', lw=1.8, ms=5, alpha=0.7, label='Observed Plain Supernatant')
        ax_s.plot(passages_num, pred_pl_s, 'v--', color='#fdae61', lw=1.5, ms=4, alpha=0.7, label='Coupled ODE Plain Supernatant')
        
        ax_s.set_title(f"[Liquid Supernatant Compartment]", fontsize=10, fontweight='bold')
        ax_s.set_ylabel("Relative Abundance", fontsize=10)
        ax_s.set_xticks(passages_num)
        ax_s.set_xticklabels([f'P{i}' for i in passages_num])
        ax_s.grid(True, ls=':', alpha=0.4)
        if col_idx == 0:
            ax_s.legend(frameon=True, fontsize=8.2, loc='best')
            
        # Row 2: Pure Liquid Broth (No Carrier)
        ax_l = axes[2, col_idx]
        obs_l = no_c_tensor[rep_idx, :, s_idx].cpu().numpy()
        pred_l = np.concatenate([[obs_l[0]], p_no_c[rep_idx, :, s_idx]])
        
        ax_l.plot(passages_num, obs_l, 'o-', color='#7570b3', lw=2.2, ms=6, label='Observed No Carrier')
        ax_l.plot(passages_num, pred_l, 's--', color='#bc80bd', lw=1.8, ms=5, label='Coupled ODE No Carrier')
        
        ax_l.set_title(f"[Pure Liquid (No Carrier)]", fontsize=10, fontweight='bold')
        ax_l.set_xlabel("Passage Cycle", fontsize=10.5, fontweight='bold')
        ax_l.set_ylabel("Relative Abundance", fontsize=10)
        ax_l.set_xticks(passages_num)
        ax_l.set_xticklabels([f'P{i}' for i in passages_num])
        ax_l.grid(True, ls=':', alpha=0.4)
        if col_idx == 0:
            ax_l.legend(frameon=True, fontsize=8.5, loc='best')
            
    plt.tight_layout()
    fig3_path = os.path.join(PROJECT_ROOT, 'Pollard_Coupled_Biofilm_Longitudinal_Trajectories.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")
    
    # Copy to Artifact directory
    artifact_dir = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
    if os.path.exists(artifact_dir):
        import shutil
        shutil.copy(fig1_path, os.path.join(artifact_dir, 'Pollard_Coupled_Biofilm_Adhesion_Traits.png'))
        shutil.copy(fig2_path, os.path.join(artifact_dir, 'Pollard_Coupled_Biofilm_Simultaneous_Fit.png'))
        shutil.copy(fig3_path, os.path.join(artifact_dir, 'Pollard_Coupled_Biofilm_Longitudinal_Trajectories.png'))
        print(f"Copied figures to artifact directory: {artifact_dir}")
        
    print("ALL FIGURES SUCCESSFULLY GENERATED!")

if __name__ == '__main__':
    main()

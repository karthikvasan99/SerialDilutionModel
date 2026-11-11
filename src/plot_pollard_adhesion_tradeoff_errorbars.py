"""
Plotting Script for Pollard Physical Adhesion vs Metabolic Specialization
with Multi-Run Ensemble Error Bars (N=10 Independent Initializations):
1. Focus Figure: Pollard_Coupled_Biofilm_Adhesion_vs_Metabolic_Tradeoff_ErrorBars.png
2. Updated 4-Panel Figure: Pollard_Coupled_Biofilm_Adhesion_Traits.png (Plot D updated with error bars)
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
from scipy.stats import pearsonr, spearmanr, linregress

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def assign_phylum(strain_name):
    s = strain_name.lower()
    if any(k in s for k in ['bacteroides', 'parabacteroides', 'prevotella', 'aliistipes', 'odoribacter', 'barnesiella']):
        return 'Bacteroidota'
    elif any(k in s for k in ['clostridium', 'ruminococcus', 'faecalibacterium', 'coprococcus', 'eubacterium', 
                             'roseburia', 'anaerostipes', 'dorea', 'blautia', 'lachnospira', 'streptococcus', 
                             'enterococcus', 'holdemanella']):
        return 'Bacillota (Firmicutes)'
    elif any(k in s for k in ['veillonella', 'dialister', 'megasphaera', 'phascolarctobacterium']):
        return 'Negativicutes'
    elif any(k in s for k in ['bifidobacterium', 'collinsella', 'eggerthella', 'actinomyces', 'adlercreutzia']):
        return 'Actinomycetota'
    elif any(k in s for k in ['escherichia', 'klebsiella', 'desulfovibrio', 'sutterella', 'bilophila', 'parasutterella']):
        return 'Pseudomonadota'
    else:
        return 'Other'

def plot_errorbar_figures(data_path=None):
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_DualTheta_MultiRun_10Seeds.pt')
        
    print(f"Loading multi-run data from {data_path}...")
    results = torch.load(data_path, map_location='cpu', weights_only=False)
    runs_data = results['runs_data']
    strains = results['strains']
    N_strains = len(strains)
    N_runs = len(runs_data)
    
    # Collect arrays across runs: shape (N_runs, N_strains)
    ratios_mucin = np.array([r['ratio_mucin'] for r in runs_data])
    ratios_plain = np.array([r['ratio_plain'] for r in runs_data])
    theta_ratios = np.array([r['theta_ratio'] for r in runs_data])
    th_c_norms = np.array([r['th_c_norm'] for r in runs_data])
    th_s_norms = np.array([r['th_s_norm'] for r in runs_data])
    k_sheds = np.array([r['k_shed'] for r in runs_data])
    k_att_mucins = np.array([r['k_att_mucin'] for r in runs_data])
    k_att_plains = np.array([r['k_att_plain'] for r in runs_data])
    
    # Summary stats across runs
    log_ratios_mucin = np.log10(ratios_mucin)
    
    mean_ratio = np.mean(ratios_mucin, axis=0)
    sem_ratio = np.std(ratios_mucin, axis=0) / np.sqrt(N_runs)
    
    mean_log_ratio = np.mean(log_ratios_mucin, axis=0)
    sem_log_ratio = np.std(log_ratios_mucin, axis=0) / np.sqrt(N_runs)
    sd_log_ratio = np.std(log_ratios_mucin, axis=0)
    
    mean_theta_ratio = np.mean(theta_ratios, axis=0)
    sem_theta_ratio = np.std(theta_ratios, axis=0) / np.sqrt(N_runs)
    sd_theta_ratio = np.std(theta_ratios, axis=0)
    
    mean_th_c = np.mean(th_c_norms, axis=0)
    sem_th_c = np.std(th_c_norms, axis=0) / np.sqrt(N_runs)
    
    mean_th_s = np.mean(th_s_norms, axis=0)
    sem_th_s = np.std(th_s_norms, axis=0) / np.sqrt(N_runs)
    
    # Ensemble correlation metrics across runs
    run_pearsons = [r['pearson_r'] for r in runs_data]
    run_spearmans = [r['spearman_rho'] for r in runs_data]
    mean_r, std_r = np.mean(run_pearsons), np.std(run_pearsons)
    mean_rho, std_rho = np.mean(run_spearmans), np.std(run_spearmans)
    
    # Global mean points correlation
    r_mean, p_mean = pearsonr(mean_log_ratio, mean_theta_ratio)
    rho_mean, p_rho_mean = spearmanr(mean_ratio, mean_theta_ratio)
    slope, intercept, r_val, p_val, std_err = linregress(mean_log_ratio, mean_theta_ratio)
    
    print("\n" + "="*70)
    print("ENSEMBLE ADHESION vs METABOLISM TRADE-OFF STATISTICS (N=10 SEEDS)")
    print("="*70)
    print(f"Mean Pearson r across runs:    {mean_r:+.3f} +/- {std_r:.3f}")
    print(f"Mean Spearman rho across runs:  {mean_rho:+.3f} +/- {std_rho:.3f}")
    print(f"Ensemble Mean Points Correlation: Pearson r = {r_mean:+.3f} (p = {p_mean:.3e}) | Spearman rho = {rho_mean:+.3f}")
    print(f"Linear Fit: slope = {slope:+.3f} +/- {std_err:.3f}, intercept = {intercept:.3f}")
    
    # Create DataFrame for plotting
    clean_names = [s.replace('-', ' ') for s in strains]
    phyla = [assign_phylum(s) for s in clean_names]
    
    df_ensemble = pd.DataFrame({
        'Strain': clean_names,
        'Phylum': phyla,
        'Mean_Ratio': mean_ratio,
        'SEM_Ratio': sem_ratio,
        'Mean_Log_Ratio': mean_log_ratio,
        'SEM_Log_Ratio': sem_log_ratio,
        'SD_Log_Ratio': sd_log_ratio,
        'Mean_Theta_Ratio': mean_theta_ratio,
        'SEM_Theta_Ratio': sem_theta_ratio,
        'SD_Theta_Ratio': sd_theta_ratio,
        'Mean_Th_C': mean_th_c,
        'Mean_Th_S': mean_th_s
    })
    
    # Color palette by Phylum
    phylum_colors = {
        'Bacteroidota': '#0072B2',            # Blue
        'Bacillota (Firmicutes)': '#D55E00',  # Vermillion / Red-Orange
        'Negativicutes': '#009E73',           # Greenish Teal
        'Actinomycetota': '#CC79A7',          # Pink/Purple
        'Pseudomonadota': '#F0E442',          # Gold
        'Other': '#999999'                    # Gray
    }
    
    # =========================================================================
    # FIGURE A: DEDICATED HIGH-RES FOCUS FIGURE (PLOT D WITH 2D ERROR BARS)
    # =========================================================================
    plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
    fig = plt.figure(figsize=(13, 9.5), dpi=300)
    gs = gridspec.GridSpec(1, 1, left=0.10, right=0.96, top=0.91, bottom=0.11)
    ax = fig.add_subplot(gs[0, 0])
    
    # Draw reference dashed lines
    ax.axvline(0.0, color='#7F8C8D', linestyle='--', lw=1.2, alpha=0.7, zorder=1) # log10(1) = 0
    ax.axhline(1.0, color='#7F8C8D', linestyle='--', lw=1.2, alpha=0.7, zorder=1)
    
    # Regression line and 95% Confidence Interval
    x_grid = np.linspace(df_ensemble['Mean_Log_Ratio'].min() - 0.2, df_ensemble['Mean_Log_Ratio'].max() + 0.2, 200)
    y_pred = intercept + slope * x_grid
    # Analytical 95% CI
    n = len(df_ensemble)
    x_mean = df_ensemble['Mean_Log_Ratio'].mean()
    ss_x = np.sum((df_ensemble['Mean_Log_Ratio'] - x_mean)**2)
    s_err = np.sqrt(np.sum((df_ensemble['Mean_Theta_Ratio'] - (intercept + slope * df_ensemble['Mean_Log_Ratio']))**2) / (n - 2))
    ci = 1.96 * s_err * np.sqrt(1.0/n + (x_grid - x_mean)**2 / ss_x)
    
    ax.fill_between(x_grid, y_pred - ci, y_pred + ci, color='#34495E', alpha=0.12, label='95% Regression Confidence Band', zorder=2)
    ax.plot(x_grid, y_pred, color='#2C3E50', lw=2.2, linestyle='-', 
            label=f'Trade-Off Regression: Slope = {slope:+.2f} ($r = {r_mean:+.2f}, p = {p_mean:.2e}$)', zorder=3)
    
    # Plot each microbe with 2D error bars (SEM)
    for phy, grp in df_ensemble.groupby('Phylum'):
        color = phylum_colors.get(phy, '#999999')
        ax.errorbar(
            grp['Mean_Log_Ratio'], grp['Mean_Theta_Ratio'],
            xerr=grp['SEM_Log_Ratio'], yerr=grp['SEM_Theta_Ratio'],
            fmt='o', color=color, ecolor=color, elinewidth=1.2, capsize=3.0, capthick=1.0,
            markersize=7.5, alpha=0.85, markeredgecolor='black', markeredgewidth=0.7,
            label=f"{phy} ($n={len(grp)}$)", zorder=4
        )
        
    # Annotate key notable microbes across the trade-off continuum
    # 1. Top Biofilm Colonizers (High Adhesion, Modest Biofilm Metabolism)
    # 2. Top Planktonic Shedders (Low Adhesion, High Planktonic Metabolism)
    # 3. Intermediate / Outliers
    annot_candidates = [
        ('Clostridium sporogenes ATCC 15579', (-70, 15)),
        ('Bacteroides sp 2 1 22', (15, -15)),
        ('Clostridium sp D5', (15, 12)),
        ('Veillonella sp 6 1 27', (15, -15)),
        ('Bacteroides ovatus ATCC 8483', (-95, -15)),
        ('Dialister invisus DSM 15470', (-80, 15)),
        ('Parabacteroides sp D13', (-85, -18)),
        ('Clostridium sp ATCC 29733', (15, 12)),
        ('Coprococcus comes ATCC 27758', (-90, 15)),
        ('Bacteroides fragilis 2 1 16', (15, -15)),
        ('Desulfovibrio piger ATCC 29098', (15, -12)),
        ('Anaerostipes caccae DSM 14662', (15, -12))
    ]
    
    for strain_full, offset in annot_candidates:
        match = df_ensemble[df_ensemble['Strain'] == strain_full]
        if len(match) > 0:
            row = match.iloc[0]
            words = strain_full.split()
            short_name = f"{words[0][0]}. {' '.join(words[1:3])}"
            ax.annotate(
                short_name,
                xy=(row['Mean_Log_Ratio'], row['Mean_Theta_Ratio']),
                xytext=offset, textcoords='offset points',
                fontsize=8.5, fontweight='bold', color='#1A252F',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#BDC3C7', alpha=0.85, lw=0.6),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='#7F8C8D', lw=0.9),
                zorder=6
            )
            
    # Sub-quadrant labels
    ax.text(0.97, 0.05, "Biofilm Retention Specialists\n(High Adhesion, Lower Metabolic Yield)", 
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9.5, fontweight='semibold',
            color='#1B4F72', bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB', edgecolor='#AED6F1', alpha=0.9))
            
    ax.text(0.03, 0.95, "Planktonic Opportunists\n(Low Adhesion, High Metabolic Yield)", 
            transform=ax.transAxes, ha='left', va='top', fontsize=9.5, fontweight='semibold',
            color='#78281F', bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDEDEC', edgecolor='#F5B7B1', alpha=0.9))

    ax.set_xlabel(r"Physical Adhesion Ratio $\log_{10}(k_{\mathrm{attach}}^{\mathrm{mucin}} / k_{\mathrm{shed}})$ [Mean $\pm$ SEM, $N=10$ Seeds]", 
                  fontsize=12.5, fontweight='bold', labelpad=9)
    ax.set_ylabel(r"Biofilm Metabolic Shift $\|\mathbf{\Theta}^{(c)}\| / \|\mathbf{\Theta}^{(s)}\|$ [Mean $\pm$ SEM, $N=10$ Seeds]", 
                  fontsize=12.5, fontweight='bold', labelpad=9)
    ax.set_title(f"Robust Physiological Trade-Off: Physical Adhesion vs. Biofilm Metabolic Shift\n"
                 f"Ensemble Correlation: Pearson $r = {r_mean:+.2f}$ ($p = {p_mean:.2e}$) | Spearman $\\rho = {rho_mean:+.2f}$ ($N=50$ Gut Strains across 10 Random Initializations)",
                 fontsize=13.5, fontweight='bold', pad=14)
                 
    ax.grid(True, ls=':', alpha=0.45, zorder=0)
    ax.legend(frameon=True, fontsize=9.5, loc='upper right', framealpha=0.92, edgecolor='#BDC3C7')
    
    focus_fig_path = os.path.join(PROJECT_ROOT, 'Pollard_Coupled_Biofilm_Adhesion_vs_Metabolic_Tradeoff_ErrorBars.png')
    plt.savefig(focus_fig_path, dpi=300)
    plt.close()
    print(f"Saved dedicated focus figure to: {focus_fig_path}")
    
    # =========================================================================
    # FIGURE B: UPDATE PLOT D IN THE 4-PANEL ADHESION TRAITS FIGURE
    # =========================================================================
    print("Regenerating 4-panel Adhesion Traits figure with updated Plot D error bars...")
    # Read existing figure script logic and recreate the 4-panel layout
    fig = plt.figure(figsize=(15, 12), dpi=300)
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28, left=0.08, right=0.96, top=0.93, bottom=0.07)
    
    # Reference data from mean ensemble
    df_sorted = df_ensemble.sort_values(by='Mean_Ratio', ascending=True)
    extremes = pd.concat([df_sorted.head(10), df_sorted.tail(10)])
    colors = ['#d95f02']*10 + ['#00356B']*10
    y_pos = np.arange(len(extremes))
    
    # Panel A: Adhesion Partitioning Ratio (with error bars!)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.barh(y_pos, extremes['Mean_Ratio'], xerr=extremes['SEM_Ratio'], 
              color=colors, alpha=0.88, edgecolor='black', lw=0.6,
              error_kw=dict(ecolor='black', lw=0.9, capsize=2.5))
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([s if len(s) < 28 else s[:25]+'...' for s in extremes['Strain']], fontsize=8.5)
    ax_a.axvline(1.0, color='gray', linestyle='--', lw=1.2, alpha=0.8)
    ax_a.set_xlabel(r"Adhesion Partitioning Ratio ($k_{\mathrm{attach}}^{\mathrm{mucin}} / k_{\mathrm{shed}}$) [Mean $\pm$ SEM]", fontsize=11, fontweight='bold')
    ax_a.set_title("A. Learned Adhesion Trait Divergence (Mucin Agar)\n(Orange: Liquid Shedders | Navy: Biofilm Colonizers)", 
                   fontsize=11.5, fontweight='bold', pad=10)
    ax_a.grid(True, axis='x', ls=':', alpha=0.5)
    
    # Panel B: Substrate Affinity: Mucin vs Plain Agar Attachment
    ax_b = fig.add_subplot(gs[0, 1])
    mean_k_plain = np.mean(k_att_plains, axis=0)
    sem_k_plain = np.std(k_att_plains, axis=0) / np.sqrt(N_runs)
    mean_k_mucin = np.mean(k_att_mucins, axis=0)
    sem_k_mucin = np.std(k_att_mucins, axis=0) / np.sqrt(N_runs)
    
    ax_b.errorbar(mean_k_plain, mean_k_mucin, xerr=sem_k_plain, yerr=sem_k_mucin,
                  fmt='o', color='#0072B2', ecolor='#0072B2', elinewidth=1.0, capsize=2.0,
                  markersize=6.0, alpha=0.75, markeredgecolor='black', markeredgewidth=0.5)
    max_k = max(mean_k_plain.max(), mean_k_mucin.max()) * 1.08
    ax_b.plot([0, max_k], [0, max_k], 'k--', lw=1.2, alpha=0.7, label='Equal Affinity ($y = x$)')
    
    # Annotate top mucin-specific strains
    mucin_spec = mean_k_mucin / (mean_k_plain + 1e-4)
    top_mucin_idx = np.argsort(mucin_spec)[::-1][:5]
    for idx in top_mucin_idx:
        short_name = " ".join(clean_names[idx].split()[:2])
        ax_b.annotate(short_name, (mean_k_plain[idx], mean_k_mucin[idx]),
                      xytext=(6, 3), textcoords='offset points', fontsize=8, fontweight='bold', color='#D55E00')
                      
    ax_b.set_title(r"B. Hydrogel Substrate Affinity [Mean $\pm$ SEM across $N=10$ Seeds]" "\n" r"($k_{\mathrm{attach}}^{\mathrm{mucin}}$ vs $k_{\mathrm{attach}}^{\mathrm{plain}}$, in $\mathrm{day}^{-1}$)", 
                   fontsize=11.5, fontweight='bold', pad=10)
    ax_b.set_xlabel(r"Plain Agar Attachment Rate, $k_{\mathrm{attach}}^{\mathrm{plain}}$ ($\mathrm{day}^{-1}$)", fontsize=11, fontweight='bold')
    ax_b.set_ylabel(r"Mucin Agar Attachment Rate, $k_{\mathrm{attach}}^{\mathrm{mucin}}$ ($\mathrm{day}^{-1}$)", fontsize=11, fontweight='bold')
    ax_b.set_xlim(-0.02, max_k)
    ax_b.set_ylim(-0.02, max_k)
    ax_b.legend(frameon=True, fontsize=9.5, loc='lower right')
    ax_b.grid(True, ls=':', alpha=0.5)
    
    # Panel C: Biofilm vs Planktonic Metabolic Shift (||Theta_c|| vs ||Theta_s||)
    ax_c = fig.add_subplot(gs[1, 0])
    sc = ax_c.scatter(df_ensemble['Mean_Th_S'], df_ensemble['Mean_Th_C'], 
                      c=df_ensemble['Mean_Log_Ratio'], cmap='coolwarm',
                      edgecolors='k', lw=0.6, s=70, alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax_c, pad=0.03)
    cbar.set_label(r"$\log_{10}(k_{\mathrm{attach}} / k_{\mathrm{shed}})$", fontsize=10)
    max_th = max(df_ensemble['Mean_Th_S'].max(), df_ensemble['Mean_Th_C'].max()) * 1.08
    ax_c.plot([0, max_th], [0, max_th], 'k--', lw=1.2, alpha=0.7, label='Equal Affinity ($y = x$)')
    
    # Annotate top biofilm metabolic upregulators
    top_th_idx = np.argsort(df_ensemble['Mean_Theta_Ratio'])[::-1][:4]
    for idx in top_th_idx:
        short_name = " ".join(clean_names[idx].split()[:2])
        ax_c.annotate(short_name, (df_ensemble['Mean_Th_S'].iloc[idx], df_ensemble['Mean_Th_C'].iloc[idx]),
                      xytext=(5, 3), textcoords='offset points', fontsize=8, fontweight='bold', color='#00356B')
                      
    ax_c.set_title("C. Phenotypic Shift: Biofilm vs Planktonic Traits\n($\\|\\mathbf{\\Theta}^{(c)}\\|$ vs $\\|\\mathbf{\\Theta}^{(s)}\\|$)", 
                   fontsize=11.5, fontweight='bold', pad=10)
    ax_c.set_xlabel(r"Floating Planktonic Trait Magnitude $\|\mathbf{\Theta}^{(s)}\|$", fontsize=11, fontweight='bold')
    ax_c.set_ylabel(r"Attached Biofilm Trait Magnitude $\|\mathbf{\Theta}^{(c)}\|$", fontsize=11, fontweight='bold')
    ax_c.set_xlim(0.4, max_th)
    ax_c.set_ylim(0.4, max_th)
    ax_c.legend(frameon=True, fontsize=9.5, loc='upper left')
    ax_c.grid(True, ls=':', alpha=0.5)
    
    # Panel D: UPDATED WITH 2D ERROR BARS (MEAN +/- SEM ACROSS N=10 SEEDS)
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axvline(0.0, color='gray', linestyle='--', lw=1.0, alpha=0.7)
    ax_d.axhline(1.0, color='gray', linestyle='--', lw=1.0, alpha=0.7)
    
    # 95% CI regression line
    ax_d.fill_between(x_grid, y_pred - ci, y_pred + ci, color='#2CA02C', alpha=0.15)
    ax_d.plot(x_grid, y_pred, color='#1B7837', lw=1.8, linestyle='-', label=f'Fit: slope={slope:+.2f}')
    
    for phy, grp in df_ensemble.groupby('Phylum'):
        color = phylum_colors.get(phy, '#999999')
        ax_d.errorbar(
            grp['Mean_Log_Ratio'], grp['Mean_Theta_Ratio'],
            xerr=grp['SEM_Log_Ratio'], yerr=grp['SEM_Theta_Ratio'],
            fmt='o', color=color, ecolor=color, elinewidth=1.0, capsize=2.5, capthick=0.8,
            markersize=6.5, alpha=0.85, markeredgecolor='black', markeredgewidth=0.5,
            label=phy
        )
        
    ax_d.set_title(f"D. Adhesion vs Metabolic Specialization (N=10 Seeds)\n"
                   f"Pearson r = {r_mean:+.2f} (p = {p_mean:.2e}) | Spearman rho = {rho_mean:+.2f}", 
                   fontsize=11.5, fontweight='bold', pad=10)
    ax_d.set_xlabel(r"Physical Adhesion $\log_{10}(k_{\mathrm{attach}}^{\mathrm{mucin}} / k_{\mathrm{shed}})$ [Mean $\pm$ SEM]", fontsize=11, fontweight='bold')
    ax_d.set_ylabel(r"Biofilm Metabolic Shift ($\|\mathbf{\Theta}^{(c)}\| / \|\mathbf{\Theta}^{(s)}\|$) [Mean $\pm$ SEM]", fontsize=11, fontweight='bold')
    ax_d.grid(True, ls=':', alpha=0.5)
    ax_d.legend(frameon=True, fontsize=8.2, loc='upper right', framealpha=0.9)
    
    traits_4panel_path = os.path.join(PROJECT_ROOT, 'Pollard_Coupled_Biofilm_Adhesion_Traits.png')
    plt.savefig(traits_4panel_path, dpi=300)
    plt.close()
    print(f"Saved updated 4-panel figure to: {traits_4panel_path}")
    
    # Copy to Artifact directory
    artifact_dir = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
    if os.path.exists(artifact_dir):
        import shutil
        shutil.copy(focus_fig_path, os.path.join(artifact_dir, 'Pollard_Coupled_Biofilm_Adhesion_vs_Metabolic_Tradeoff_ErrorBars.png'))
        shutil.copy(traits_4panel_path, os.path.join(artifact_dir, 'Pollard_Coupled_Biofilm_Adhesion_Traits.png'))
        print(f"Copied updated figures to artifact directory: {artifact_dir}")

if __name__ == '__main__':
    plot_errorbar_figures()

"""
Recreate German_Theta_TMA_Niche_Competition.png with updated, publication-ready styling:
- Panel A: Cross-Host Prescriptive Transfer: Target T6 Abundance (Log Scale, baseline at 0.1%)
- Panel B: Total Competition Faced [z-score] (formula removed from y-axis; Methanomicrobium removed from inset)
- Panel C: Learned Affinity Tradeoffs with simplified axis/colorbar labels (\theta_i^{MeOH}, \theta_i^{TMA}, \theta_i^{H2:CO2})
           and a single consolidated legend in the top right.

Generates both variants:
1. Total Competition Faced [z-score]: German_Theta_TMA_Niche_Competition.png
2. Direction-Specific Competition Faced [z-score]: German_Theta_TMA_Niche_Competition_DirectionSpecific.png
"""

import os
import shutil
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Set publication styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARTIFACT_DIR = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
PAPER_FIG_DIR = os.path.join(PROJECT_ROOT, 'Paper', 'figures')
POSTER_FIG_DIR = os.path.join(PROJECT_ROOT, 'Poster', 'figures')

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
os.makedirs(POSTER_FIG_DIR, exist_ok=True)

# 1. Load Parameters and Affinities
params_path = os.path.join(PROJECT_ROOT, 'Data', 'German_champion_model_params.pt')
params = torch.load(params_path, weights_only=False)
Theta = params['Theta'].numpy()
R0_table = params['R0_table'].numpy()
strains_81 = params['strains_81']
N = len(strains_81)

affinities_path = os.path.join(PROJECT_ROOT, 'Data', 'German_Contrast_Vector_Strain_Affinities.csv')
df_aff = pd.read_csv(affinities_path)
proj_meoh = df_aff['Projected_Affinity_MeOH_5b'].values
proj_tma  = df_aff['Projected_Affinity_TMA_5b'].values

# Basal H2:CO2 direction and projections
r_basal = R0_table[0]
u_h2co2 = r_basal / np.linalg.norm(r_basal)
proj_h2co2 = Theta @ u_h2co2

# 2. Load Processed Trajectory Data
data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
raw_data = torch.load(data_path, weights_only=False)
raw_traj = torch.stack(list(raw_data['trajectories'].values()), dim=0)

peaks = raw_traj.amax(dim=(0, 1))
mask_1pct = peaks >= 0.01
data_91 = raw_traj[:, :, mask_1pct]

shared_mask = (data_91[:6].sum(dim=(0, 1)) > 0) & (data_91[6:].sum(dim=(0, 1)) > 0)
raw_81 = data_91[:, :, shared_mask]
rel_raw = raw_81 / raw_81.sum(dim=-1, keepdim=True)
rel_pct = rel_raw.numpy() * 100

floored_raw = raw_81 + 1e-4
floored_data = floored_raw / floored_raw.sum(dim=-1, keepdim=True)
floored_np = floored_data.numpy()

target_idx = 0   # Methanomicrobium sp.
pyr_idx      = 63  # Pyramidobacter sp. C12-8
vicugnae_idx = [i for i, s in enumerate(strains_81) if 'vicugnae' in s][0]
iso4_idx     = [i for i, s in enumerate(strains_81) if 'ISO4-H5' in s][0]

def generate_figure(comp_type="total", panel_letters=True):
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 5.2), dpi=300)

    # =============================================================================
    # PANEL A: Cross-Host Prescriptive Transfer: Target T6 Abundance (Log Scale)
    # Baseline at 0.1%
    # =============================================================================
    ax1 = axes[0]
    cases_a = [
        "Cow OSU\n(+MeOH, 5 bar)",
        "Cow OSU\n(+TMA, 5 bar)",
        "Cow BRRS\n(+MeOH, 5 bar)",
        "Cow BRRS\n(+TMA, 5 bar)"
    ]
    eval_cases_idx = [8, 10, 2, 4]
    obs_t1_cases = floored_np[eval_cases_idx, 0, target_idx] * 100
    obs_t6_cases = floored_np[eval_cases_idx, -1, target_idx] * 100
    x_pred_t6 = np.array([47.7, 7.6, 1.7, 10.5])

    x_a = np.arange(len(cases_a))
    bar_w_a = 0.25
    y_floor = 0.1   # User requested: move baseline up to 0.1%

    h_t1 = np.maximum(obs_t1_cases - y_floor, 0.0)
    h_obs = np.array([
        max(obs_t6_cases[0] - y_floor, 0.0),
        max(obs_t6_cases[1] - y_floor, 0.0),
        0.0,  # Cow BRRS (+MeOH) is strictly 0.0% observed (below LOD): NO bar drawn!
        max(obs_t6_cases[3] - y_floor, 0.0)
    ])
    h_pred = np.maximum(x_pred_t6 - y_floor, 0.0)

    rects_t1 = ax1.bar(
        x_a - bar_w_a, h_t1, width=bar_w_a, bottom=y_floor,
        color='#95a5a6', edgecolor='black', linewidth=0.9, alpha=0.85,
        label='Inoculum ($T_1$)', zorder=3
    )
    rects_obs = ax1.bar(
        x_a, h_obs, width=bar_w_a, bottom=y_floor,
        color='#1b365d', edgecolor='black', linewidth=0.9, alpha=0.90,
        label='Observed ($T_6$)', zorder=3
    )
    rects_fit = ax1.bar(
        x_a[:2] + bar_w_a, h_pred[:2], width=bar_w_a, bottom=y_floor,
        color='#5d6d7e', edgecolor='black', linewidth=0.9,
        hatch='//', label='Model Fit (In-Sample, $T_6$)', zorder=3
    )
    rects_pred = ax1.bar(
        x_a[2:] + bar_w_a, h_pred[2:], width=bar_w_a, bottom=y_floor,
        color='#e67e22', edgecolor='black', linewidth=0.9,
        hatch='//', label='Model Prediction (Zero-Shot, $T_6$)', zorder=3
    )

    for i in range(len(cases_a)):
        v_t1 = obs_t1_cases[i]
        ax1.text(x_a[i] - bar_w_a, max(v_t1, y_floor) * 1.35, f"{v_t1:.1f}%",
                 ha='center', va='bottom', fontsize=7.6, fontweight='bold', color='#555555')
        
        if i == 2:
            # 0.0% observed sits neatly just above the 0.1% baseline axis
            ax1.text(x_a[i], 0.13, "0.0%",
                     ha='center', va='bottom', fontsize=7.6, fontweight='bold', color='#1b365d')
        else:
            v_obs = obs_t6_cases[i]
            ax1.text(x_a[i], max(v_obs, y_floor) * 1.35, f"{v_obs:.1f}%",
                     ha='center', va='bottom', fontsize=7.6, fontweight='bold', color='#1b365d')
        
        v_pred = x_pred_t6[i]
        col = '#495057' if i < 2 else '#d35400'
        ax1.text(x_a[i] + bar_w_a, max(v_pred, y_floor) * 1.35, f"{v_pred:.1f}%",
                 ha='center', va='bottom', fontsize=7.6, fontweight='bold', color=col)

    ax1.set_yscale('log')
    ax1.set_ylim(0.1, 250)
    ax1.set_yticks([0.1, 1, 10, 100])
    ax1.set_yticklabels(['0.1%', '1%', '10%', '100%'], fontsize=8.8, fontweight='bold')

    ax1.set_xticks(x_a)
    ax1.set_xticklabels(cases_a, fontsize=9.0, fontweight='bold')
    ax1.set_ylabel(r'$\it{Methanomicrobium\ sp.}$ Abundance (%) [Log Scale]', fontsize=9.8, fontweight='bold')
    ax1.grid(axis='y', which='major', linestyle='--', alpha=0.45, zorder=0)
    ax1.legend(loc='upper right', fontsize=7.8, framealpha=0.92, edgecolor='#cccccc', facecolor='white')

    if panel_letters:
        ax1.set_title(r'$\mathbf{A}$', fontsize=12.5, pad=8, loc='left', fontweight='bold')

    # =============================================================================
    # PANEL B: Competition Faced ($T_1 \to T_6$) [$z$-score] + Inset Scatter
    # =============================================================================
    ax2 = axes[1]
    positions_d = [0.80, 1.20, 2.80, 3.20]
    colors_4 = ['#2980b9', '#16a085', '#8e44ad', '#d35400']

    cond_info = [
        ('Cow OSU (+MeOH, 5b)', 8, R0_table[8 % 6], proj_meoh, colors_4[0], 'o'),
        ('Cow BRRS (+MeOH, 5b)', 2, R0_table[2 % 6], proj_meoh, colors_4[1], 'o'),
        ('Cow OSU (+TMA, 5b)', 10, R0_table[10 % 6], proj_tma, colors_4[2], '^'),
        ('Cow BRRS (+TMA, 5b)', 4, R0_table[4 % 6], proj_tma, colors_4[3], '^')
    ]

    all_comp_data = []
    target_comp_vals = []
    inset_data = []

    for label, c_idx, r0_vec, p_dir, host_col, marker in cond_info:
        x_mean = np.mean(floored_np[c_idx, :, :], axis=0)
        x_t6 = floored_np[c_idx, -1, :]
        
        if comp_type == "total":
            proj_tot = Theta @ r0_vec
            c_val = np.array([np.sum(x_mean[np.arange(N) != i] * proj_tot[np.arange(N) != i]) for i in range(N)])
        else:
            c_val = np.array([np.sum(x_mean[np.arange(N) != i] * p_dir[np.arange(N) != i]) for i in range(N)])
        
        mu = np.mean(c_val)
        sigma = np.std(c_val)
        z_val = (c_val - mu) / sigma
        
        sp, p_sp = stats.spearmanr(z_val, x_t6)
        all_comp_data.append(z_val)
        target_comp_vals.append(z_val[target_idx])
        inset_data.append({
            'label': label,
            'z_val': z_val,
            't6_pct': x_t6 * 100,
            'color': host_col,
            'marker': marker,
            'sp': sp,
            'p_sp': p_sp
        })

    vp = ax2.violinplot(all_comp_data, positions=positions_d, widths=0.32, showmeans=False, showmedians=True, showextrema=True)

    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(colors_4[i])
        pc.set_edgecolor('black')
        pc.set_linewidth(1.0)
        pc.set_alpha(0.65)

    for k in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        if k in vp:
            vp[k].set_color('#222222')
            vp[k].set_linewidth(1.1)

    np.random.seed(42)
    for i, data in enumerate(all_comp_data):
        pos = positions_d[i]
        jitter = np.random.normal(0, 0.032, size=len(data))
        ax2.scatter(pos + jitter, data, color=colors_4[i], alpha=0.35, s=14, edgecolor='none', zorder=3)

    ax2.axhline(0, color='#7f8c8d', linestyle='--', linewidth=0.9, alpha=0.7, zorder=2)
    ax2.text(0.35, 0.35, 'Mean ($z=0$)', color='#7f8c8d', fontsize=7.5, fontstyle='italic')

    # Y-axis label without formula
    if comp_type == "total":
        ax2.set_ylim(-9.8, 8.8)
        ax2.set_ylabel(r'Total Competition Faced [$z$-score]', fontsize=9.8, fontweight='bold')
        callout_configs = [
            (positions_d[0], target_comp_vals[0], -26, 0, 'right', 'center'),
            (positions_d[1], target_comp_vals[1], 26, 0, 'left', 'center'),
            (positions_d[2], target_comp_vals[2], -26, -16, 'right', 'top'),
            (positions_d[3], target_comp_vals[3], 26, -16, 'left', 'top'),
        ]
        ins_h = '30%'
    else:
        ax2.set_ylim(-9.8, 12.5)
        ax2.set_ylabel(r'Direction-Specific Competition Faced [$z$-score]', fontsize=9.8, fontweight='bold')
        callout_configs = [
            (positions_d[0], target_comp_vals[0], -26, 0, 'right', 'center'),
            (positions_d[1], target_comp_vals[1], 26, 0, 'left', 'center'),
            (positions_d[2], target_comp_vals[2], -26, -16, 'right', 'top'),
            (positions_d[3], target_comp_vals[3], 26, 0, 'left', 'center'),
        ]
        ins_h = '28%'

    for pos, t_val, off_x, off_y, ha, va in callout_configs:
        ax2.scatter(pos, t_val, color='#e74c3c', edgecolor='black', linewidth=1.2, s=95, marker='D', zorder=10)
        ax2.annotate(
            f'{t_val:+.2f}',
            xy=(pos, t_val),
            xytext=(off_x, off_y),
            textcoords='offset points',
            fontsize=8.5,
            fontweight='bold',
            color='#c0392b',
            ha=ha,
            va=va,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='#fdedec', edgecolor='#e74c3c', alpha=0.92, lw=0.8),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.9),
            zorder=11
        )

    ax2.set_xticks([1.0, 3.0])
    ax2.set_xticklabels(['Methanol (5 bar)', 'Trimethylamine (5 bar)'], fontsize=9.8, fontweight='bold')
    ax2.set_xlim(0.25, 3.85)
    ax2.grid(axis='y', linestyle=':', alpha=0.5, zorder=1)

    leg_elements = [
        Patch(facecolor=colors_4[0], edgecolor='black', alpha=0.7, label='Cow OSU (MeOH)'),
        Patch(facecolor=colors_4[1], edgecolor='black', alpha=0.7, label='Cow BRRS (MeOH)'),
        Patch(facecolor=colors_4[2], edgecolor='black', alpha=0.7, label='Cow OSU (TMA)'),
        Patch(facecolor=colors_4[3], edgecolor='black', alpha=0.7, label='Cow BRRS (TMA)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#e74c3c', markeredgecolor='black', markersize=8, label=r'$\it{Methanomicrobium\ sp.}$')
    ]
    ax2.legend(handles=leg_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=7.2, frameon=True, facecolor='white', framealpha=0.92)

    # Inset: T6 Abundance (Log Scale) vs z-score (NO Methanomicrobium marker)
    ax_ins = inset_axes(ax2, width='44%', height=ins_h, loc='upper right', borderpad=0.8)
    for d in inset_data:
        ax_ins.scatter(d['z_val'], d['t6_pct'], color=d['color'], alpha=0.50, s=14, marker=d['marker'], edgecolor='none', zorder=3)

    ax_ins.set_yscale('log')
    ax_ins.set_ylim(0.007, 150)
    ax_ins.set_yticks([0.01, 0.1, 1, 10, 100])
    ax_ins.set_yticklabels(['0.01%', '0.1%', '1%', '10%', '100%'], fontsize=6.2)
    x_lbl = r'$z(C_{\mathrm{total}})$' if comp_type == "total" else r"$z(C_{\mathbf{v}})$"
    ax_ins.set_xlabel(x_lbl, fontsize=7.5, labelpad=2, fontweight='bold')
    ax_ins.set_ylabel(r'$T_6$ (%) [Log Scale]', fontsize=7.2, labelpad=2, fontweight='bold')
    ax_ins.tick_params(labelsize=6.2, pad=1.5)
    ax_ins.grid(True, which='both', linestyle=':', alpha=0.35)

    leg_inset = []
    for d in inset_data:
        sp_val = d['sp']
        short_host = 'OSU' if 'OSU' in d['label'] else 'BRRS'
        med = 'MeOH' if 'MeOH' in d['label'] else 'TMA'
        lbl = rf"{short_host} ({med}): $\rho = {sp_val:.2f}$"
        leg_inset.append(Line2D([0], [0], marker=d['marker'], color='w', markerfacecolor=d['color'], markeredgecolor='none', markersize=6.0, label=lbl))

    ax_ins.legend(handles=leg_inset, loc='lower left', fontsize=5.8, frameon=True, facecolor='white', framealpha=0.92, edgecolor='#cccccc', labelspacing=0.25, handletextpad=0.3)

    if panel_letters:
        ax2.set_title(r'$\mathbf{B}$', fontsize=12.5, pad=8, loc='left', fontweight='bold')

    # =============================================================================
    # =============================================================================
    # PANEL C: Barycentric Niche Allocation Simplex (Min-Offset Representation)
    # Relative metabolic budget allocation across H2:CO2, MeOH, and TMA (all \tilde{\theta} >= 0)
    # =============================================================================
    ax3 = axes[2]

    # Min-offset representation
    off_meoh = proj_meoh - proj_meoh.min()
    off_tma  = proj_tma - proj_tma.min()
    off_h2   = proj_h2co2

    tot = off_h2 + off_meoh + off_tma
    p_h2   = off_h2 / tot
    p_meoh = off_meoh / tot
    p_tma  = off_tma / tot

    h_tri = np.sqrt(3) / 2
    x_tern = p_meoh + 0.5 * p_tma
    y_tern = p_tma * h_tri

    # Equilateral triangle boundary
    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, h_tri, 0.0]
    ax3.plot(triangle_x, triangle_y, 'k-', lw=1.8, zorder=2)

    # Grid lines (20%, 40%, 60%, 80%)
    for f in [0.2, 0.4, 0.6, 0.8]:
        y_line = f * h_tri
        ax3.plot([0.5 * f, 1.0 - 0.5 * f], [y_line, y_line], color='#bdc3c7', linestyle=':', lw=0.7, zorder=1)
        ax3.plot([f, 0.5 * f], [0, f * h_tri], color='#bdc3c7', linestyle=':', lw=0.7, zorder=1)
        ax3.plot([1.0 - f, 1.0 - 0.5 * f], [0, f * h_tri], color='#bdc3c7', linestyle=':', lw=0.7, zorder=1)

    sc = ax3.scatter(
        x_tern, y_tern,
        c=proj_h2co2, cmap='viridis',
        s=55, edgecolor='black', linewidth=0.7, alpha=0.90, zorder=5
    )

    # Highlight archetypes
    ax3.scatter(x_tern[target_idx], y_tern[target_idx], color='#e74c3c', edgecolor='black', linewidth=1.5, s=115, marker='D', zorder=10,
                label=r'$\it{Methanomicrobium\ sp.}$')
    ax3.scatter(x_tern[vicugnae_idx], y_tern[vicugnae_idx], color='#e67e22', edgecolor='black', linewidth=1.2, s=85, marker='s', zorder=9,
                label=r'$\it{Streptococcus\ vicugnae}$')
    ax3.scatter(x_tern[iso4_idx], y_tern[iso4_idx], color='#27ae60', edgecolor='black', linewidth=1.2, s=85, marker='^', zorder=9,
                label=r'ISO4-H5')
    ax3.scatter(x_tern[pyr_idx], y_tern[pyr_idx], color='#2980b9', edgecolor='black', linewidth=1.2, s=85, marker='o', zorder=9,
                label=r'$\it{Pyramidobacter\ sp.}$')

    # Vertex labels
    ax3.text(-0.02, -0.04, r'$\mathbf{u}_{\mathrm{H}_2:\mathrm{CO}_2}$' + '\n(Basal Syngas)', ha='center', va='top', fontsize=9.2, fontweight='bold', color='#1b365d')
    ax3.text(1.02, -0.04, r'$\mathbf{v}_{\mathrm{MeOH}}$' + '\n(Methanol)', ha='center', va='top', fontsize=9.2, fontweight='bold', color='#16a085')
    ax3.text(0.50, h_tri + 0.04, r'$\mathbf{v}_{\mathrm{TMA}}$' + '\n(Trimethylamine)', ha='center', va='bottom', fontsize=9.2, fontweight='bold', color='#8e44ad')

    # Edge percentage guide
    ax3.text(0.50, -0.035, '50%', fontsize=6.8, color='#7f8c8d', ha='center', va='top')
    ax3.text(0.25 - 0.025, h_tri/2, '50%', fontsize=6.8, color='#7f8c8d', ha='right', va='center')
    ax3.text(0.75 + 0.025, h_tri/2, '50%', fontsize=6.8, color='#7f8c8d', ha='left', va='center')

    # Tradeoff annotation box (preserved Pearson R)
    r_val, p_val = stats.pearsonr(off_meoh, off_tma)
    trade_text = f'Trade-off: $r = {r_val:.2f}$\n($p = 8.2 \\times 10^{{-13}}$)'
    ax3.text(0.98, 0.98, trade_text, transform=ax3.transAxes, fontsize=8.0, fontweight='bold', color='#c0392b',
             ha='right', va='top', bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdedec', edgecolor='#e74c3c', alpha=0.92, lw=0.8), zorder=12)

    ax3.set_xlim(-0.16, 1.16)
    ax3.set_ylim(-0.15, 1.05)
    ax3.set_aspect('equal')
    ax3.axis('off')

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax3, pad=0.04, aspect=22, shrink=0.82)
    cbar.set_label(r'$\theta_i^{\mathrm{H}_2:\mathrm{CO}_2}$ (Basal Affinity)', fontsize=9.5, fontweight='bold')
    cbar.ax.tick_params(labelsize=7.8)

    # Legend
    ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.2, frameon=True, facecolor='white', framealpha=0.92, edgecolor='#cccccc')

    if panel_letters:
        ax3.set_title(r'$\mathbf{C}$', fontsize=12.5, pad=8, loc='left', fontweight='bold')

    fig.subplots_adjust(top=0.92, bottom=0.14, left=0.06, right=0.98, wspace=0.25)

    if comp_type == "total":
        out_poster   = os.path.join(POSTER_FIG_DIR, 'German_Theta_TMA_Niche_Competition.png')
        out_paper    = os.path.join(PAPER_FIG_DIR, 'German_Theta_TMA_Niche_Competition.png')
        out_artifact = os.path.join(ARTIFACT_DIR, 'German_Theta_TMA_Niche_Competition.png')
    else:
        out_poster   = os.path.join(POSTER_FIG_DIR, 'German_Theta_TMA_Niche_Competition_DirectionSpecific.png')
        out_paper    = os.path.join(PAPER_FIG_DIR, 'German_Theta_TMA_Niche_Competition_DirectionSpecific.png')
        out_artifact = os.path.join(ARTIFACT_DIR, 'German_Direction_Specific_Competition.png')

    fig.savefig(out_poster, dpi=300, bbox_inches='tight')
    fig.savefig(out_paper, dpi=300, bbox_inches='tight')
    fig.savefig(out_artifact, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Successfully generated {comp_type} figure in:")
    print(f"  - {out_paper}")
    print(f"  - {out_poster}")
    print(f"  - {out_artifact}")

if __name__ == '__main__':
    generate_figure("total", panel_letters=True)
    generate_figure("direction", panel_letters=True)

"""
Generate Niche Space Figures with X and Y Axes Set to Growth Medium Affinity:
  X-axis: Affinity towards Methanol Medium (Theta_i^T u_MeOH or Theta_i^T R0_MeOH)
  Y-axis: Affinity towards TMA Medium (Theta_i^T u_TMA or Theta_i^T R0_TMA)

Features:
  - Constant Opacity (alpha = 0.85, no opacity scaling)
  - Linear point size scaling (no intercept: s = k * x_i)
  - Exclusion of strains with 0% abundance
  - Panel A (BRRS): Target and MeOH competitor sink (Pyramidobacter C12-8) marked
  - Panel B (OSU): Target and resident TMA competitors (S. vicugnae, D. detoxificans, Bacteroidales WCE2008) marked
  - Clean diagonal dashed line y = x (no text, no legend box)
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Publication styling
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

# 1. Load Precomputed Champion Parameters
params_path = os.path.join(PROJECT_ROOT, 'Data', 'German_champion_model_params.pt')
params = torch.load(params_path, weights_only=False)
Theta = params['Theta'].numpy() # (81, 3)
R0_table = params['R0_table'].numpy() # (6, 3)
strains_81 = params['strains_81']
N = len(strains_81)

r_meoh = R0_table[2] # MeOH 5b
r_tma  = R0_table[4] # TMA 5b
u_meoh = r_meoh / np.linalg.norm(r_meoh)
u_tma  = r_tma  / np.linalg.norm(r_tma)

# Unit medium affinities
aff_meoh_unit = np.dot(Theta, u_meoh)
aff_tma_unit  = np.dot(Theta, u_tma)

# Raw medium affinities
aff_meoh_raw = np.dot(Theta, r_meoh)
aff_tma_raw  = np.dot(Theta, r_tma)

# 2. Load Processed Data to Extract Exact Unfloored Raw T1 Abundances
data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
raw_data = torch.load(data_path, weights_only=False)
raw_traj = torch.stack(list(raw_data['trajectories'].values()), dim=0)

peaks = raw_traj.amax(dim=(0, 1))
mask_1pct = peaks >= 0.01
data_91 = raw_traj[:, :, mask_1pct]

shared_mask = (data_91[:6].sum(dim=(0, 1)) > 0) & (data_91[6:].sum(dim=(0, 1)) > 0)
raw_81 = data_91[:, :, shared_mask]
rel_raw = raw_81 / raw_81.sum(dim=-1, keepdim=True)

brrs_t1_pct = rel_raw[[0, 2, 4], 0].mean(dim=0).numpy() * 100 # (81,)
osu_t1_pct  = rel_raw[[6, 8, 10], 0].mean(dim=0).numpy() * 100 # (81,)

target_idx   = 0   # Methanomicrobium sp.
pyramido_idx = [i for i, s in enumerate(strains_81) if 'Pyramidobacter sp. C12-8' in s][0]
vicugnae_idx = 8   # Streptococcus vicugnae
denitro_idx  = 16  # Denitrobacterium detoxificans
bactero_idx  = 5   # Bacteroidales bacterium WCE2008

norm_linear = mcolors.Normalize(vmin=0.0, vmax=40.0)
k_area = 18.0

cmap_trunc = mcolors.LinearSegmentedColormap.from_list(
    'trunc_blues', plt.cm.Blues(np.linspace(0.25, 1.0, 256))
)
dark_edge_rgb = np.array([0.05, 0.15, 0.35])

def get_styles(abund, mask):
    a_sub = abund[mask]
    sizes = k_area * a_sub
    norm_v = norm_linear(a_sub)
    rgba = cmap_trunc(norm_v)
    # Constant Opacity (alpha = 0.85, no scaling with abundance)
    rgba[:, 3] = 0.85
    edges = np.zeros((len(a_sub), 4))
    edges[:, :3] = dark_edge_rgb
    edges[:, 3] = 0.95
    return sizes, rgba, edges

def plot_medium_niche(coord_type='unit'):
    if coord_type == 'unit':
        x_pts = aff_meoh_unit
        y_pts = aff_tma_unit
        x_label = r'Affinity towards Methanol Medium ($\mathbf{\Theta}_i^\top \mathbf{u}_{\mathrm{MeOH}}$)'
        y_label = r'Affinity towards TMA Medium ($\mathbf{\Theta}_i^\top \mathbf{u}_{\mathrm{TMA}}$)'
        min_lim, max_lim = 0.4, 2.5
        xlims = (0.4, 2.55)
        ylims = (0.5, 2.60)
        file_tag = 'Unit'
    else:
        x_pts = aff_meoh_raw
        y_pts = aff_tma_raw
        x_label = r'Initial Growth Potential in Methanol Medium ($\mathbf{\Theta}_i^\top \mathbf{R}_0^{(\mathrm{MeOH})}$)'
        y_label = r'Initial Growth Potential in TMA Medium ($\mathbf{\Theta}_i^\top \mathbf{R}_0^{(\mathrm{TMA})}$)'
        min_lim, max_lim = 1.5, 9.2
        xlims = (1.5, 9.5)
        ylims = (2.0, 9.5)
        file_tag = 'Raw'

    # =========================================================================
    # SIDE-BY-SIDE VS FIGURE
    # =========================================================================
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.2, 6.0), dpi=300)

    # Panel A: Cow BRRS Inoculum
    mask_b = brrs_t1_pct > 0
    idx_b = np.where(mask_b)[0]
    sizes_b, rgba_b, edges_b = get_styles(brrs_t1_pct, mask_b)
    sort_b = np.argsort(brrs_t1_pct[mask_b])

    ax_l.scatter(
        x_pts[idx_b][sort_b], y_pts[idx_b][sort_b],
        c=rgba_b[sort_b], s=sizes_b[sort_b],
        edgecolors=edges_b[sort_b], linewidth=0.8,
        zorder=3
    )

    # Annotate Target
    ax_l.annotate(
        r'$\mathbf{Methanomicrobium\ sp.}$' + f'\n(Target, $T_1={brrs_t1_pct[target_idx]:.1f}\\%$)',
        (x_pts[target_idx], y_pts[target_idx]),
        textcoords='offset points', xytext=(15, 12),
        fontsize=8.8, fontweight='bold', color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdedec', alpha=0.92, edgecolor='#c0392b', linewidth=0.9),
        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2),
        zorder=7
    )

    # Annotate Pyramidobacter
    ax_l.annotate(
        r'$\mathbf{Pyramidobacter\ C12-8}$' + f'\n(MeOH Competitor, $T_1={brrs_t1_pct[pyramido_idx]:.1f}\\%$)',
        (x_pts[pyramido_idx], y_pts[pyramido_idx]),
        textcoords='offset points', xytext=(15, -16),
        fontsize=8.8, fontweight='bold', color='#d35400',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=0.9),
        arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.2),
        zorder=7
    )

    # Diagonal Equal Growth Ray
    ax_l.plot([min_lim, max_lim], [min_lim, max_lim], linestyle='--', color='#7f8c8d', lw=1.2, alpha=0.7, zorder=2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_trunc, norm=norm_linear)
    sm.set_array([])
    cbar_l = plt.colorbar(sm, ax=ax_l, pad=0.03, shrink=0.88)
    cbar_l.set_label('Cow BRRS ' + r'$T_1$ Inoculum Abundance (%)', fontsize=9.2, fontweight='bold')
    cbar_l.set_ticks([0, 10, 20, 30, 40])
    cbar_l.set_ticklabels(['0%', '10%', '20%', '30%', '40%'])
    cbar_l.ax.tick_params(labelsize=8.2)

    ax_l.set_xlabel(x_label, fontsize=10.5, fontweight='bold')
    ax_l.set_ylabel(y_label, fontsize=10.5, fontweight='bold')
    ax_l.set_title(r'$\mathbf{A}$  Cow BRRS Inoculum (Medium Affinity Space)', fontsize=11.5, pad=12, loc='left')
    ax_l.grid(True, linestyle=':', alpha=0.45)
    ax_l.set_xlim(xlims)
    ax_l.set_ylim(ylims)

    # Panel B: Cow OSU Inoculum
    mask_o = osu_t1_pct > 0
    idx_o = np.where(mask_o)[0]
    sizes_o, rgba_o, edges_o = get_styles(osu_t1_pct, mask_o)
    sort_o = np.argsort(osu_t1_pct[mask_o])

    ax_r.scatter(
        x_pts[idx_o][sort_o], y_pts[idx_o][sort_o],
        c=rgba_o[sort_o], s=sizes_o[sort_o],
        edgecolors=edges_o[sort_o], linewidth=0.8,
        zorder=3
    )

    # Target
    ax_r.annotate(
        r'$\mathbf{Methanomicrobium\ sp.}$' + f'\n(Target, $T_1={osu_t1_pct[target_idx]:.1f}\\%$)',
        (x_pts[target_idx], y_pts[target_idx]),
        textcoords='offset points', xytext=(15, 12),
        fontsize=8.8, fontweight='bold', color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdedec', alpha=0.92, edgecolor='#c0392b', linewidth=0.9),
        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2),
        zorder=7
    )

    # S. vicugnae (high TMA affinity, above diagonal)
    ax_r.annotate(
        r'$\mathbf{Streptococcus\ vicugnae}$' + '\n(TMA Specialist)',
        (x_pts[vicugnae_idx], y_pts[vicugnae_idx]),
        textcoords='offset points', xytext=(-35, 14),
        fontsize=8.5, fontweight='bold', color='#d35400',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=0.9),
        arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.2),
        zorder=7
    )

    # D. detoxificans (top TMA sink)
    ax_r.annotate(
        r'$\mathbf{Denitrobacterium\ detoxificans}$' + '\n(Top TMA Sink)',
        (x_pts[denitro_idx], y_pts[denitro_idx]),
        textcoords='offset points', xytext=(-40, -25),
        fontsize=8.5, fontweight='bold', color='#d35400',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=0.9),
        arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.2),
        zorder=7
    )

    # Bacteroidales WCE2008
    ax_r.annotate(
        r'$\mathbf{Bacteroidales\ WCE2008}$' + '\n(Generalist Competitor)',
        (x_pts[bactero_idx], y_pts[bactero_idx]),
        textcoords='offset points', xytext=(15, -16),
        fontsize=8.5, fontweight='bold', color='#d35400',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=0.9),
        arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.2),
        zorder=7
    )

    # Diagonal Equal Growth Ray
    ax_r.plot([min_lim, max_lim], [min_lim, max_lim], linestyle='--', color='#7f8c8d', lw=1.2, alpha=0.7, zorder=2)

    # Colorbar
    cbar_r = plt.colorbar(sm, ax=ax_r, pad=0.03, shrink=0.88)
    cbar_r.set_label('Cow OSU ' + r'$T_1$ Inoculum Abundance (%)', fontsize=9.2, fontweight='bold')
    cbar_r.set_ticks([0, 10, 20, 30, 40])
    cbar_r.set_ticklabels(['0%', '10%', '20%', '30%', '40%'])
    cbar_r.ax.tick_params(labelsize=8.2)

    ax_r.set_xlabel(x_label, fontsize=10.5, fontweight='bold')
    ax_r.set_ylabel(y_label, fontsize=10.5, fontweight='bold')
    ax_r.set_title(r'$\mathbf{B}$  Cow OSU Inoculum (Medium Affinity Space)', fontsize=11.5, pad=12, loc='left')
    ax_r.grid(True, linestyle=':', alpha=0.45)
    ax_r.set_xlim(xlims)
    ax_r.set_ylim(ylims)

    plt.tight_layout()

    out_paper = os.path.join(PAPER_FIG_DIR, f'German_Niche_Space_Host_Comparison_Medium_{file_tag}.png')
    out_poster = os.path.join(POSTER_FIG_DIR, f'German_Niche_Space_Host_Comparison_Medium_{file_tag}.png')
    out_artifact = os.path.join(ARTIFACT_DIR, f'German_Niche_Space_Host_Comparison_Medium_{file_tag}.png')

    fig.savefig(out_paper, dpi=300, bbox_inches='tight')
    fig.savefig(out_poster, dpi=300, bbox_inches='tight')
    fig.savefig(out_artifact, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Generated {coord_type} medium affinity comparison figures (Constant Opacity):")
    print(f"  - {out_paper}")
    print(f"  - {out_artifact}")

plot_medium_niche('unit')
plot_medium_niche('raw')

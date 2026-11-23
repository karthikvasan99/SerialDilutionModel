"""
Generate Separate Figure of Plot C with Cow OSU Abundances,
as well as a Side-by-Side Host Comparison (Cow BRRS vs. Cow OSU) for Internal Verification:
- Single-color colormap (Blues, with alternative Reds version)
- Point sizes scale LINEARLY with abundance with STRICTLY ZERO INTERCEPT (s = k * x_i)
- Point OPACITY scales directly with abundance with STRICTLY ZERO INTERCEPT (alpha = x_i / 40.0)
- Excludes any strain with zero abundance (abund > 0 only)
- In the Cow OSU plot: marks the TMA competitors that outcompete Methanomicrobium sp.
  (Streptococcus vicugnae, Denitrobacterium detoxificans, Bacteroidales WCE2008) instead of Pyramidobacter!
- NO legend box
- NO equal affinity text on the plot (clean dashed line only)
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

# 1. Load Precomputed Affinities
affinities_path = os.path.join(PROJECT_ROOT, 'Data', 'German_Contrast_Vector_Strain_Affinities.csv')
df_aff = pd.read_csv(affinities_path)
N = len(df_aff)
strains_81 = df_aff['Strain_Name'].tolist()
proj_meoh = df_aff['Projected_Affinity_MeOH_5b'].values
proj_tma  = df_aff['Projected_Affinity_TMA_5b'].values

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

# Inoculum T1 relative abundances (5-bar conditions: BRRS=[0, 2, 4], OSU=[6, 8, 10])
brrs_t1_pct = rel_raw[[0, 2, 4], 0].mean(dim=0).numpy() * 100 # (81,)
osu_t1_pct  = rel_raw[[6, 8, 10], 0].mean(dim=0).numpy() * 100 # (81,)

target_idx   = 0   # Methanomicrobium sp.
pyramido_idx = [i for i, s in enumerate(strains_81) if 'Pyramidobacter sp. C12-8' in s][0]
vicugnae_idx = 8   # Streptococcus vicugnae
denitro_idx  = 16  # Denitrobacterium detoxificans
bactero_idx  = 5   # Bacteroidales bacterium WCE2008

norm_linear = mcolors.Normalize(vmin=0.0, vmax=40.0)
k_area = 18.0  # Area = k * abund (Zero intercept: 0% -> 0 pt^2)

def generate_niche_figures(color_theme='Blues'):
    if color_theme == 'Blues':
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list(
            'trunc_blues', plt.cm.Blues(np.linspace(0.25, 1.0, 256))
        )
        dark_edge_rgb = np.array([0.05, 0.15, 0.35])
        suffix = ''
    else:
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list(
            'trunc_reds', plt.cm.Reds(np.linspace(0.25, 1.0, 256))
        )
        dark_edge_rgb = np.array([0.35, 0.05, 0.05])
        suffix = '_Reds'

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

    min_lim, max_lim = -0.5, 1.6

    # =========================================================================
    # 1. STANDALONE FIGURE: Cow OSU Inoculum
    # =========================================================================
    fig_osu, ax = plt.subplots(figsize=(6.8, 5.8), dpi=300)
    
    mask_osu = osu_t1_pct > 0
    idx_osu = np.where(mask_osu)[0]
    sizes_osu, rgba_osu, edges_osu = get_styles(osu_t1_pct, mask_osu)

    sort_sub = np.argsort(osu_t1_pct[mask_osu])
    ax.scatter(
        proj_meoh[idx_osu][sort_sub], proj_tma[idx_osu][sort_sub],
        c=rgba_osu[sort_sub], s=sizes_osu[sort_sub],
        edgecolors=edges_osu[sort_sub], linewidth=0.8,
        zorder=3
    )

    # Annotate Target Microbe
    ax.annotate(
        r'$\mathbf{Methanomicrobium\ sp.}$' + f'\n(Target, $T_1={osu_t1_pct[target_idx]:.1f}\\%$)',
        (proj_meoh[target_idx], proj_tma[target_idx]),
        textcoords='offset points', xytext=(15, 15),
        fontsize=8.8, fontweight='bold', color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#fdedec', alpha=0.92, edgecolor='#c0392b', linewidth=1.0),
        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.3),
        zorder=7
    )

    # Annotate TMA Competitor 1: S. vicugnae (highest TMA affinity)
    ax.annotate(
        r'$\mathbf{Streptococcus\ vicugnae}$' + '\n(TMA Specialist)',
        (proj_meoh[vicugnae_idx], proj_tma[vicugnae_idx]),
        textcoords='offset points', xytext=(14, -14),
        fontsize=8.5, fontweight='bold', color='#d35400',
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=1.0),
        arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.3),
        zorder=7
    )

    # Annotate TMA Competitor 2: D. detoxificans (top TMA sink)
    ax.annotate(
        r'$\mathbf{Denitrobacterium\ detoxificans}$' + '\n(Top TMA Sink)',
        (proj_meoh[denitro_idx], proj_tma[denitro_idx]),
        textcoords='offset points', xytext=(-30, -25),
        fontsize=8.5, fontweight='bold', color='#d35400',
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=1.0),
        arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.3),
        zorder=7
    )

    # Annotate TMA Competitor 3: Bacteroidales WCE2008
    ax.annotate(
        r'$\mathbf{Bacteroidales\ WCE2008}$' + '\n(TMA Competitor)',
        (proj_meoh[bactero_idx], proj_tma[bactero_idx]),
        textcoords='offset points', xytext=(14, -14),
        fontsize=8.5, fontweight='bold', color='#d35400',
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=1.0),
        arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.3),
        zorder=7
    )

    # Diagonal Equal Affinity Ray (NO TEXT ON PLOT)
    ax.plot([min_lim, max_lim], [min_lim, max_lim], linestyle='--', color='#7f8c8d', lw=1.2, alpha=0.7, zorder=2)

    # Single-color Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_trunc, norm=norm_linear)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.03, shrink=0.88)
    cbar.set_label('Cow OSU $T_1$ Inoculum Abundance (%)', fontsize=9.5, fontweight='bold')
    cbar.set_ticks([0, 10, 20, 30, 40])
    cbar.set_ticklabels(['0%', '10%', '20%', '30%', '40%'])
    cbar.ax.tick_params(labelsize=8.5)

    ax.set_xlabel(r'Projected Methanol Affinity ($\mathbf{\Theta}_i^\top \mathbf{v}_{\mathrm{MeOH}}$)', fontsize=11.0, fontweight='bold')
    ax.set_ylabel(r'Projected TMA Affinity ($\mathbf{\Theta}_i^\top \mathbf{v}_{\mathrm{TMA}}$)', fontsize=11.0, fontweight='bold')
    ax.set_title(r'Strain Niche Space on 5-Bar Contrast Vectors' + '\n' + r'(Cow OSU Inoculum — Resident TMA Competitors Marked)', fontsize=11.5, pad=12)
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.set_xlim(-0.75, 1.85)
    ax.set_ylim(-0.50, 1.65)

    plt.tight_layout()

    out_osu_paper = os.path.join(PAPER_FIG_DIR, f'German_Theta_TMA_Niche_Competition_PlotC_OSU{suffix}.png')
    out_osu_poster = os.path.join(POSTER_FIG_DIR, f'German_Theta_TMA_Niche_Competition_PlotC_OSU{suffix}.png')
    out_osu_artifact = os.path.join(ARTIFACT_DIR, f'German_Theta_TMA_Niche_Competition_PlotC_OSU{suffix}.png')

    fig_osu.savefig(out_osu_paper, dpi=300, bbox_inches='tight')
    fig_osu.savefig(out_osu_poster, dpi=300, bbox_inches='tight')
    fig_osu.savefig(out_osu_artifact, dpi=300, bbox_inches='tight')
    plt.close(fig_osu)

    # =========================================================================
    # 2. COMPARATIVE VS FIGURE: Cow BRRS vs. Cow OSU Niche Space
    # =========================================================================
    fig_cmp, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.2, 6.0), dpi=300)

    # Panel 1: Cow BRRS Inoculum (Marking MeOH Competitor Sink: Pyramidobacter C12-8)
    mask_b = brrs_t1_pct > 0
    idx_b = np.where(mask_b)[0]
    sizes_b, rgba_b, edges_b = get_styles(brrs_t1_pct, mask_b)
    sort_b = np.argsort(brrs_t1_pct[mask_b])

    ax_l.scatter(
        proj_meoh[idx_b][sort_b], proj_tma[idx_b][sort_b],
        c=rgba_b[sort_b], s=sizes_b[sort_b],
        edgecolors=edges_b[sort_b], linewidth=0.8,
        zorder=3
    )

    ax_l.annotate(
        r'$\mathbf{Methanomicrobium\ sp.}$' + f'\n(Target, $T_1={brrs_t1_pct[target_idx]:.1f}\\%$)',
        (proj_meoh[target_idx], proj_tma[target_idx]),
        textcoords='offset points', xytext=(15, 15),
        fontsize=8.8, fontweight='bold', color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdedec', alpha=0.92, edgecolor='#c0392b', linewidth=0.9),
        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2),
        zorder=7
    )

    ax_l.annotate(
        r'$\mathbf{Pyramidobacter\ C12-8}$' + f'\n(MeOH Competitor Sink, $T_1={brrs_t1_pct[pyramido_idx]:.1f}\\%$)',
        (proj_meoh[pyramido_idx], proj_tma[pyramido_idx]),
        textcoords='offset points', xytext=(14, -15),
        fontsize=8.8, fontweight='bold', color='#d35400',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=0.9),
        arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.2),
        zorder=7
    )

    ax_l.plot([min_lim, max_lim], [min_lim, max_lim], linestyle='--', color='#7f8c8d', lw=1.2, alpha=0.7, zorder=2)

    sm = plt.cm.ScalarMappable(cmap=cmap_trunc, norm=norm_linear)
    sm.set_array([])
    cbar_l = plt.colorbar(sm, ax=ax_l, pad=0.03, shrink=0.88)
    cbar_l.set_label('Cow BRRS ' + r'$T_1$ Inoculum Abundance (%)', fontsize=9.2, fontweight='bold')
    cbar_l.set_ticks([0, 10, 20, 30, 40])
    cbar_l.set_ticklabels(['0%', '10%', '20%', '30%', '40%'])
    cbar_l.ax.tick_params(labelsize=8.2)

    ax_l.set_xlabel(r'Projected Methanol Affinity ($\mathbf{\Theta}_i^\top \mathbf{v}_{\mathrm{MeOH}}$)', fontsize=10.5, fontweight='bold')
    ax_l.set_ylabel(r'Projected TMA Affinity ($\mathbf{\Theta}_i^\top \mathbf{v}_{\mathrm{TMA}}$)', fontsize=10.5, fontweight='bold')
    ax_l.set_title(r'$\mathbf{A}$  Cow BRRS Inoculum (MeOH Competitor Sink Marked)', fontsize=11.5, pad=12, loc='left')
    ax_l.grid(True, linestyle=':', alpha=0.45)
    ax_l.set_xlim(-0.75, 1.85)
    ax_l.set_ylim(-0.50, 1.65)

    # Panel 2: Cow OSU Inoculum (Marking TMA Competitors instead of Pyramidobacter!)
    ax_r.scatter(
        proj_meoh[idx_osu][sort_sub], proj_tma[idx_osu][sort_sub],
        c=rgba_osu[sort_sub], s=sizes_osu[sort_sub],
        edgecolors=edges_osu[sort_sub], linewidth=0.8,
        zorder=3
    )

    # Target Methanomicrobium sp.
    ax_r.annotate(
        r'$\mathbf{Methanomicrobium\ sp.}$' + f'\n(Target, $T_1={osu_t1_pct[target_idx]:.1f}\\%$)',
        (proj_meoh[target_idx], proj_tma[target_idx]),
        textcoords='offset points', xytext=(15, 15),
        fontsize=8.8, fontweight='bold', color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdedec', alpha=0.92, edgecolor='#c0392b', linewidth=0.9),
        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2),
        zorder=7
    )

    # TMA Competitor 1: S. vicugnae
    ax_r.annotate(
        r'$\mathbf{Streptococcus\ vicugnae}$' + '\n(TMA Specialist)',
        (proj_meoh[vicugnae_idx], proj_tma[vicugnae_idx]),
        textcoords='offset points', xytext=(14, -14),
        fontsize=8.5, fontweight='bold', color='#d35400',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=0.9),
        arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.2),
        zorder=7
    )

    # TMA Competitor 2: D. detoxificans
    ax_r.annotate(
        r'$\mathbf{Denitrobacterium\ detoxificans}$' + '\n(Top TMA Sink)',
        (proj_meoh[denitro_idx], proj_tma[denitro_idx]),
        textcoords='offset points', xytext=(-30, -25),
        fontsize=8.5, fontweight='bold', color='#d35400',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=0.9),
        arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.2),
        zorder=7
    )

    # TMA Competitor 3: Bacteroidales WCE2008
    ax_r.annotate(
        r'$\mathbf{Bacteroidales\ WCE2008}$' + '\n(TMA Competitor)',
        (proj_meoh[bactero_idx], proj_tma[bactero_idx]),
        textcoords='offset points', xytext=(14, -14),
        fontsize=8.5, fontweight='bold', color='#d35400',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=0.9),
        arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.2),
        zorder=7
    )

    ax_r.plot([min_lim, max_lim], [min_lim, max_lim], linestyle='--', color='#7f8c8d', lw=1.2, alpha=0.7, zorder=2)

    cbar_r = plt.colorbar(sm, ax=ax_r, pad=0.03, shrink=0.88)
    cbar_r.set_label('Cow OSU ' + r'$T_1$ Inoculum Abundance (%)', fontsize=9.2, fontweight='bold')
    cbar_r.set_ticks([0, 10, 20, 30, 40])
    cbar_r.set_ticklabels(['0%', '10%', '20%', '30%', '40%'])
    cbar_r.ax.tick_params(labelsize=8.2)

    ax_r.set_xlabel(r'Projected Methanol Affinity ($\mathbf{\Theta}_i^\top \mathbf{v}_{\mathrm{MeOH}}$)', fontsize=10.5, fontweight='bold')
    ax_r.set_ylabel(r'Projected TMA Affinity ($\mathbf{\Theta}_i^\top \mathbf{v}_{\mathrm{TMA}}$)', fontsize=10.5, fontweight='bold')
    ax_r.set_title(r'$\mathbf{B}$  Cow OSU Inoculum (Resident TMA Competitors Marked)', fontsize=11.5, pad=12, loc='left')
    ax_r.grid(True, linestyle=':', alpha=0.45)
    ax_r.set_xlim(-0.75, 1.85)
    ax_r.set_ylim(-0.50, 1.65)

    plt.tight_layout()

    out_cmp_paper = os.path.join(PAPER_FIG_DIR, f'German_Niche_Space_Host_Comparison_BRRS_vs_OSU{suffix}.png')
    out_cmp_poster = os.path.join(POSTER_FIG_DIR, f'German_Niche_Space_Host_Comparison_BRRS_vs_OSU{suffix}.png')
    out_cmp_artifact = os.path.join(ARTIFACT_DIR, f'German_Niche_Space_Host_Comparison_BRRS_vs_OSU{suffix}.png')

    fig_cmp.savefig(out_cmp_paper, dpi=300, bbox_inches='tight')
    fig_cmp.savefig(out_cmp_poster, dpi=300, bbox_inches='tight')
    fig_cmp.savefig(out_cmp_artifact, dpi=300, bbox_inches='tight')
    plt.close(fig_cmp)

    print(f"Generated {color_theme} figures (OSU marked with TMA competitors):")
    print(f"  - {out_cmp_paper}")
    print(f"  - {out_osu_paper}")

# Generate primary Blues and alternative Reds
generate_niche_figures('Blues')
generate_niche_figures('Reds')

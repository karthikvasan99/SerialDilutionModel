"""
Generate 2x2 (4-panel) Figure of Strain Niche Space for:
(Cow BRRS, Cow OSU) x (+MeOH, +TMA) at 5 bar:
- Top-Left:     Cow BRRS (+MeOH, 5 bar)
- Top-Right:    Cow OSU  (+MeOH, 5 bar)
- Bottom-Left:  Cow BRRS (+TMA, 5 bar)
- Bottom-Right: Cow OSU  (+TMA, 5 bar)

Minimalist labeling requested by user:
- Text is simplified to strictly Taxon Name and (T1 -> T6).
- For Pyramidobacter in Panel A, replaced 'Dominant MeOH Sink' with its T6 trajectory (T1=31.9% -> T6=2.7%).
- All editorial phrases (top winner, tma specialist, negative tma affinity, target, rescued, dominant meoh sink) removed.
- Color coding preserved exactly:
  * Green box:  #1 Winner (T6)
  * Orange box: #2 Competitor (T6)
  * Red box:    Target Methanomicrobium sp. (when excluded / suppressed)
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

# 2. Load Trajectories to Extract Exact Condition-Specific T1 and T6 Abundances
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

# Key strain indices:
target_idx   = 0   # Methanomicrobium sp.
pyramido_idx = [i for i, s in enumerate(strains_81) if 'Pyramidobacter sp. C12-8' in s][0]
thermo_idx   = [i for i, s in enumerate(strains_81) if 'Thermoplasmata archaeon' in s][0]
vicugnae_idx = 8   # Streptococcus vicugnae
denitro_idx  = 16  # Denitrobacterium detoxificans
bactero_idx  = 5   # Bacteroidales bacterium WCE2008

norm_linear = mcolors.Normalize(vmin=0.0, vmax=50.0)
k_area = 16.0  # Linear scaling s = k * abund (zero intercept)

# Color styling for callout boxes (PRESERVED EXACTLY AS REQUESTED)
style_winner = dict(boxstyle='round,pad=0.25', facecolor='#eafaf1', alpha=0.92, edgecolor='#27ae60', linewidth=1.1)
color_winner = '#27ae60'

style_second = dict(boxstyle='round,pad=0.25', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=1.0)
color_second = '#d35400'

style_target_red = dict(boxstyle='round,pad=0.25', facecolor='#fdedec', alpha=0.92, edgecolor='#c0392b', linewidth=1.0)
color_target_red = '#c0392b'

def create_4panel_figure(color_theme='Blues'):
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
        rgba[:, 3] = 0.85
        edges = np.zeros((len(a_sub), 4))
        edges[:, :3] = dark_edge_rgb
        edges[:, 3] = 0.95
        return sizes, rgba, edges

    min_lim, max_lim = -0.5, 1.6

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 14.0), dpi=300)

    panels = [
        # (row, col, cond_idx, panel_letter, title, host_name, additive_name)
        (0, 0, 2,  r'$\mathbf{A}$', 'Cow BRRS — Inoculum in Methanol (+MeOH, 5 bar)', 'Cow BRRS', '+MeOH'),
        (0, 1, 8,  r'$\mathbf{B}$', 'Cow OSU — Inoculum in Methanol (+MeOH, 5 bar)', 'Cow OSU', '+MeOH'),
        (1, 0, 4,  r'$\mathbf{C}$', 'Cow BRRS — Inoculum in Trimethylamine (+TMA, 5 bar)', 'Cow BRRS', '+TMA'),
        (1, 1, 10, r'$\mathbf{D}$', 'Cow OSU — Inoculum in Trimethylamine (+TMA, 5 bar)', 'Cow OSU', '+TMA')
    ]

    for row, col, c_idx, letter, title, host, additive in panels:
        ax = axes[row, col]
        ax.set_xlim(-0.85, 2.20)
        ax.set_ylim(-0.55, 1.75)

        t1 = rel_pct[c_idx, 0]
        t6 = rel_pct[c_idx, -1]

        mask = t1 > 0
        idx_nz = np.where(mask)[0]
        sizes, rgba, edges = get_styles(t1, mask)
        sort_order = np.argsort(t1[mask])

        # Plot all resident taxa (including Methanomicrobium naturally styled)
        ax.scatter(
            proj_meoh[idx_nz][sort_order], proj_tma[idx_nz][sort_order],
            c=rgba[sort_order], s=sizes[sort_order],
            edgecolors=edges[sort_order], linewidth=0.8,
            zorder=3
        )

        # Equal affinity ray (dashed line)
        ax.plot([min_lim, max_lim], [min_lim, max_lim], linestyle='--', color='#7f8c8d', lw=1.2, alpha=0.65, zorder=2)

        t_t1 = t1[target_idx]
        t_t6 = t6[target_idx]

        if row == 0 and col == 0:
            # BRRS + MeOH (Condition 2)
            # 1. Target (Red: excluded)
            ax.annotate(
                r'$\mathbf{Methanomicrobium\ sp.}$' + f'\n($T_1={t_t1:.1f}\\% \\to T_6={t_t6:.1f}\\%$)',
                (proj_meoh[target_idx], proj_tma[target_idx]),
                textcoords='offset points', xytext=(-35, 20), ha='right', va='bottom',
                fontsize=8.5, fontweight='bold', color=color_target_red,
                bbox=style_target_red, arrowprops=dict(arrowstyle='->', color=color_target_red, lw=1.2), zorder=7
            )
            # 2. Pyramidobacter C12-8 (Orange: #2)
            ax.annotate(
                r'$\mathbf{Pyramidobacter\ C12-8}$' + f'\n($T_1={t1[pyramido_idx]:.1f}\\% \\to T_6={t6[pyramido_idx]:.1f}\\%$)',
                (proj_meoh[pyramido_idx], proj_tma[pyramido_idx]),
                textcoords='offset points', xytext=(-30, -32), ha='right', va='top',
                fontsize=8.2, fontweight='bold', color=color_second,
                bbox=style_second, arrowprops=dict(arrowstyle='->', color=color_second, lw=1.2), zorder=7
            )
            # 3. Thermoplasmata archaeon (Green: #1 Winner)
            ax.annotate(
                r'$\mathbf{Thermoplasmata\ arch.}$' + f'\n($T_1={t1[thermo_idx]:.1f}\\% \\to T_6={t6[thermo_idx]:.1f}\\%$)',
                (proj_meoh[thermo_idx], proj_tma[thermo_idx]),
                textcoords='offset points', xytext=(0, 32), ha='center', va='bottom',
                fontsize=8.2, fontweight='bold', color=color_winner,
                bbox=style_winner, arrowprops=dict(arrowstyle='->', color=color_winner, lw=1.3), zorder=7
            )

        elif row == 0 and col == 1:
            # OSU + MeOH (Condition 8)
            # 1. Target (Green: #1 Winner)
            ax.annotate(
                r'$\mathbf{Methanomicrobium\ sp.}$' + f'\n($T_1={t_t1:.1f}\\% \\to T_6={t_t6:.1f}\\%$)',
                (proj_meoh[target_idx], proj_tma[target_idx]),
                textcoords='offset points', xytext=(-35, 20), ha='right', va='bottom',
                fontsize=8.5, fontweight='bold', color=color_winner,
                bbox=style_winner, arrowprops=dict(arrowstyle='->', color=color_winner, lw=1.3), zorder=7
            )
            # 2. Bacteroidales WCE2008 (Orange: #2)
            ax.annotate(
                r'$\mathbf{Bacteroidales\ WCE2008}$' + f'\n($T_1={t1[bactero_idx]:.1f}\\% \\to T_6={t6[bactero_idx]:.1f}\\%$)',
                (proj_meoh[bactero_idx], proj_tma[bactero_idx]),
                textcoords='offset points', xytext=(12, -28), ha='left', va='top',
                fontsize=8.2, fontweight='bold', color=color_second,
                bbox=style_second, arrowprops=dict(arrowstyle='->', color=color_second, lw=1.2), zorder=7
            )

        elif row == 1 and col == 0:
            # BRRS + TMA (Condition 4)
            # 1. Target (Green: #1 Winner)
            ax.annotate(
                r'$\mathbf{Methanomicrobium\ sp.}$' + f'\n($T_1={t_t1:.1f}\\% \\to T_6={t_t6:.1f}\\%$)',
                (proj_meoh[target_idx], proj_tma[target_idx]),
                textcoords='offset points', xytext=(-35, 20), ha='right', va='bottom',
                fontsize=8.5, fontweight='bold', color=color_winner,
                bbox=style_winner, arrowprops=dict(arrowstyle='->', color=color_winner, lw=1.3), zorder=7
            )
            # 2. Pyramidobacter C12-8 (Orange: #2)
            ax.annotate(
                r'$\mathbf{Pyramidobacter\ C12-8}$' + f'\n($T_1={t1[pyramido_idx]:.1f}\\% \\to T_6={t6[pyramido_idx]:.1f}\\%$)',
                (proj_meoh[pyramido_idx], proj_tma[pyramido_idx]),
                textcoords='offset points', xytext=(-30, -32), ha='right', va='top',
                fontsize=8.2, fontweight='bold', color=color_second,
                bbox=style_second, arrowprops=dict(arrowstyle='->', color=color_second, lw=1.2), zorder=7
            )

        elif row == 1 and col == 1:
            # OSU + TMA (Condition 10)
            # 1. Target (Red: suppressed)
            ax.annotate(
                r'$\mathbf{Methanomicrobium\ sp.}$' + f'\n($T_1={t_t1:.1f}\\% \\to T_6={t_t6:.1f}\\%$)',
                (proj_meoh[target_idx], proj_tma[target_idx]),
                textcoords='offset points', xytext=(-25, 24), ha='right', va='bottom',
                fontsize=8.5, fontweight='bold', color=color_target_red,
                bbox=style_target_red, arrowprops=dict(arrowstyle='->', color=color_target_red, lw=1.2), zorder=7
            )
            # 2. Denitrobacterium detoxificans (Green: #1 Winner)
            ax.annotate(
                r'$\mathbf{Denitrobacterium\ detox.}$' + f'\n($T_1={t1[denitro_idx]:.1f}\\% \\to T_6={t6[denitro_idx]:.1f}\\%$)',
                (proj_meoh[denitro_idx], proj_tma[denitro_idx]),
                textcoords='offset points', xytext=(-35, -30), ha='right', va='top',
                fontsize=8.2, fontweight='bold', color=color_winner,
                bbox=style_winner, arrowprops=dict(arrowstyle='->', color=color_winner, lw=1.3), zorder=7
            )
            # 3. Streptococcus vicugnae (Orange: #2)
            ax.annotate(
                r'$\mathbf{Streptococcus\ vicugnae}$' + f'\n($T_1={t1[vicugnae_idx]:.1f}\\% \\to T_6={t6[vicugnae_idx]:.1f}\\%$)',
                (proj_meoh[vicugnae_idx], proj_tma[vicugnae_idx]),
                textcoords='offset points', xytext=(15, -15), ha='left', va='top',
                fontsize=8.0, fontweight='bold', color=color_second,
                bbox=style_second, arrowprops=dict(arrowstyle='->', color=color_second, lw=1.2), zorder=7
            )

        # Labels & limits
        ax.set_xlabel(r'Projected Methanol Affinity ($\mathbf{\Theta}_i^\top \mathbf{v}_{\mathrm{MeOH}}$)', fontsize=10.2, fontweight='bold')
        ax.set_ylabel(r'Projected TMA Affinity ($\mathbf{\Theta}_i^\top \mathbf{v}_{\mathrm{TMA}}$)', fontsize=10.2, fontweight='bold')
        ax.set_title(f'{letter}  {title}', fontsize=11.2, pad=10, loc='left')
        ax.grid(True, linestyle=':', alpha=0.45)

        # Colorbar for each panel with generous pad
        sm = plt.cm.ScalarMappable(cmap=cmap_trunc, norm=norm_linear)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.045, shrink=0.88)
        cbar.set_label(f'{host} ' + r'$T_1$ Inoculum Abundance (%)', fontsize=8.8, fontweight='bold')
        cbar.set_ticks([0, 10, 20, 30, 40, 50])
        cbar.set_ticklabels(['0%', '10%', '20%', '30%', '40%', '50%'])
        cbar.ax.tick_params(labelsize=8.0)

    # Subtle legend in Panel A explaining the box border colors
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label=r'#1 Winner ($T_6$)', markerfacecolor='#eafaf1', markeredgecolor='#27ae60', markersize=10, markeredgewidth=1.2),
        plt.Line2D([0], [0], marker='s', color='w', label=r'#2 Competitor ($T_6$)', markerfacecolor='#fdf2e9', markeredgecolor='#d35400', markersize=10, markeredgewidth=1.1),
        plt.Line2D([0], [0], marker='s', color='w', label=r'Target Excluded / Low ($T_6$)', markerfacecolor='#fdedec', markeredgecolor='#c0392b', markersize=10, markeredgewidth=1.1)
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper left', fontsize=8.0, framealpha=0.92, edgecolor='#cccccc', facecolor='white')

    plt.tight_layout()

    out_paper = os.path.join(PAPER_FIG_DIR, f'German_Niche_Space_Host_Comparison_BRRS_vs_OSU{suffix}.png')
    out_poster = os.path.join(POSTER_FIG_DIR, f'German_Niche_Space_Host_Comparison_BRRS_vs_OSU{suffix}.png')
    out_artifact = os.path.join(ARTIFACT_DIR, f'German_Niche_Space_Host_Comparison_BRRS_vs_OSU{suffix}.png')

    fig.savefig(out_paper, dpi=300, bbox_inches='tight')
    fig.savefig(out_poster, dpi=300, bbox_inches='tight')
    fig.savefig(out_artifact, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Successfully saved minimalist 4-panel figure ({color_theme}) to {out_paper}')

if __name__ == '__main__':
    create_4panel_figure('Blues')
    create_4panel_figure('Reds')

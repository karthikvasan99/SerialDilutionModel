import os
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform

# Global matplotlib style settings
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# 1. Load Data
DATA_PATH = 'Data/German_processed_data.pt'
data = torch.load(DATA_PATH, weights_only=False)
trajectories = data['trajectories']
strains = data['strains']
num_species = len(strains)

# 2. Enhanced Taxon Grouping
def clean_taxon_name(s):
    if 'Methanomicrobium' in s:
        return 'Methanomicrobium sp.'
    if 'Thermoplasmata' in s:
        return 'Thermoplasmata archaeon'
    if 'methanogenic archaeon' in s:
        return 'Methanogenic archaeon'
    if 'Pyramidobacter sp. C12-8' in s:
        return 'Pyramidobacter sp. C12-8'
    if 'Pyramidobacter' in s:
        return 'Pyramidobacter other'
    if 'Prevotellaceae' in s:
        return 'Prevotellaceae UBA7028'
    if 'WCE2008' in s:
        return 'Bacteroidales WCE2008'
    if 'Streptococcus' in s:
        return 'Streptococcus spp.'
    if 'Clostridium' in s or 'aminophilum' in s:
        return 'Clostridium spp.'
    if 'Megasphaera' in s:
        return 'Megasphaera elsdenii'
    if 'Denitrobacterium' in s:
        return 'Denitrobacterium detoxificans'
    if 'Anaerovibrio' in s:
        return 'Anaerovibrio spp.'
    if 'Kandleria' in s:
        return 'Kandleria spp.'
    if 'Bacteroides' in s:
        return 'Bacteroides spp.'
    return 'Other taxa'

cleaned_taxa = [clean_taxon_name(s) for s in strains]

key_taxa_ordered = [
    'Methanomicrobium sp.',          # Hydrogenotrophic Methanogen (Teal)
    'Thermoplasmata archaeon',       # Methylotrophic Methanogen (Orange)
    'Methanogenic archaeon',         # Methylotrophic Methanogen (Warm Amber)
    'Pyramidobacter sp. C12-8',      # Synergistetes Fermenter (Dark Crimson)
    'Pyramidobacter other',          # Synergistetes Fermenter (Soft Coral)
    'Prevotellaceae UBA7028',        # Bacteroidetes (Purple)
    'Bacteroidales WCE2008',         # Bacteroidetes (Violet/Mauve)
    'Streptococcus spp.',            # Firmicutes (Royal Blue)
    'Clostridium spp.',              # Firmicutes (Turquoise)
    'Megasphaera elsdenii',          # Firmicutes (Tan/Gold)
    'Denitrobacterium detoxificans', # Actinobacteria (Forest Green)
    'Anaerovibrio spp.',             # Firmicutes (Brown)
    'Kandleria spp.',                # Firmicutes (Rose Pink)
    'Bacteroides spp.',              # Bacteroidetes (Mint Green)
    'Other taxa'                     # Remainder (Cool Slate Grey)
]

taxa_colors = {
    'Methanomicrobium sp.': '#008080',          # Teal
    'Thermoplasmata archaeon': '#FF7F0E',       # Vivid Orange
    'Methanogenic archaeon': '#FDB462',         # Warm Amber
    'Pyramidobacter sp. C12-8': '#D62728',      # Dark Crimson
    'Pyramidobacter other': '#FB8072',          # Soft Coral
    'Prevotellaceae UBA7028': '#9467BD',        # Purple
    'Bacteroidales WCE2008': '#BC80BD',         # Violet/Mauve
    'Streptococcus spp.': '#1F77B4',            # Royal Blue
    'Clostridium spp.': '#17BECF',              # Turquoise
    'Megasphaera elsdenii': '#E6AB02',          # Mustard/Gold
    'Denitrobacterium detoxificans': '#2CA02C', # Forest Green
    'Anaerovibrio spp.': '#8C564B',             # Brown
    'Kandleria spp.': '#E377C2',                # Rose Pink
    'Bacteroides spp.': '#66C2A5',              # Mint/Sea Green
    'Other taxa': '#CBD5E1'                     # Cool Slate Grey
}

# Metric helpers
def bray_curtis(x, y):
    return np.sum(np.abs(x - y)) / np.sum(x + y)

def jensen_shannon(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0); p = p / np.sum(p)
    q = np.clip(q, eps, 1.0); q = q / np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return np.sqrt(0.5 * (kl_pm + kl_qm))

def get_taxa_matrix(tensor):
    seq = tensor.numpy() # (6, 751)
    df_taxa = pd.DataFrame(index=range(6), columns=key_taxa_ordered, data=0.0)
    for t in range(6):
        vec = seq[t]
        for s_idx, s_val in enumerate(vec):
            c_name = cleaned_taxa[s_idx]
            if c_name in key_taxa_ordered:
                df_taxa.loc[t, c_name] += s_val
            else:
                df_taxa.loc[t, 'Other taxa'] += s_val
    return df_taxa

# Conditions structure
conditions_order = [
    # Host BRRS
    ('BRRS', 'H2:CO2', 5.0),
    ('BRRS', 'H2:CO2', 30.0),
    ('BRRS', 'H2:CO2 + MeOH', 5.0),
    ('BRRS', 'H2:CO2 + MeOH', 30.0),
    ('BRRS', 'H2:CO2 + TMA', 5.0),
    ('BRRS', 'H2:CO2 + TMA', 30.0),
    # Host OSU
    ('OSU', 'H2:CO2', 5.0),
    ('OSU', 'H2:CO2', 30.0),
    ('OSU', 'H2:CO2 + MeOH', 5.0),
    ('OSU', 'H2:CO2 + MeOH', 30.0),
    ('OSU', 'H2:CO2 + TMA', 5.0),
    ('OSU', 'H2:CO2 + TMA', 30.0)
]

output_dir = 'Figures'
os.makedirs(output_dir, exist_ok=True)
artifact_dir = '/home/karthik/.gemini/antigravity-cli/brain/9c7d67f0-4221-486d-b843-9eb798320cb8'

# ==============================================================================
# FIGURE 1: MASTER OVERVIEW FIGURE (Composition, BC Dissimilarity, Drift, PCoA)
# ==============================================================================
print("Generating Figure 1: Master Overview...")
fig = plt.figure(figsize=(24, 18), facecolor='white')

# Suptitle at top
fig.suptitle("Microbiome Composition Dynamics in German's Serial Dilution Data (T1 to T6)\nAcross Media, Hydrogen Pressure, and Inoculum Host Conditions",
             fontsize=19, fontweight='bold', y=0.985)

# Gridspec with ample top space for suptitle and legend
gs = fig.add_gridspec(3, 6, height_ratios=[1.1, 1.1, 1.3], hspace=0.38, wspace=0.28,
                      top=0.88, bottom=0.05, left=0.05, right=0.98)

# Panel A Section Header
fig.text(0.05, 0.905, "A. Passage-by-Passage Community Composition Across 12 Regimes (T1 $\\to$ T6)",
         fontsize=14, fontweight='bold', color='#111111')

# Taxa Legend across top (placed between suptitle and Panel A title)
legend_handles = [mpatches.Patch(facecolor=taxa_colors[t], edgecolor='#333333', linewidth=0.5, label=t) for t in key_taxa_ordered]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.948),
           ncol=5, fontsize=10, frameon=True, edgecolor='#cccccc', facecolor='#fafafa')

# ----------------- ROW 1 & 2: 12-Panel Stacked Composition Area Plots -----------------
for idx, cond_key in enumerate(conditions_order):
    origin, substrate, pressure = cond_key
    row = 0 if origin == 'BRRS' else 1
    col = idx % 6
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor('#ffffff')
    
    seq_tensor = trajectories[cond_key]
    df_taxa = get_taxa_matrix(seq_tensor)
    
    x = np.arange(1, 7)
    bottom = np.zeros(6)
    
    for t_name in key_taxa_ordered:
        vals = df_taxa[t_name].values
        ax.bar(x, vals, bottom=bottom, color=taxa_colors[t_name], edgecolor='white', linewidth=0.4, width=0.78)
        bottom += vals
        
    ax.set_xlim(0.4, 6.6)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f'T{t}' for t in range(1, 7)], fontsize=9.5, fontweight='bold')
    
    # Subtitle formatting
    sub_title = f"{substrate}\n({int(pressure)} psi)"
    ax.set_title(sub_title, fontsize=11, fontweight='bold', pad=4)
    
    if col == 0:
        ax.set_ylabel(f"{origin} Host\nRel. Abundance", fontsize=12, fontweight='bold')
    else:
        ax.set_yticklabels([])
        
    # Annotate T1 -> T6 BC distance
    bc_val = bray_curtis(seq_tensor[0].numpy(), seq_tensor[5].numpy())
    ax.text(0.96, 0.94, f"$\\Delta_{{BC}}$={bc_val:.2f}", transform=ax.transAxes,
            ha='right', va='top', fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.92, edgecolor='#888888'))

# ----------------- ROW 3: QUANTITATIVE DRIFT & ORDINATION -----------------
# Subplot 3A: Bray-Curtis Dissimilarity (T1 vs T6)
ax_bc = fig.add_subplot(gs[2, 0:2])
ax_bc.set_facecolor('#ffffff')

bc_data = []
for cond_key in conditions_order:
    origin, substrate, pressure = cond_key
    seq = trajectories[cond_key].numpy()
    bc_val = bray_curtis(seq[0], seq[5])
    jsd_val = jensen_shannon(seq[0], seq[5])
    bc_data.append({
        'Host': origin,
        'Substrate': substrate,
        'Pressure': f'{int(pressure)} psi',
        'Condition': f"{origin}\n{substrate}\n{int(pressure)}psi",
        'BC': bc_val,
        'JSD': jsd_val
    })
df_bc = pd.DataFrame(bc_data)

x_pos = np.arange(len(df_bc))
media_bar_colors = []
for _, r in df_bc.iterrows():
    if r['Substrate'] == 'H2:CO2':
        c = '#2E86AB' if r['Host'] == 'BRRS' else '#66B2D6'
    elif 'MeOH' in r['Substrate']:
        c = '#D35400' if r['Host'] == 'BRRS' else '#E67E22'
    else: # TMA
        c = '#27AE60' if r['Host'] == 'BRRS' else '#58D68D'
    media_bar_colors.append(c)

hatches = ['///' if '30' in r['Pressure'] else '' for _, r in df_bc.iterrows()]

bars = ax_bc.bar(x_pos, df_bc['BC'], color=media_bar_colors, edgecolor='black', linewidth=0.8, alpha=0.9, width=0.75)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# Add values on top
for bar in bars:
    h = bar.get_height()
    ax_bc.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.2f}",
               ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax_bc.axvline(5.5, color='gray', linestyle='--', lw=1.5)
ax_bc.text(2.5, 0.90, "BRRS Inoculum (Mean: 0.67)", ha='center', fontsize=9.5, fontweight='bold', color='#8B0000',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff0f0', edgecolor='#ffcccc'))
ax_bc.text(8.5, 0.90, "OSU Inoculum (Mean: 0.49)", ha='center', fontsize=9.5, fontweight='bold', color='#104E8B',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f8ff', edgecolor='#cce6ff'))

ax_bc.set_xticks(x_pos)
cond_labels = [f"{'MeOH' if 'MeOH' in r['Substrate'] else ('TMA' if 'TMA' in r['Substrate'] else 'H2')}\n{r['Pressure']}" for _, r in df_bc.iterrows()]
ax_bc.set_xticklabels(cond_labels, fontsize=8.5)
ax_bc.set_ylabel("Bray-Curtis Dissimilarity ($T_1 \\to T_6$)", fontsize=11, fontweight='bold')
ax_bc.set_title("B. Total Community Turnover ($T_1$ vs $T_6$)", fontsize=12.5, fontweight='bold', loc='left')
ax_bc.set_ylim(0, 0.98)
ax_bc.grid(axis='y', linestyle=':', alpha=0.6)

bc_legend_elements = [
    mpatches.Patch(facecolor='#2E86AB', edgecolor='k', label='H2:CO2'),
    mpatches.Patch(facecolor='#D35400', edgecolor='k', label='H2:CO2 + MeOH'),
    mpatches.Patch(facecolor='#27AE60', edgecolor='k', label='H2:CO2 + TMA'),
    mpatches.Patch(facecolor='white', edgecolor='k', hatch='///', label='30 psi (vs 5 psi solid)')
]
ax_bc.legend(handles=bc_legend_elements, loc='upper left', fontsize=8, framealpha=0.9)


# Subplot 3B: Cumulative Divergence from Initial T1
ax_drift = fig.add_subplot(gs[2, 2:4])
ax_drift.set_facecolor('#ffffff')
palette_lines = {
    'H2:CO2': '#2E86AB',
    'H2:CO2 + MeOH': '#D35400',
    'H2:CO2 + TMA': '#27AE60'
}

for cond_key in conditions_order:
    origin, substrate, pressure = cond_key
    seq = trajectories[cond_key].numpy()
    drifts = [bray_curtis(seq[0], seq[t]) for t in range(6)]
    
    col_line = palette_lines[substrate]
    ls = '-' if origin == 'BRRS' else '--'
    lw = 2.4 if pressure == 30.0 else 1.5
    marker = 'o' if origin == 'BRRS' else 's'
    
    ax_drift.plot(range(1, 7), drifts, color=col_line, linestyle=ls, linewidth=lw,
                  marker=marker, markersize=5.5, alpha=0.85)

ax_drift.set_xticks(range(1, 7))
ax_drift.set_xticklabels([f'T{t}' for t in range(1, 7)], fontsize=10, fontweight='bold')
ax_drift.set_xlabel("Passage (Transfer Number)", fontsize=11, fontweight='bold')
ax_drift.set_ylabel("Bray-Curtis Distance from $T_1$ ($d_{BC}(T_1, T_t)$)", fontsize=11, fontweight='bold')
ax_drift.set_title("C. Cumulative Divergence from Inoculum", fontsize=12.5, fontweight='bold', loc='left')
ax_drift.set_ylim(-0.02, 0.92)
ax_drift.grid(True, linestyle=':', alpha=0.6)

drift_handles = [
    Line2D([0], [0], color='#2E86AB', lw=2, label='H2:CO2'),
    Line2D([0], [0], color='#D35400', lw=2, label='H2:CO2 + MeOH'),
    Line2D([0], [0], color='#27AE60', lw=2, label='H2:CO2 + TMA'),
    Line2D([0], [0], color='black', ls='-', marker='o', lw=1.5, label='BRRS Host Inoculum'),
    Line2D([0], [0], color='black', ls='--', marker='s', lw=1.5, label='OSU Host Inoculum'),
    Line2D([0], [0], color='black', ls='-', lw=2.4, label='30 psi (thick vs 5 psi thin)')
]
ax_drift.legend(handles=drift_handles, loc='lower right', fontsize=8, framealpha=0.9)


# Subplot 3C: PCoA Ordination Trajectories
ax_pcoa = fig.add_subplot(gs[2, 4:6])
ax_pcoa.set_facecolor('#ffffff')

# Build full sample matrix for PCoA
sample_mat = []
sample_meta = []
for cond_key in conditions_order:
    origin, substrate, pressure = cond_key
    seq = trajectories[cond_key].numpy()
    for t in range(6):
        sample_mat.append(seq[t])
        sample_meta.append({
            'Origin': origin,
            'Substrate': substrate,
            'Pressure': pressure,
            'Passage': t + 1,
            'CondKey': cond_key
        })

X_all = np.array(sample_mat)
df_meta = pd.DataFrame(sample_meta)
D = squareform(pdist(X_all, metric='braycurtis'))
n_samples = len(X_all)
H_mat = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
B_mat = -0.5 * H_mat.dot(D ** 2).dot(H_mat)
evals, evecs = np.linalg.eigh(B_mat)
sort_i = np.argsort(evals)[::-1]
evals = evals[sort_i]
evecs = evecs[:, sort_i]
var1 = evals[0] / np.sum(np.maximum(evals, 0)) * 100
var2 = evals[1] / np.sum(np.maximum(evals, 0)) * 100

coords = evecs[:, :2] * np.sqrt(np.maximum(evals[:2], 0))
df_meta['PCoA1'] = coords[:, 0]
df_meta['PCoA2'] = coords[:, 1]

for cond_key in conditions_order:
    origin, substrate, pressure = cond_key
    sub_df = df_meta[df_meta['CondKey'] == cond_key].sort_values('Passage')
    
    col_line = palette_lines[substrate]
    ls = '-' if origin == 'BRRS' else '--'
    lw = 2.0 if pressure == 30.0 else 1.2
    
    # Plot path
    ax_pcoa.plot(sub_df['PCoA1'], sub_df['PCoA2'], color=col_line, linestyle=ls, linewidth=lw, alpha=0.7)
    
    # Arrows along trajectory
    for k in range(len(sub_df) - 1):
        x1, y1 = sub_df.iloc[k][['PCoA1', 'PCoA2']]
        x2, y2 = sub_df.iloc[k+1][['PCoA1', 'PCoA2']]
        if k == 2: # Add arrow at mid passage
            ax_pcoa.annotate('', xy=(x2, y2), xytext=(x1, y1),
                             arrowprops=dict(arrowstyle="->", color=col_line, lw=1.5, alpha=0.85))
            
    # T1 Point (circle)
    p1 = sub_df.iloc[0]
    ax_pcoa.scatter(p1['PCoA1'], p1['PCoA2'], s=60, facecolors='white', edgecolors=col_line,
                    linewidth=2.0, zorder=5)
    
    # T6 Point (star for BRRS, square for OSU)
    p6 = sub_df.iloc[5]
    marker_t6 = '*' if origin == 'BRRS' else 's'
    ax_pcoa.scatter(p6['PCoA1'], p6['PCoA2'], s=110, color=col_line, edgecolors='black',
                    marker=marker_t6, linewidth=1.0, zorder=6)

ax_pcoa.set_xlabel(f"PCoA Axis 1 ({var1:.1f}% Variance)", fontsize=11, fontweight='bold')
ax_pcoa.set_ylabel(f"PCoA Axis 2 ({var2:.1f}% Variance)", fontsize=11, fontweight='bold')
ax_pcoa.set_title("D. Community State Trajectories (Bray-Curtis PCoA)", fontsize=12.5, fontweight='bold', loc='left')
ax_pcoa.grid(True, linestyle=':', alpha=0.6)

# Annotate clusters in PCoA space
ax_pcoa.text(-0.27, -0.32, "BRRS Inoculum (T1)\nPyramidobacter-rich", fontsize=8.5, color='#a01010', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffeeee', edgecolor='#cc9999', alpha=0.85))
ax_pcoa.text(0.24, 0.16, "H2:CO2 Enriched (T6)\nMethanomicrobium-rich", fontsize=8.5, color='#006666', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#e6f7f7', edgecolor='#99cccc', alpha=0.85))
ax_pcoa.text(-0.25, 0.41, "BRRS MeOH (5 psi, T6)\nThermoplasmata-rich", fontsize=8.5, color='#d95f02', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff5eb', edgecolor='#fdbe85', alpha=0.85))

pcoa_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='white', markersize=8, label='T1 (Passage 1)'),
    Line2D([0], [0], marker='*', color='w', markeredgecolor='black', markerfacecolor='gray', markersize=10, label='T6 (BRRS Host)'),
    Line2D([0], [0], marker='s', color='w', markeredgecolor='black', markerfacecolor='gray', markersize=8, label='T6 (OSU Host)')
]
ax_pcoa.legend(handles=pcoa_legend_elements, loc='lower left', fontsize=8, framealpha=0.9)

fig1_path = os.path.join(output_dir, 'German_microbiome_dynamics_overview.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"Saved Figure 1 to {fig1_path}")
plt.close()

# ==============================================================================
# FIGURE 2: DETAILED 12-PANEL COMPOSITION & STEP-WISE DISSIMILARITY
# ==============================================================================
print("Generating Figure 2: Detailed 12-Panel Composition Dynamics...")
fig2, axes2 = plt.subplots(4, 6, figsize=(26, 16), facecolor='white',
                           gridspec_kw={'height_ratios': [2, 1, 2, 1], 'hspace': 0.38, 'wspace': 0.26, 'top': 0.88, 'bottom': 0.05})

fig2.suptitle("Passage-to-Passage Community Composition & Step-wise Shifts Across 12 Enrichment Regimes",
              fontsize=18, fontweight='bold', y=0.975)

for idx, cond_key in enumerate(conditions_order):
    origin, substrate, pressure = cond_key
    seq_tensor = trajectories[cond_key]
    df_taxa = get_taxa_matrix(seq_tensor)
    
    is_brrs = (origin == 'BRRS')
    col = idx % 6
    row_top = 0 if is_brrs else 2
    row_bot = 1 if is_brrs else 3
    
    ax_top = axes2[row_top, col]
    ax_bot = axes2[row_bot, col]
    ax_top.set_facecolor('#ffffff')
    ax_bot.set_facecolor('#ffffff')
    
    # 1. Stacked bar plot
    x = np.arange(1, 7)
    bottom = np.zeros(6)
    for t_name in key_taxa_ordered:
        vals = df_taxa[t_name].values
        ax_top.bar(x, vals, bottom=bottom, color=taxa_colors[t_name], edgecolor='white', linewidth=0.4, width=0.78)
        bottom += vals
        
    ax_top.set_xlim(0.4, 6.6)
    ax_top.set_ylim(0, 1.0)
    ax_top.set_xticks(range(1, 7))
    ax_top.set_xticklabels([f'T{t}' for t in range(1, 7)], fontsize=9, fontweight='bold')
    ax_top.set_title(f"{origin} | {substrate}\n({int(pressure)} psi)", fontsize=11, fontweight='bold', pad=3)
    ax_top.grid(axis='y', linestyle=':', alpha=0.5)
    
    if col == 0:
        ax_top.set_ylabel(f"{origin} Host\nRelative Abundance", fontsize=11, fontweight='bold')
    else:
        ax_top.set_yticklabels([])
        
    # 2. Step-wise Bray-Curtis shift bars
    seq = seq_tensor.numpy()
    step_bcs = [bray_curtis(seq[t], seq[t+1]) for t in range(5)]
    step_labels = ['T1-2', 'T2-3', 'T3-4', 'T4-5', 'T5-6']
    
    bar_col = '#C0392B' if is_brrs else '#2980B9'
    bars = ax_bot.bar(range(5), step_bcs, color=bar_col, edgecolor='black', alpha=0.85, width=0.68)
    for b in bars:
        h = b.get_height()
        ax_bot.text(b.get_x() + b.get_width()/2., h + 0.02, f"{h:.2f}",
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')
                    
    ax_bot.set_xticks(range(5))
    ax_bot.set_xticklabels(step_labels, fontsize=8, fontweight='bold')
    ax_bot.set_ylim(0, 0.85)
    ax_bot.grid(axis='y', linestyle=':', alpha=0.5)
    
    if col == 0:
        ax_bot.set_ylabel("Step $\\Delta_{BC}$", fontsize=10, fontweight='bold')
    else:
        ax_bot.set_yticklabels([])

# Common legend for figure 2
fig2.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.935),
            ncol=5, fontsize=10, frameon=True, edgecolor='#cccccc', facecolor='#fafafa')

fig2_path = os.path.join(output_dir, 'German_passage_dynamics_12panels.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"Saved Figure 2 to {fig2_path}")
plt.close()

# ==============================================================================
# FIGURE 3: DIFFERENTIAL ABUNDANCE HEATMAP & FUNCTIONAL GUILD BLOOMS
# ==============================================================================
print("Generating Figure 3: Taxa Shifts & Functional Clade Blooms...")
fig3, (ax_heat, ax_guilds) = plt.subplots(1, 2, figsize=(22, 10), facecolor='white',
                                          gridspec_kw={'width_ratios': [1.25, 1.0], 'wspace': 0.28, 'top': 0.90, 'bottom': 0.12})

fig3.suptitle("Taxonomic Selection & Functional Guild Trajectories in German's Microbiome Enrichments",
              fontsize=17, fontweight='bold', y=0.97)

# Panel A: Differential Abundance Heatmap (T6 - T1)
diff_taxa_ordered = [t for t in key_taxa_ordered if t != 'Other taxa']

diff_matrix = np.zeros((len(diff_taxa_ordered), len(conditions_order)))
cond_col_labels = []

for c_idx, cond_key in enumerate(conditions_order):
    origin, substrate, pressure = cond_key
    seq = trajectories[cond_key].numpy()
    df_taxa = get_taxa_matrix(trajectories[cond_key])
    diff_vec = df_taxa.loc[5, diff_taxa_ordered].values - df_taxa.loc[0, diff_taxa_ordered].values
    diff_matrix[:, c_idx] = diff_vec
    
    cond_col_labels.append(f"{origin}\n{substrate}\n{int(pressure)}psi")

# Plot Heatmap
im = ax_heat.imshow(diff_matrix, cmap='RdBu_r', vmin=-0.45, vmax=0.45, aspect='auto')

# Colorbar
cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
cbar.set_label("Change in Relative Abundance ($\\Delta = T_6 - T_1$)", fontsize=11, fontweight='bold')

ax_heat.set_xticks(range(len(conditions_order)))
ax_heat.set_xticklabels(cond_col_labels, fontsize=8.5, fontweight='bold')
ax_heat.set_yticks(range(len(diff_taxa_ordered)))
ax_heat.set_yticklabels(diff_taxa_ordered, fontsize=10.5, fontweight='bold')

# Write numerical changes in heatmap cells
for i in range(len(diff_taxa_ordered)):
    for j in range(len(conditions_order)):
        val = diff_matrix[i, j]
        if abs(val) >= 0.03:
            text_color = 'white' if abs(val) > 0.2 else 'black'
            ax_heat.text(j, i, f"{val:+.2f}", ha='center', va='center', fontsize=7.5,
                         fontweight='bold', color=text_color)

ax_heat.axvline(5.5, color='black', linewidth=2.5)
ax_heat.text(2.5, -0.9, "BRRS Inoculum Conditions", ha='center', fontsize=11, fontweight='bold', color='#8B0000')
ax_heat.text(8.5, -0.9, "OSU Inoculum Conditions", ha='center', fontsize=11, fontweight='bold', color='#104E8B')
ax_heat.set_title("A. Taxon-Specific Differential Abundance ($\\Delta = T_6 - T_1$)", fontsize=13, fontweight='bold', loc='left')

# Panel B: Key Functional Guilds Trajectories
substrates = ['H2:CO2', 'H2:CO2 + MeOH', 'H2:CO2 + TMA']
sub_styles = {'H2:CO2': '-', 'H2:CO2 + MeOH': '--', 'H2:CO2 + TMA': ':'}
passages = np.arange(1, 7)

for sub in substrates:
    brrs_5 = get_taxa_matrix(trajectories[('BRRS', sub, 5.0)])
    brrs_30 = get_taxa_matrix(trajectories[('BRRS', sub, 30.0)])
    osu_5 = get_taxa_matrix(trajectories[('OSU', sub, 5.0)])
    osu_30 = get_taxa_matrix(trajectories[('OSU', sub, 30.0)])
    
    # Hydrogenotrophic Methanogens
    h2_methano = (brrs_5['Methanomicrobium sp.'] + brrs_30['Methanomicrobium sp.'] +
                  osu_5['Methanomicrobium sp.'] + osu_30['Methanomicrobium sp.']) / 4.0
    
    # Methylotrophic Methanogens
    methyl_methano = (brrs_5['Thermoplasmata archaeon'] + brrs_5['Methanogenic archaeon'] +
                      brrs_30['Thermoplasmata archaeon'] + brrs_30['Methanogenic archaeon'] +
                      osu_5['Thermoplasmata archaeon'] + osu_5['Methanogenic archaeon'] +
                      osu_30['Thermoplasmata archaeon'] + osu_30['Methanogenic archaeon']) / 4.0
                      
    # Synergistetes Fermenters
    synergist = (brrs_5['Pyramidobacter sp. C12-8'] + brrs_5['Pyramidobacter other'] +
                 brrs_30['Pyramidobacter sp. C12-8'] + brrs_30['Pyramidobacter other'] +
                 osu_5['Pyramidobacter sp. C12-8'] + osu_5['Pyramidobacter other'] +
                 osu_30['Pyramidobacter sp. C12-8'] + osu_30['Pyramidobacter other']) / 4.0

    ax_guilds.plot(passages, h2_methano, color='#008080', linestyle=sub_styles[sub], linewidth=2.5,
                   marker='o', label=f'H2 Methanogens ({sub})')
    ax_guilds.plot(passages, methyl_methano, color='#E67E22', linestyle=sub_styles[sub], linewidth=2.5,
                   marker='s', label=f'Methylotrophic Methanogens ({sub})')
    ax_guilds.plot(passages, synergist, color='#D62728', linestyle=sub_styles[sub], linewidth=2.5,
                   marker='^', label=f'Synergistetes ({sub})')

ax_guilds.set_xticks(passages)
ax_guilds.set_xticklabels([f'T{t}' for t in passages], fontsize=11, fontweight='bold')
ax_guilds.set_xlabel("Passage (Transfer Number)", fontsize=11, fontweight='bold')
ax_guilds.set_ylabel("Mean Relative Abundance", fontsize=11, fontweight='bold')
ax_guilds.set_title("B. Functional Guild Trajectories Across Media Substrates", fontsize=13, fontweight='bold', loc='left')
ax_guilds.grid(True, linestyle=':', alpha=0.6)
ax_guilds.legend(fontsize=8.5, loc='upper right', framealpha=0.9)

fig3_path = os.path.join(output_dir, 'German_taxa_shifts_and_guilds.png')
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"Saved Figure 3 to {fig3_path}")
plt.close()

# Copy figures to artifact directory
shutil.copy(fig1_path, os.path.join(artifact_dir, 'German_microbiome_dynamics_overview.png'))
shutil.copy(fig2_path, os.path.join(artifact_dir, 'German_passage_dynamics_12panels.png'))
shutil.copy(fig3_path, os.path.join(artifact_dir, 'German_taxa_shifts_and_guilds.png'))
print("All figures updated and copied to artifact directory.")

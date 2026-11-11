import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import shutil

# Load cached data
cache = np.load('Data/Gore_Comm5_Oscillation_Cache.npz')

rel_abund = cache['rel_abund']         # (80, 3)
R_pca = cache['R_pca']                 # (80, 120, 2)
pca_var = cache['pca_var']             # (2,)
species_idx = cache['species_idx']     # [18, 21, 22]

# Set typography and styling to match Ophelia/poster style with prominent labels
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.8
plt.rcParams['xtick.major.width'] = 1.6
plt.rcParams['ytick.major.width'] = 1.6
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

fig = plt.figure(figsize=(18, 5.8))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

sp_colors = ['#1f77b4', '#2ca02c', '#d62728']
sp_labels = [f'Species {idx}' for idx in species_idx]

# ── Panel A: Relative Abundance Oscillations ─────────────────────────────────
ax_a = fig.add_subplot(gs[0])
passages = np.arange(1, 41)
for i in range(3):
    ax_a.plot(passages, rel_abund[:40, i], color=sp_colors[i], lw=3.0, label=sp_labels[i])
ax_a.set_xlabel('Serial Dilution Passage', fontsize=18, fontweight='bold', labelpad=8)
ax_a.set_ylabel('Relative Abundance', fontsize=18, fontweight='bold', labelpad=8)
ax_a.tick_params(axis='both', which='major', labelsize=15)
ax_a.set_ylim(-0.02, 1.05)
ax_a.set_xlim(1, 40)
ax_a.legend(loc='upper right', fontsize=14, framealpha=0.92, edgecolor='none')
ax_a.grid(True, alpha=0.3, linestyle='--')

# ── Panel B: Species Phase Portrait in Relative Abundance ───────────────────
ax_b = fig.add_subplot(gs[1])
cmap_b = plt.cm.plasma
start_idx = 40
n_show = 80 - start_idx

# Plot full phase trajectory (gray background curve)
ax_b.plot(rel_abund[start_idx:, 0], rel_abund[start_idx:, 1], '-', color='#888888', lw=1.5, alpha=0.5, zorder=1)

# Color points by passage number
for i in range(n_show):
    c = cmap_b(i / (n_show - 1))
    ax_b.plot(rel_abund[start_idx+i, 0], rel_abund[start_idx+i, 1], 'o', color=c, markersize=8.0, alpha=0.9, zorder=2)

sm_b = plt.cm.ScalarMappable(cmap=cmap_b, norm=plt.Normalize(40, 80))
sm_b.set_array([])
cbar_b = plt.colorbar(sm_b, ax=ax_b, shrink=0.82, pad=0.04)
cbar_b.set_label('Passage #', fontsize=15, fontweight='bold', labelpad=6)
cbar_b.ax.tick_params(labelsize=13)

ax_b.set_xlabel('Species 18 Relative Abundance', fontsize=18, fontweight='bold', labelpad=8)
ax_b.set_ylabel('Species 21 Relative Abundance', fontsize=18, fontweight='bold', labelpad=8)
ax_b.tick_params(axis='both', which='major', labelsize=15)
ax_b.grid(True, alpha=0.3, linestyle='--')

# ── Panel C: Resource PCA Trajectories ───────────────────────────────────────
ax_c = fig.add_subplot(gs[2])
cmap_c = plt.cm.viridis

# Transient passages 5-39 in light gray
for p in range(5, 40):
    ax_c.plot(R_pca[p, :, 0], R_pca[p, :, 1], color='#cccccc', lw=1.0, alpha=0.6, zorder=1)

# Converged passages 40-79 color-coded
for p in range(40, 80):
    color = cmap_c((p - 40) / 39)
    ax_c.plot(R_pca[p, :, 0], R_pca[p, :, 1], lw=2.2, color=color, alpha=0.85, zorder=2)
    # Start (circle) and end (triangle) of each intra-passage cycle
    ax_c.plot(R_pca[p, 0, 0], R_pca[p, 0, 1], 'o', color=color, ms=5.5, zorder=3)
    ax_c.plot(R_pca[p, -1, 0], R_pca[p, -1, 1], '^', color=color, ms=5.5, zorder=3)

sm_c = plt.cm.ScalarMappable(cmap=cmap_c, norm=plt.Normalize(40, 80))
sm_c.set_array([])
cbar_c = plt.colorbar(sm_c, ax=ax_c, shrink=0.82, pad=0.04)
cbar_c.set_label('Passage #', fontsize=15, fontweight='bold', labelpad=6)
cbar_c.ax.tick_params(labelsize=13)

ax_c.set_xlabel(f'Resource PC1 ({pca_var[0]*100:.1f}%)', fontsize=18, fontweight='bold', labelpad=8)
ax_c.set_ylabel(f'Resource PC2 ({pca_var[1]*100:.1f}%)', fontsize=18, fontweight='bold', labelpad=8)
ax_c.tick_params(axis='both', which='major', labelsize=15)

legend_elements = [
    Line2D([0], [0], color='#cccccc', lw=2.5, label='Transients (5–39)'),
    Line2D([0], [0], color=cmap_c(0.6), lw=2.5, label='Converged (40–80)')
]
ax_c.legend(handles=legend_elements, fontsize=13, framealpha=0.92, loc='upper right', edgecolor='none')
ax_c.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

out_fig = 'Poster/figures/Gore_Oscillation_Combined_Panel.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"Saved: {out_fig}")

# Copy to brain artifact directory
shutil.copy(out_fig, '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143/Gore_Oscillation_Combined_Panel.png')
print("Copied to artifacts.")

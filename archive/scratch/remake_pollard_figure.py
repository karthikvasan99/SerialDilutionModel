import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.cluster.hierarchy import linkage, leaves_list

# Load cached data
cache_path = 'Data/Poster_Cache_Pollard.npz'
data = np.load(cache_path, allow_pickle=True)

Theta_np = data['Theta']
W_np = data['W']
true_r2 = data['full_r2']
retrained_w0_r2 = data['no_w_r2']
persistence_r2 = data['persist_r2']
shuff_a_r2 = data['shuff_a_r2']
shuff_th_r2 = data['shuff_th_r2']
clean_env_names = data['env_names']

# Hierarchical clustering of Theta for Panel A
Z = linkage(Theta_np, method='ward')
order = leaves_list(Z)
Theta_clustered = Theta_np[order]

# Multi-panel figure in 2 rows:
# Row 1: Panels A and B as square heatmaps side by side
# Row 2: Panel C spanning the full width, taller for legend and labels
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.8
plt.rcParams['xtick.major.width'] = 1.6
plt.rcParams['ytick.major.width'] = 1.6
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

fig = plt.figure(figsize=(13.6, 10.8), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.30, 1.15], hspace=0.34, wspace=0.18,
                       left=0.085, right=0.96, top=0.96, bottom=0.08)

# Panel A: Uptake Traits Theta (Square Box, anchored West to align left border with Row 2)
ax0 = fig.add_subplot(gs[0, 0])
im0 = ax0.imshow(Theta_clustered, aspect='auto', cmap='YlGnBu', interpolation='nearest')
ax0.set_box_aspect(1)
ax0.set_anchor('W')
badge_a = dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#00356B', alpha=0.92, lw=1.5)
ax0.text(0.05, 0.94, r"$\mathbf{Uptake\ Traits\ (\Theta)}$", transform=ax0.transAxes, fontsize=16.0, fontweight='bold', va='top', bbox=badge_a)
ax0.set_xlabel('Resource', fontsize=18.0, fontweight='bold', color='black', labelpad=6)
ax0.set_ylabel('Strains (N=50)', fontsize=18.0, fontweight='bold', color='black', labelpad=6)
ax0.set_xticks(range(5))
ax0.set_xticklabels([f'$R_{k+1}$' for k in range(5)], fontsize=15.0, fontweight='bold')
ax0.set_yticks([])

cax0 = inset_axes(ax0, width='4.5%', height='100%', loc='lower left',
                  bbox_to_anchor=(1.04, 0.0, 1.0, 1.0), bbox_transform=ax0.transAxes, borderpad=0)
cbar0 = fig.colorbar(im0, cax=cax0)
cbar0.set_label('Uptake Rate', fontsize=16.0, fontweight='bold', color='black')
cbar0.ax.tick_params(labelsize=14.0)

# Panel B: Cross-Feeding W (Square Box, anchored West to fill row width)
ax1 = fig.add_subplot(gs[0, 1])
im1 = ax1.imshow(W_np, aspect='equal', cmap='magma_r', interpolation='nearest')
ax1.set_box_aspect(1)
ax1.set_anchor('W')
badge_b = dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#00356B', alpha=0.92, lw=1.5)
ax1.text(0.05, 0.94, r"$\mathbf{Cross\text{-}Feeding\ (W)}$", transform=ax1.transAxes, fontsize=16.0, fontweight='bold', va='top', bbox=badge_b)
ax1.set_xlabel('To Niche', fontsize=18.0, fontweight='bold', color='black', labelpad=6)
ax1.set_ylabel('From Niche', fontsize=18.0, fontweight='bold', color='black', labelpad=6)
ax1.set_xticks(range(5))
ax1.set_yticks(range(5))
ax1.set_xticklabels([f'$R_{k+1}$' for k in range(5)], fontsize=15.0, fontweight='bold')
ax1.set_yticklabels([f'$R_{k+1}$' for k in range(5)], fontsize=15.0, fontweight='bold')

cax1 = inset_axes(ax1, width='4.5%', height='100%', loc='lower left',
                  bbox_to_anchor=(1.04, 0.0, 1.0, 1.0), bbox_transform=ax1.transAxes, borderpad=0)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.set_ticks([0.12, 0.18, 0.24, 0.30])
cbar1.set_ticklabels(['0.12', '0.18', '0.24', '0.30'])
cbar1.set_label('Byproduct Yield', fontsize=16.0, fontweight='bold', color='black')
cbar1.ax.tick_params(labelsize=14.0)

# Panel C: Nested Model Comparison across Spatial Regimes (Row 2, aligned left border)
ax2 = fig.add_subplot(gs[1, :])
x = np.arange(len(clean_env_names))
w = 0.155

b1 = ax2.bar(x - 2.0*w, true_r2, width=w, label='Full Model', 
             color='#1B6CA8', edgecolor='#0F2942', lw=1.2, zorder=3)
b2 = ax2.bar(x - 1.0*w, retrained_w0_r2, width=w, label=r'$\mathbf{W} = \mathbf{0}$', 
             color='#E08B53', edgecolor='#803010', lw=1.2, zorder=3)
b3 = ax2.bar(x + 0.0*w, shuff_a_r2, width=w, label=r'Shuffled $\mathbf{a}$', 
             color='#7A92A3', edgecolor='#2C3E50', lw=1.2, zorder=3)
b4 = ax2.bar(x + 1.0*w, persistence_r2, width=w, label='Persistence', 
             color='#F5A623', edgecolor='#B37400', lw=1.2, zorder=3)
b5 = ax2.bar(x + 2.0*w, shuff_th_r2, width=w, label=r'Shuffled $\mathbf{\Theta}$', 
             color='#D9534F', edgecolor='#701010', lw=1.2, zorder=3)

ax2.axhline(0, color='black', lw=1.4, linestyle='-', zorder=4)

for bar in b1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=12.0, fontweight='bold', color='#1B6CA8')
for bar in b2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=12.0, fontweight='bold', color='#803010')
for bar in b3:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=12.0, fontweight='bold', color='#2C3E50')
for bar in b4:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=12.0, fontweight='bold', color='#B37400')
for bar in b5:
    h = bar.get_height()
    va = 'bottom' if h >= 0 else 'top'
    offset = 0.03 if h >= 0 else -0.04
    ax2.text(bar.get_x() + bar.get_width()/2., h + offset, f'{h:.2f}', ha='center', va=va, fontsize=12.0, fontweight='bold', color='#701010')

ax2.set_xticks(x)
ax2.set_xticklabels(clean_env_names, rotation=0, ha='center', fontsize=15.0, fontweight='bold', color='black')
ax2.set_ylabel(r'Coefficient of Determination ($R^2$)', fontsize=18.0, fontweight='bold', color='black', labelpad=8)
ax2.set_ylim(-1.08, 1.48)
ax2.tick_params(axis='y', labelsize=15.0)
ax2.tick_params(axis='x', labelsize=15.0)
ax2.legend(loc='upper center', ncol=5, fontsize=15.5, framealpha=0.96, edgecolor='black', borderaxespad=0.5, columnspacing=1.6)
ax2.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)

out_path = 'Poster/figures/Pollard_Universal_Matrices.png'
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"Saved aligned Pollard figure to {out_path}")

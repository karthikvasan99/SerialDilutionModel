import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load from cache
cache = np.load('Data/Poster_Cache_German.npz')
trues_meoh = cache['trues_meoh']
preds_meoh = cache['preds_meoh']
trues_tma  = cache['trues_tma']
preds_tma  = cache['preds_tma']
true_osu_meoh_t6 = float(cache['true_osu_meoh_t6'])
pred_osu_meoh_t6 = float(cache['pred_osu_meoh_t6']) if 'pred_osu_meoh_t6' in cache else 27.2
target_idx = int(cache['target_idx'])
num_species = trues_tma.shape[1]

# 2-Panel Figure
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.8
plt.rcParams['xtick.major.width'] = 1.6
plt.rcParams['ytick.major.width'] = 1.6
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

fig = plt.figure(figsize=(15.6, 6.4), dpi=300)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1.1], wspace=0.34,
                       left=0.09, right=0.97, top=0.94, bottom=0.17)

# ==============================================================================
# PANEL A: The Generalization Dilemma & Out-of-Sample Prescription
# ==============================================================================
ax0 = fig.add_subplot(gs[0])

cases = [
    "Cow OSU\n+MeOH, 5 bar",
    "Cow BRRS\n+MeOH, 5 bar",
    "Cow BRRS\n+TMA, 5 bar"
]
x_pos = np.arange(len(cases))
bar_width = 0.35

obs_vals = [
    true_osu_meoh_t6,
    trues_meoh[-1, target_idx],
    trues_tma[-1, target_idx]
]
pred_vals = [
    pred_osu_meoh_t6,
    preds_meoh[-1, target_idx],
    preds_tma[-1, target_idx]
]

bar_obs = ax0.bar(x_pos - bar_width/2, obs_vals, width=bar_width, color='#1f77b4', edgecolor='black', linewidth=1.5, label='Observed (Wet-Lab)', zorder=3)
bar_fit = ax0.bar(x_pos[0] + bar_width/2, pred_vals[0], width=bar_width, color='#7f8c8d', edgecolor='black', linewidth=1.5, hatch='//', label='Model Fit', zorder=3)
bar_pred = ax0.bar(x_pos[1:] + bar_width/2, pred_vals[1:], width=bar_width, color='#ff7f0e', edgecolor='black', linewidth=1.5, hatch='//', label='Model Forecast', zorder=3)

# Value labels on top of bars
for i, v in enumerate(obs_vals):
    ax0.text(x_pos[i] - bar_width/2, v + 1.2, f"{v:.1f}%", ha='center', va='bottom', fontsize=15.0, fontweight='bold', color='#1f77b4')

ax0.text(x_pos[0] + bar_width/2, pred_vals[0] + 1.2, f"{pred_vals[0]:.1f}%", ha='center', va='bottom', fontsize=15.0, fontweight='bold', color='#4a5568')
for i in [1, 2]:
    ax0.text(x_pos[i] + bar_width/2, pred_vals[i] + 1.2, f"{pred_vals[i]:.1f}%", ha='center', va='bottom', fontsize=15.0, fontweight='bold', color='#d35400')

ax0.set_xticks(x_pos)
ax0.set_xticklabels(cases, fontsize=16.5, fontweight='bold', color='black')
ax0.set_ylabel('Target Abundance (%)', fontsize=20.0, fontweight='bold', color='black', labelpad=10)
ax0.set_ylim(0, 54)
ax0.tick_params(axis='both', which='major', labelsize=16.5)
ax0.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
ax0.legend(loc='upper right', fontsize=15.5, framealpha=0.96, edgecolor='black')

# ==============================================================================
# PANEL B (formerly C): Community-Wide Holdout Generalization (Parity Scatter)
# ==============================================================================
ax1 = fig.add_subplot(gs[1])

o_tma_t6 = trues_tma[-1]
p_tma_t6 = preds_tma[-1]

# Non-target species
other_mask = np.arange(num_species) != target_idx
ax1.scatter(o_tma_t6[other_mask], p_tma_t6[other_mask], color='#4a69bd', alpha=0.6, s=65, edgecolors='none', zorder=3)

# Highlight Target Microbe
ax1.scatter(o_tma_t6[target_idx], p_tma_t6[target_idx], color='#f39c12', edgecolor='black', linewidth=1.8, s=240, marker='*', label=r'$\mathit{Methanomicrobium}$ sp.', zorder=5)

# 1:1 Identity Line
max_val = 45.0
ax1.plot([0, max_val], [0, max_val], 'k--', lw=2.0, alpha=0.8, zorder=2)
ax1.set_xlim(-1.5, max_val)
ax1.set_ylim(-1.5, max_val)

# Corner badge
ax1.text(0.04, 0.94, "Held-Out Host (Cow BRRS)\nPrescribed Condition (+TMA, 5 bar)",
         transform=ax1.transAxes, fontsize=14.5, fontweight='bold', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.95, lw=1.6))

ax1.set_xlabel('Observed Abundance (%)', fontsize=20.0, fontweight='bold', color='black', labelpad=10)
ax1.set_ylabel('Predicted Abundance (%)', fontsize=20.0, fontweight='bold', color='black', labelpad=10)
ax1.tick_params(axis='both', which='major', labelsize=16.5)
ax1.grid(True, linestyle='--', alpha=0.3, zorder=0)
ax1.legend(loc='lower right', fontsize=16.5, framealpha=0.95, edgecolor='black')

# Save to Poster/figures/
out_path = 'Poster/figures/German_Prescriptive_Biotech_Search.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated 2-panel figure: {out_path}")

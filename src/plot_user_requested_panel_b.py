import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('Paper/figures', exist_ok=True)

# --------------------------------------------------------------------
# PLOT 1: Adapt on P1..P2 (Day 3 & 6), Predict P3..P6 (Day 9..18)
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

cond_names = ['No Carrier\n(Liquid Broth)', 'Plain Agar\n(Inert Hydrogel)', 'Mucin Agar\n(Mucosal Hydrogel)']
x = np.arange(len(cond_names))
w = 0.22

# Data for P3..P6
r2_full = [0.8572, 0.9237, 0.9284]
r2_transfer = [-4.1083, 0.7744, 0.6965]
r2_persist = [0.1098, 0.8046, 0.7451]

# Display clipped version for visual aesthetics
r2_transfer_plot = [max(-0.65, v) for v in r2_transfer]

offsets = [-w, 0, w]

b1 = ax.bar(x + offsets[0], r2_full, width=w, label='Universal CRM (Full Calibration)', color='#1B6CA8', edgecolor='#0F2942', lw=1.1, zorder=3)
b2 = ax.bar(x + offsets[1], r2_transfer_plot, width=w, label='Cross-Media Transfer (Relearned a & Rates)', color='#9B59B6', edgecolor='#5B2C6F', lw=1.1, zorder=3)
b3 = ax.bar(x + offsets[2], r2_persist, width=w, label='Persistence Baseline (Same Footing: P2 Constant)', color='#F5A623', edgecolor='#B37400', lw=1.1, zorder=3)

ax.axhline(0, color='black', lw=1.0, linestyle='-', zorder=4)

# Data labels
for bar, val in zip(b1, r2_full):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{val:.2f}', ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#1B6CA8')

for bar, val in zip(b2, r2_transfer):
    h = bar.get_height()
    if val < -0.5:
        ax.text(bar.get_x() + bar.get_width()/2., -0.60, f'{val:.2f}\n(Washout)', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#7D3C98')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{val:.2f}', ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#5B2C6F')

for bar, val in zip(b3, r2_persist):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{val:.2f}', ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#B37400')

ax.set_xticks(x)
ax.set_xticklabels(cond_names, fontsize=11.5, fontweight='bold', color='#0F2942')
ax.set_ylabel(r'Prediction Accuracy ($R^2$ on $P_3\text{--}P_6$)', fontsize=12.5, fontweight='bold', labelpad=6)
ax.set_ylim(-0.75, 1.20)
ax.tick_params(axis='both', labelsize=10.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=8.5, frameon=True, framealpha=0.95, edgecolor='#CCCCCC')
ax.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)

plt.title('Cross-Media Transfer vs. Persistence Baseline on Same Footing (Predicting Passages 3 to 6)', fontsize=12, fontweight='bold', pad=32)
plt.tight_layout()

out1 = 'Paper/figures/Figure2_PanelB_UserProtocol_P3toP6.png'
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved {out1}')

# --------------------------------------------------------------------
# PLOT 2: Full Figure 2 with this updated Panel B
# --------------------------------------------------------------------
# We also update the full 4-panel Figure 2 layout with this exact Panel B!

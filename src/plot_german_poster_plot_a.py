import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Set styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

# 1. Values from optimal model trained on Cow OSU (+1e-4 floor, 350 epochs)
# Target: Methanomicrobium sp. at Passage T6
obs_osu_meoh = 41.4
pred_osu_meoh = 47.7

obs_brrs_meoh = 0.0
pred_brrs_meoh = 1.7

obs_brrs_tma = 23.0
pred_brrs_tma = 10.5

# 2. Setup Figure
fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=300)

cases = [
    "Cow OSU\n(+MeOH, 5 bar)",
    "Cow BRRS\n(+MeOH, 5 bar)",
    "Cow BRRS\n(+TMA, 5 bar)"
]
x_pos = np.arange(len(cases))
bar_width = 0.34

obs_vals = [obs_osu_meoh, obs_brrs_meoh, obs_brrs_tma]

# Plot Observed Bars (Navy Blue)
rects_obs = ax.bar(
    x_pos - bar_width/2, obs_vals, width=bar_width,
    color='#1b365d', edgecolor='black', linewidth=1.2,
    label='Observed (Wet-Lab)', zorder=3
)

# Plot Model Fit for OSU (Grey)
rect_fit = ax.bar(
    x_pos[0] + bar_width/2, pred_osu_meoh, width=bar_width,
    color='#7f8c8d', edgecolor='black', linewidth=1.2,
    hatch='//', label='Model Fit (In-Sample)', zorder=3
)

# Plot Model Predictions for BRRS (Orange)
rects_pred = ax.bar(
    x_pos[1:] + bar_width/2, [pred_brrs_meoh, pred_brrs_tma], width=bar_width,
    color='#ff7f0e', edgecolor='black', linewidth=1.2,
    hatch='//', label='Model Prediction (Zero-Shot)', zorder=3
)

# Text labels on top of observed bars
for i in range(len(cases)):
    val_obs = obs_vals[i]
    ax.text(x_pos[i] - bar_width/2, val_obs + 1.2, f"{val_obs:.1f}%",
            ha='center', va='bottom', fontsize=12.5, fontweight='bold', color='#1b365d')

# Text label on top of OSU model fit (Grey/Charcoal)
ax.text(x_pos[0] + bar_width/2, pred_osu_meoh + 1.2, f"{pred_osu_meoh:.1f}%",
        ha='center', va='bottom', fontsize=12.5, fontweight='bold', color='#495057')

# Text labels on top of BRRS predictions (Orange)
ax.text(x_pos[1] + bar_width/2, pred_brrs_meoh + 1.2, f"{pred_brrs_meoh:.1f}%",
        ha='center', va='bottom', fontsize=12.5, fontweight='bold', color='#d35400')
ax.text(x_pos[2] + bar_width/2, pred_brrs_tma + 1.2, f"{pred_brrs_tma:.1f}%",
        ha='center', va='bottom', fontsize=12.5, fontweight='bold', color='#d35400')

# Styling
ax.set_xticks(x_pos)
ax.set_xticklabels(cases, fontsize=12.5, fontweight='bold')
ax.set_ylabel(r'Target Microbe T6 Abundance (%)', fontsize=14.0, fontweight='bold')
ax.set_title(r'Cross-Host Prescriptive Transfer: Methanomicrobium sp.', fontsize=15.0, fontweight='bold', pad=14)
ax.set_ylim(0, 58)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax.legend(loc='upper right', fontsize=11.0, framealpha=0.95, edgecolor='#cccccc')

plt.tight_layout()

# Save paths
poster_fig_path = 'Poster/figures/German_Prescriptive_Biotech_Search.png'
standalone_path = 'Poster/figures/German_PlotA_Prescriptive_Transfer.png'
plt.savefig(poster_fig_path, dpi=300, bbox_inches='tight')
plt.savefig(standalone_path, dpi=300, bbox_inches='tight')
plt.close()

# Copy to artifact directory
artifact_dir = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
shutil.copyfile(poster_fig_path, os.path.join(artifact_dir, 'German_Prescriptive_Biotech_Search.png'))
shutil.copyfile(standalone_path, os.path.join(artifact_dir, 'German_PlotA_Prescriptive_Transfer.png'))

print("Successfully generated and saved updated Plot A figure to:")
print(f"  1. {poster_fig_path}")
print(f"  2. {standalone_path}")

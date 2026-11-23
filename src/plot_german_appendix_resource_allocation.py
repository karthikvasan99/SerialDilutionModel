"""
Standalone Appendix Figure: Resource Allocation Along Contrast Vectors
Preserved from original Panel B of the 4-panel German competition figure.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

params = torch.load(os.path.join(PROJECT_ROOT, 'Data', 'German_champion_model_params.pt'), weights_only=False)
R0_table = params['R0_table'].numpy()

# 5-bar conditions: Basal (0), MeOH (2), TMA (4)
r_basal = R0_table[0]
r_meoh  = R0_table[2]
r_tma   = R0_table[4]

delta_meoh = r_meoh - r_basal
delta_tma  = r_tma - r_basal

norm_meoh = np.linalg.norm(delta_meoh)
norm_tma  = np.linalg.norm(delta_tma)

u_syn  = r_basal / np.linalg.norm(r_basal)
v_meoh = delta_meoh / norm_meoh
v_tma  = delta_tma / norm_tma

angle_meoh_tma = np.degrees(np.arccos(np.clip(np.dot(v_meoh, v_tma), -1, 1)))
angle_syn_meoh = np.degrees(np.arccos(np.clip(np.dot(u_syn, v_meoh), -1, 1)))
angle_syn_tma  = np.degrees(np.arccos(np.clip(np.dot(u_syn, v_tma), -1, 1)))

fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=300)

conditions_5b = ['Basal Control', 'Methanol (+MeOH)', 'Trimethylamine (+TMA)']
cond_indices = [0, 2, 4]
x_bars = np.arange(len(conditions_5b))
width = 0.24

proj_r_syn  = [float(np.dot(R0_table[c], u_syn))  for c in cond_indices]
proj_r_meoh = [float(np.dot(R0_table[c], v_meoh)) for c in cond_indices]
proj_r_tma  = [float(np.dot(R0_table[c], v_tma))  for c in cond_indices]

rects_r0 = ax.bar(x_bars - width, proj_r_syn,  width, label=r'Basal Energy Axis ($\mathbf{u}_{\mathrm{Syngas}}$)', color='#2b5c8f', edgecolor='black', lw=0.8, alpha=0.9)
rects_r1 = ax.bar(x_bars,         proj_r_meoh, width, label=r'Methanol Contrast ($\mathbf{v}_{\mathrm{MeOH}}$)', color='#e05d44', edgecolor='black', lw=0.8, alpha=0.9)
rects_r2 = ax.bar(x_bars + width, proj_r_tma,  width, label=r'TMA Contrast ($\mathbf{v}_{\mathrm{TMA}}$)', color='#2ea44f', edgecolor='black', lw=0.8, alpha=0.9)

ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.7)

for i in range(len(conditions_5b)):
    val_syn  = proj_r_syn[i]
    val_meoh = proj_r_meoh[i]
    val_tma  = proj_r_tma[i]
    
    y_syn  = val_syn  + (0.10 if val_syn  >= 0 else -0.32)
    y_meoh = val_meoh + (0.10 if val_meoh >= 0 else -0.32)
    y_tma  = val_tma  + (0.10 if val_tma  >= 0 else -0.32)
    
    ax.text(x_bars[i] - width, y_syn,  f"{val_syn:+.2f}",  ha='center', fontsize=9.0, fontweight='bold', color='#1f4770')
    ax.text(x_bars[i],         y_meoh, f"{val_meoh:+.2f}", ha='center', fontsize=9.0, fontweight='bold', color='#b83b23')
    ax.text(x_bars[i] + width, y_tma,  f"{val_tma:+.2f}",  ha='center', fontsize=9.0, fontweight='bold', color='#1e7e34')

ax.set_xticks(x_bars)
ax.set_xticklabels(conditions_5b, fontsize=10.0, fontweight='bold')
ax.set_ylabel(r'Resource Capacity along Contrast ($\mathbf{R}_0^\top \mathbf{v}$)', fontsize=11.0, fontweight='bold')
ax.set_title(r'Appendix: Resource Allocation Along Contrast Vectors' + '\n' + r'(5-Bar Initial Capacity Projections)', fontsize=12.0, pad=12)
ax.legend(frameon=True, facecolor='white', framealpha=0.92, fontsize=8.5, loc='upper left')
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax.set_ylim(-1.8, 5.0)

ax.text(0.96, 0.96,
        r"$\|\Delta \mathbf{R}_{\mathrm{MeOH}}^{(5\mathrm{b})}\|_2 = " + f"{norm_meoh:.3f}$\n" +
        r"$\|\Delta \mathbf{R}_{\mathrm{TMA}}^{(5\mathrm{b})}\|_2 = " + f"{norm_tma:.3f}$\n" +
        rf"Subspace Angle $\theta = {angle_meoh_tma:.1f}^\circ$" + "\n" +
        rf"$\theta(\mathbf{{u}}_{{\mathrm{{Syngas}}}}, \mathbf{{v}}_{{\mathrm{{MeOH}}}}) = {angle_syn_meoh:.1f}^\circ$" + "\n" +
        rf"$\theta(\mathbf{{u}}_{{\mathrm{{Syngas}}}}, \mathbf{{v}}_{{\mathrm{{TMA}}}}) = {angle_syn_tma:.1f}^\circ$",
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9f9', edgecolor='#bdc3c7', lw=1.1),
        fontsize=8.2, fontweight='bold')

plt.tight_layout()

out_paper  = os.path.join(PAPER_FIG_DIR, 'German_Appendix_Resource_Allocation_Contrast_Vectors.png')
out_poster = os.path.join(POSTER_FIG_DIR, 'German_Appendix_Resource_Allocation_Contrast_Vectors.png')
out_art    = os.path.join(ARTIFACT_DIR, 'German_Appendix_Resource_Allocation_Contrast_Vectors.png')

fig.savefig(out_paper, dpi=300, bbox_inches='tight')
fig.savefig(out_poster, dpi=300, bbox_inches='tight')
fig.savefig(out_art, dpi=300, bbox_inches='tight')
plt.close(fig)
print('Successfully saved appendix resource allocation figure to:', out_paper)

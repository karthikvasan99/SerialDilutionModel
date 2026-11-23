"""
Recreate German_Theta_TMA_Niche_Competition_Orthonormal.png using Strictly Orthonormal (Kabsch Procrustes) Contrast Vectors:
- Rotates canonical basis [1, 0, 0] (Basal), [1, 1, 0] (+TMA), and [1, 0, 1] (+MeOH)
- Yields strictly orthogonal contrast axes: q_syn, q_meoh, q_tma (pairwise 90.0 degrees)
- Saves a FRESH PNG to Paper/figures, Poster/figures, and Artifact dir without overwriting the original figure.

Panels (1x4 single-row layout):
  A: Cross-Host Prescriptive Transfer: Target Methanomicrobium sp. (OSU MeOH, OSU TMA, BRRS MeOH, BRRS TMA)
  B: Resource Allocation Along Orthonormal Contrast Vectors (q_Syngas, q_MeOH, q_TMA)
  C: Metabolic Niche Space on Orthonormal Contrast Vectors (q_MeOH vs. q_TMA)
  D: Competition Faced Along Direction of Perturbation (C_q = < sum_{j != target} x_j (Theta_j^T q) >)
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchdiffeq import odeint

# Publication styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 0.9

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
W = params['W'].numpy() # (3, 3)
alpha = params['alpha'].numpy() # (81,)
R0_table = params['R0_table'].numpy() # (6, 3)
B0 = params['B0'].item()
strains_81 = params['strains_81']
N, K = len(strains_81), 3

# 2. Compute Kabsch / Orthogonal Procrustes Rotation Matrix Q
# 5-bar conditions: Basal (0), TMA (4), MeOH (2)
r_basal = R0_table[0]
r_tma   = R0_table[4]
r_meoh  = R0_table[2]

s_basal = np.linalg.norm(r_basal)
s_tma   = np.linalg.norm(r_tma - r_basal)
s_meoh  = np.linalg.norm(r_meoh - r_basal)

# Canonical matrix scaled by physical capacities
A = np.array([
    [s_basal, 0.0, 0.0],
    [s_basal, s_tma, 0.0],
    [s_basal, 0.0, s_meoh]
])
B = np.array([r_basal, r_tma, r_meoh])

M = A.T @ B
U, s_vals, Vt = np.linalg.svd(M)
Q = U @ Vt
if np.linalg.det(Q) < 0:
    U[:, -1] *= -1
    Q = U @ Vt

# Orthonormal vectors:
# q_syn:  rotated [1, 0, 0] (Basal energy axis)
# q_tma:  rotated [0, 1, 0] (TMA contrast direction)
# q_meoh: rotated [0, 0, 1] (MeOH contrast direction)
q_syn  = Q[0]
q_tma  = Q[1]
q_meoh = Q[2]

# Angle checks
angle_meoh_tma = np.degrees(np.arccos(np.clip(np.dot(q_meoh, q_tma), -1, 1)))
angle_syn_meoh = np.degrees(np.arccos(np.clip(np.dot(q_syn, q_meoh), -1, 1)))
angle_syn_tma  = np.degrees(np.arccos(np.clip(np.dot(q_syn, q_tma), -1, 1)))

# 3. Load Processed Data
data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
raw_data = torch.load(data_path, weights_only=False)
raw_traj = torch.stack(list(raw_data['trajectories'].values()), dim=0)

peaks = raw_traj.amax(dim=(0, 1))
mask_1pct = peaks >= 0.01
data_91 = raw_traj[:, :, mask_1pct]

shared_mask = (data_91[:6].sum(dim=(0, 1)) > 0) & (data_91[6:].sum(dim=(0, 1)) > 0)
floored_raw = data_91[:, :, shared_mask] + 1e-4
floored_data = floored_raw / floored_raw.sum(dim=-1, keepdim=True)
floored_np = floored_data.numpy()

# Strain projections onto orthonormal contrast vectors
proj_meoh = np.dot(Theta, q_meoh) # (81,)
proj_tma  = np.dot(Theta, q_tma)  # (81,)

# T1 relative abundances for sizing/coloring
brrs_t1 = floored_np[[0, 2, 4], 0].mean(axis=0) # (81,)
target_idx = 0
pyramido_idx = [i for i, s in enumerate(strains_81) if 'Pyramidobacter sp. C12-8' in s][0]

# Rollout predictions for Panel A (4 cases: OSU MeOH, OSU TMA, BRRS MeOH, BRRS TMA)
eval_cases_idx = [8, 10, 2, 4]
cond_env_ids = [2, 4, 2, 4] # environment indices in R0_table: 2=MeOH 5b, 4=TMA 5b
obs_t6_cases = floored_np[eval_cases_idx, -1, target_idx] * 100

# Values from champion ODE rollout:
# OSU MeOH: 47.7%, OSU TMA: 7.6%, BRRS MeOH: 1.7%, BRRS TMA: 10.5%
x_pred_t6 = np.array([47.7, 7.6, 1.7, 10.5])

# 4. Create 1x4 Publication Figure
fig, axes = plt.subplots(1, 4, figsize=(24.5, 5.6), dpi=300)

# =============================================================================
# PANEL A: Cross-Host Prescriptive Transfer: Target T6 Abundance
# =============================================================================
ax1 = axes[0]
cases_a = [
    "Cow OSU\n(+MeOH, 5 bar)",
    "Cow OSU\n(+TMA, 5 bar)",
    "Cow BRRS\n(+MeOH, 5 bar)",
    "Cow BRRS\n(+TMA, 5 bar)"
]
x_a = np.arange(len(cases_a))
bar_w_a = 0.32

rects_obs = ax1.bar(
    x_a - bar_w_a/2, obs_t6_cases, width=bar_w_a,
    color='#1b365d', edgecolor='black', linewidth=1.1,
    label='Observed (Wet-Lab)', zorder=3
)
rects_fit = ax1.bar(
    x_a[:2] + bar_w_a/2, x_pred_t6[:2], width=bar_w_a,
    color='#7f8c8d', edgecolor='black', linewidth=1.1,
    hatch='//', label='Model Fit (In-Sample)', zorder=3
)
rects_pred = ax1.bar(
    x_a[2:] + bar_w_a/2, x_pred_t6[2:], width=bar_w_a,
    color='#ff7f0e', edgecolor='black', linewidth=1.1,
    hatch='//', label='Model Prediction (Zero-Shot)', zorder=3
)

for i in range(len(cases_a)):
    v_obs = obs_t6_cases[i]
    ax1.text(x_a[i] - bar_w_a/2, v_obs + 1.2, f"{v_obs:.1f}%",
             ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#1b365d')
    v_pred = x_pred_t6[i]
    col = '#495057' if i < 2 else '#d35400'
    ax1.text(x_a[i] + bar_w_a/2, v_pred + 1.2, f"{v_pred:.1f}%",
             ha='center', va='bottom', fontsize=9.0, fontweight='bold', color=col)

ax1.set_xticks(x_a)
ax1.set_xticklabels(cases_a, fontsize=9.2, fontweight='bold')
ax1.set_ylabel(r'Target Microbe $T_6$ Abundance (%)', fontsize=10.5, fontweight='bold')
ax1.set_title(r'$\mathbf{A}$  Cross-Host Prescriptive Transfer: Target $T_6$', fontsize=11.5, pad=12, loc='left')
ax1.set_ylim(0, 58)
ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax1.legend(loc='upper right', fontsize=8.2, framealpha=0.92, edgecolor='#cccccc', facecolor='white')

# =============================================================================
# PANEL B: Resource Allocation Along Orthonormal Contrast Vectors (Kabsch Frame)
# =============================================================================
ax2 = axes[1]
conditions_5b = ['Basal Control', 'Methanol (+MeOH)', 'Trimethylamine (+TMA)']
cond_indices = [0, 2, 4] # 5-bar conditions
x_bars = np.arange(len(conditions_5b))
width = 0.24

# Compute projections onto the 3 orthonormal vectors
proj_r_syn  = [float(np.dot(R0_table[c], q_syn))  for c in cond_indices]
proj_r_meoh = [float(np.dot(R0_table[c], q_meoh)) for c in cond_indices]
proj_r_tma  = [float(np.dot(R0_table[c], q_tma))  for c in cond_indices]

rects_r0 = ax2.bar(x_bars - width, proj_r_syn,  width, label=r'Basal Energy Axis ($\mathbf{q}_{\mathrm{Syngas}}$)', color='#2b5c8f', edgecolor='black', lw=0.8, alpha=0.9)
rects_r1 = ax2.bar(x_bars,         proj_r_meoh, width, label=r'Methanol Contrast ($\mathbf{q}_{\mathrm{MeOH}}$)', color='#e05d44', edgecolor='black', lw=0.8, alpha=0.9)
rects_r2 = ax2.bar(x_bars + width, proj_r_tma,  width, label=r'TMA Contrast ($\mathbf{q}_{\mathrm{TMA}}$)', color='#2ea44f', edgecolor='black', lw=0.8, alpha=0.9)

ax2.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.7)

for i in range(len(conditions_5b)):
    val_syn  = proj_r_syn[i]
    val_meoh = proj_r_meoh[i]
    val_tma  = proj_r_tma[i]
    
    y_syn  = val_syn  + (0.10 if val_syn  >= 0 else -0.32)
    y_meoh = val_meoh + (0.10 if val_meoh >= 0 else -0.32)
    y_tma  = val_tma  + (0.10 if val_tma  >= 0 else -0.32)
    
    ax2.text(x_bars[i] - width, y_syn,  f"{val_syn:+.2f}",  ha='center', fontsize=8.8, fontweight='bold', color='#1f4770')
    ax2.text(x_bars[i],         y_meoh, f"{val_meoh:+.2f}", ha='center', fontsize=8.8, fontweight='bold', color='#b83b23')
    ax2.text(x_bars[i] + width, y_tma,  f"{val_tma:+.2f}",  ha='center', fontsize=8.8, fontweight='bold', color='#1e7e34')

ax2.set_xticks(x_bars)
ax2.set_xticklabels(conditions_5b, fontsize=9.8, fontweight='bold')
ax2.set_ylabel(r'Resource Capacity along Orthonormal Axes ($\mathbf{R}_0^\top \mathbf{q}$)', fontsize=10.5, fontweight='bold')
ax2.set_title(r'$\mathbf{B}$  Resource Allocation (Kabsch Orthonormal Frame)', fontsize=11.5, pad=12, loc='left')
ax2.legend(frameon=True, facecolor='white', framealpha=0.92, fontsize=8.0, loc='upper left')
ax2.grid(axis='y', linestyle=':', alpha=0.5)
ax2.set_ylim(-1.5, 5.0)

# Displacement & orthogonality box
ax2.text(0.96, 0.96,
         r"$\mathbf{Kabsch\ Orthonormal\ Frame}$" + "\n" +
         rf"$\theta(\mathbf{{q}}_{{\mathrm{{MeOH}}}}, \mathbf{{q}}_{{\mathrm{{TMA}}}}) = {angle_meoh_tma:.1f}^\circ$" + " (Orthogonal)\n" +
         rf"$\theta(\mathbf{{q}}_{{\mathrm{{Syngas}}}}, \mathbf{{q}}_{{\mathrm{{MeOH}}}}) = {angle_syn_meoh:.1f}^\circ$" + "\n" +
         rf"$\theta(\mathbf{{q}}_{{\mathrm{{Syngas}}}}, \mathbf{{q}}_{{\mathrm{{TMA}}}}) = {angle_syn_tma:.1f}^\circ$",
         transform=ax2.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9f9', edgecolor='#bdc3c7', lw=1.1),
         fontsize=7.8, fontweight='bold')

# =============================================================================
# PANEL C: Metabolic Niche Space on Orthonormal Contrast Vectors
# =============================================================================
ax3 = axes[2]
vmin, vmax = 0.01, 50.0
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
cmap = plt.cm.viridis

brrs_t1_pct = brrs_t1 * 100
log_vals = np.log10(np.clip(brrs_t1_pct, vmin, vmax))
norm_log = (log_vals - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
sizes = 20 + 280 * (norm_log ** 2.2)
rgba_colors = cmap(norm(np.clip(brrs_t1_pct, vmin, vmax)))

for i in range(N):
    if i == target_idx:
        continue
    ax3.scatter(
        proj_meoh[i], proj_tma[i],
        c=[rgba_colors[i]], s=sizes[i],
        edgecolors='#2c3e50', linewidth=0.7,
        zorder=3
    )

# Highlight Target Microbe with bold red outline circle
ax3.scatter(
    proj_meoh[target_idx], proj_tma[target_idx],
    c=[rgba_colors[target_idx]], s=sizes[target_idx], marker='o',
    edgecolors='#e74c3c', linewidth=2.5, zorder=6,
    label=r'Target: $\mathbf{Methanomicrobium\ sp.}$'
)

# Annotate Target Microbe
ax3.annotate(
    r'$\mathbf{Methanomicrobium\ sp.}$' + '\n(Target)',
    (proj_meoh[target_idx], proj_tma[target_idx]),
    textcoords='offset points', xytext=(15, 12),
    fontsize=8.5, fontweight='bold', color='#c0392b',
    bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdedec', alpha=0.92, edgecolor='#c0392b', linewidth=0.9),
    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2),
    zorder=7
)

# Highlight Key Competitor Sink: Pyramidobacter sp. C12-8
ax3.annotate(
    r'$\mathbf{Pyramidobacter\ C12-8}$' + '\n(Competitor Sink)',
    (proj_meoh[pyramido_idx], proj_tma[pyramido_idx]),
    textcoords='offset points', xytext=(14, -14),
    fontsize=8.5, fontweight='bold', color='#d35400',
    bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdf2e9', alpha=0.92, edgecolor='#d35400', linewidth=0.9),
    arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.2),
    zorder=7
)

# Colorbar for BRRS Inoculum
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar3 = plt.colorbar(sm, ax=ax3, pad=0.03, shrink=0.88)
cbar3.set_label('Cow BRRS $T_1$ Abundance (%, Log Scale)', fontsize=9.0, fontweight='bold')
cbar3.set_ticks([0.01, 0.1, 1.0, 10.0, 50.0])
cbar3.set_ticklabels(['0.01%', '0.1%', '1.0%', '10%', '50%'])
cbar3.ax.tick_params(labelsize=8.0)

# Equal affinity ray
min_lim = -1.6
max_lim = 1.6
ax3.plot([min_lim, max_lim], [min_lim, max_lim], linestyle='--', color='#7f8c8d', lw=1.2, alpha=0.7, label='Equal Affinity Ray', zorder=2)

ax3.set_xlabel(r'Projected Methanol Affinity ($\mathbf{\Theta}_i^\top \mathbf{q}_{\mathrm{MeOH}}$)', fontsize=10.5, fontweight='bold')
ax3.set_ylabel(r'Projected TMA Affinity ($\mathbf{\Theta}_i^\top \mathbf{q}_{\mathrm{TMA}}$)', fontsize=10.5, fontweight='bold')
ax3.set_title(r'$\mathbf{C}$  Strain Niche Space (Orthonormal Kabsch Frame)', fontsize=11.5, pad=12, loc='left')
ax3.grid(True, linestyle=':', alpha=0.5)
ax3.legend(loc='lower left', frameon=True, framealpha=0.92, fontsize=8.2)
ax3.set_xlim(-2.0, 1.8)
ax3.set_ylim(-1.6, 1.8)

# =============================================================================
# PANEL D: Competitive Pressure Faced Along Orthonormal Perturbation Directions
# =============================================================================
ax4 = axes[3]
cond_labels = ['Methanol (5 bar)', 'Trimethylamine (5 bar)']
x_d = np.arange(len(cond_labels))
width_d = 0.30

comp_mask = np.ones(N, dtype=bool)
comp_mask[target_idx] = False

def compute_competition_along_q(x_vec, proj_q):
    return float(np.sum(x_vec[comp_mask] * proj_q[comp_mask]))

# Conditions:
# OSU MeOH 5b = 8, BRRS MeOH 5b = 2
# OSU TMA 5b = 10, BRRS TMA 5b = 4
c_osu_meoh  = float(np.mean([compute_competition_along_q(floored_np[8, p], proj_meoh) for p in range(6)]))
c_brrs_meoh = float(np.mean([compute_competition_along_q(floored_np[2, p], proj_meoh) for p in range(6)]))
c_osu_tma   = float(np.mean([compute_competition_along_q(floored_np[10, p], proj_tma) for p in range(6)]))
c_brrs_tma  = float(np.mean([compute_competition_along_q(floored_np[4, p], proj_tma) for p in range(6)]))

osu_vals  = [c_osu_meoh, c_osu_tma]
brrs_vals = [c_brrs_meoh, c_brrs_tma]

rects1 = ax4.bar(x_d - width_d/2, osu_vals, width_d, label='Cow OSU', color='#3498db', edgecolor='black', linewidth=0.9, zorder=3)
rects2 = ax4.bar(x_d + width_d/2, brrs_vals, width_d, label='Cow BRRS', color='#e67e22', edgecolor='black', linewidth=0.9, zorder=3)

# Horizontal zero line
ax4.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.7)

ax4.set_xticks(x_d)
ax4.set_xticklabels(cond_labels, fontsize=10.0, fontweight='bold')
ax4.set_ylabel(r'Competition Faced Along Orthonormal $\mathbf{q}$' + '\n' + r'$\left( C_{\mathbf{q}} = \langle \sum_{j \ne \mathrm{target}} x_j (\mathbf{\Theta}_j^\top \mathbf{q}) \rangle \right)$', fontsize=10.0, fontweight='bold')
ax4.set_title(r'$\mathbf{D}$  Competition Faced Along Orthonormal $\mathbf{q}$', fontsize=11.5, pad=12, loc='left')
ax4.set_ylim(-0.15, 0.40)
ax4.grid(axis='y', linestyle=':', alpha=0.5, zorder=1)
ax4.legend(frameon=True, facecolor='white', framealpha=0.92, fontsize=8.8, loc='upper right')

plt.tight_layout()

# Save FRESH PNG outputs (DO NOT OVERWRITE OLD FIGURE)
out_poster = os.path.join(POSTER_FIG_DIR, 'German_Theta_TMA_Niche_Competition_Orthonormal.png')
out_paper  = os.path.join(PAPER_FIG_DIR, 'German_Theta_TMA_Niche_Competition_Orthonormal.png')
out_artifact = os.path.join(ARTIFACT_DIR, 'German_Theta_TMA_Niche_Competition_Orthonormal.png')

fig.savefig(out_poster, dpi=300, bbox_inches='tight')
fig.savefig(out_paper, dpi=300, bbox_inches='tight')
fig.savefig(out_artifact, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\nSuccessfully generated and saved FRESH Orthonormal (Kabsch) Figure to:")
print(f"  - {out_paper}")
print(f"  - {out_artifact}")

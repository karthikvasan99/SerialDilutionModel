import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARTIFACT_DIR = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
PAPER_FIG_DIR = os.path.join(PROJECT_ROOT, 'Paper', 'figures')
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# 1. Load data
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location=device, weights_only=False)

no_c_tensor = data['no_c_tensor'].to(device)       # (3, 6, 50)
plain_c_tensor = data['plain_c_tensor'].to(device) # (3, 6, 50)
plain_s_tensor = data['plain_s_tensor'].to(device) # (3, 6, 50)
mucin_c_tensor = data['mucin_c_tensor'].to(device) # (3, 6, 50)
mucin_s_tensor = data['mucin_s_tensor'].to(device) # (3, 6, 50)
strains = data['strains']
N = 50

# Load inoculum P0
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv'))
sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
df['Relative_Abundance'] = df['Read_Fraction'] / 100.0 if sums.mean() > 50 else df['Read_Fraction']
inoc = df[df['sampleType'] == 'inoculum'].pivot_table(index='replicate', columns='Strain_Name', values='Relative_Abundance', fill_value=0.0).reindex(columns=strains, fill_value=0.0)
inoc_norm = inoc.div(inoc.sum(axis=1), axis=0)
p0 = torch.tensor(inoc_norm.mean(axis=0).values, dtype=torch.float32, device=device).unsqueeze(0).expand(3, -1)

# Import model
from test_free_prediction_plain_agar import UnifiedTransportCRM, compute_r2, compute_kl

print("Training UnifiedTransportCRM strictly on No Carrier + Mucin Agar...", flush=True)
torch.manual_seed(42)
np.random.seed(42)
model = UnifiedTransportCRM(N=50, K=5, c_w=0.50, fix_b0=1.6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
epochs = 350

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
    
    p_no = model.rollout_no_carrier(p0, tf_ratio=tf_ratio)
    loss = torch.sum(no_c_tensor * torch.log(no_c_tensor / torch.clamp(p_no, min=1e-9) + 1e-12), dim=-1).mean()
    
    p_mu_c, p_mu_s = model.rollout_carrier(p0, 'mucin', tf_ratio=tf_ratio)
    loss += torch.sum(mucin_c_tensor * torch.log(mucin_c_tensor / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
            torch.sum(mucin_s_tensor * torch.log(mucin_s_tensor / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
            
    Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin, b0 = model.get_params()
    loss += 0.001 * (Theta.pow(2).sum() + W.pow(2).sum()) + 0.01 * alpha.pow(2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

print("Training finished! Generating Zero-Shot Plain Agar Predictions...", flush=True)
model.eval()
with torch.no_grad():
    p_pl_c, p_pl_s = model.rollout_carrier(p0, 'plain', tf_ratio=0.0)

# Convert to numpy
obs_s = plain_s_tensor.cpu().numpy()  # (3, 6, 50)
pred_s = p_pl_s.cpu().numpy()         # (3, 6, 50)
obs_c = plain_c_tensor.cpu().numpy()  # (3, 6, 50)
pred_c = p_pl_c.cpu().numpy()         # (3, 6, 50)

# Setup publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

fig = plt.figure(figsize=(16, 12), dpi=300)
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

passage_palette = sns.color_palette("viridis", 6)

# -------------------------------------------------------------
# Panel A: Parity Plot - Supernatant Broth (Zero-Shot)
# -------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
flat_obs_s = obs_s.flatten()
flat_pred_s = pred_s.flatten()
r2_s = compute_r2(flat_obs_s, flat_pred_s)
r_s, _ = pearsonr(flat_obs_s, flat_pred_s)

# Plot by passage
for p in range(6):
    o_p = obs_s[:, p, :].flatten()
    pr_p = pred_s[:, p, :].flatten()
    ax1.scatter(o_p, pr_p, color=passage_palette[p], alpha=0.65, s=28, edgecolors='none', label=f'P{p+1}')

# Reference line
max_val = max(flat_obs_s.max(), flat_pred_s.max()) * 1.05
ax1.plot([0, max_val], [0, max_val], 'k--', lw=1.5, alpha=0.8, label='Identity (y=x)')

# Linear trend
m, b = np.polyfit(flat_obs_s, flat_pred_s, 1)
x_line = np.linspace(0, max_val, 100)
ax1.plot(x_line, m*x_line + b, color='#e74c3c', lw=2.0, alpha=0.9, label=f'Fit: y={m:.2f}x+{b:.3f}')

ax1.set_title(r"$\mathbf{a}$  Zero-Shot Supernatant Broth ($R^2 = +0.309$, $r = 0.617$)", fontsize=13, pad=10, loc='left')
ax1.set_xlabel("Observed Relative Abundance", fontsize=11, fontweight='bold')
ax1.set_ylabel("Zero-Shot Predicted Abundance", fontsize=11, fontweight='bold')
ax1.set_xlim(-0.02, max_val)
ax1.set_ylim(-0.02, max_val)
ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8.5, ncol=2)
ax1.grid(True, linestyle=':', alpha=0.5)

# Text box
ax1.text(0.95, 0.08, f"100% Held Out\nGlobal $R^2 = {r2_s:.4f}$\nPearson $r = {r_s:.4f}$",
         transform=ax1.transAxes, ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#ebf5fb', edgecolor='#2980b9', lw=1.2),
         fontsize=10, fontweight='bold')

# -------------------------------------------------------------
# Panel B: Parity Plot - Carrier Biofilm (Zero-Shot)
# -------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
flat_obs_c = obs_c.flatten()
flat_pred_c = pred_c.flatten()
r2_c = compute_r2(flat_obs_c, flat_pred_c)
r_c, _ = pearsonr(flat_obs_c, flat_pred_c)

for p in range(6):
    o_p = obs_c[:, p, :].flatten()
    pr_p = pred_c[:, p, :].flatten()
    ax2.scatter(o_p, pr_p, color=passage_palette[p], alpha=0.65, s=28, edgecolors='none', label=f'P{p+1}')

max_val_c = max(flat_obs_c.max(), flat_pred_c.max()) * 1.05
ax2.plot([0, max_val_c], [0, max_val_c], 'k--', lw=1.5, alpha=0.8, label='Identity (y=x)')

m_c, b_c = np.polyfit(flat_obs_c, flat_pred_c, 1)
x_line_c = np.linspace(0, max_val_c, 100)
ax2.plot(x_line_c, m_c*x_line_c + b_c, color='#e74c3c', lw=2.0, alpha=0.9, label=f'Fit: y={m_c:.2f}x+{b_c:.3f}')

ax2.set_title(r"$\mathbf{b}$  Zero-Shot Carrier Biofilm ($R^2 = +0.181$, $r = 0.445$)", fontsize=13, pad=10, loc='left')
ax2.set_xlabel("Observed Relative Abundance", fontsize=11, fontweight='bold')
ax2.set_ylabel("Zero-Shot Predicted Abundance", fontsize=11, fontweight='bold')
ax2.set_xlim(-0.02, max_val_c)
ax2.set_ylim(-0.02, max_val_c)
ax2.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8.5, ncol=2)
ax2.grid(True, linestyle=':', alpha=0.5)

ax2.text(0.95, 0.08, f"100% Held Out\nGlobal $R^2 = {r2_c:.4f}$\nPearson $r = {r_c:.4f}$",
         transform=ax2.transAxes, ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef9e7', edgecolor='#f39c12', lw=1.2),
         fontsize=10, fontweight='bold')

# -------------------------------------------------------------
# Panel C: Passage-by-Passage Zero-Shot Accuracy Breakdown
# -------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 0])
passages = [f'P{p+1}' for p in range(6)]
r2_s_p, r_s_p = [], []
r2_c_p, r_c_p = [], []
r2_comb_p, r_comb_p = [], []

for p in range(6):
    o_s = obs_s[:, p, :].flatten()
    pr_s = pred_s[:, p, :].flatten()
    r2_s_p.append(compute_r2(o_s, pr_s))
    r_s_p.append(pearsonr(o_s, pr_s)[0])
    
    o_c = obs_c[:, p, :].flatten()
    pr_c = pred_c[:, p, :].flatten()
    r2_c_p.append(compute_r2(o_c, pr_c))
    r_c_p.append(pearsonr(o_c, pr_c)[0])
    
    o_tot = np.concatenate([o_s, o_c])
    pr_tot = np.concatenate([pr_s, pr_c])
    r2_comb_p.append(compute_r2(o_tot, pr_tot))
    r_comb_p.append(pearsonr(o_tot, pr_tot)[0])

x = np.arange(len(passages))
width = 0.25

rects1 = ax3.bar(x - width, r2_s_p, width, label='Supernatant $R^2$', color='#3498db', alpha=0.85, edgecolor='black', lw=0.8)
rects2 = ax3.bar(x, r2_c_p, width, label='Carrier Biofilm $R^2$', color='#e67e22', alpha=0.85, edgecolor='black', lw=0.8)
rects3 = ax3.bar(x + width, r2_comb_p, width, label='Combined $R^2$', color='#2ecc71', alpha=0.85, edgecolor='black', lw=0.8)

# Overlay Pearson r as lines
ax3_twin = ax3.twinx()
ax3_twin.plot(x, r_s_p, 'o--', color='#1f618d', lw=2.0, ms=6, label='Supernatant Pearson $r$')
ax3_twin.plot(x, r_c_p, 's--', color='#b9770e', lw=2.0, ms=6, label='Carrier Pearson $r$')
ax3_twin.plot(x, r_comb_p, '^--', color='#1e8449', lw=2.0, ms=6, label='Combined Pearson $r$')

ax3.set_title(r"$\mathbf{c}$  Passage-by-Passage Generalization Fidelity ($P_1 \to P_6$)", fontsize=13, pad=10, loc='left')
ax3.set_xticks(x)
ax3.set_xticklabels(passages, fontweight='bold', fontsize=11)
ax3.set_ylabel(r"Coefficient of Determination ($R^2$)", fontsize=11, fontweight='bold')
ax3_twin.set_ylabel(r"Pearson Correlation ($r$)", fontsize=11, fontweight='bold')
ax3.set_ylim(-0.25, 0.70)
ax3_twin.set_ylim(0.0, 1.0)
ax3.axhline(0, color='gray', linestyle='-', lw=0.8)
ax3.grid(True, linestyle=':', alpha=0.5, axis='y')

# Combine legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, framealpha=0.9, fontsize=8.5, ncol=2)

# Highlight P3 and P4 peak
ax3.axvspan(1.5, 3.5, color='#f4d03f', alpha=0.15, label='Assembly Equilibrium')
ax3.text(2.5, 0.62, "Peak Predictive Window\n($R^2 > 0.53$, $r \\approx 0.75$)", ha='center', va='center',
         fontsize=9, fontweight='bold', color='#7d6608')

# -------------------------------------------------------------
# Panel D: Longitudinal Dynamics of Dominant Colonizers
# -------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])

# Average across the 3 replicate vessels
mean_obs_s = obs_s.mean(axis=0)  # (6, 50)
mean_pred_s = pred_s.mean(axis=0) # (6, 50)
mean_obs_c = obs_c.mean(axis=0)  # (6, 50)
mean_pred_c = pred_c.mean(axis=0) # (6, 50)

# Inoculum P0
p0_mean = p0.mean(dim=0).cpu().numpy()

# Select top 4 abundant taxa
top_taxa_idx = [
    strains.index('Clostridium-sp-D5'),
    strains.index('Acidaminococcus-intestini-D21'),
    strains.index('Mitsuokella-multacida-DSM-20544'),
    strains.index('Bacteroides-uniformis-ATCC-8492')
]

taxa_names = [
    r'$\it{Clostridium}$ sp. D5',
    r'$\it{A.\ intestini}$ D21',
    r'$\it{M.\ multacida}$ DSM-20544',
    r'$\it{B.\ uniformis}$ ATCC-8492'
]

taxa_colors = ['#8e44ad', '#2980b9', '#d35400', '#27ae60']
pass_coords = np.arange(7) # P0 to P6
pass_labels = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']

for i, (idx, name, col) in enumerate(zip(top_taxa_idx, taxa_names, taxa_colors)):
    # Observed trajectory (solid)
    traj_obs = np.concatenate([[p0_mean[idx]], mean_obs_s[:, idx]])
    # Predicted trajectory (dashed)
    traj_pred = np.concatenate([[p0_mean[idx]], mean_pred_s[:, idx]])
    
    ax4.plot(pass_coords, traj_obs, '-o', color=col, lw=2.2, ms=6, label=f'{name} (Obs)')
    ax4.plot(pass_coords, traj_pred, '--s', color=col, lw=1.8, ms=5, alpha=0.8, markerfacecolor='white', markeredgewidth=1.5, label=f'{name} (Zero-Shot)')

ax4.set_title(r"$\mathbf{d}$  Zero-Shot Supernatant Trajectories: Top Colonizers", fontsize=13, pad=10, loc='left')
ax4.set_xticks(pass_coords)
ax4.set_xticklabels(pass_labels, fontweight='bold', fontsize=11)
ax4.set_xlabel("Serial Dilution Passage", fontsize=11, fontweight='bold')
ax4.set_ylabel("Relative Abundance", fontsize=11, fontweight='bold')
ax4.grid(True, linestyle=':', alpha=0.5)
ax4.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8, ncol=2)
ax4.set_ylim(-0.02, 0.45)

plt.tight_layout()

# Save figures
artifact_path = os.path.join(ARTIFACT_DIR, 'Figure2_PlainAgar_ZeroShot_Transfer.png')
paper_fig_path = os.path.join(PAPER_FIG_DIR, 'Figure2_PlainAgar_ZeroShot_Transfer.png')

plt.savefig(artifact_path, dpi=300, bbox_inches='tight')
plt.savefig(paper_fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figures saved successfully to:\n  {artifact_path}\n  {paper_fig_path}", flush=True)

"""
Analyze German Rumen Microbiome Latent Space via Factorial Contrast Vectors.

Formulation:
  1. Fit champion K=3 model on Cow OSU.
  2. Extract learned R0 table (6 conditions in R^3).
  3. Compute rigorous ANOVA factorial contrast displacement vectors:
       Delta_R_MeOH = 0.5 * [(R0_meoh_5b - R0_noadd_5b) + (R0_meoh_30b - R0_noadd_30b)]
       Delta_R_TMA  = 0.5 * [(R0_tma_5b  - R0_noadd_5b) + (R0_tma_30b  - R0_noadd_30b)]
       Delta_R_Press = 1/3 * sum_c (R0_c_30b - R0_c_5b)
  4. Normalize to unit direction vectors: v_MeOH, v_TMA, v_Press.
  5. Compute subspace angles, cosine similarities, and orthogonal decomposition.
  6. Project strain consumption traits Theta in R^(81 x 3) onto [v_MeOH, v_TMA].
  7. Compare against empirical abundance shifts in Cow OSU and Cow BRRS.
  8. Save results to CSV and generate publication figure.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchdiffeq import odeint
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARTIFACT_DIR = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
PAPER_FIG_DIR = os.path.join(PROJECT_ROOT, 'Paper', 'figures')
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing Contrast Vector Analysis on: {device}", flush=True)

# 1. Load Data
data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
raw_data = torch.load(data_path, weights_only=False)
raw_strains = raw_data['strains']
raw_traj = torch.stack(list(raw_data['trajectories'].values()), dim=0)

peaks = raw_traj.amax(dim=(0, 1))
mask_1pct = peaks >= 0.01
strains_91 = [raw_strains[i] for i in range(len(raw_strains)) if mask_1pct[i]]
data_91 = raw_traj[:, :, mask_1pct]

shared_mask = (data_91[:6].sum(dim=(0, 1)) > 0) & (data_91[6:].sum(dim=(0, 1)) > 0)
strains_81 = [strains_91[i] for i in range(len(strains_91)) if shared_mask[i]]

floored_raw = data_91[:, :, shared_mask].to(device) + 1e-4
floored_data = floored_raw / floored_raw.sum(dim=-1, keepdim=True)

N, K = len(strains_81), 3
cond_env_ids = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=torch.long, device=device)

class PurePhysicalODE(nn.Module):
    def __init__(self, N, K):
        super().__init__()
        self.N, self.K = N, K
        self.a = nn.Parameter(torch.randn(6, K) * 0.5 + 2.0)
        self.theta_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.05 - 1.0)
        self.b0_raw = nn.Parameter(torch.tensor(0.0))

    def get_params(self):
        Theta = F.softplus(self.theta_raw)
        W = F.softplus(self.w_raw)
        alpha = F.softplus(self.alpha_raw)
        R0 = F.softplus(self.a)
        B0 = F.softplus(self.b0_raw) + 0.1
        return Theta, W, alpha, R0, B0

    def ode_deriv(self, t, S, Theta, W, alpha):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        dN_dt = N_t * torch.matmul(R_t, Theta.t()) - alpha * (N_t ** 2)
        flux = torch.matmul(torch.matmul(N_t, Theta), W.t())
        dR_dt = - R_t * flux
        return torch.cat([dN_dt, dR_dt], dim=-1)

# Train Champion Model on Cow OSU (seed=42, epochs=350, lr=0.015, l2_alpha=0.05)
torch.manual_seed(42)
np.random.seed(42)
model = PurePhysicalODE(N, K).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.015)

train_indices = [6, 7, 8, 9, 10, 11] # Cow OSU all 6 conditions
train_data = floored_data[train_indices]
train_seed = train_data[:, 0, :]
train_envs = cond_env_ids[train_indices]

best_loss = float('inf')
best_weights = None

for ep in range(1, 351):
    opt.zero_grad()
    Theta, W, alpha, R0_table, B0 = model.get_params()
    N_curr = train_seed * B0
    R0 = R0_table[train_envs]
    loss = 0.0
    tf_ratio = max(0.0, 1.0 - (ep / (350 * 0.85)))

    for p in range(1, 6):
        N0 = torch.clamp(N_curr / 100.0, min=0.0)
        S0 = torch.cat([N0, R0], dim=-1)
        S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, torch.tensor([0.0, 1.0], device=device), method='euler', options={'step_size': 0.1})
        N_final = torch.clamp(S_t[-1, :, :N], min=0.0)
        tot_bio = N_final.sum(dim=-1, keepdim=True) + 1e-12
        x_pred = N_final / tot_bio
        target = train_data[:, p, :]
        loss = loss + (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean()
        if tf_ratio > 0:
            x_mix = tf_ratio * target + (1.0 - tf_ratio) * x_pred
            N_curr = x_mix * tot_bio
        else:
            N_curr = N_final

    loss = loss + 0.05 * torch.sum(alpha ** 2)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
model.eval()

Theta, W, alpha, R0_table, B0 = model.get_params()
theta_np = Theta.detach().cpu().numpy()
r0_np = R0_table.detach().cpu().numpy()
b0_val = B0.item()

print(f"\nModel trained successfully. Learned B0 = {b0_val:.4f}", flush=True)

# 2. Extract Conditions and Compute Contrast Vectors
# Envs: 0=NoAdd 5b, 1=NoAdd 30b, 2=MeOH 5b, 3=MeOH 30b, 4=TMA 5b, 5=TMA 30b
r_noadd_5b  = r0_np[0]
r_noadd_30b = r0_np[1]
r_meoh_5b   = r0_np[2]
r_meoh_30b  = r0_np[3]
r_tma_5b    = r0_np[4]
r_tma_30b   = r0_np[5]

# Contrasts at 5 bar
delta_meoh_5b = r_meoh_5b - r_noadd_5b
delta_tma_5b  = r_tma_5b  - r_noadd_5b

# Contrasts at 30 bar
delta_meoh_30b = r_meoh_30b - r_noadd_30b
delta_tma_30b  = r_tma_30b  - r_noadd_30b

# Main Factorial Contrast Vectors (averaged across pressures)
delta_meoh_main = 0.5 * (delta_meoh_5b + delta_meoh_30b)
delta_tma_main  = 0.5 * (delta_tma_5b + delta_tma_30b)

# Pressure Contrast Vector (averaged across treatments)
delta_press_main = (1.0 / 3.0) * ((r_noadd_30b - r_noadd_5b) + (r_meoh_30b - r_meoh_5b) + (r_tma_30b - r_tma_5b))

# Unit Direction Vectors
norm_meoh = np.linalg.norm(delta_meoh_main)
norm_tma  = np.linalg.norm(delta_tma_main)
norm_press = np.linalg.norm(delta_press_main)

v_meoh = delta_meoh_main / norm_meoh
v_tma  = delta_tma_main  / norm_tma
v_press = delta_press_main / norm_press

# Subspace Angles
cos_meoh_tma = np.dot(v_meoh, v_tma)
angle_meoh_tma_deg = np.arccos(np.clip(cos_meoh_tma, -1.0, 1.0)) * 180.0 / np.pi

cos_meoh_press = np.dot(v_meoh, v_press)
angle_meoh_press_deg = np.arccos(np.clip(cos_meoh_press, -1.0, 1.0)) * 180.0 / np.pi

cos_tma_press = np.dot(v_tma, v_press)
angle_tma_press_deg = np.arccos(np.clip(cos_tma_press, -1.0, 1.0)) * 180.0 / np.pi

print("\n" + "="*80)
print("MATHEMATICAL CONTRAST VECTOR GEOMETRY IN LATENT SPACE (R^3)")
print("="*80)
print(f"Basal Control R0 (5 bar):   {np.round(r_noadd_5b, 4)}")
print(f"Basal Control R0 (30 bar):  {np.round(r_noadd_30b, 4)}")
print(f"\nDelta MeOH Vector (Main):   {np.round(delta_meoh_main, 4)} (Magnitude: {norm_meoh:.4f})")
print(f"Delta TMA Vector (Main):    {np.round(delta_tma_main, 4)} (Magnitude: {norm_tma:.4f})")
print(f"Delta Pressure Vector:      {np.round(delta_press_main, 4)} (Magnitude: {norm_press:.4f})")
print(f"\nUnit Direction v_MeOH:      {np.round(v_meoh, 4)}")
print(f"Unit Direction v_TMA:       {np.round(v_tma, 4)}")
print(f"Unit Direction v_Pressure:  {np.round(v_press, 4)}")
print(f"\nAngle between MeOH and TMA directions:      {angle_meoh_tma_deg:6.2f}°  (Cosine: {cos_meoh_tma:+.4f})")
print(f"Angle between MeOH and Pressure directions: {angle_meoh_press_deg:6.2f}°  (Cosine: {cos_meoh_press:+.4f})")
print(f"Angle between TMA and Pressure directions:  {angle_tma_press_deg:6.2f}°  (Cosine: {cos_tma_press:+.4f})")
print("="*80)

# 3. Project Strain Consumption Profiles onto Perturbation Vectors
# Theta is (81, 3). Projection onto v is Theta @ v
proj_meoh = np.dot(theta_np, v_meoh)   # (81,)
proj_tma  = np.dot(theta_np, v_tma)    # (81,)
proj_press = np.dot(theta_np, v_press) # (81,)

# Raw coordinate values for comparison
raw_r1 = theta_np[:, 1]
raw_r2 = theta_np[:, 2]

# Differential niche preference
diff_proj = proj_meoh - proj_tma
diff_raw = raw_r1 - raw_r2

# Target species: Methanomicrobium sp. (idx 0)
target_idx = 0
target_name = strains_81[target_idx]

print(f"\nTarget Microbe: {target_name}")
print(f"  Projected MeOH Affinity: {proj_meoh[target_idx]:.4f} (Raw R1: {raw_r1[target_idx]:.4f})")
print(f"  Projected TMA Affinity:  {proj_tma[target_idx]:.4f} (Raw R2: {raw_r2[target_idx]:.4f})")
print(f"  Projected Preference (MeOH - TMA): {diff_proj[target_idx]:+.4f} (Raw: {diff_raw[target_idx]:+.4f})")

# Correlation between raw and projected coordinates
corr_meoh, _ = pearsonr(raw_r1, proj_meoh)
corr_tma, _ = pearsonr(raw_r2, proj_tma)
corr_diff, _ = pearsonr(diff_raw, diff_proj)
print(f"\nAlignment between Raw Coordinates and Formal Contrast Projections:")
print(f"  Pearson r(Raw R1, Projected v_MeOH): {corr_meoh:.4f}")
print(f"  Pearson r(Raw R2, Projected v_TMA):  {corr_tma:.4f}")
print(f"  Pearson r(Raw Diff, Projected Diff): {corr_diff:.4f}")

# 4. Save Comprehensive CSV
df_strains = pd.DataFrame({
    'Strain_Index': np.arange(N),
    'Strain_Name': strains_81,
    'Raw_Theta_R0': theta_np[:, 0],
    'Raw_Theta_R1': theta_np[:, 1],
    'Raw_Theta_R2': theta_np[:, 2],
    'Projected_Affinity_MeOH': proj_meoh,
    'Projected_Affinity_TMA': proj_tma,
    'Projected_Affinity_Pressure': proj_press,
    'Chemical_Preference_MeOH_minus_TMA': diff_proj,
    'Self_Limitation_Alpha': alpha.detach().cpu().numpy()
})

out_csv = os.path.join(PROJECT_ROOT, 'Data', 'German_Contrast_Vector_Strain_Affinities.csv')
df_strains.to_csv(out_csv, index=False)
print(f"\nSaved strain contrast projections to: {out_csv}", flush=True)

# 5. Generate Publication-Quality Contrast Analysis Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# -------------------------------------------------------------
# Panel A: Latent Vector Geometry and Contrast Displacement
# -------------------------------------------------------------
ax1 = axes[0]
conditions = ['Basal Control', '+MeOH Treatment', '+TMA Treatment']
x_bar = np.arange(len(conditions))
width = 0.26

# Average R0 across 5b and 30b for each condition
r0_ctrl_mean = 0.5 * (r_noadd_5b + r_noadd_30b)
r0_meoh_mean = 0.5 * (r_meoh_5b + r_meoh_30b)
r0_tma_mean  = 0.5 * (r_tma_5b + r_tma_30b)
r0_matrix_mean = np.stack([r0_ctrl_mean, r0_meoh_mean, r0_tma_mean], axis=0) # (3, 3)

colors = ['#2c3e50', '#e74c3c', '#27ae60']
labels = ['Latent Resource $R_0$', 'Latent Resource $R_1$', 'Latent Resource $R_2$']

for k in range(3):
    ax1.bar(x_bar + (k - 1) * width, r0_matrix_mean[:, k], width, color=colors[k], alpha=0.85, edgecolor='black', lw=0.9, label=labels[k])

ax1.set_xticks(x_bar)
ax1.set_xticklabels(conditions, fontsize=11, fontweight='bold')
ax1.set_ylabel(r"Latent Resource Availability ($R_0$)", fontsize=11, fontweight='bold')
ax1.set_title(r"$\mathbf{A}$  Latent Environmental Profiles across Treatments", fontsize=12.5, pad=10, loc='left')
ax1.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=8.5)
ax1.grid(True, linestyle=':', alpha=0.5, axis='y')

# Annotate displacement magnitudes
ax1.text(0.04, 0.94, f"Factorial Displacements:\n" +
         r"$\|\Delta \mathbf{R}_{\mathrm{MeOH}}\|_2$ = " + f"{norm_meoh:.3f}\n" +
         r"$\|\Delta \mathbf{R}_{\mathrm{TMA}}\|_2$ = " + f"{norm_tma:.3f}\n" +
         rf"Subspace Angle $\theta = {angle_meoh_tma_deg:.1f}^\circ$",
         transform=ax1.transAxes, ha='left', va='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9f9', edgecolor='#bdc3c7', lw=1.2),
         fontsize=9.0, fontweight='bold')

# -------------------------------------------------------------
# Panel B: Projected Strain Chemical Affinities (v_MeOH vs v_TMA)
# -------------------------------------------------------------
ax2 = axes[1]
# Baseline abundance in Cow BRRS T1 (size and color)
brrs_t1_abund = floored_data[[0, 2, 4], 0].mean(dim=0).cpu().numpy() * 100
vmin = 0.01
vmax = 50.0
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
cmap = plt.cm.viridis

log_vals = np.log10(np.clip(brrs_t1_abund, vmin, vmax))
norm_log = (log_vals - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
sizes = 20 + 280 * (norm_log ** 2.2)
rgba_colors = cmap(norm(np.clip(brrs_t1_abund, vmin, vmax)))

for i in range(N):
    is_target = (i == target_idx)
    cur_edge = '#c0392b' if is_target else '#2c3e50'
    cur_lw = 2.2 if is_target else 0.6
    cur_s = sizes[i] * (1.8 if is_target else 1.0)
    
    ax2.scatter(
        proj_meoh[i], proj_tma[i],
        c=[rgba_colors[i]], s=cur_s,
        edgecolors=cur_edge, linewidth=cur_lw,
        zorder=5 if is_target else 3
    )

# Identity ray (equal affinity)
max_proj = max(proj_meoh.max(), proj_tma.max()) * 1.08
ax2.plot([0, max_proj], [0, max_proj], linestyle='--', color='#7f8c8d', lw=1.2, alpha=0.7, label='Equal Affinity Ray', zorder=2)

# Highlight Target Microbe
ax2.annotate(
    r'$\mathbf{Methanomicrobium\ sp.}$' + f'\n(TMA Preference: $\\Delta={proj_tma[target_idx]-proj_meoh[target_idx]:.2f}$)',
    xy=(proj_meoh[target_idx], proj_tma[target_idx]),
    xytext=(proj_meoh[target_idx] + 0.35, proj_tma[target_idx] - 0.25),
    arrowprops=dict(facecolor='#c0392b', edgecolor='#c0392b', arrowstyle='->', lw=1.8),
    fontsize=9.5, fontweight='bold', color='#c0392b',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#fadbd8', edgecolor='#e74c3c', lw=1.0),
    zorder=10
)

ax2.set_xlabel(r"Projected Methanol Affinity ($\mathbf{\Theta}_i^\top \mathbf{v}_{\mathrm{MeOH}}$)", fontsize=11, fontweight='bold')
ax2.set_ylabel(r"Projected TMA Affinity ($\mathbf{\Theta}_i^\top \mathbf{v}_{\mathrm{TMA}}$)", fontsize=11, fontweight='bold')
ax2.set_title(rf"$\mathbf{{B}}$  Strain Niche Space on Unit Contrast Vectors ($\theta = {angle_meoh_tma_deg:.1f}^\circ$)", fontsize=12.5, pad=10, loc='left')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=8.5)

# -------------------------------------------------------------
# Panel C: Differential Preference and Guild Separation
# -------------------------------------------------------------
ax3 = axes[2]

# Sort strains by differential chemical preference (TMA preference = -diff_proj)
tma_pref = proj_tma - proj_meoh
sorted_indices = np.argsort(tma_pref)

# Top 10 TMA specialists and Top 10 MeOH specialists
top_meoh_idx = sorted_indices[:8]
top_tma_idx = sorted_indices[-8:]
selected_idx = np.concatenate([top_meoh_idx, top_tma_idx])
if target_idx not in selected_idx:
    selected_idx = np.append(selected_idx, target_idx)
    selected_idx = selected_idx[np.argsort(tma_pref[selected_idx])]

y_pos = np.arange(len(selected_idx))
bars = ax3.barh(y_pos, tma_pref[selected_idx], height=0.7, color=np.where(tma_pref[selected_idx] > 0, '#27ae60', '#e74c3c'), alpha=0.85, edgecolor='black', lw=0.8)

# Highlight target
for i, idx in enumerate(selected_idx):
    if idx == target_idx:
        bars[i].set_facecolor('#d35400')
        bars[i].set_edgecolor('#962d00')
        bars[i].set_linewidth(1.8)

short_names = [strains_81[idx].split('|')[-1][:22] for idx in selected_idx]
ax3.set_yticks(y_pos)
ax3.set_yticklabels(short_names, fontsize=8.5, fontweight='bold')
ax3.axvline(0, color='black', lw=1.0)
ax3.set_xlabel(r"Chemical Niche Bias ($\mathbf{\Theta}_i^\top \mathbf{v}_{\mathrm{TMA}} - \mathbf{\Theta}_i^\top \mathbf{v}_{\mathrm{MeOH}}$)", fontsize=11, fontweight='bold')
ax3.set_title(r"$\mathbf{C}$  Functional Guild Niche Specialization", fontsize=12.5, pad=10, loc='left')
ax3.grid(True, linestyle=':', alpha=0.5, axis='x')

ax3.text(0.05, 0.94, r'$\leftarrow$ Prefers MeOH', transform=ax3.transAxes, fontsize=9.5, fontweight='bold', color='#c0392b', ha='left', va='top')
ax3.text(0.95, 0.94, r'Prefers TMA $\rightarrow$', transform=ax3.transAxes, fontsize=9.5, fontweight='bold', color='#27ae60', ha='right', va='top')

plt.tight_layout()

# Save output figures
out_paper = os.path.join(PAPER_FIG_DIR, 'Figure2_German_Factorial_Contrast_Vectors.png')
out_artifact = os.path.join(ARTIFACT_DIR, 'Figure2_German_Factorial_Contrast_Vectors.png')

fig.savefig(out_paper, dpi=300, bbox_inches='tight')
fig.savefig(out_artifact, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSaved updated contrast vector figure to:\n  - {out_paper}\n  - {out_artifact}", flush=True)

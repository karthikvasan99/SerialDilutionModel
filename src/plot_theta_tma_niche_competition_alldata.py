import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr

# Set styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Generating Publication Figure with Substrate-Specific Niche Competition on: {device}")

# 1. Load Data
raw_data = torch.load('Data/German_processed_data.pt', weights_only=False)
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
target_idx = 0

def get_W(w_raw, K, max_crossfeed=0.5):
    W_sp = F.softplus(w_raw)
    W_sp = W_sp * (1.0 - torch.eye(K, device=W_sp.device))
    row_sums = W_sp.sum(dim=-1, keepdim=True)
    scale = torch.clamp(max_crossfeed / torch.clamp(row_sums, min=1e-6), max=1.0)
    return W_sp * scale

def ode_step(S, Theta, W, alpha, N, dt=0.1):
    for _ in range(10):
        N_t = torch.clamp(S[..., :N], min=0.0)
        R_t = torch.clamp(S[..., N:], min=0.0)
        growth = torch.matmul(R_t, Theta.t())
        dN_dt = N_t * growth - alpha * (N_t ** 2)
        flux = R_t * torch.matmul(N_t, Theta)
        byproduct = torch.matmul(flux, W)
        dR = - flux + byproduct
        dS = torch.cat([dN_dt, dR], dim=-1)
        S = S + dt * dS
    return S

torch.manual_seed(42)
np.random.seed(42)

# Optimal hyperparameters from 6-fold CV
lr = 0.020
l2_alpha = 0.05
epochs = 300

a = nn.Parameter(torch.randn(6, K, device=device) * 0.5 + 2.0)
theta_raw = nn.Parameter(torch.randn(N, K, device=device) * 0.1)
w_raw = nn.Parameter(torch.randn(K, K, device=device) * 0.1 - 1.0)
alpha_raw = nn.Parameter(torch.randn(N, device=device) * 0.05 - 1.0)
b0_raw = nn.Parameter(torch.tensor(0.0, device=device))

opt = torch.optim.Adam([a, theta_raw, w_raw, alpha_raw, b0_raw], lr=lr)

train_indices = list(range(12)) # ALL 12 conditions
train_data = floored_data[train_indices]
train_seed = train_data[:, 0, :]
train_envs = cond_env_ids[train_indices]

best_loss = float('inf')
best_params = None

for ep in range(1, epochs + 1):
    opt.zero_grad()
    Theta = F.softplus(theta_raw)
    W = get_W(w_raw, K)
    alpha = F.softplus(alpha_raw)
    R0_table = F.softplus(a)
    B0 = F.softplus(b0_raw) + 0.1

    N_curr = train_seed * B0
    R0 = R0_table[train_envs]
    loss = 0.0
    tf_ratio = max(0.0, 1.0 - (ep / (epochs * 0.85)))

    for p in range(1, 6):
        N0 = torch.clamp(N_curr / 100.0, min=0.0)
        S0 = torch.cat([N0, R0], dim=-1)
        S_fin = ode_step(S0, Theta, W, alpha, N)
        N_final = torch.clamp(S_fin[..., :N], min=0.0)
        tot_bio = N_final.sum(dim=-1, keepdim=True) + 1e-12
        x_pred = N_final / tot_bio
        target = train_data[:, p, :]
        loss = loss + (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean()
        if tf_ratio > 0:
            x_mix = tf_ratio * target + (1.0 - tf_ratio) * x_pred
            N_curr = x_mix * tot_bio
        else:
            N_curr = N_final

    if l2_alpha > 0:
        loss = loss + l2_alpha * torch.sum(alpha ** 2)

    loss.backward()
    nn.utils.clip_grad_norm_([a, theta_raw, w_raw, alpha_raw, b0_raw], 5.0)
    opt.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_params = (a.detach().clone(), theta_raw.detach().clone(), w_raw.detach().clone(), alpha_raw.detach().clone(), b0_raw.detach().clone())

b_a, b_th, b_w, b_al, b_b0 = best_params
Theta = F.softplus(b_th).cpu().numpy()
R0_table = F.softplus(b_a).cpu().numpy()

# Chemical Axes Mapping in Champion Model:
# Column 0: spikes in TMA 5b & 30b -> TMA Axis
# Column 1: high across all conditions -> Syngas H2:CO2 Baseline
# Column 2: spikes in MeOH 5b & 30b -> MeOH Axis
idx_tma = 0
idx_syn = 1
idx_meoh = 2

# Calculate Baseline Inoculum Abundances
brrs_t1_abund = floored_data[[0, 2, 4], 0].mean(dim=0).cpu().numpy() * 100
osu_t1_abund = floored_data[[6, 8, 10], 0].mean(dim=0).cpu().numpy() * 100
mean_t1_abund = 0.5 * (brrs_t1_abund + osu_t1_abund)

# 2x2 Layout: 4 panels (A, B, C, D)
fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)

# =========================================================================
# PANEL A: Focus on 5 bar conditions (No Additives, MeOH, TMA)
# =========================================================================
ax1 = axes[0, 0]
conditions_5b = ['No Additives', 'MeOH', 'TMA']
cond_indices = [0, 2, 4] # Envs: 0=NoAdd 5b, 2=MeOH 5b, 4=TMA 5b
x_bars = np.arange(len(conditions_5b))
width = 0.24

ax1.bar(x_bars - width, R0_table[cond_indices, idx_syn],  width, label=r'Resource 1 (Syngas $H_2:CO_2$)', color='#2b5c8f')
ax1.bar(x_bars,         R0_table[cond_indices, idx_meoh], width, label='Resource 2 (MeOH Axis)', color='#e05d44')
ax1.bar(x_bars + width, R0_table[cond_indices, idx_tma],  width, label='Resource 0 (TMA Axis)', color='#2ea44f')

for i in range(len(conditions_5b)):
    c_idx = cond_indices[i]
    ax1.text(x_bars[i] - width, R0_table[c_idx, idx_syn] + 0.08, f"{R0_table[c_idx, idx_syn]:.2f}", ha='center', fontsize=9.5, fontweight='bold', color='#2b5c8f')
    ax1.text(x_bars[i],         R0_table[c_idx, idx_meoh] + 0.08, f"{R0_table[c_idx, idx_meoh]:.2f}", ha='center', fontsize=9.5, fontweight='bold', color='#e05d44')
    ax1.text(x_bars[i] + width, R0_table[c_idx, idx_tma] + 0.08, f"{R0_table[c_idx, idx_tma]:.2f}", ha='center', fontsize=9.5, fontweight='bold', color='#2ea44f')

ax1.set_xticks(x_bars)
ax1.set_xticklabels(conditions_5b, fontsize=11.5, fontweight='bold')
ax1.set_ylabel(r'Learned Resource Capacity ($R_0$)', fontsize=11.5, fontweight='bold')
ax1.set_title('A. Unsupervised Chemical Axis Discovery (Full Dataset Champion)', fontsize=12.5, fontweight='bold', pad=12)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='upper left')
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.set_ylim(0, 4.0)

# =========================================================================
# PANEL B: Niche Map (Completely Clean: No Text Boxes or Arrows)
# =========================================================================
ax2 = axes[0, 1]

vmin = 0.01
vmax = 50.0
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
cmap = plt.cm.viridis

log_vals = np.log10(np.clip(brrs_t1_abund, vmin, vmax))
norm_log = (log_vals - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))

sizes = 22 + 250 * (norm_log ** 2.0)
alphas = 0.35 + 0.65 * (norm_log ** 1.5)

rgba_colors = cmap(norm(np.clip(brrs_t1_abund, vmin, vmax)))

for i in range(N):
    if i == target_idx:
        continue
    cur_color = rgba_colors[i].copy()
    cur_color[3] = alphas[i]
    ax2.scatter(
        Theta[i, idx_meoh], Theta[i, idx_tma],
        c=[cur_color],
        s=sizes[i],
        edgecolors='black',
        linewidth=0.6,
        zorder=3
    )

# Highlight Target Microbe with distinct star and add to legend (NO text box or arrow)
ax2.scatter(
    Theta[target_idx, idx_meoh], Theta[target_idx, idx_tma],
    c='#e74c3c', s=250, marker='*', edgecolors='black', linewidth=1.2, zorder=6,
    label='Target: Methanomicrobium sp.'
)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar2 = plt.colorbar(sm, ax=ax2, pad=0.03, shrink=0.88)
cbar2.set_label('BRRS Inoculum (T1) Abundance (%, Log Scale)', fontsize=9.5, fontweight='bold')
cbar2.set_ticks([0.01, 0.1, 1.0, 10.0, 50.0])
cbar2.set_ticklabels(['0.01%', '0.1%', '1.0%', '10%', '50%'])
cbar2.ax.tick_params(labelsize=8.5)

ax2.plot([0, 2.0], [0, 2.0], linestyle=':', color='#7f8c8d', alpha=0.6, label='Symmetric Utilization Ray', zorder=1)
ax2.set_xlabel(r'Learned Methanol Affinity ($\Theta_{i, \mathrm{MeOH}}$)', fontsize=11.5, fontweight='bold')
ax2.set_ylabel(r'Learned TMA Affinity ($\Theta_{i, \mathrm{TMA}}$)', fontsize=11.5, fontweight='bold')
ax2.set_title('B. Metabolic Niche Space (Full Dataset Champion)', fontsize=12.5, fontweight='bold', pad=12)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
ax2.set_xlim(-0.05, 1.9)
ax2.set_ylim(-0.05, 1.9)

# =========================================================================
# PANEL C: Host Asymmetry vs. Chemical Niche Bias (Clean: No Text Boxes or Arrows)
# =========================================================================
ax3 = axes[1, 0]

x_affinity_diff = Theta[:, idx_meoh] - Theta[:, idx_tma]
y_abund_diff = brrs_t1_abund - osu_t1_abund

ax3.axvline(0, color='#7f8c8d', linestyle='--', linewidth=1.0, alpha=0.7)
ax3.axhline(0, color='#7f8c8d', linestyle='--', linewidth=1.0, alpha=0.7)

# Log-scale colormap for Mean Abundance at T1 across hosts
vmin_c = 0.01
vmax_c = 30.0
norm_c = mcolors.LogNorm(vmin=vmin_c, vmax=vmax_c)
cmap_c = plt.cm.viridis

log_c = np.log10(np.clip(mean_t1_abund, vmin_c, vmax_c))
norm_log_c = (log_c - np.log10(vmin_c)) / (np.log10(vmax_c) - np.log10(vmin_c))
sizes_c = 25 + 240 * (norm_log_c ** 1.8)

# Scatter background points
for i in range(N):
    if i == target_idx:
        continue
    ax3.scatter(
        x_affinity_diff[i], y_abund_diff[i],
        c=[cmap_c(norm_c(np.clip(mean_t1_abund[i], vmin_c, vmax_c)))],
        s=sizes_c[i],
        edgecolors='black',
        linewidth=0.7,
        alpha=0.85,
        zorder=4
    )

# Highlight Target Microbe with distinct star and add to legend (NO text box or arrow)
ax3.scatter(
    x_affinity_diff[target_idx], y_abund_diff[target_idx],
    c='#e74c3c', s=250, marker='*', edgecolors='black', linewidth=1.2, zorder=6,
    label='Target: Methanomicrobium sp.'
)

sm_c = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_c)
sm_c.set_array([])
cbar3 = plt.colorbar(sm_c, ax=ax3, pad=0.03, shrink=0.88)
cbar3.set_label('Mean Inoculum Abundance at T1 (%, Log Scale)', fontsize=9.5, fontweight='bold')
cbar3.set_ticks([0.01, 0.1, 1.0, 10.0, 30.0])
cbar3.set_ticklabels(['0.01%', '0.1%', '1.0%', '10%', '30%'])
cbar3.ax.tick_params(labelsize=8.5)

# Clean, unboxed quadrant labels
ax3.text(1.0, 31, 'BRRS-Biased\n(Prefers MeOH)', fontsize=9.5, fontweight='bold', color='#b03a2e', ha='center', alpha=0.85)
ax3.text(-1.0, 31, 'BRRS-Biased\n(Prefers TMA)', fontsize=9.5, fontweight='bold', color='#b03a2e', ha='center', alpha=0.85)
ax3.text(-1.0, -22, 'OSU-Biased\n(Prefers TMA)', fontsize=9.5, fontweight='bold', color='#2471a3', ha='center', alpha=0.85)
ax3.text(1.0, -22, 'OSU-Biased\n(Prefers MeOH)', fontsize=9.5, fontweight='bold', color='#2471a3', ha='center', alpha=0.85)

ax3.set_xlabel(r'Chemical Niche Preference ($\Theta_{\mathrm{MeOH}} - \Theta_{\mathrm{TMA}}$)', fontsize=11.5, fontweight='bold')
ax3.set_ylabel('Host Inoculum Asymmetry: BRRS - OSU (%)', fontsize=11.5, fontweight='bold')
ax3.set_title('C. Host Asymmetry vs. Chemical Niche Bias', fontsize=12.5, fontweight='bold', pad=12)
ax3.grid(True, linestyle='--', alpha=0.3)
ax3.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
ax3.set_xlim(-1.5, 1.6)
ax3.set_ylim(-26, 36)

# =========================================================================
# PANEL D: Substrate-Specific Competitive Pressure (Clean: No Text Boxes or Arrows)
# =========================================================================
ax4 = axes[1, 1]
cond_labels = ['Methanol (5 bar)', 'Trimethylamine (5 bar)']
x_d = np.arange(len(cond_labels))
width_d = 0.32

comp_mask = np.ones(N, dtype=bool)
comp_mask[target_idx] = False

# Compute Substrate-Specific Competition on the added nutrient axis:
ov_sub_meoh = Theta[target_idx, idx_meoh] * R0_table[2, idx_meoh] * Theta[:, idx_meoh]
brrs_meoh_t1 = np.sum(ov_sub_meoh[comp_mask] * floored_data[2, 0].cpu().numpy()[comp_mask])
brrs_meoh_t6 = np.sum(ov_sub_meoh[comp_mask] * floored_data[2, -1].cpu().numpy()[comp_mask])
osu_meoh_t1  = np.sum(ov_sub_meoh[comp_mask] * floored_data[8, 0].cpu().numpy()[comp_mask])
osu_meoh_t6  = np.sum(ov_sub_meoh[comp_mask] * floored_data[8, -1].cpu().numpy()[comp_mask])

ov_sub_tma = Theta[target_idx, idx_tma] * R0_table[4, idx_tma] * Theta[:, idx_tma]
brrs_tma_t1 = np.sum(ov_sub_tma[comp_mask] * floored_data[4, 0].cpu().numpy()[comp_mask])
brrs_tma_t6 = np.sum(ov_sub_tma[comp_mask] * floored_data[4, -1].cpu().numpy()[comp_mask])
osu_tma_t1  = np.sum(ov_sub_tma[comp_mask] * floored_data[10, 0].cpu().numpy()[comp_mask])
osu_tma_t6  = np.sum(ov_sub_tma[comp_mask] * floored_data[10, -1].cpu().numpy()[comp_mask])

osu_t1_vals = [osu_meoh_t1, osu_tma_t1]
osu_t6_vals = [osu_meoh_t6, osu_tma_t6]

brrs_t1_vals = [brrs_meoh_t1, brrs_tma_t1]
brrs_t6_vals = [brrs_meoh_t6, brrs_tma_t6]

rects1 = ax4.bar(x_d - width_d/2, osu_t1_vals, width_d, label='Cow OSU (Inoculum T1)', color='#3498db', edgecolor='black', linewidth=0.8)
rects2 = ax4.bar(x_d + width_d/2, brrs_t1_vals, width_d, label='Cow BRRS (Inoculum T1)', color='#e67e22', edgecolor='black', linewidth=0.8)

for i in range(len(cond_labels)):
    # OSU labels
    ax4.text(x_d[i] - width_d/2, osu_t1_vals[i] + 0.03, f'{osu_t1_vals[i]:.2f}', ha='center', fontsize=10.5, fontweight='bold', color='#2980b9')
    diff_osu = ((osu_t6_vals[i] - osu_t1_vals[i]) / osu_t1_vals[i]) * 100
    shift_osu = f"-> {osu_t6_vals[i]:.2f} ({diff_osu:+.0f}%)"
    ax4.annotate(shift_osu, (x_d[i] - width_d/2, osu_t1_vals[i]/2), ha='center', va='center', fontsize=9.0, color='white', fontweight='bold')
    
    # BRRS labels
    ax4.text(x_d[i] + width_d/2, brrs_t1_vals[i] + 0.03, f'{brrs_t1_vals[i]:.2f}', ha='center', fontsize=10.5, fontweight='bold', color='#d35400')
    diff_pct = ((brrs_t6_vals[i] - brrs_t1_vals[i]) / brrs_t1_vals[i]) * 100
    shift_brrs = f"-> {brrs_t6_vals[i]:.2f} ({diff_pct:+.0f}%)"
    ax4.annotate(shift_brrs, (x_d[i] + width_d/2, brrs_t1_vals[i]/2), ha='center', va='center', fontsize=9.0, color='white', fontweight='bold')

# Differences across hosts
diff_meoh = ((brrs_meoh_t1 - osu_meoh_t1) / osu_meoh_t1) * 100
diff_tma = ((brrs_tma_t1 - osu_tma_t1) / osu_tma_t1) * 100

# Clean titles above bars without boxes or arrows
max_meoh = max(osu_meoh_t1, brrs_meoh_t1)
max_tma = max(osu_tma_t1, brrs_tma_t1)
y_max_d = max(max_meoh, max_tma) * 1.20
ax4.set_ylim(0, y_max_d)

ax4.text(x_d[0], max_meoh + 0.16, f'BRRS > OSU ({diff_meoh:+.0f}%)\n[Transfer Fails]',
         ha='center', va='bottom', fontsize=10.0, fontweight='bold', color='#c0392b')

ax4.text(x_d[1], max_tma + 0.16, f'BRRS < OSU ({diff_tma:+.0f}%)\n[Prescription Rescues]',
         ha='center', va='bottom', fontsize=10.0, fontweight='bold', color='#27ae60')

ax4.set_xticks(x_d)
ax4.set_xticklabels(cond_labels, fontsize=11.5, fontweight='bold')
ax4.set_ylabel('Substrate-Specific Niche Pressure', fontsize=11.5, fontweight='bold')
ax4.set_title('D. Substrate-Specific Competitive Inversion at T1', fontsize=12.5, fontweight='bold', pad=12)
ax4.grid(axis='y', linestyle='--', alpha=0.3)
ax4.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='upper left')

plt.tight_layout()

# Save paths
out_path_alldata = 'Poster/figures/German_Theta_TMA_Niche_Competition_AllData.png'
out_path_orig = 'Poster/figures/German_Theta_TMA_Niche_Competition.png'
plt.savefig(out_path_alldata, dpi=300, bbox_inches='tight')
plt.savefig(out_path_orig, dpi=300, bbox_inches='tight')
plt.close()

# Copy to artifact directory
artifact_dir = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
shutil.copyfile(out_path_alldata, os.path.join(artifact_dir, 'German_Theta_TMA_Niche_Competition_AllData.png'))
shutil.copyfile(out_path_orig, os.path.join(artifact_dir, 'German_Theta_TMA_Niche_Competition.png'))

print("Updated 4-panel figure with substrate-specific competitive inversion successfully saved to:")
print(f"  1. {out_path_alldata}")
print(f"  2. {out_path_orig}")

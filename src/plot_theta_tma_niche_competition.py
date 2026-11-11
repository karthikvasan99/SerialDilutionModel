import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchdiffeq import odeint

# Set styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

torch.manual_seed(42)
np.random.seed(42)

model = PurePhysicalODE(N, K).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.015)

train_indices = [6, 7, 8, 9, 10, 11]
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
Theta, W, alpha, R0_table, B0 = model.get_params()

theta_np = Theta.detach().cpu().numpy()
r0_np = R0_table.detach().cpu().numpy()

# 2x2 Layout: 4 panels (A, B, C, D)
fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)

# =========================================================================
# PANEL A: Focus strictly on 5 bar conditions (No Additives, MeOH, TMA)
# =========================================================================
ax1 = axes[0, 0]
conditions_5b = ['No Additives', 'MeOH', 'TMA']
cond_indices = [0, 2, 4]
x_bars = np.arange(len(conditions_5b))
width = 0.24

ax1.bar(x_bars - width, r0_np[cond_indices, 0], width, label='Resource 0 (Syngas)', color='#2b5c8f')
ax1.bar(x_bars,         r0_np[cond_indices, 1], width, label='Resource 1 (MeOH Axis)', color='#e05d44')
ax1.bar(x_bars + width, r0_np[cond_indices, 2], width, label='Resource 2 (TMA Axis)', color='#2ea44f')

ax1.set_xticks(x_bars)
ax1.set_xticklabels(conditions_5b, fontsize=11, fontweight='bold')
# Clean label with R0 properly formatted
ax1.set_ylabel(r'Learned Resource Capacity ($R_0$)', fontsize=11, fontweight='bold')
ax1.set_title('A. Unsupervised Chemical Axis Discovery (5 bar)', fontsize=12.5, fontweight='bold', pad=12)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='upper left')
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.set_ylim(0, 3.6)

# =========================================================================
# PANEL B: Niche Map with High-Contrast Log Scale & Dynamic Alpha
# =========================================================================
ax2 = axes[0, 1]
brrs_t1_abund = floored_data[[0, 2, 4], 0].mean(dim=0).cpu().numpy() * 100

vmin = 0.01
vmax = 50.0
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
cmap = plt.cm.viridis

log_vals = np.log10(np.clip(brrs_t1_abund, vmin, vmax))
norm_log = (log_vals - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))

sizes = 15 + 265 * (norm_log ** 2.2)
alphas = 0.20 + 0.80 * (norm_log ** 1.8)

rgba_colors = cmap(norm(np.clip(brrs_t1_abund, vmin, vmax)))

for i in range(N):
    cur_color = rgba_colors[i].copy()
    cur_color[3] = alphas[i]
    ax2.scatter(
        theta_np[i, 1], theta_np[i, 2],
        c=[cur_color],
        s=sizes[i],
        edgecolors='black',
        linewidth=0.7,
        zorder=3
    )

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar2 = plt.colorbar(sm, ax=ax2, pad=0.03, shrink=0.88)
cbar2.set_label('BRRS Inoculum (T1) Abundance (%, Log Scale)', fontsize=9.5, fontweight='bold')
cbar2.set_ticks([0.01, 0.1, 1.0, 10.0, 50.0])
cbar2.set_ticklabels(['0.01%', '0.1%', '1.0%', '10%', '50%'])
cbar2.ax.tick_params(labelsize=8.5)

ax2.plot([0, 2.6], [0, 2.6], linestyle=':', color='#7f8c8d', alpha=0.4, label='Symmetric Utilization Ray', zorder=1)

highlight_species_b = [
    (0, 'Methanomicrobium sp.', (-25, 14)),
    ([i for i, s in enumerate(strains_81) if 'Denitrobacterium detoxificans' in s][0], 'Denitrobacterium', (10, -5)),
    ([i for i, s in enumerate(strains_81) if 'Bacteroides pyogenes' in s][0], 'B. pyogenes', (10, 5)),
    ([i for i, s in enumerate(strains_81) if 'Streptococcus vicugnae' in s][0], 'S. vicugnae', (10, -12)),
    ([i for i, s in enumerate(strains_81) if 'Bacteroidales bacterium WCE2008' in s][0], 'Bacteroidales WCE2008', (-80, -18)),
    ([i for i, s in enumerate(strains_81) if 'Pyramidobacter sp. C12-8' in s][0], 'Pyramidobacter C12-8', (10, -5)),
    ([i for i, s in enumerate(strains_81) if 'Thermoplasmata archaeon' in s][0], 'Thermoplasmata', (10, 5))
]

for idx, label, offset in highlight_species_b:
    ax2.annotate(
        label, (theta_np[idx, 1], theta_np[idx, 2]),
        textcoords='offset points', xytext=offset,
        fontsize=8.5, fontweight='bold', color='#111111',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.90, edgecolor='#cccccc', linewidth=0.6),
        zorder=6
    )

ax2.set_xlabel(r'Learned Methanol Affinity ($\Theta_{i, \mathrm{MeOH}}$)', fontsize=11, fontweight='bold')
ax2.set_ylabel(r'Learned TMA Affinity ($\Theta_{i, \mathrm{TMA}}$)', fontsize=11, fontweight='bold')
ax2.set_title('B. Metabolic Niche Space (High Contrast Log T1)', fontsize=12.5, fontweight='bold', pad=12)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_xlim(-0.1, 2.7)
ax2.set_ylim(-0.1, 2.7)

# =========================================================================
# PANEL C: Abundance in (BRRS - OSU) vs Affinity (MeOH - TMA)
# =========================================================================
ax3 = axes[1, 0]

x_affinity_diff = theta_np[:, 1] - theta_np[:, 2]
osu_t1_abund = floored_data[[6, 8, 10], 0].mean(dim=0).cpu().numpy() * 100
y_abund_diff = brrs_t1_abund - osu_t1_abund

ax3.axvline(0, color='#7f8c8d', linestyle='--', linewidth=1.0, alpha=0.7)
ax3.axhline(0, color='#7f8c8d', linestyle='--', linewidth=1.0, alpha=0.7)

norm_diff = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-25.0, vmax=35.0)
scatter_c = ax3.scatter(
    x_affinity_diff, y_abund_diff,
    c=y_abund_diff,
    cmap='coolwarm',
    norm=norm_diff,
    s=65,
    edgecolors='black',
    linewidth=0.8,
    zorder=4
)

cbar3 = plt.colorbar(scatter_c, ax=ax3, pad=0.03, shrink=0.88)
cbar3.set_label('Inoculum Bias: BRRS - OSU (%)', fontsize=9.5, fontweight='bold')
cbar3.ax.tick_params(labelsize=8.5)

ax3.text(1.3, 32, 'BRRS-Biased\n(Prefers MeOH)', fontsize=9, fontweight='bold', color='#c0392b', ha='center',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdf2e9', edgecolor='#e67e22', alpha=0.85))
ax3.text(-1.1, 32, 'BRRS-Biased\n(Prefers TMA)', fontsize=9, fontweight='bold', color='#c0392b', ha='center',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdf2e9', edgecolor='#e67e22', alpha=0.85))
ax3.text(-1.1, -22, 'OSU-Biased\n(Prefers TMA)', fontsize=9, fontweight='bold', color='#2980b9', ha='center',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#ebf5fb', edgecolor='#3498db', alpha=0.85))
ax3.text(1.3, -22, 'OSU-Biased\n(Prefers MeOH)', fontsize=9, fontweight='bold', color='#2980b9', ha='center',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#ebf5fb', edgecolor='#3498db', alpha=0.85))

highlight_species_c = [
    (0, 'Methanomicrobium sp.', (10, 5)),
    ([i for i, s in enumerate(strains_81) if 'Denitrobacterium detoxificans' in s][0], 'Denitrobacterium', (-95, -15)),
    ([i for i, s in enumerate(strains_81) if 'Bacteroides pyogenes' in s][0], 'B. pyogenes', (10, -8)),
    ([i for i, s in enumerate(strains_81) if 'Streptococcus vicugnae' in s][0], 'S. vicugnae', (10, -5)),
    ([i for i, s in enumerate(strains_81) if 'Bacteroidales bacterium WCE2008' in s][0], 'Bacteroidales WCE2008', (10, 5)),
    ([i for i, s in enumerate(strains_81) if 'Pyramidobacter sp. C12-8' in s][0], 'Pyramidobacter C12-8', (-120, -5)),
    ([i for i, s in enumerate(strains_81) if 'Thermoplasmata archaeon' in s][0], 'Thermoplasmata', (-90, 8))
]

for idx, label, offset in highlight_species_c:
    ax3.annotate(
        label, (x_affinity_diff[idx], y_abund_diff[idx]),
        textcoords='offset points', xytext=offset,
        fontsize=8.5, fontweight='bold', color='#111111',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.90, edgecolor='#cccccc', linewidth=0.6),
        zorder=6
    )

ax3.set_xlabel(r'Chemical Niche Preference ($\Theta_{\mathrm{MeOH}} - \Theta_{\mathrm{TMA}}$)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Host Inoculum Asymmetry: BRRS - OSU (%)', fontsize=11, fontweight='bold')
ax3.set_title('C. Host Asymmetry vs. Chemical Niche Bias', fontsize=12.5, fontweight='bold', pad=12)
ax3.grid(True, linestyle='--', alpha=0.3)
ax3.set_xlim(-1.8, 1.8)
ax3.set_ylim(-26, 36)

# =========================================================================
# PANEL D: Initial Inoculum (T1) Competition with Dynamic Shifts to T6
# =========================================================================
ax4 = axes[1, 1]
cond_labels = ['MeOH', 'TMA']
x_d = np.arange(len(cond_labels))
width_d = 0.32

osu_t1 = [5.798, 9.640]
brrs_t1 = [9.583, 8.964]

osu_t6 = [7.404, 10.159]
brrs_t6 = [11.375, 6.263]

rects1 = ax4.bar(x_d - width_d/2, osu_t1, width_d, label='Cow OSU (Inoculum T1)', color='#3498db', edgecolor='black', linewidth=0.6)
rects2 = ax4.bar(x_d + width_d/2, brrs_t1, width_d, label='Cow BRRS (Inoculum T1)', color='#e67e22', edgecolor='black', linewidth=0.6)

for i in range(len(cond_labels)):
    ax4.text(x_d[i] - width_d/2, osu_t1[i] + 0.3, f'{osu_t1[i]:.2f}', ha='center', fontsize=10, fontweight='bold', color='#2980b9')
    ax4.annotate(f'-> {osu_t6[i]:.1f} (T6)', (x_d[i] - width_d/2, osu_t1[i]/2), ha='center', fontsize=8.5, color='white', fontweight='bold')
    
    ax4.text(x_d[i] + width_d/2, brrs_t1[i] + 0.3, f'{brrs_t1[i]:.2f}', ha='center', fontsize=10, fontweight='bold', color='#d35400')
    shift_label = f'-> {brrs_t6[i]:.1f} (T6)'
    if i == 1:
        shift_label = f'-> {brrs_t6[i]:.1f} (-30%)'
    ax4.annotate(shift_label, (x_d[i] + width_d/2, brrs_t1[i]/2), ha='center', fontsize=8.5, color='white', fontweight='bold')

ax4.set_xticks(x_d)
ax4.set_xticklabels(cond_labels, fontsize=11.5, fontweight='bold')
ax4.set_ylabel('Competitive Pressure on Methanomicrobium', fontsize=11, fontweight='bold')
ax4.set_title('D. Initial Inoculum (T1) Competition', fontsize=12.5, fontweight='bold', pad=12)
ax4.set_ylim(0, 12.5)
ax4.grid(axis='y', linestyle='--', alpha=0.3)
ax4.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='upper left')

plt.tight_layout()
out_path = 'Poster/figures/German_Theta_TMA_Niche_Competition.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

# Copy to artifact
artifact_dir = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
shutil.copyfile(out_path, os.path.join(artifact_dir, 'German_Theta_TMA_Niche_Competition.png'))
print('Successfully regenerated and saved clean figure to:', out_path)

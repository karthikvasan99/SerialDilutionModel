"""
Generate Poster Figure 4: Non-Equilibrium Phase Dynamics (Gore et al. Science 2022)
100-passage long-term simulation with adaptive Dormand-Prince integration (dopri5).
Resource depletion traces for each passage rendered with uniform line width and uniform color.
Eliminates styling artifacts: reveals true geometric contrast between stable and chaotic regimes.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchdiffeq import odeint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load fitted model checkpoint and data
checkpoint_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Fitted_Niche_Chaos.pth')
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
sd = checkpoint['model_state_dict']

data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
data = torch.load(data_path, weights_only=False)

Theta = F.softplus(sd['Theta_raw']).to(device)
W_sp = F.softplus(sd['W_raw']).to(device)
W_sp = W_sp * (1.0 - torch.eye(5, device=device))
row_sums = W_sp.sum(dim=-1, keepdim=True)
scale = torch.clamp(0.85 / torch.clamp(row_sums, min=1e-6), max=1.0)
W = (W_sp * scale).to(device)
alpha = (0.005 * torch.sigmoid(sd['alpha_raw']) + 1e-5).to(device)
Km = (F.softplus(sd['Km_raw']) + 0.05).to(device)

# LN at R0=0.50 (stable fixed point regime), HN at R0=3.24 (non-equilibrium regime)
R0_ln = (torch.ones(5, device=device) * 0.50)
R0_hn = (F.softplus(sd['a_ln'].to(device)) + F.softplus(sd['delta_mn'].to(device)) + F.softplus(sd['delta_hn'].to(device)))

def ode_deriv(t, S):
    N_t = torch.clamp(S[..., :48], min=0.0)
    R_t = torch.clamp(S[..., 48:], min=0.0)
    monod = R_t / (R_t + Km)
    growth = torch.matmul(monod, Theta.t())
    dN = N_t * growth - alpha * (N_t ** 2)
    flux = monod * torch.matmul(N_t, Theta)
    byproduct = torch.matmul(flux, W)
    dR = - flux + byproduct
    return torch.cat([dN, dR], dim=-1)

mask = data['presence_masks'][(48, 'Community1')].float().to(device)
rel = data['relative_abundance'][('HN', 48, 'Community1')].to(device)
od = data['biomass_od'][('HN', 48, 'Community1')].mean(dim=0).to(device)
N0 = (rel[0] * od[0]) * mask
t_span = torch.linspace(0.0, 1.0, 31, device=device)

def simulate_100(r_init):
    N_curr = N0.clone()
    N_all, R_all = [N_curr.cpu().numpy()], []
    for p in range(100):
        N_in = torch.clamp(N_curr / 30.0 + 1e-5 * mask, min=0.0)
        S0 = torch.cat([N_in, r_init], dim=-1)
        S_t = odeint(ode_deriv, S0, t_span, method='dopri5', rtol=1e-5, atol=1e-6)
        S_t = torch.clamp(S_t, min=0.0)
        R_all.append(S_t[..., 48:].cpu().numpy())
        N_curr = torch.clamp(S_t[-1, :48] * mask, min=0.0)
        N_all.append(N_curr.cpu().numpy())
    return np.array(N_all), np.array(R_all)

print("Simulating 100 serial dilution passages for LN and HN with dopri5...")
N_ln, R_ln = simulate_100(R0_ln)
N_hn, R_hn = simulate_100(R0_hn)

passages = N_hn.shape[0]  # 101 points (passages 0..100)
p_axis = np.arange(passages)

plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig = plt.figure(figsize=(15.0, 4.7), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.15], hspace=0.36, wspace=0.20)

species_colors = plt.cm.turbo(np.linspace(0.02, 0.98, 48))

# =====================================================================
# Panel A1: Low Nutrients Trajectories (All 48 Species -> Stable Fixed Point)
# =====================================================================
ax0 = fig.add_subplot(gs[0, 0])
for sp in range(48):
    ax0.plot(p_axis, N_ln[:, sp], '-', color=species_colors[sp], lw=1.1, alpha=0.75)
ax0.text(0.04, 0.86, r'$\mathbf{Low\ Nutrients:\ Point\ Attractor\ (All\ 48\ Species)}$', transform=ax0.transAxes,
         fontsize=13.5, fontweight='bold', color='#00356B',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#F0F4F8', edgecolor='#00356B', alpha=0.92, lw=1.5))
ax0.set_ylabel('Biomass (OD)', fontsize=14.0, fontweight='bold', color='#1A365D')
ax0.set_xlabel('Passage', fontsize=13.0, fontweight='bold', color='#1A365D')
ax0.set_xlim(-1, 101)
ax0.set_xticks(range(0, 101, 10))
ax0.set_ylim(-0.01, 0.20)
ax0.tick_params(labelsize=12.0)
ax0.grid(True, linestyle='--', alpha=0.5)

# =====================================================================
# Panel A2: High Nutrients Trajectories (All 48 Species -> Non-Equilibrium Dynamics)
# =====================================================================
ax1 = fig.add_subplot(gs[0, 1])
for sp in range(48):
    ax1.plot(p_axis, N_hn[:, sp], '-', color=species_colors[sp], lw=1.1, alpha=0.75)
ax1.text(0.04, 0.86, r'$\mathbf{High\ Nutrients:\ Non\text{-}Equilibrium\ Dynamics\ (All\ 48\ Species)}$', transform=ax1.transAxes,
         fontsize=13.5, fontweight='bold', color='#802010',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#FDF2E9', edgecolor='#802010', alpha=0.92, lw=1.5))
ax1.set_ylabel('Biomass (OD)', fontsize=14.0, fontweight='bold', color='#1A365D')
ax1.set_xlabel('Passage', fontsize=13.0, fontweight='bold', color='#1A365D')
ax1.set_xlim(-1, 101)
ax1.set_xticks(range(0, 101, 10))
ax1.set_ylim(-0.5, 27.0)
ax1.tick_params(labelsize=12.0)
ax1.grid(True, linestyle='--', alpha=0.5)

# =====================================================================
# Panel B1: Low Nutrients Phase Space (Uniform Line Width and Color)
# =====================================================================
ax2 = fig.add_subplot(gs[1, 0])
uniform_blue = '#00356B'
for p in range(100):
    r_path = R_ln[p]
    ax2.plot(r_path[:, 0], r_path[:, 1], color=uniform_blue, lw=1.2, alpha=0.45)

ax2.text(0.04, 0.86, r'$\mathbf{Point\ Attractor\ (Single\ Invariant\ Path)}$', transform=ax2.transAxes,
         fontsize=13.5, fontweight='bold', color='#00356B',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#F0F4F8', edgecolor='#00356B', alpha=0.92, lw=1.5))
ax2.set_xlabel('Resource 1 ($R_1$)', fontsize=14.0, fontweight='bold', color='#1A365D')
ax2.set_ylabel('Resource 2 ($R_2$)', fontsize=14.0, fontweight='bold', color='#1A365D')
ax2.set_xlim(0.485, 0.505)
ax2.set_ylim(0.495, 0.505)
ax2.tick_params(labelsize=12.0)
ax2.grid(True, linestyle='--', alpha=0.5)

# =====================================================================
# Panel B2: High Nutrients Phase Space (Uniform Line Width and Color)
# =====================================================================
ax3 = fig.add_subplot(gs[1, 1])
uniform_red = '#802010'
for p in range(100):
    r_path = R_hn[p]
    ax3.plot(r_path[:, 0], r_path[:, 1], color=uniform_red, lw=1.2, alpha=0.35)

ax3.text(0.04, 0.86, r'$\mathbf{Non\text{-}Equilibrium\ Attractor\ (Wandering\ Loops)}$', transform=ax3.transAxes,
         fontsize=13.5, fontweight='bold', color='#802010',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#FDF2E9', edgecolor='#802010', alpha=0.92, lw=1.5))
ax3.set_xlabel('Resource 1 ($R_1$)', fontsize=14.0, fontweight='bold', color='#1A365D')
ax3.set_ylabel('Resource 2 ($R_2$)', fontsize=14.0, fontweight='bold', color='#1A365D')
ax3.set_xlim(-0.15, 3.50)
ax3.set_ylim(-0.15, 3.80)
ax3.tick_params(labelsize=12.0)
ax3.grid(True, linestyle='--', alpha=0.5)

# Save figure
out_path = os.path.join(PROJECT_ROOT, 'Poster', 'figures', 'Gore_Phase_Space_Dynamics.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated 100-passage authentic figure at: {out_path}")

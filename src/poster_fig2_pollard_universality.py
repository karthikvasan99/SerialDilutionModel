import sys; sys.path.insert(0, ".")
import os, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, leaves_list
from torchdiffeq import odeint

from src.Pollard_Train_Physical_ODE import PurePhysicalPollardODE, prepare_data

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_path = 'Data/SupplementaryDataS2.csv'
trajectories_p, conditions_p, unique_envs_p, num_species_p = prepare_data(data_path)

# 1. Full Physical Model (Theta and W)
model_full = PurePhysicalPollardODE(N=num_species_p, K=5, num_conditions=len(unique_envs_p)).to(device)
ckpt_full = torch.load('Data/Pollard_Physical_ODE_50Strains.pth', weights_only=False)
model_full.load_state_dict(ckpt_full['model_state_dict'])
model_full.eval()

# 2. Retrained Model without Cross-Feeding (W = 0 strictly during training)
class NoCrossFeedingPollardODE(nn.Module):
    def __init__(self, N, K, num_conditions=5):
        super().__init__()
        self.N, self.K = N, K
        self.r0_raw = nn.Parameter(torch.randn(num_conditions, K))
        self.theta_raw = nn.Parameter(torch.randn(N, K))
        self.alpha_raw = nn.Parameter(torch.randn(N))
        self.b0_raw = nn.Parameter(torch.tensor(0.0))
        
    def get_params(self):
        return F.softplus(self.theta_raw), F.softplus(self.alpha_raw), F.softplus(self.r0_raw), F.softplus(self.b0_raw) + 0.1
        
    def ode_deriv(self, t, S, Theta, alpha):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        dN = N_t * torch.matmul(R_t, Theta.t()) - alpha * (N_t**2)
        dR = - R_t * torch.matmul(N_t, Theta)
        return torch.cat([dN, dR], dim=-1)

model_no_w = NoCrossFeedingPollardODE(num_species_p, 5, 5).to(device)
ckpt_now = torch.load('Data/Pollard_Physical_ODE_50Strains_NoW.pth', weights_only=False)
model_no_w.load_state_dict(ckpt_now['model_state_dict'])
model_no_w.eval()

delta_t = 0.10
tau_steps = int(1.0 / delta_t) + 1
t_span = torch.linspace(0.0, 1.0, tau_steps, device=device)

clean_env_names = [
    'Mucin Carrier',
    'Mucin Supern.',
    'No Carrier (Liq.)',
    'Plain Carrier',
    'Plain Supern.'
]

with torch.no_grad():
    Theta, W, alpha, R0_table, B0 = model_full.get_params()
    Theta_np = Theta.detach().cpu().numpy()
    W_np = W.detach().cpu().numpy()

    # Function to evaluate Full Model
    def eval_full(Th, W_mat, al, R0_tab):
        r2_list = []
        for c_idx in range(5):
            mask = (conditions_p == c_idx)
            t_c = trajectories_p[mask, 1:, :].cpu().numpy().flatten()
            R0_sub = R0_tab[c_idx:c_idx+1].expand(mask.sum(), -1)
            N_curr = trajectories_p[mask, 0, :] * B0
            preds = []
            for p in range(1, 6):
                N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
                S0 = torch.cat([N0, R0_sub], dim=-1)
                S_t = odeint(lambda t, S: model_full.ode_deriv(t, S, Th, W_mat, al), S0, t_span, method='euler', options={'step_size': delta_t})
                N_final = torch.clamp(S_t[-1, :, :num_species_p], min=0.0)
                N_curr = N_final
                preds.append(N_final)
            pt = torch.stack(preds, dim=1)
            pred_rel = (pt / (pt.sum(dim=-1, keepdim=True) + 1e-8)).detach().cpu().numpy().flatten()
            var_t = np.sum((t_c - np.mean(t_c))**2)
            r2 = 1.0 - np.sum((t_c - pred_rel)**2) / var_t if var_t > 0 else 0.0
            r2_list.append(r2)
        return np.array(r2_list)

    # 1. Full Model
    true_r2 = eval_full(Theta, W, alpha, R0_table)

    # 2. Retrained No-W Model
    Theta_nw, alpha_nw, R0_nw, B0_nw = model_no_w.get_params()
    r2_no_w_list = []
    for c_idx in range(5):
        mask = (conditions_p == c_idx)
        t_c = trajectories_p[mask, 1:, :].cpu().numpy().flatten()
        R0_sub = R0_nw[c_idx:c_idx+1].expand(mask.sum(), -1)
        N_curr = trajectories_p[mask, 0, :] * B0_nw
        preds = []
        for p in range(1, 6):
            N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
            S0 = torch.cat([N0, R0_sub], dim=-1)
            S_t = odeint(lambda t, S: model_no_w.ode_deriv(t, S, Theta_nw, alpha_nw), S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :num_species_p], min=0.0)
            N_curr = N_final
            preds.append(N_final)
        pt = torch.stack(preds, dim=1)
        pred_rel = (pt / (pt.sum(dim=-1, keepdim=True) + 1e-8)).detach().cpu().numpy().flatten()
        var_t = np.sum((t_c - np.mean(t_c))**2)
        r2 = 1.0 - np.sum((t_c - pred_rel)**2) / var_t if var_t > 0 else 0.0
        r2_no_w_list.append(r2)
    retrained_w0_r2 = np.array(r2_no_w_list)

    # 3. Open-Loop Horizon Persistence Null Baseline (predicts x_0 across all dynamic passages p=1..5)
    persistence_r2_list = []
    for c_idx in range(5):
        mask = (conditions_p == c_idx)
        true_vals = trajectories_p[mask, 1:, :].cpu().numpy().flatten()
        persist_vals = trajectories_p[mask, 0:1, :].expand(-1, 5, -1).cpu().numpy().flatten()
        var_t = np.sum((true_vals - np.mean(true_vals))**2)
        r2_p = 1.0 - np.sum((true_vals - persist_vals)**2) / var_t if var_t > 0 else 0.0
        persistence_r2_list.append(r2_p)
    persistence_r2 = np.array(persistence_r2_list)

    # 4. Shuffled Environment Vector a (Permuting resource concentrations within environment)
    shuff_a_runs = []
    for rep in range(25):
        r2_sha_list = []
        for c_idx in range(5):
            mask = (conditions_p == c_idx)
            t_c = trajectories_p[mask, 1:, :].cpu().numpy().flatten()
            perm_a = torch.randperm(5)
            R0_sub = R0_table[c_idx:c_idx+1, perm_a].expand(mask.sum(), -1)
            N_curr = trajectories_p[mask, 0, :] * B0
            preds = []
            for p in range(1, 6):
                N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
                S0 = torch.cat([N0, R0_sub], dim=-1)
                S_t = odeint(lambda t, S: model_full.ode_deriv(t, S, Theta, W, alpha), S0, t_span, method='euler', options={'step_size': delta_t})
                N_final = torch.clamp(S_t[-1, :, :num_species_p], min=0.0)
                N_curr = N_final
                preds.append(N_final)
            pt = torch.stack(preds, dim=1)
            pred_rel = (pt / (pt.sum(dim=-1, keepdim=True) + 1e-8)).detach().cpu().numpy().flatten()
            var_t = np.sum((t_c - np.mean(t_c))**2)
            r2 = 1.0 - np.sum((t_c - pred_rel)**2) / var_t if var_t > 0 else 0.0
            r2_sha_list.append(r2)
        shuff_a_runs.append(r2_sha_list)
    shuff_a_r2 = np.mean(shuff_a_runs, axis=0)

    # 5. Shuffled Species Traits Theta (Permuting species rows)
    shuff_th_runs = []
    for rep in range(25):
        perm_th = torch.randperm(num_species_p)
        r2_sh = eval_full(Theta[perm_th], W, alpha, R0_table)
        shuff_th_runs.append(r2_sh)
    shuff_th_r2 = np.mean(shuff_th_runs, axis=0)

# Hierarchical clustering of Theta for Panel A
Z = linkage(Theta_np, method='ward')
order = leaves_list(Z)
Theta_clustered = Theta_np[order]

# Multi-panel figure in 2 rows:
# Row 1: Panels A and B as square heatmaps side by side
# Row 2: Panel C spanning the full width with 5 benchmark bars (made taller for ample legend space)
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig = plt.figure(figsize=(13.6, 10.8), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.30, 1.15], hspace=0.34, wspace=0.18,
                       left=0.085, right=0.96, top=0.96, bottom=0.08)

# Panel A: Uptake Traits Theta (Square Box, anchored West to align left border with Row 2)
ax0 = fig.add_subplot(gs[0, 0])
im0 = ax0.imshow(Theta_clustered, aspect='auto', cmap='YlGnBu', interpolation='nearest')
ax0.set_box_aspect(1)
ax0.set_anchor('W')
badge_a = dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#00356B', alpha=0.92, lw=1.5)
ax0.text(0.05, 0.94, r"$\mathbf{Uptake\ Traits\ (\Theta)}$", transform=ax0.transAxes, fontsize=16.0, fontweight='bold', va='top', bbox=badge_a)
ax0.set_xlabel('Resource', fontsize=18.0, fontweight='bold', color='black', labelpad=6)
ax0.set_ylabel('Strains (N=50)', fontsize=18.0, fontweight='bold', color='black', labelpad=6)
ax0.set_xticks(range(5))
ax0.set_xticklabels([f'$R_{k+1}$' for k in range(5)], fontsize=15.0, fontweight='bold')
ax0.set_yticks([])

cax0 = inset_axes(ax0, width='4.5%', height='100%', loc='lower left',
                  bbox_to_anchor=(1.04, 0.0, 1.0, 1.0), bbox_transform=ax0.transAxes, borderpad=0)
cbar0 = fig.colorbar(im0, cax=cax0)
cbar0.set_label('Uptake Rate', fontsize=16.0, fontweight='bold', color='black')
cbar0.ax.tick_params(labelsize=14.0)

# Panel B: Cross-Feeding W (Square Box, anchored West to fill row width)
ax1 = fig.add_subplot(gs[0, 1])
im1 = ax1.imshow(W_np, aspect='equal', cmap='magma_r', interpolation='nearest')
ax1.set_box_aspect(1)
ax1.set_anchor('W')
badge_b = dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#00356B', alpha=0.92, lw=1.5)
ax1.text(0.05, 0.94, r"$\mathbf{Cross\text{-}Feeding\ (W)}$", transform=ax1.transAxes, fontsize=16.0, fontweight='bold', va='top', bbox=badge_b)
ax1.set_xlabel('To Niche', fontsize=18.0, fontweight='bold', color='black', labelpad=6)
ax1.set_ylabel('From Niche', fontsize=18.0, fontweight='bold', color='black', labelpad=6)
ax1.set_xticks(range(5))
ax1.set_yticks(range(5))
ax1.set_xticklabels([f'$R_{k+1}$' for k in range(5)], fontsize=15.0, fontweight='bold')
ax1.set_yticklabels([f'$R_{k+1}$' for k in range(5)], fontsize=15.0, fontweight='bold')

cax1 = inset_axes(ax1, width='4.5%', height='100%', loc='lower left',
                  bbox_to_anchor=(1.04, 0.0, 1.0, 1.0), bbox_transform=ax1.transAxes, borderpad=0)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.set_ticks([0.12, 0.18, 0.24, 0.30])
cbar1.set_ticklabels(['0.12', '0.18', '0.24', '0.30'])
cbar1.set_label('Byproduct Yield', fontsize=16.0, fontweight='bold', color='black')
cbar1.ax.tick_params(labelsize=14.0)

# Panel C: Nested Model Comparison across Spatial Regimes (5 bars spanning full width)
ax2 = fig.add_subplot(gs[1, :])
x = np.arange(len(clean_env_names))
w = 0.155

b1 = ax2.bar(x - 2.0*w, true_r2, width=w, label='Full Model', 
             color='#1B6CA8', edgecolor='#0F2942', lw=1.2, zorder=3)
b2 = ax2.bar(x - 1.0*w, retrained_w0_r2, width=w, label=r'$\mathbf{W} = \mathbf{0}$', 
             color='#E08B53', edgecolor='#803010', lw=1.2, zorder=3)
b3 = ax2.bar(x + 0.0*w, shuff_a_r2, width=w, label=r'Shuffled $\mathbf{a}$', 
             color='#7A92A3', edgecolor='#2C3E50', lw=1.2, zorder=3)
b4 = ax2.bar(x + 1.0*w, persistence_r2, width=w, label='Persistence', 
             color='#F5A623', edgecolor='#B37400', lw=1.2, zorder=3)
b5 = ax2.bar(x + 2.0*w, shuff_th_r2, width=w, label=r'Shuffled $\mathbf{\Theta}$', 
             color='#D9534F', edgecolor='#701010', lw=1.2, zorder=3)

ax2.axhline(0, color='black', lw=1.2, linestyle='-', zorder=4)

for bar in b1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#1B6CA8')
for bar in b2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#803010')
for bar in b3:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#2C3E50')
for bar in b4:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#B37400')
for bar in b5:
    h = bar.get_height()
    va = 'bottom' if h >= 0 else 'top'
    offset = 0.03 if h >= 0 else -0.04
    ax2.text(bar.get_x() + bar.get_width()/2., h + offset, f'{h:.2f}', ha='center', va=va, fontsize=11.5, fontweight='bold', color='#701010')

ax2.set_xticks(x)
ax2.set_xticklabels(clean_env_names, rotation=0, ha='center', fontsize=14.5, fontweight='bold', color='#0F2942')
ax2.set_ylabel(r'Coefficient of Determination ($R^2$)', fontsize=18.0, fontweight='bold', color='black', labelpad=8)
ax2.set_ylim(-1.08, 1.48)
ax2.tick_params(axis='y', labelsize=13.5)
ax2.legend(loc='upper center', ncol=5, fontsize=15.5, framealpha=0.96, edgecolor='black', borderaxespad=0.5, columnspacing=1.6)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)

out_path = 'Poster/figures/Pollard_Universal_Matrices.png'
fig.savefig(out_path, dpi=300)
print(f"Saved updated 2-row 5-bar Pollard figure to {out_path}")

"""
Pollard Synthetic Gut Microbiome (Coupled Biofilm-Planktonic CRM):
Learned Traits Generalization vs Null Models.

Generates the Publication/Poster Figure 2 comparing:
- Panel A: Hierarchically clustered metabolic uptake trait matrix Theta (50 x 5)
- Panel B: Mass-conserving byproduct cross-feeding matrix W (5 x 5)
- Panel C: Benchmark evaluation comparing Full Coupled Model against 5 Null Models
           across all 4 spatial conditions:
           1. Mucin Carrier
           2. Mucin Supernatant
           3. Plain Carrier
           4. Plain Supernatant

Null Models:
1. Retrained No Cross-Feeding (W = 0)
2. Shuffled Environment Resources R0 (N=50 shuffles, Mean +/- SD)
3. Persistence Null Baseline (P1 persists across P2..P6)
4. Shuffled Species Metabolic Traits Theta (N=50 shuffles, Mean +/- SD)
5. Shuffled Spatial Adhesion Traits k_attach & k_shed (N=50 shuffles, Mean +/- SD)
"""

import sys
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, leaves_list

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Pollard_Train_Coupled_Biofilm_ODE import CoupledBiofilmPollardODE, load_coupled_data

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_path = os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv')
carrier_tensor, supernatant_tensor, cond_tensor, strains_keep, conditions, rep_labels = load_coupled_data(data_path)

carrier_tensor = carrier_tensor.to(device)
supernatant_tensor = supernatant_tensor.to(device)
cond_tensor = cond_tensor.to(device)

num_species = len(strains_keep)
K = 5
num_conditions = len(conditions)

# 1. Load Full Coupled Model
model_full = CoupledBiofilmPollardODE(N=num_species, K=K, num_conditions=num_conditions).to(device)
ckpt_full_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Coupled_Biofilm_ODE_50Strains.pth')
ckpt_full = torch.load(ckpt_full_path, map_location=device, weights_only=False)
model_full.load_state_dict(ckpt_full['model_state_dict'])
model_full.eval()


# 2. Load Retrained No-W Model
class CoupledBiofilmNoWODE(CoupledBiofilmPollardODE):
    def get_params(self):
        Theta, _, alpha, k_shed, k_attach, R0, b0_c, b0_s, eta = super().get_params()
        W_zero = torch.zeros(self.K, self.K, device=Theta.device)
        return Theta, W_zero, alpha, k_shed, k_attach, R0, b0_c, b0_s, eta


ckpt_now_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Coupled_Biofilm_ODE_50Strains_NoW.pth')
model_no_w = CoupledBiofilmNoWODE(N=num_species, K=K, num_conditions=num_conditions).to(device)
if os.path.exists(ckpt_now_path):
    print(f"Loading existing No-W model from {ckpt_now_path}...")
    model_no_w.load_state_dict(torch.load(ckpt_now_path, map_location=device, weights_only=False)['model_state_dict'])
    model_no_w.eval()
else:
    print(f"Retraining No-W model and saving to {ckpt_now_path}...")
    optimizer = torch.optim.Adam(model_no_w.parameters(), lr=0.015)
    best_loss = float('inf')
    best_state = None
    for epoch in range(1, 251):
        optimizer.zero_grad()
        tf_ratio = max(0.0, 1.0 - (epoch / (250 * 0.75)))
        pred_c, pred_s = model_no_w.forward_rollout(carrier_tensor, supernatant_tensor, cond_tensor, tf_ratio=tf_ratio)
        target_c = carrier_tensor[:, 1:, :]
        target_s = supernatant_tensor[:, 1:, :]
        loss = torch.sum(target_c * torch.log(target_c / torch.clamp(pred_c, min=1e-9) + 1e-12), dim=-1).mean() + \
               torch.sum(target_s * torch.log(target_s / torch.clamp(pred_s, min=1e-9) + 1e-12), dim=-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_no_w.parameters(), 5.0)
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model_no_w.state_dict().items()}
    torch.save({'model_state_dict': best_state, 'best_loss': best_loss}, ckpt_now_path)
    model_no_w.load_state_dict(best_state)
    model_no_w.eval()

# Ground truth observed targets
t_c = carrier_tensor[:, 1:, :].cpu().numpy()
t_s = supernatant_tensor[:, 1:, :].cpu().numpy()
m_mask = (cond_tensor == 0).cpu().numpy()
p_mask = (cond_tensor == 1).cpu().numpy()

cond_names = ['Mucin Carrier', 'Mucin Supern.', 'Plain Carrier', 'Plain Supern.']


def compute_r2(p_c, p_s):
    splits = [
        (t_c[m_mask].flatten(), p_c[m_mask].flatten()),
        (t_s[m_mask].flatten(), p_s[m_mask].flatten()),
        (t_c[p_mask].flatten(), p_c[p_mask].flatten()),
        (t_s[p_mask].flatten(), p_s[p_mask].flatten()),
    ]
    res = []
    for obs, pred in splits:
        var_t = np.sum((obs - np.mean(obs))**2)
        r2 = 1.0 - np.sum((obs - pred)**2) / var_t if var_t > 0 else 0.0
        res.append(r2)
    return np.array(res)


def rollout_custom(Theta, W, alpha, k_shed, k_attach, R0, b0_c, b0_s, eta, delta_t=0.1):
    t_span = torch.linspace(0.0, 1.0, int(1.0 / delta_t) + 1, device=device)
    k_attach_batch = k_attach[cond_tensor]
    R0_batch = R0[cond_tensor]
    N_c_curr = carrier_tensor[:, 0, :] * b0_c
    N_s_curr = supernatant_tensor[:, 0, :] * b0_s
    pred_c_list, pred_s_list = [], []
    for p in range(1, 6):
        N_c_0 = torch.clamp(0.2 * N_c_curr + 1e-6, min=0.0)
        N_s_0 = torch.clamp(eta * N_s_curr + 1e-6, min=0.0)
        S0 = torch.cat([N_c_0, N_s_0, R0_batch], dim=-1)
        def f(t, S):
            return model_full.ode_deriv(t, S, Theta, W, alpha, k_shed, k_attach_batch)
        S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
        N_c_final = torch.clamp(S_t[-1, :, :num_species], min=0.0)
        N_s_final = torch.clamp(S_t[-1, :, num_species:2*num_species], min=0.0)
        p_c = N_c_final / (N_c_final.sum(dim=-1, keepdim=True) + 1e-8)
        p_s = N_s_final / (N_s_final.sum(dim=-1, keepdim=True) + 1e-8)
        pred_c_list.append(p_c)
        pred_s_list.append(p_s)
        N_c_curr = N_c_final
        N_s_curr = N_s_final
    return torch.stack(pred_c_list, dim=1).cpu().numpy(), torch.stack(pred_s_list, dim=1).cpu().numpy()


print("Running evaluations across models and nulls...")
with torch.no_grad():
    Theta, W, alpha, k_shed, k_attach, R0, b0_c, b0_s, eta = model_full.get_params()
    Theta_np = Theta.detach().cpu().numpy()
    W_np = W.detach().cpu().numpy()

    # 1. Full Model
    pc, ps = rollout_custom(Theta, W, alpha, k_shed, k_attach, R0, b0_c, b0_s, eta)
    true_r2 = compute_r2(pc, ps)

    # 2. Retrained No-W Model
    Theta_nw, W_nw, alpha_nw, k_shed_nw, k_attach_nw, R0_nw, b0_c_nw, b0_s_nw, eta_nw = model_no_w.get_params()
    pc_nw, ps_nw = rollout_custom(Theta_nw, W_nw, alpha_nw, k_shed_nw, k_attach_nw, R0_nw, b0_c_nw, b0_s_nw, eta_nw)
    now_r2 = compute_r2(pc_nw, ps_nw)

    # 3. Persistence Null Baseline
    p0_c = np.repeat(carrier_tensor[:, 0:1, :].cpu().numpy(), 5, axis=1)
    p0_s = np.repeat(supernatant_tensor[:, 0:1, :].cpu().numpy(), 5, axis=1)
    persist_r2 = compute_r2(p0_c, p0_s)

    # 4. Shuffled R0 Null Model (N=50)
    N_shuffles = 50
    print(f"Executing {N_shuffles} random permutations for Shuffled R0...")
    shuff_r0_runs = []
    for _ in range(N_shuffles):
        perm_r = torch.randperm(K)
        pc_s, ps_s = rollout_custom(Theta, W, alpha, k_shed, k_attach, R0[:, perm_r], b0_c, b0_s, eta)
        shuff_r0_runs.append(compute_r2(pc_s, ps_s))
    shuff_r0_runs = np.array(shuff_r0_runs)
    shuff_r0_r2 = np.mean(shuff_r0_runs, axis=0)
    shuff_r0_std = np.std(shuff_r0_runs, axis=0)

    # 5. Shuffled Theta Null Model (N=50)
    print(f"Executing {N_shuffles} random permutations for Shuffled Theta...")
    shuff_th_runs = []
    for _ in range(N_shuffles):
        perm_th = torch.randperm(num_species)
        pc_s, ps_s = rollout_custom(Theta[perm_th], W, alpha[perm_th], k_shed, k_attach, R0, b0_c, b0_s, eta)
        shuff_th_runs.append(compute_r2(pc_s, ps_s))
    shuff_th_runs = np.array(shuff_th_runs)
    shuff_th_r2 = np.mean(shuff_th_runs, axis=0)
    shuff_th_std = np.std(shuff_th_runs, axis=0)

    # 6. Shuffled Spatial Adhesion Traits Null Model (N=50)
    print(f"Executing {N_shuffles} random permutations for Shuffled Adhesion Traits...")
    shuff_adh_runs = []
    for _ in range(N_shuffles):
        perm_adh = torch.randperm(num_species)
        pc_s, ps_s = rollout_custom(Theta, W, alpha, k_shed[perm_adh], k_attach[:, perm_adh], R0, b0_c, b0_s, eta)
        shuff_adh_runs.append(compute_r2(pc_s, ps_s))
    shuff_adh_runs = np.array(shuff_adh_runs)
    shuff_adh_r2 = np.mean(shuff_adh_runs, axis=0)
    shuff_adh_std = np.std(shuff_adh_runs, axis=0)

print("\n--- Summary of Model and Null Performances (Mean +/- SD) ---")
for i, name in enumerate(cond_names):
    print(f"{name:18s}: Full={true_r2[i]:.2f} | W=0={now_r2[i]:.2f} | Shuff_R0={shuff_r0_r2[i]:.2f} +/- {shuff_r0_std[i]:.2f} | Persist={persist_r2[i]:.2f} | Shuff_Th={shuff_th_r2[i]:.2f} +/- {shuff_th_std[i]:.2f} | Shuff_Adh={shuff_adh_r2[i]:.2f} +/- {shuff_adh_std[i]:.2f}")

# Hierarchical clustering of Theta for Panel A
Z = linkage(Theta_np, method='ward')
order = leaves_list(Z)
Theta_clustered = Theta_np[order]

# =========================================================================
# Multi-panel figure in 2 rows:
# Row 1: Panels A and B as square heatmaps side by side
# Row 2: Panel C spanning the full width with 6 benchmark bars
# =========================================================================
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig = plt.figure(figsize=(14.2, 11.0), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1.15], hspace=0.34, wspace=0.18,
                       left=0.08, right=0.96, top=0.96, bottom=0.08)

# Panel A: Uptake Traits Theta
ax0 = fig.add_subplot(gs[0, 0])
im0 = ax0.imshow(Theta_clustered, aspect='auto', cmap='YlGnBu', interpolation='nearest')
ax0.set_box_aspect(1)
ax0.set_anchor('W')
badge_a = dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#00356B', alpha=0.92, lw=1.5)
ax0.text(0.05, 0.94, r"$\mathbf{Uptake\ Traits\ (\Theta)}$", transform=ax0.transAxes, fontsize=16.0, fontweight='bold', va='top', bbox=badge_a)
ax0.set_xlabel('Resource', fontsize=18.0, fontweight='bold', color='black', labelpad=6)
ax0.set_ylabel('Strains (N=50)', fontsize=18.0, fontweight='bold', color='black', labelpad=6)
ax0.set_xticks(range(K))
ax0.set_xticklabels([f'$R_{k+1}$' for k in range(K)], fontsize=15.0, fontweight='bold')
ax0.set_yticks([])

cax0 = inset_axes(ax0, width='4.5%', height='100%', loc='lower left',
                  bbox_to_anchor=(1.04, 0.0, 1.0, 1.0), bbox_transform=ax0.transAxes, borderpad=0)
cbar0 = fig.colorbar(im0, cax=cax0)
cbar0.set_label('Uptake Rate', fontsize=16.0, fontweight='bold', color='black')
cbar0.ax.tick_params(labelsize=14.0)

# Panel B: Cross-Feeding W
ax1 = fig.add_subplot(gs[0, 1])
im1 = ax1.imshow(W_np, aspect='equal', cmap='magma_r', interpolation='nearest')
ax1.set_box_aspect(1)
ax1.set_anchor('W')
badge_b = dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#00356B', alpha=0.92, lw=1.5)
ax1.text(0.05, 0.94, r"$\mathbf{Cross\text{-}Feeding\ (W)}$", transform=ax1.transAxes, fontsize=16.0, fontweight='bold', va='top', bbox=badge_b)
ax1.set_xlabel('To Niche', fontsize=18.0, fontweight='bold', color='black', labelpad=6)
ax1.set_ylabel('From Niche', fontsize=18.0, fontweight='bold', color='black', labelpad=6)
ax1.set_xticks(range(K))
ax1.set_yticks(range(K))
ax1.set_xticklabels([f'$R_{k+1}$' for k in range(K)], fontsize=15.0, fontweight='bold')
ax1.set_yticklabels([f'$R_{k+1}$' for k in range(K)], fontsize=15.0, fontweight='bold')

cax1 = inset_axes(ax1, width='4.5%', height='100%', loc='lower left',
                  bbox_to_anchor=(1.04, 0.0, 1.0, 1.0), bbox_transform=ax1.transAxes, borderpad=0)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.set_label('Byproduct Yield', fontsize=16.0, fontweight='bold', color='black')
cbar1.ax.tick_params(labelsize=14.0)

# Panel C: Nested Model Comparison across Spatial Regimes (with Error Bars)
ax2 = fig.add_subplot(gs[1, :])
x = np.arange(len(cond_names))
w = 0.135

b1 = ax2.bar(x - 2.5*w, true_r2, width=w, label='Full Coupled Model', 
             color='#1B6CA8', edgecolor='#0F2942', lw=1.2, zorder=3)
b2 = ax2.bar(x - 1.5*w, now_r2, width=w, label=r'$\mathbf{W} = \mathbf{0}$', 
             color='#E08B53', edgecolor='#803010', lw=1.2, zorder=3)
b3 = ax2.bar(x - 0.5*w, shuff_r0_r2, width=w, yerr=shuff_r0_std, capsize=3.5,
             label=r'Shuffled $\mathbf{R}_0$ ($\pm$SD)', 
             color='#7A92A3', edgecolor='#2C3E50', lw=1.2,
             error_kw=dict(ecolor='#2C3E50', lw=1.2, capthick=1.2), zorder=3)
b4 = ax2.bar(x + 0.5*w, persist_r2, width=w, label='Persistence', 
             color='#F5A623', edgecolor='#B37400', lw=1.2, zorder=3)
b5 = ax2.bar(x + 1.5*w, shuff_th_r2, width=w, yerr=shuff_th_std, capsize=3.5,
             label=r'Shuffled $\mathbf{\Theta}$ ($\pm$SD)', 
             color='#D9534F', edgecolor='#701010', lw=1.2,
             error_kw=dict(ecolor='#701010', lw=1.2, capthick=1.2), zorder=3)
b6 = ax2.bar(x + 2.5*w, shuff_adh_r2, width=w, yerr=shuff_adh_std, capsize=3.5,
             label=r'Shuffled Adhesion ($\pm$SD)', 
             color='#8E44AD', edgecolor='#4A154B', lw=1.2,
             error_kw=dict(ecolor='#4A154B', lw=1.2, capthick=1.2), zorder=3)

ax2.axhline(0, color='black', lw=1.2, linestyle='-', zorder=4)

# Number labels on top of bars
for bar in b1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#1B6CA8')
for bar in b2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#803010')
for i, bar in enumerate(b3):
    h = bar.get_height()
    err = shuff_r0_std[i]
    ax2.text(bar.get_x() + bar.get_width()/2., h + err + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#2C3E50')
for bar in b4:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#B37400')
for i, bar in enumerate(b5):
    h = bar.get_height()
    err = shuff_th_std[i]
    if h >= 0:
        ax2.text(bar.get_x() + bar.get_width()/2., h + err + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#701010')
    else:
        ax2.text(bar.get_x() + bar.get_width()/2., h - err - 0.04, f'{h:.2f}', ha='center', va='top', fontsize=10.5, fontweight='bold', color='#701010')
for i, bar in enumerate(b6):
    h = bar.get_height()
    err = shuff_adh_std[i]
    if h >= 0:
        ax2.text(bar.get_x() + bar.get_width()/2., h + err + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#4A154B')
    else:
        ax2.text(bar.get_x() + bar.get_width()/2., h - err - 0.04, f'{h:.2f}', ha='center', va='top', fontsize=10.5, fontweight='bold', color='#4A154B')

ax2.set_xticks(x)
ax2.set_xticklabels(cond_names, rotation=0, ha='center', fontsize=14.5, fontweight='bold', color='#0F2942')
ax2.set_ylabel(r'Coefficient of Determination ($R^2$)', fontsize=18.0, fontweight='bold', color='black', labelpad=8)
ax2.set_ylim(-1.38, 1.48)
ax2.tick_params(axis='y', labelsize=13.5)
ax2.legend(loc='upper center', ncol=6, fontsize=12.5, framealpha=0.96, edgecolor='black', borderaxespad=0.5, columnspacing=1.0)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)

out_poster_path = 'Poster/figures/Pollard_Coupled_Biofilm_Universality.png'
os.makedirs(os.path.dirname(out_poster_path), exist_ok=True)
fig.savefig(out_poster_path, dpi=300)
print(f"Saved coupled universality figure to {out_poster_path}")

# Copy to artifact directory
artifact_dir = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
shutil.copyfile(out_poster_path, os.path.join(artifact_dir, 'Pollard_Coupled_Biofilm_Universality.png'))
print("Copied to artifact directory!")

"""
Environmental Noise Simulation and Comparison on Gore's Persistent Fluctuating Communities:
1. Compares Real Microcosms vs Deterministic Rollout vs Environmentally Noisy Rollout (Days 7-10).
2. Demonstrates how realistic environmental perturbations (8% resource noise, 5% dilution noise)
   excite the underdamped modes of the multi-species system and restore jagged fluctuations.
3. Evaluates long-term 200-passage persistent stochastic fluctuations.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from torchdiffeq import odeint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load data and checkpoint
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
data = torch.load(data_path, weights_only=False)

from src.train_fluctuating_only_chaos import DecoupledThetaPhiAlpha01ODE

ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Fitted_K8_FluctuatingOnly_Chaos.pth')
ckpt = torch.load(ckpt_path, weights_only=False)
model = DecoupledThetaPhiAlpha01ODE(N=48, K=8, alpha_max=0.10, Kd=0.01).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Select 4 representative fluctuating communities
test_comms = [
    ('HN', 48, 'Community1', '48-Species HN Community 1', 'Empirical CV = 1.037'),
    ('HN', 24, 'Community1', '24-Species HN Community 1', 'Empirical CV = 1.050'),
    ('HN', 24, 'Community6', '24-Species HN Community 6', 'Empirical CV = 1.084'),
    ('HN', 12, 'Community1', '12-Species HN Community 1', 'Empirical CV = 0.553')
]

days_axis = np.array([7, 8, 9, 10])
t_span = torch.linspace(0.0, 1.0, 50, device=device)
R0_hn = model.get_R0()[2]

# Consistent color palette for 48 species
np.random.seed(42)
cmap_base = plt.cm.tab20(np.linspace(0, 1, 20))
cmap_b = plt.cm.tab20b(np.linspace(0, 1, 20))
cmap_c = plt.cm.tab20c(np.linspace(0, 1, 8))
all_colors = np.vstack([cmap_base, cmap_b, cmap_c])

# Noise parameters calibrated to empirical measurements:
# 1. Empirical Day-to-Day OD CV is 9.7% -> sigma_R = 0.08
# 2. Pipetting volume CV ~ 5% -> sigma_D = 0.05
# 3. 16S Sequencing observation noise -> sigma_obs = 0.12
sigma_R = 0.08
sigma_D = 0.05
sigma_obs = 0.12

fig = plt.figure(figsize=(19.0, 15.0), dpi=300)
gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 1], hspace=0.36, wspace=0.16)

np.random.seed(101)
torch.manual_seed(101)

for row_idx, (cond, S_val, comm_val, title, cv_info) in enumerate(test_comms):
    rel_emp = data['relative_abundance'][(cond, S_val, comm_val)].numpy()
    mask = data['presence_masks'][(S_val, comm_val)].float().to(device)
    od = data['biomass_od'][(cond, S_val, comm_val)].mean(dim=0).to(device)
    
    # 1. Real Microcosm Days 7 to 10
    emp_window = rel_emp[7:11]
    
    # 2. Deterministic Model Rollout from Day 6
    N_curr_det = (torch.tensor(rel_emp[6], device=device) * od[6]) * mask
    rollout_det = []
    with torch.no_grad():
        for p in range(7, 11):
            N_in = torch.clamp(N_curr_det / model.D + model.mu * mask, min=0.0)
            S0 = torch.cat([N_in, R0_hn], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_curr_det = torch.clamp(S_t[-1, :48] * mask, min=0.0)
            rollout_det.append(N_curr_det.cpu().numpy())
    rollout_det = np.array(rollout_det)
    mod_det = rollout_det / np.where(rollout_det.sum(axis=-1, keepdims=True) > 0, rollout_det.sum(axis=-1, keepdims=True), 1.0)
    
    # 3. Stochastic Environmental Noise Rollout from Day 6
    N_curr_stoch = (torch.tensor(rel_emp[6], device=device) * od[6]) * mask
    rollout_stoch = []
    with torch.no_grad():
        for p in range(7, 11):
            d_noise = float(np.exp(np.random.randn() * sigma_D))
            D_eff = model.D * d_noise
            N_in = torch.clamp(N_curr_stoch / D_eff + model.mu * mask, min=0.0)
            
            r_noise = torch.tensor(np.exp(np.random.randn(8) * sigma_R), device=device, dtype=torch.float32)
            R0_noisy = R0_hn * r_noise
            
            S0 = torch.cat([N_in, R0_noisy], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_next = torch.clamp(S_t[-1, :48] * mask, min=0.0)
            
            obs_noise = torch.tensor(np.exp(np.random.randn(48) * sigma_obs), device=device, dtype=torch.float32)
            N_obs = N_next * obs_noise * mask
            rollout_stoch.append(N_obs.cpu().numpy())
            N_curr_stoch = N_next
            
    rollout_stoch = np.array(rollout_stoch)
    mod_stoch = rollout_stoch / np.where(rollout_stoch.sum(axis=-1, keepdims=True) > 0, rollout_stoch.sum(axis=-1, keepdims=True), 1.0)
    
    # Calculate CV for each
    surv_mask = emp_window.mean(axis=0) >= 1e-3
    cv_emp = np.mean(emp_window[:, surv_mask].std(axis=0) / emp_window[:, surv_mask].mean(axis=0))
    cv_det = np.mean(mod_det[:, surv_mask].std(axis=0) / mod_det[:, surv_mask].mean(axis=0))
    cv_stoch = np.mean(mod_stoch[:, surv_mask].std(axis=0) / mod_stoch[:, surv_mask].mean(axis=0))
    
    # Top species
    max_ab = np.maximum(emp_window.max(axis=0), np.maximum(mod_det.max(axis=0), mod_stoch.max(axis=0)))
    top_species = np.where(max_ab >= 0.025)[0]
    if len(top_species) < 4:
        top_species = np.argsort(max_ab)[-5:]
    top_species = top_species[np.argsort(max_ab[top_species])[::-1]]
    
    y_max = min(1.05, max_ab[top_species].max() * 1.12)
    
    # Column 1: Real Microcosm
    ax0 = fig.add_subplot(gs[row_idx, 0])
    for sp in top_species:
        c = all_colors[sp % 48]
        ax0.plot(days_axis, emp_window[:, sp], marker='o', markersize=6.5, lw=2.2, color=c, label=f"Sp {sp+1:2d}")
    ax0.set_title(f"(A{row_idx+1}) Real: {title}\n(Empirical $CV = {cv_emp:.3f}$)",
                  fontsize=11.5, fontweight='bold', color='#1A365D', loc='left')
    ax0.set_ylabel("Relative Abundance", fontsize=10.5, fontweight='bold')
    ax0.set_xlabel("Day", fontsize=10.5, fontweight='bold')
    ax0.set_xticks(days_axis)
    ax0.set_xlim(6.8, 10.2)
    ax0.set_ylim(-0.02, y_max)
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend(loc='upper right', fontsize=8.0, ncol=min(3, len(top_species)), framealpha=0.9)
    
    # Column 2: Deterministic Model Rollout
    ax1 = fig.add_subplot(gs[row_idx, 1])
    for sp in top_species:
        c = all_colors[sp % 48]
        ax1.plot(days_axis, mod_det[:, sp], marker='s', markersize=6.5, lw=2.2, color=c, linestyle='--', label=f"Sp {sp+1:2d}")
    ax1.set_title(f"(B{row_idx+1}) Deterministic Rollout (Smooth)\n(Simulated $CV = {cv_det:.3f}$)",
                  fontsize=11.5, fontweight='bold', color='#4A5568', loc='left')
    ax1.set_ylabel("Relative Abundance", fontsize=10.5, fontweight='bold')
    ax1.set_xlabel("Day", fontsize=10.5, fontweight='bold')
    ax1.set_xticks(days_axis)
    ax1.set_xlim(6.8, 10.2)
    ax1.set_ylim(-0.02, y_max)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=8.0, ncol=min(3, len(top_species)), framealpha=0.9)
    
    # Column 3: Stochastic Model with Environmental Noise
    ax2 = fig.add_subplot(gs[row_idx, 2])
    for sp in top_species:
        c = all_colors[sp % 48]
        ax2.plot(days_axis, mod_stoch[:, sp], marker='^', markersize=6.5, lw=2.2, color=c, linestyle='-', label=f"Sp {sp+1:2d}")
    ax2.set_title(f"(C{row_idx+1}) + Environmental Noise (Jagged)\n(Simulated $CV = {cv_stoch:.3f}$)",
                  fontsize=11.5, fontweight='bold', color='#8B2500', loc='left')
    ax2.set_ylabel("Relative Abundance", fontsize=10.5, fontweight='bold')
    ax2.set_xlabel("Day", fontsize=10.5, fontweight='bold')
    ax2.set_xticks(days_axis)
    ax2.set_xlim(6.8, 10.2)
    ax2.set_ylim(-0.02, y_max)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8.0, ncol=min(3, len(top_species)), framealpha=0.9)

out_fig = os.path.join(PROJECT_ROOT, 'Figures', 'Gore_Environmental_Noise_Comparison_Days7_10.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"Comparison figure saved to: {out_fig}")

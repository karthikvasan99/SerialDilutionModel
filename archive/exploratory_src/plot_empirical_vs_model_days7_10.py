"""
Plot Side-by-Side Trajectories: Empirical Data vs. Consumer-Resource Model Simulation
for the Days 7 to 10 Window on Gore's Persistent Fluctuating Communities.
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
    ('HN', 48, 'Community1', '48-Species Community 1 (HN)', 'Empirical CV = 1.037'),
    ('HN', 24, 'Community1', '24-Species Community 1 (HN)', 'Empirical CV = 1.050'),
    ('HN', 24, 'Community6', '24-Species Community 6 (HN)', 'Empirical CV = 1.084'),
    ('HN', 12, 'Community1', '12-Species Community 1 (HN)', 'Empirical CV = 0.553')
]

days_axis = np.array([7, 8, 9, 10])
t_span = torch.linspace(0.0, 1.0, 50, device=device)
R0_hn = model.get_R0()[2]

# Prepare consistent color map for 48 species
np.random.seed(42)
cmap_base = plt.cm.tab20(np.linspace(0, 1, 20))
cmap_b = plt.cm.tab20b(np.linspace(0, 1, 20))
cmap_c = plt.cm.tab20c(np.linspace(0, 1, 8))
all_colors = np.vstack([cmap_base, cmap_b, cmap_c])

fig = plt.figure(figsize=(15.5, 14.5), dpi=300)
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.38, wspace=0.18)

for row_idx, (cond, S_val, comm_val, title, cv_info) in enumerate(test_comms):
    rel_emp = data['relative_abundance'][(cond, S_val, comm_val)].numpy()
    mask = data['presence_masks'][(S_val, comm_val)].float().to(device)
    od = data['biomass_od'][(cond, S_val, comm_val)].mean(dim=0).to(device)
    
    # 1. Empirical Days 7 to 10
    emp_window = rel_emp[7:11] # shape (4, 48)
    
    # 2. Model Autonomous Rollout from Day 6
    N_curr = (torch.tensor(rel_emp[6], device=device) * od[6]) * mask
    rollout = []
    with torch.no_grad():
        for p in range(7, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * mask, min=0.0)
            S0 = torch.cat([N_in, R0_hn], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_curr = torch.clamp(S_t[-1, :48] * mask, min=0.0)
            rollout.append(N_curr.cpu().numpy())
            
    rollout = np.array(rollout) # (4, 48)
    mod_window = rollout / np.where(rollout.sum(axis=-1, keepdims=True) > 0, rollout.sum(axis=-1, keepdims=True), 1.0)
    
    # Compute correlation across all present species in this window
    mask_np = mask.cpu().numpy() > 0
    y_emp_flat = emp_window[:, mask_np].flatten()
    y_mod_flat = mod_window[:, mask_np].flatten()
    r_val, _ = pearsonr(y_emp_flat, y_mod_flat)
    r2_val = r_val ** 2
    
    # Identify top species in this window to plot with individual labels
    max_ab = np.maximum(emp_window.max(axis=0), mod_window.max(axis=0))
    top_species = np.where(max_ab >= 0.025)[0]
    # If too few, take top 5
    if len(top_species) < 4:
        top_species = np.argsort(max_ab)[-5:]
    # Sort by abundance descending
    top_species = top_species[np.argsort(max_ab[top_species])[::-1]]
    
    # Left subplot: Real Experimental Microcosm
    ax_emp = fig.add_subplot(gs[row_idx, 0])
    for sp in top_species:
        c = all_colors[sp % 48]
        ax_emp.plot(days_axis, emp_window[:, sp], marker='o', markersize=6.5, lw=2.2, color=c,
                    label=f"Sp {sp+1:2d}")
    ax_emp.set_title(f"Real Microcosm: {title}\n({cv_info})", fontsize=11.5, fontweight='bold', color='#1A365D', loc='left')
    ax_emp.set_ylabel("Relative Abundance", fontsize=10.5, fontweight='bold')
    ax_emp.set_xlabel("Day", fontsize=10.5, fontweight='bold')
    ax_emp.set_xticks(days_axis)
    ax_emp.set_xlim(6.8, 10.2)
    ax_emp.grid(True, linestyle='--', alpha=0.5)
    ax_emp.legend(loc='upper right', fontsize=8.5, ncol=min(3, len(top_species)), framealpha=0.9)
    
    # Determine common y-axis limit
    y_max = max(emp_window[:, top_species].max(), mod_window[:, top_species].max()) * 1.12
    ax_emp.set_ylim(-0.02, min(1.05, y_max))
    
    # Right subplot: Model Simulation (Autonomous Rollout)
    ax_mod = fig.add_subplot(gs[row_idx, 1])
    for sp in top_species:
        c = all_colors[sp % 48]
        ax_mod.plot(days_axis, mod_window[:, sp], marker='s', markersize=6.5, lw=2.2, color=c,
                    linestyle='--', label=f"Sp {sp+1:2d}")
    ax_mod.set_title(f"Model Simulation: {title}\n(Rollout $r = {r_val:.3f}$, $R^2 = {r2_val*100:.1f}\\%$)",
                     fontsize=11.5, fontweight='bold', color='#8B2500', loc='left')
    ax_mod.set_ylabel("Relative Abundance", fontsize=10.5, fontweight='bold')
    ax_mod.set_xlabel("Day", fontsize=10.5, fontweight='bold')
    ax_mod.set_xticks(days_axis)
    ax_mod.set_xlim(6.8, 10.2)
    ax_mod.set_ylim(-0.02, min(1.05, y_max))
    ax_mod.grid(True, linestyle='--', alpha=0.5)
    ax_mod.legend(loc='upper right', fontsize=8.5, ncol=min(3, len(top_species)), framealpha=0.9)

out_fig = os.path.join(PROJECT_ROOT, 'Figures', 'Gore_Empirical_vs_Model_Days7_10.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {out_fig}")

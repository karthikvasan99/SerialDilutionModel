import sys; sys.path.insert(0, ".")
import os, torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, leaves_list
from torchdiffeq import odeint

from src.Pollard_Train_Physical_ODE import PurePhysicalPollardODE, prepare_data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_path = 'Data/SupplementaryDataS2.csv'
trajectories_p, conditions_p, unique_envs_p, num_species_p = prepare_data(data_path)

model = PurePhysicalPollardODE(N=num_species_p, K=5, num_conditions=len(unique_envs_p)).to(device)
ckpt = torch.load('Data/Pollard_Physical_ODE_50Strains.pth', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

Theta, W, alpha, R0_table, B0 = model.get_params()
Theta_np = Theta.detach().cpu().numpy()
W_np = W.detach().cpu().numpy()

delta_t = 0.10
tau_steps = int(1.0 / delta_t) + 1
t_span = torch.linspace(0.0, 1.0, tau_steps, device=device)

matched_r2, mismatched_r2, shuffled_r2 = [], [], []
matched_r, mismatched_r, shuffled_r = [], [], []

clean_env_names = [
    'Mucin Carrier',
    'Mucin Supern.',
    'No Carrier (Liq.)',
    'Plain Carrier',
    'Plain Supern.'
]

with torch.no_grad():
    def eval_subset(mask, R0_sub):
        N_curr = trajectories_p[mask, 0, :] * B0
        preds = []
        for p in range(1, 6):
            N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
            S0 = torch.cat([N0, R0_sub], dim=-1)
            S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :num_species_p], min=0.0)
            N_curr = N_final
            preds.append(N_final)
        pt = torch.stack(preds, dim=1)
        return pt / (pt.sum(dim=-1, keepdim=True) + 1e-8)

    for c_idx in range(5):
        mask = (conditions_p == c_idx)
        t_c = trajectories_p[mask, 1:, :].cpu().numpy().flatten()
        
        # Matched
        R0_mat = R0_table[c_idx:c_idx+1].expand(mask.sum(), -1)
        p_mat = eval_subset(mask, R0_mat).cpu().numpy().flatten()
        r_mat, _ = pearsonr(t_c, p_mat)
        r2_mat = 1.0 - np.sum((t_c - p_mat)**2) / np.sum((t_c - np.mean(t_c))**2)
        matched_r2.append(r2_mat)
        matched_r.append(r_mat)
        
        # Mismatched
        mis_r2_list, mis_r_list = [], []
        for other_c in range(5):
            if other_c != c_idx:
                R0_mis = R0_table[other_c:other_c+1].expand(mask.sum(), -1)
                p_mis = eval_subset(mask, R0_mis).cpu().numpy().flatten()
                r_m, _ = pearsonr(t_c, p_mis)
                r2_m = 1.0 - np.sum((t_c - p_mis)**2) / np.sum((t_c - np.mean(t_c))**2)
                mis_r2_list.append(r2_m)
                mis_r_list.append(r_m)
        mismatched_r2.append(np.mean(mis_r2_list))
        mismatched_r.append(np.mean(mis_r_list))
        
        # Shuffled
        shuff_r2_list, shuff_r_list = [], []
        for _ in range(5):
            perm = torch.randperm(5)
            R0_shuff = R0_table[c_idx:c_idx+1, perm].expand(mask.sum(), -1)
            p_sh = eval_subset(mask, R0_shuff).cpu().numpy().flatten()
            r_s, _ = pearsonr(t_c, p_sh)
            r2_s = 1.0 - np.sum((t_c - p_sh)**2) / np.sum((t_c - np.mean(t_c))**2)
            shuff_r2_list.append(r2_s)
            shuff_r_list.append(r_s)
        shuffled_r2.append(np.mean(shuff_r2_list))
        shuffled_r.append(np.mean(shuff_r_list))

Z = linkage(Theta_np, method='ward')
order = leaves_list(Z)
Theta_clustered = Theta_np[order]

# Multi-panel figure with extra-large fonts for poster readability
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig = plt.figure(figsize=(15.0, 4.6), dpi=300)
gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 0.9, 1.45], wspace=0.32)

# Panel A
ax0 = fig.add_subplot(gs[0])
im0 = ax0.imshow(Theta_clustered, aspect='auto', cmap='YlGnBu', interpolation='nearest')
ax0.set_title(r'$\mathbf{(A)\;Uptake\;Traits\;}\mathbf{\Theta}\;(50 \times 5)$', fontsize=14.0, fontweight='bold', color='#0F2942', pad=10)
ax0.set_xlabel('Latent Resource', fontsize=13.0, fontweight='bold', color='#1A365D')
ax0.set_ylabel('Clustered Gut Strains ($N=50$)', fontsize=13.0, fontweight='bold', color='#1A365D')
ax0.set_xticks(range(5))
ax0.set_xticklabels([f'$R_{k+1}$' for k in range(5)], fontsize=12.0, fontweight='bold')
ax0.set_yticks([])
cbar0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
cbar0.set_label('Uptake Rate', fontsize=12.0, fontweight='bold')
cbar0.ax.tick_params(labelsize=10.5)

# Panel B
ax1 = fig.add_subplot(gs[1])
im1 = ax1.imshow(W_np, aspect='auto', cmap='magma_r', interpolation='nearest')
ax1.set_title(r'$\mathbf{(B)\;Cross\text{-}Feeding\;}\mathbf{W}\;(5 \times 5)$', fontsize=14.0, fontweight='bold', color='#0F2942', pad=10)
ax1.set_xlabel('To Niche ($R_j$)', fontsize=13.0, fontweight='bold', color='#1A365D')
ax1.set_ylabel('From Niche ($R_i$)', fontsize=13.0, fontweight='bold', color='#1A365D')
ax1.set_xticks(range(5))
ax1.set_yticks(range(5))
ax1.set_xticklabels([f'$R_{k+1}$' for k in range(5)], fontsize=12.0, fontweight='bold')
ax1.set_yticklabels([f'$R_{k+1}$' for k in range(5)], fontsize=12.0, fontweight='bold')
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Byproduct Yield', fontsize=12.0, fontweight='bold')
cbar1.ax.tick_params(labelsize=10.5)

# Panel C
ax2 = fig.add_subplot(gs[2])
x = np.arange(len(clean_env_names))
bar_width = 0.27

b1 = ax2.bar(x - bar_width, matched_r2, width=bar_width, label=r'Matched $\mathbf{a}_c$ (True Env)', 
             color='#1B6CA8', edgecolor='#0F2942', lw=1.3, zorder=3)
b2 = ax2.bar(x, mismatched_r2, width=bar_width, label=r'Mismatched $\mathbf{a}$ (Swap)', 
             color='#E06D53', edgecolor='#802010', lw=1.3, zorder=3)
b3 = ax2.bar(x + bar_width, shuffled_r2, width=bar_width, label=r'Shuffled $\mathbf{a}$ (Null)', 
             color='#A0AAB2', edgecolor='#4A5568', lw=1.3, zorder=3)

for bar in b1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#1B6CA8')
for bar in b2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#802010')

ax2.set_xticks(x)
ax2.set_xticklabels(clean_env_names, rotation=20, ha='right', fontsize=11.5, fontweight='bold')
ax2.set_ylabel(r'Predictive Accuracy ($R^2$)', fontsize=13.0, fontweight='bold', color='#1A365D')
ax2.set_title(r'$\mathbf{(C)\;Universality\;Across\;Spatial\;Carriers}$', fontsize=14.0, fontweight='bold', color='#0F2942', pad=10)
ax2.set_ylim(0, 1.05)
ax2.tick_params(axis='y', labelsize=11.5)
# Extra-large legend
ax2.legend(loc='upper right', fontsize=12.5, framealpha=0.95, edgecolor='#0F2942')
ax2.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)

out_path = 'Poster/figures/Pollard_Universal_Matrices.png'
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved large font Pollard figure to {out_path}")

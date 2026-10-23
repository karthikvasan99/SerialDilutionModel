import os, sys
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from torchdiffeq import odeint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1. Load Data
raw_data = torch.load('Data/German_processed_data.pt', weights_only=False)
raw_strains = raw_data['strains']
trajectories_dict = raw_data['trajectories']
raw_traj = torch.stack(list(trajectories_dict.values()), dim=0)
peaks = raw_traj.amax(dim=(0, 1))
mask_1pct = peaks >= 0.01

data_91 = raw_traj[:, :, mask_1pct].to(device)
data_91 = data_91 / (data_91.sum(dim=-1, keepdim=True) + 1e-12)

num_conditions, num_passages, num_species = data_91.shape
cond_ids = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=torch.long, device=device)
num_unique_envs = 6

brrs_mask = (data_91[:6].sum(dim=(0, 1)) > 0).float()
osu_mask = (data_91[6:].sum(dim=(0, 1)) > 0).float()
host_masks = torch.cat([brrs_mask.unsqueeze(0).repeat(6, 1), osu_mask.unsqueeze(0).repeat(6, 1)], dim=0).to(device)

x0 = data_91[:, 0, :]
seed_t1 = torch.where(x0 > 0, x0, torch.tensor(1e-4, device=device)) * host_masks
seed_t1 = seed_t1 / seed_t1.sum(dim=-1, keepdim=True)

# 2. Model Class
class GermanPurePhysicalODE(nn.Module):
    def __init__(self, N, K, num_conditions=6):
        super().__init__()
        self.N, self.K = N, K
        self.r0_raw = nn.Parameter(torch.randn(num_conditions, K))
        self.theta_raw = nn.Parameter(torch.randn(N, K))
        self.w_raw = nn.Parameter(torch.randn(K, K))
        self.alpha_raw = nn.Parameter(torch.randn(N))
        self.b0_raw = nn.Parameter(torch.tensor(0.0))
    def get_params(self):
        return F.softplus(self.theta_raw), F.softplus(self.w_raw), F.softplus(self.alpha_raw), F.softplus(self.r0_raw), F.softplus(self.b0_raw) + 0.1
    def ode_deriv(self, t, S, Theta, W, alpha):
        N_t, R_t = torch.clamp(S[..., :self.N], min=0.0), torch.clamp(S[..., self.N:], min=0.0)
        return torch.cat([N_t * torch.matmul(R_t, Theta.t()) - alpha * (N_t**2), - R_t * torch.matmul(torch.matmul(N_t, Theta), W.t())], dim=-1)

model = GermanPurePhysicalODE(N=91, K=12, num_conditions=6).to(device)
ckpt = torch.load('Data/German_Physical_ODE_91Strains_Best.pth', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

Theta, W, alpha, R0_table, B0 = model.get_params()
T_SPAN = torch.linspace(0.0, 1.0, 11, device=device)

def run_simulation(R0_input):
    with torch.no_grad():
        N_curr = seed_t1 * B0
        eval_preds = [seed_t1]
        for p in range(1, num_passages):
            N0 = torch.clamp((N_curr / 100.0) * host_masks, min=0.0)
            S0 = torch.cat([N0, R0_input], dim=-1)
            S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, T_SPAN, method='euler', options={'step_size': 0.1})
            N_final = torch.clamp(S_t[-1, :, :num_species] * host_masks, min=0.0)
            x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
            eval_preds.append(x_pred)
            N_curr = N_final
        return torch.stack(eval_preds, dim=1)

R0_true = R0_table[cond_ids]
pred_true = run_simulation(R0_true).cpu().numpy()
true_np = data_91.cpu().numpy()

persist_preds = [data_91[:, 0:1, :]]
for p in range(1, num_passages):
    persist_preds.append(data_91[:, p-1:p, :])
persist_np = torch.cat(persist_preds, dim=1).cpu().numpy()

cond_ids_mis = (cond_ids + 1) % 6
pred_mis = run_simulation(R0_table[cond_ids_mis]).cpu().numpy()

shuff_preds = []
for _ in range(10):
    perm = torch.randperm(12)
    R0_shuff = R0_table[:, perm][cond_ids]
    shuff_preds.append(run_simulation(R0_shuff).cpu().numpy())
pred_shuff = np.mean(shuff_preds, axis=0)

r2_true_p, r2_pers_p, r2_mis_p, r2_shuff_p = [], [], [], []
for p in range(num_passages):
    t_p = true_np[:, p, :].flatten()
    var_p = np.sum((t_p - np.mean(t_p))**2)
    r2_true_p.append(1.0 - np.sum((t_p - pred_true[:, p, :].flatten())**2) / var_p)
    r2_pers_p.append(1.0 - np.sum((t_p - persist_np[:, p, :].flatten())**2) / var_p)
    r2_mis_p.append(1.0 - np.sum((t_p - pred_mis[:, p, :].flatten())**2) / var_p)
    r2_shuff_p.append(1.0 - np.sum((t_p - pred_shuff[:, p, :].flatten())**2) / var_p)

t_dyn = true_np[:, 1:, :].flatten()
var_dyn = np.sum((t_dyn - np.mean(t_dyn))**2)

overall_r2 = [
    1.0 - np.sum((t_dyn - pred_true[:, 1:, :].flatten())**2) / var_dyn,
    1.0 - np.sum((t_dyn - persist_np[:, 1:, :].flatten())**2) / var_dyn,
    1.0 - np.sum((t_dyn - pred_mis[:, 1:, :].flatten())**2) / var_dyn,
    np.clip(1.0 - np.sum((t_dyn - pred_shuff[:, 1:, :].flatten())**2) / var_dyn, -0.5, None)
]

overall_r = [
    np.corrcoef(t_dyn, pred_true[:, 1:, :].flatten())[0, 1],
    np.corrcoef(t_dyn, persist_np[:, 1:, :].flatten())[0, 1],
    np.corrcoef(t_dyn, pred_mis[:, 1:, :].flatten())[0, 1],
    np.corrcoef(t_dyn, pred_shuff[:, 1:, :].flatten())[0, 1]
]

pca = PCA(n_components=2)
R0_clean = R0_table.detach().cpu().numpy()
r0_pca = pca.fit_transform(R0_clean)

cond_labels = [
    r'$\mathrm{H_2:CO_2}$ (5b)',
    r'$\mathrm{H_2:CO_2}$ (30b)',
    r'$\mathrm{+ MeOH}$ (5b)',
    r'$\mathrm{+ MeOH}$ (30b)',
    r'$\mathrm{+ TMA}$ (5b)',
    r'$\mathrm{+ TMA}$ (30b)'
]
cond_colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a', '#d62728', '#ff9896']

plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig = plt.figure(figsize=(15.0, 4.6), dpi=300)
gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.35, 1.05], wspace=0.32)

# Panel A
ax0 = fig.add_subplot(gs[0])
models = [r'True $\mathbf{a}$', 'Persistence', r'Mismatched $\mathbf{a}$', r'Shuffled $\mathbf{a}$']
bar_colors = ['#1B6CA8', '#F5A623', '#E06D53', '#8C9BA5']
edge_colors = ['#0F2942', '#B37400', '#802010', '#4A5568']

bars = ax0.bar(models, overall_r2, color=bar_colors, edgecolor=edge_colors, lw=1.3, width=0.55, zorder=3)
for bar, r_val in zip(bars, overall_r):
    h = bar.get_height()
    va = 'bottom' if h >= 0 else 'top'
    ypos = h + 0.02 if h >= 0 else h - 0.05
    ax0.text(bar.get_x() + bar.get_width()/2., ypos, f'$R^2={h:.2f}$\n($r={r_val:.2f}$)', 
             ha='center', va=va, fontsize=11.0, fontweight='bold', color='#1A365D')

ax0.axhline(0, color='black', lw=1.0, alpha=0.7)
ax0.set_ylim(-0.25, 1.05)
ax0.set_ylabel(r'Predictive Accuracy ($R^2$)', fontsize=13.0, fontweight='bold', color='#1A365D')
ax0.set_title(r'$\mathbf{(A)\;Environmental\;Specificity}$', fontsize=14.0, fontweight='bold', color='#0F2942', pad=10)
ax0.tick_params(axis='x', rotation=18, labelsize=11.5)
ax0.tick_params(axis='y', labelsize=11.5)
ax0.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)

# Panel B
ax1 = fig.add_subplot(gs[1])
p_idx = np.arange(1, 6)
ax1.plot(p_idx, r2_true_p[1:], 'o-', color='#1B6CA8', lw=2.4, ms=7, label=r'True Matched $\mathbf{a}$', zorder=4)
ax1.plot(p_idx, r2_pers_p[1:], 's--', color='#F5A623', lw=2.0, ms=6, label=r'Persistence Null ($\mathbf{x}_{t-1}$)', zorder=3)
ax1.plot(p_idx, r2_mis_p[1:], '^:', color='#E06D53', lw=2.0, ms=6, label=r'Mismatched $\mathbf{a}$ (Swap)', zorder=2)
ax1.plot(p_idx, np.clip(r2_shuff_p[1:], -0.3, None), 'x-.', color='#8C9BA5', lw=1.8, ms=6, label=r'Shuffled $\mathbf{a}$ Null', zorder=1)

ax1.set_xticks(p_idx)
ax1.set_xticklabels(['T2 (4d)', 'T3 (8d)', 'T4 (12d)', 'T5 (16d)', 'T6 (20d)'], fontsize=11.5, fontweight='bold')
ax1.set_xlabel('Bioreactor Timepoint', fontsize=13.0, fontweight='bold', color='#1A365D')
ax1.set_ylabel(r'Passage Accuracy ($R^2$)', fontsize=13.0, fontweight='bold', color='#1A365D')
ax1.set_title(r'$\mathbf{(B)\;20\text{-}Day\;Predictive\;Horizon}$', fontsize=14.0, fontweight='bold', color='#0F2942', pad=10)
ax1.set_ylim(-0.25, 1.0)
ax1.tick_params(axis='y', labelsize=11.5)
ax1.legend(loc='lower left', fontsize=12.0, framealpha=0.95, edgecolor='#0F2942')
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel C
ax2 = fig.add_subplot(gs[2])
for i in range(6):
    ax2.scatter(r0_pca[i, 0], r0_pca[i, 1], color=cond_colors[i], s=120, edgecolors='#0F2942', lw=1.5, zorder=4)
    offset_x = 0.08 if i not in [1, 3] else -0.16
    offset_y = 0.06 if i % 2 == 0 else -0.09
    ax2.annotate(cond_labels[i], (r0_pca[i, 0], r0_pca[i, 1]), 
                 xytext=(r0_pca[i, 0] + offset_x, r0_pca[i, 1] + offset_y),
                 fontsize=10.5, fontweight='bold', color='#1A365D')

ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12.5, fontweight='bold', color='#1A365D')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12.5, fontweight='bold', color='#1A365D')
ax2.set_title(r'$\mathbf{(C)\;Learned\;Resource\;Space\;}\mathbf{a}$', fontsize=14.0, fontweight='bold', color='#0F2942', pad=10)
ax2.tick_params(labelsize=11.0)
ax2.grid(True, linestyle='--', alpha=0.5)

out_path = 'Poster/figures/German_A_Vector_Null_Comparison.png'
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved large font German figure to {out_path}")

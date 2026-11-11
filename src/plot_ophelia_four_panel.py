import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr, linregress
from torchdiffeq import odeint
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

# 1. Device and Styling
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.3
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.35

# 2. Distinct Colors for Timepoints t = 12h, 24h, 36h, 48h, 60h
t_colors = ['#2c3e50', '#2980b9', '#16a085', '#e67e22', '#c0392b']
t_labels = ['12 h', '24 h', '36 h', '48 h', '60 h']

# 3. Load Venturelli Data & Model
train_mat = loadmat('Data/train_data.mat')['train_data']
train_valid = [np.nan_to_num(train_mat[i], nan=0.0) for i in range(train_mat.shape[0]) 
              if not np.isnan(train_mat[i][0]).any() and np.sum(train_mat[i][0]) > 1e-4]
train_tensor = torch.tensor(np.array(train_valid), dtype=torch.float32, device=device)

test_mat = loadmat('Data/test_data.mat')['test_data']
test_valid = [np.nan_to_num(test_mat[i], nan=0.0) for i in range(test_mat.shape[0]) 
              if not np.isnan(test_mat[i][0]).any() and np.sum(test_mat[i][0]) > 1e-4]
val_tensor = torch.tensor(np.array(test_valid), dtype=torch.float32, device=device)

class OpheliaODEK(nn.Module):
    def __init__(self, K=7, max_crossfeed=0.40):
        super().__init__()
        self.N, self.K = 12, K
        self.max_crossfeed = max_crossfeed
        self.a = nn.Parameter(torch.randn(1, K))
        self.theta_raw = nn.Parameter(torch.randn(12, K))
        self.w_raw = nn.Parameter(torch.randn(K, K))
        self.alpha_raw = nn.Parameter(torch.randn(12))
        self.log_c = nn.Parameter(torch.tensor(2.0))
        
    def get_params(self):
        Theta = F.softplus(self.theta_raw)
        alpha = F.softplus(self.alpha_raw)
        R0 = F.softplus(self.a)
        c = torch.exp(self.log_c)
        W_sp = F.softplus(self.w_raw)
        W_sp = W_sp * (1.0 - torch.eye(self.K, device=W_sp.device))
        row_sums = W_sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_crossfeed / torch.clamp(row_sums, min=1e-6), max=1.0)
        W = W_sp * scale
        return Theta, W, alpha, R0, c
        
    def ode_deriv(self, t, S, Theta, W, alpha, c):
        N_t, R_t = torch.clamp(S[..., :12], min=0.0), torch.clamp(S[..., 12:], min=0.0)
        growth = torch.matmul(R_t, Theta.t())
        dN = N_t * growth - alpha * (N_t**2)
        flux = c * R_t * torch.matmul(N_t, Theta)
        byproduct = torch.matmul(flux, W)
        dR = - flux + byproduct
        return torch.cat([dN, dR], dim=-1)

model = OpheliaODEK(K=7).to(device)
ckpt = torch.load('Data/Ophelia_Physical_ODE_12Strains.pth', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

t_span = torch.linspace(0.0, 5.0, 6, device=device)
with torch.no_grad():
    Theta, W, alpha, R0_table, c = model.get_params()
    
    # Train rollout
    N0_tr = train_tensor[:, 0, :]
    S0_tr = torch.cat([N0_tr, R0_table[0].unsqueeze(0).expand(N0_tr.size(0), -1)], dim=-1)
    S_t_tr = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha, c), S0_tr, t_span, method='euler', options={'step_size': 0.1})
    train_pred = torch.clamp(S_t_tr[..., :12].permute(1, 0, 2), min=0.0)
    
    # Test rollout
    N0_val = val_tensor[:, 0, :]
    S0_val = torch.cat([N0_val, R0_table[0].unsqueeze(0).expand(N0_val.size(0), -1)], dim=-1)
    S_t_val = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha, c), S0_val, t_span, method='euler', options={'step_size': 0.1})
    val_pred = torch.clamp(S_t_val[..., :12].permute(1, 0, 2), min=0.0)

# 4. Load Pairwise Held-Out Data
pairwise_files = sorted(glob.glob('archive/pairwise_results/*.npz'))
print(f"Loaded {len(pairwise_files)} pairwise files.")

all_pair_true_by_t = {t: [] for t in ['12h', '24h', '36h', '48h', '60h']}
all_pair_pred_by_t = {t: [] for t in ['12h', '24h', '36h', '48h', '60h']}
pairwise_r2_by_t = {t: [] for t in ['12h', '24h', '36h', '48h', '60h']}

for f in pairwise_files:
    d = np.load(f)
    for t in ['12h', '24h', '36h', '48h', '60h']:
        kt, kp, kr = f'true_t={t}', f'pred_t={t}', f'r2_t={t}'
        if kt in d and kp in d:
            all_pair_true_by_t[t].append(d[kt])
            all_pair_pred_by_t[t].append(d[kp])
            pairwise_r2_by_t[t].append(float(d[kr]))

# 5. Load Diversity Invariance Data
df_features = pd.read_csv('Data/Ophelia_Test_Community_Accuracy_Features.csv')

# 6. Create Publication 4-Panel Figure (2x2)
fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.5), dpi=300)

# =========================================================================
# PANEL A: Training Fit (Observed vs Predicted OD, colored by t)
# =========================================================================
axA = axes[0, 0]
tr_all_true, tr_all_pred = [], []
for idx, t_idx in enumerate(range(1, 6)):
    yt = train_tensor[:, t_idx, :].cpu().numpy().flatten()
    yp = train_pred[:, t_idx, :].cpu().numpy().flatten()
    tr_all_true.append(yt)
    tr_all_pred.append(yp)
    axA.scatter(yt, yp, color=t_colors[idx], alpha=0.35, s=22, edgecolors='none', label=f'$t = {t_labels[idx]}$')

tr_true_flat = np.concatenate(tr_all_true)
tr_pred_flat = np.concatenate(tr_all_pred)
r2_tr = r2_score(tr_true_flat, tr_pred_flat)
r_tr = pearsonr(tr_true_flat, tr_pred_flat)[0]

max_A = max(tr_true_flat.max(), tr_pred_flat.max()) * 1.05
axA.plot([0, max_A], [0, max_A], 'k--', lw=1.8, alpha=0.8, label='$y = x$')
axA.set_xlim(-0.02 * max_A, max_A)
axA.set_ylim(-0.02 * max_A, max_A)
axA.set_xlabel(r'Observed Optical Density ($\mathrm{OD}_{600}$)', fontsize=12, fontweight='bold')
axA.set_ylabel(r'Model Predicted $\mathrm{OD}_{600}$', fontsize=12, fontweight='bold')
axA.set_title('A. In-Sample Training Fit (102 Consortia)', fontsize=13, fontweight='bold', pad=10)

badge_A = rf"$\mathbf{{Training\ Consortia}}$" + "\n" + f"$R^2 = {r2_tr:.3f}$\nPearson $r = {r_tr:.3f}$\n$N = {len(tr_true_flat):,}$"
axA.text(0.05, 0.95, badge_A, transform=axA.transAxes, fontsize=10.5, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#2c3e50', alpha=0.92, lw=1.2))

# =========================================================================
# PANEL B: Testing Fit (Observed vs Predicted OD on Held-out 73 Communities)
# =========================================================================
axB = axes[0, 1]
te_all_true, te_all_pred = [], []
for idx, t_idx in enumerate(range(1, 6)):
    yt = val_tensor[:, t_idx, :].cpu().numpy().flatten()
    yp = val_pred[:, t_idx, :].cpu().numpy().flatten()
    te_all_true.append(yt)
    te_all_pred.append(yp)
    axB.scatter(yt, yp, color=t_colors[idx], alpha=0.45, s=26, edgecolors='none', label=f'$t = {t_labels[idx]}$')

te_true_flat = np.concatenate(te_all_true)
te_pred_flat = np.concatenate(te_all_pred)
r2_te = r2_score(te_true_flat, te_pred_flat)
r_te = pearsonr(te_true_flat, te_pred_flat)[0]

max_B = max(te_true_flat.max(), te_pred_flat.max()) * 1.05
axB.plot([0, max_B], [0, max_B], 'k--', lw=1.8, alpha=0.8, label='$y = x$')
axB.set_xlim(-0.02 * max_B, max_B)
axB.set_ylim(-0.02 * max_B, max_B)
axB.set_xlabel(r'Observed Optical Density ($\mathrm{OD}_{600}$)', fontsize=12, fontweight='bold')
axB.set_ylabel(r'Model Predicted $\mathrm{OD}_{600}$', fontsize=12, fontweight='bold')
axB.set_title('B. Out-of-Sample Test Fit (73 Consortia)', fontsize=13, fontweight='bold', pad=10)

badge_B = rf"$\mathbf{{Held\text{{-}}Out\ Test\ Consortia}}$" + "\n" + f"$R^2 = {r2_te:.3f}$\nPearson $r = {r_te:.3f}$\n$N = {len(te_true_flat):,}$"
axB.text(0.05, 0.95, badge_B, transform=axB.transAxes, fontsize=10.5, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#2c3e50', alpha=0.92, lw=1.2))

axB.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc', fontsize=10.0)

# =========================================================================
# PANEL C: Trajectory Pearson Invariance to Shannon Diversity
# =========================================================================
axC = axes[1, 0]
x_shannon = df_features['Shannon_Entropy'].values
y_traj_r = df_features['Trajectory_Pearson_r'].values
richness_vals = df_features['Richness'].values

# Scatter points colored by richness
scC = axC.scatter(
    x_shannon, y_traj_r,
    c=richness_vals, cmap='plasma', s=65,
    edgecolors='black', linewidth=0.8, alpha=0.88, zorder=3
)

# Linear regression fit
slope, intercept, r_val, p_val, std_err = linregress(x_shannon, y_traj_r)
x_line = np.linspace(x_shannon.min(), x_shannon.max(), 100)
y_line = slope * x_line + intercept
axC.plot(x_line, y_line, color='#2c3e50', linestyle='--', lw=2.2, zorder=4, label='Linear Fit')

# Confidence band
y_err = std_err * np.sqrt(1/len(x_shannon) + (x_line - np.mean(x_shannon))**2 / np.sum((x_shannon - np.mean(x_shannon))**2))
axC.fill_between(x_line, y_line - 1.96*y_err, y_line + 1.96*y_err, color='#2c3e50', alpha=0.15, zorder=2)

rho_val, sp_pval = spearmanr(x_shannon, y_traj_r)
badge_C = rf"$\mathbf{{Invariance\ to\ Diversity}}$" + "\n" + f"Pearson $r = {r_val:+.3f}$ ($p = {p_val:.3f}$)\nSpearman $\\rho = {rho_val:+.3f}$ ($p = {sp_pval:.3f}$)"
axC.text(0.05, 0.15, badge_C, transform=axC.transAxes, fontsize=10.5, fontweight='bold',
         verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', edgecolor='#cccccc', alpha=0.92, lw=1.2))

axC.set_xlabel('Initial Community Shannon Diversity ($H$)', fontsize=12, fontweight='bold')
axC.set_ylabel('Trajectory Pearson Correlation ($r$)', fontsize=12, fontweight='bold')
axC.set_title('C. Complexity Invariance (Negative Result)', fontsize=13, fontweight='bold', pad=10)
axC.set_ylim(0.40, 1.05)

cbarC = fig.colorbar(scC, ax=axC, pad=0.03, shrink=0.85)
cbarC.set_label('Species Richness ($S$)', fontsize=10.5, fontweight='bold')

# =========================================================================
# PANEL D: Pairwise Plot (Held-Out Species Pairs, Leave-Pair-Out)
# =========================================================================
axD = axes[1, 1]
pair_all_true, pair_all_pred = [], []

np.random.seed(42)
for idx, t in enumerate(['12h', '24h', '36h', '48h', '60h']):
    yt_t = np.concatenate(all_pair_true_by_t[t])
    yp_t = np.concatenate(all_pair_pred_by_t[t])
    pair_all_true.append(yt_t)
    pair_all_pred.append(yp_t)
    
    # Subsample for crisp visualization without rendering lag
    sub_idx = np.random.choice(len(yt_t), size=min(1500, len(yt_t)), replace=False)
    axD.scatter(yt_t[sub_idx], yp_t[sub_idx], color=t_colors[idx], alpha=0.35, s=18, edgecolors='none', label=f'$t = {t_labels[idx]}$')

pair_true_flat = np.concatenate(pair_all_true)
pair_pred_flat = np.concatenate(pair_all_pred)
r2_pair = r2_score(pair_true_flat, pair_pred_flat)
r_pair = pearsonr(pair_true_flat, pair_pred_flat)[0]

max_D = max(pair_true_flat.max(), pair_pred_flat.max()) * 1.05
axD.plot([0, max_D], [0, max_D], 'k--', lw=1.8, alpha=0.8, label='$y = x$')
axD.set_xlim(-0.02 * max_D, max_D)
axD.set_ylim(-0.02 * max_D, max_D)
axD.set_xlabel('Observed Relative Abundance', fontsize=12, fontweight='bold')
axD.set_ylabel('Model Predicted Relative Abundance', fontsize=12, fontweight='bold')
axD.set_title('D. Leave-Pair-Out Validation (66 Unseen Pairs)', fontsize=13, fontweight='bold', pad=10)

mean_pair_r2 = np.mean([np.mean(pairwise_r2_by_t[t]) for t in ['12h', '24h', '36h', '48h', '60h']])
badge_D = rf"$\mathbf{{Held\text{{-}}Out\ Species\ Pairs}}$" + "\n" + f"Pooled $R^2 = {r2_pair:.3f}$\nPearson $r = {r_pair:.3f}$\nMean Per-Pair $R^2 = {mean_pair_r2:.3f}$\n$N = {len(pair_true_flat):,}$ points"
axD.text(0.05, 0.95, badge_D, transform=axD.transAxes, fontsize=10.5, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#2c3e50', alpha=0.92, lw=1.2))

# Add mini-inset bar chart showing R2 by timepoint across the 66 pairs
inset_ax = axD.inset_axes([0.62, 0.08, 0.35, 0.32])
mean_r2_by_t = [np.mean(pairwise_r2_by_t[t]) for t in ['12h', '24h', '36h', '48h', '60h']]
std_r2_by_t = [np.std(pairwise_r2_by_t[t]) for t in ['12h', '24h', '36h', '48h', '60h']]
bars = inset_ax.bar(range(5), mean_r2_by_t, yerr=std_r2_by_t, capsize=3, color=t_colors, edgecolor='black', lw=0.8, alpha=0.85)
inset_ax.set_xticks(range(5))
inset_ax.set_xticklabels(['12h', '24h', '36h', '48h', '60h'], fontsize=7.5, fontweight='bold')
inset_ax.set_ylabel(r'Mean $R^2$', fontsize=8.0, fontweight='bold')
inset_ax.set_ylim(0, 1.05)
inset_ax.set_title(r'Pairwise $R^2$ by Time', fontsize=8.5, fontweight='bold', pad=3)
inset_ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
out_fig = 'Paper/figures/Figure1_Ophelia_FourPanel_Publication.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"Saved publication-quality 4-panel figure to: {out_fig}")

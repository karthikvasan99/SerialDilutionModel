"""
Generate Publication Figure 1 for Ophelia Dataset with Validated K=3 Architecture
Loads Data/Ophelia_Physical_ODE_Winning_K3.pth and Data/Ophelia_Test_Community_Accuracy_Features_K3.csv
Saves Paper/figures/Figure1_Ophelia_ThreePanel_Publication.png at 300 dpi.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# 2. Colors for Timepoints t = 12h, 24h, 36h, 48h, 60h
t_colors = ['#2c3e50', '#2980b9', '#16a085', '#e67e22', '#c0392b']
t_labels = ['12 h', '24 h', '36 h', '48 h', '60 h']

# 3. Load Test Data & Model
test_mat = loadmat('Data/test_data.mat')['test_data']
test_valid = [np.nan_to_num(test_mat[i], nan=0.0) for i in range(test_mat.shape[0]) 
              if not np.isnan(test_mat[i][0]).any() and np.sum(test_mat[i][0]) > 1e-4]
val_tensor = torch.tensor(np.array(test_valid), dtype=torch.float32, device=device)

class OpheliaODEK(nn.Module):
    def __init__(self, K=3, max_crossfeed=0.20):
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

ckpt_path = 'Data/Ophelia_Physical_ODE_Winning_K3.pth'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
K = ckpt.get('K', 3)
cw = ckpt.get('c_w', 0.20)
print(f"Loaded checkpoint from {ckpt_path} (K={K}, c_w={cw})")

model = OpheliaODEK(K=K, max_crossfeed=cw).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

t_span = torch.linspace(0.0, 5.0, 6, device=device)
with torch.no_grad():
    Theta, W, alpha, R0_table, c = model.get_params()
    N0_val = val_tensor[:, 0, :]
    S0_val = torch.cat([N0_val, R0_table[0].unsqueeze(0).expand(N0_val.size(0), -1)], dim=-1)
    S_t_val = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha, c), S0_val, t_span, method='euler', options={'step_size': 0.1})
    val_pred = torch.clamp(S_t_val[..., :12].permute(1, 0, 2), min=0.0)

# 4. Load Features Data
features_path = 'Data/Ophelia_Test_Community_Accuracy_Features_K3.csv'
df_features = pd.read_csv(features_path)

# 5. Create 3-Panel Horizontal Figure (1x3 Banner Layout)
fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), dpi=300)

# =========================================================================
# PANEL A: Out-of-Sample Testing Fit (All t on Single Plot)
# =========================================================================
axA = axes[0]
te_all_true, te_all_pred = [], []
for idx, t_idx in enumerate(range(1, 6)):
    yt = val_tensor[:, t_idx, :].cpu().numpy().flatten()
    yp = val_pred[:, t_idx, :].cpu().numpy().flatten()
    te_all_true.append(yt)
    te_all_pred.append(yp)
    axA.scatter(yt, yp, color=t_colors[idx], alpha=0.45, s=28, edgecolors='none', label=f'$t = {t_labels[idx]}$', zorder=3)

te_true_flat = np.concatenate(te_all_true)
te_pred_flat = np.concatenate(te_all_pred)
r2_te = r2_score(te_true_flat, te_pred_flat)
r_te = pearsonr(te_true_flat, te_pred_flat)[0]

max_A = max(te_true_flat.max(), te_pred_flat.max()) * 1.05
axA.plot([0, max_A], [0, max_A], 'k--', lw=1.8, alpha=0.8, label='$y = x$', zorder=2)
axA.set_xlim(-0.02 * max_A, max_A)
axA.set_ylim(-0.02 * max_A, max_A)
axA.set_xlabel(r'Observed Optical Density ($\mathrm{OD}_{600}$)', fontsize=12, fontweight='bold')
axA.set_ylabel(r'Model Predicted $\mathrm{OD}_{600}$', fontsize=12, fontweight='bold')
axA.set_title(f'A. Out-of-Sample Test Fit (73 Consortia, $K={K}$)', fontsize=13, fontweight='bold', pad=10)

badge_A = rf"$\mathbf{{Held\text{{-}}Out\ Test\ Consortia}}$" + "\n" + f"$R^2 = {r2_te:.3f}$\nPearson $r = {r_te:.3f}$\n$N = {len(te_true_flat):,}$"
axA.text(0.05, 0.95, badge_A, transform=axA.transAxes, fontsize=10.5, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#2c3e50', alpha=0.92, lw=1.2))
axA.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc', fontsize=9.5)

# =========================================================================
# PANEL B: Invariance to Initial Community Diversity (Shannon H)
# =========================================================================
axB = axes[1]
x_shannon = df_features['Initial_Shannon_H'].values
y_traj_r = df_features['Trajectory_Pearson_r'].values
richness_vals = df_features['Initial_Richness_S'].values

scB = axB.scatter(
    x_shannon, y_traj_r,
    c=richness_vals, cmap='plasma', s=68,
    edgecolors='black', linewidth=0.8, alpha=0.88, zorder=3
)

# Linear regression
slope_b, intercept_b, r_b, p_b, std_err_b = linregress(x_shannon, y_traj_r)
x_line_b = np.linspace(x_shannon.min(), x_shannon.max(), 100)
y_line_b = slope_b * x_line_b + intercept_b
axB.plot(x_line_b, y_line_b, color='#2c3e50', linestyle='--', lw=2.2, zorder=4, label='Linear Fit')

y_err_b = std_err_b * np.sqrt(1/len(x_shannon) + (x_line_b - np.mean(x_shannon))**2 / np.sum((x_shannon - np.mean(x_shannon))**2))
axB.fill_between(x_line_b, y_line_b - 1.96*y_err_b, y_line_b + 1.96*y_err_b, color='#2c3e50', alpha=0.15, zorder=2)

rho_b, sp_pval_b = spearmanr(x_shannon, y_traj_r)
badge_B = rf"$\mathbf{{Invariance\ to\ Diversity}}$" + "\n" + f"Pearson $r = {r_b:+.3f}$ ($p = {p_b:.3f}$)\nSpearman $\\rho = {rho_b:+.3f}$ ($p = {sp_pval_b:.3f}$)"
axB.text(0.05, 0.12, badge_B, transform=axB.transAxes, fontsize=10.5, fontweight='bold',
         verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', edgecolor='#cccccc', alpha=0.92, lw=1.2))

axB.set_xlabel('Initial Community Shannon Diversity ($H$)', fontsize=12, fontweight='bold')
axB.set_ylabel('Trajectory Pearson Correlation ($r$)', fontsize=12, fontweight='bold')
axB.set_title('B. Invariance to Initial Diversity', fontsize=13, fontweight='bold', pad=10)
axB.set_ylim(0.40, 1.05)

cbarB = fig.colorbar(scB, ax=axB, pad=0.03, shrink=0.88)
cbarB.set_label('Species Richness ($S$)', fontsize=10.0, fontweight='bold')

# =========================================================================
# PANEL C: Invariance to Distance from Training Set (Mean Bray-Curtis)
# =========================================================================
axC = axes[2]
x_dist = df_features['Mean_Bray_Curtis'].values
y_traj_r = df_features['Trajectory_Pearson_r'].values

scC = axC.scatter(
    x_dist, y_traj_r,
    c=richness_vals, cmap='plasma', s=68,
    edgecolors='black', linewidth=0.8, alpha=0.88, zorder=3
)

slope_c, intercept_c, r_c, p_c, std_err_c = linregress(x_dist, y_traj_r)
x_line_c = np.linspace(x_dist.min(), x_dist.max(), 100)
y_line_c = slope_c * x_line_c + intercept_c
axC.plot(x_line_c, y_line_c, color='#2c3e50', linestyle='--', lw=2.2, zorder=4, label='Linear Fit')

y_err_c = std_err_c * np.sqrt(1/len(x_dist) + (x_line_c - np.mean(x_dist))**2 / np.sum((x_dist - np.mean(x_dist))**2))
axC.fill_between(x_line_c, y_line_c - 1.96*y_err_c, y_line_c + 1.96*y_err_c, color='#2c3e50', alpha=0.15, zorder=2)

rho_c, sp_pval_c = spearmanr(x_dist, y_traj_r)
badge_C = rf"$\mathbf{{Invariance\ to\ Distance}}$" + "\n" + f"Pearson $r = {r_c:+.3f}$ ($p = {p_c:.3f}$)\nSpearman $\\rho = {rho_c:+.3f}$ ($p = {sp_pval_c:.3f}$)"
axC.text(0.05, 0.12, badge_C, transform=axC.transAxes, fontsize=10.5, fontweight='bold',
         verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', edgecolor='#cccccc', alpha=0.92, lw=1.2))

axC.set_xlabel('Mean Bray-Curtis Distance to Training Set', fontsize=12, fontweight='bold')
axC.set_ylabel('Trajectory Pearson Correlation ($r$)', fontsize=12, fontweight='bold')
axC.set_title('C. Invariance to Training Set Distance', fontsize=13, fontweight='bold', pad=10)
axC.set_ylim(0.40, 1.05)

cbarC = fig.colorbar(scC, ax=axC, pad=0.03, shrink=0.88)
cbarC.set_label('Species Richness ($S$)', fontsize=10.0, fontweight='bold')

plt.tight_layout()
out_fig = 'Paper/figures/Figure1_Ophelia_ThreePanel_Publication.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"Saved publication-quality 3-panel figure to: {out_fig}")

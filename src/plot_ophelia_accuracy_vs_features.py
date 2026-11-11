import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr, linregress
from scipy.spatial.distance import jensenshannon, braycurtis, jaccard
from scipy.io import loadmat
from torchdiffeq import odeint

# Set styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing Ophelia Accuracy vs Features Analysis on: {device}")

# 1. Load Data
train_mat = loadmat('Data/train_data.mat')['train_data']
train_valid = [np.nan_to_num(train_mat[i], nan=0.0) for i in range(train_mat.shape[0]) 
              if not np.isnan(train_mat[i][0]).any() and np.sum(train_mat[i][0]) > 1e-4]
train_tensor = torch.tensor(np.array(train_valid), dtype=torch.float32, device=device)

test_mat = loadmat('Data/test_data.mat')['test_data']
test_valid = [np.nan_to_num(test_mat[i], nan=0.0) for i in range(test_mat.shape[0]) 
              if not np.isnan(test_mat[i][0]).any() and np.sum(test_mat[i][0]) > 1e-4]
val_tensor = torch.tensor(np.array(test_valid), dtype=torch.float32, device=device)

train_x0 = train_tensor[:, 0, :].cpu().numpy() # (102, 12)
test_x0 = val_tensor[:, 0, :].cpu().numpy()    # (73, 12)
N_test, N_train = test_x0.shape[0], train_x0.shape[0]

train_bin = (train_x0 > 1e-4).astype(float)
test_bin = (test_x0 > 1e-4).astype(float)
train_p0 = train_x0 / np.sum(train_x0, axis=1, keepdims=True)
test_p0 = test_x0 / np.sum(test_x0, axis=1, keepdims=True)

# 2. Model Definition with Physical Byproduct Secretion
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

# 3. Simulate Rollout on Test Data
t_span = torch.linspace(0.0, 5.0, 6, device=device)
with torch.no_grad():
    Theta, W, alpha, R0_table, c = model.get_params()
    Theta_np = Theta.detach().cpu().numpy()
    R0_np = R0_table[0].detach().cpu().numpy()
    
    N0_val = val_tensor[:, 0, :]
    batch_size_val = N0_val.size(0)
    R0_batch_val = R0_table[0].unsqueeze(0).expand(batch_size_val, -1)
    S0_val = torch.cat([N0_val, R0_batch_val], dim=-1)
    S_t_val = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha, c), S0_val, t_span, method='euler', options={'step_size': 0.1})
    val_pred = torch.clamp(S_t_val[..., :12].permute(1, 0, 2), min=0.0)

val_true_np = val_tensor.cpu().numpy()
val_pred_np = val_pred.cpu().numpy()

# 4. Compute Effective MacArthur Interaction Matrix
# A_ij = sum_k Theta_ik * R0_k * Theta_jk
A = np.dot(Theta_np * R0_np, Theta_np.T)
norms = np.linalg.norm(Theta_np, axis=1, keepdims=True)
Theta_normed = Theta_np / norms
O_pianka = np.dot(Theta_normed, Theta_normed.T)

# 5. Calculate Features for Each Test Community
jaccard_dists = [np.mean([jaccard(test_bin[i], train_bin[k]) for k in range(N_train)]) for i in range(N_test)]
jsd_dists = [np.mean([jensenshannon(test_p0[i], train_p0[k]) for k in range(N_train)]) for i in range(N_test)]
bc_dists = [np.mean([braycurtis(test_x0[i], train_x0[k]) for k in range(N_train)]) for i in range(N_test)]

min_jaccard_dists = [np.min([jaccard(test_bin[i], train_bin[k]) for k in range(N_train)]) for i in range(N_test)]
min_jsd_dists = [np.min([jensenshannon(test_p0[i], train_p0[k]) for k in range(N_train)]) for i in range(N_test)]
min_bc_dists = [np.min([braycurtis(test_x0[i], train_x0[k]) for k in range(N_train)]) for i in range(N_test)]

richness = [int(np.sum(test_bin[i])) for i in range(N_test)]
entropy = [-np.sum(test_p0[i, test_bin[i] > 0] * np.log(test_p0[i, test_bin[i] > 0] + 1e-12)) for i in range(N_test)]

avg_comp_unweighted = []
avg_comp_weighted = []
avg_pianka_unweighted = []

for i in range(N_test):
    pres = np.where(test_bin[i] > 0)[0]
    S = len(pres)
    if S > 1:
        pairs_A = [A[pres[u], pres[v]] for u in range(S) for v in range(u+1, S)]
        pairs_P = [O_pianka[pres[u], pres[v]] for u in range(S) for v in range(u+1, S)]
        avg_comp_unweighted.append(np.mean(pairs_A))
        avg_pianka_unweighted.append(np.mean(pairs_P))
        
        p = test_p0[i, pres] / np.sum(test_p0[i, pres])
        w_mat = np.outer(p, p)
        np.fill_diagonal(w_mat, 0.0)
        norm_w = np.sum(w_mat)
        A_sub = A[np.ix_(pres, pres)].copy()
        np.fill_diagonal(A_sub, 0.0)
        avg_comp_weighted.append(np.sum(w_mat * A_sub) / norm_w if norm_w > 0 else 0.0)
    else:
        avg_comp_unweighted.append(0.0)
        avg_pianka_unweighted.append(0.0)
        avg_comp_weighted.append(0.0)

# 6. Calculate Accuracy Metrics for Each Test Community
traj_r = []
traj_r2 = []
traj_rmse = []
traj_mae = []
final_bc_sim = []
final_r = []

for i in range(N_test):
    y_t = val_true_np[i, 1:, :].flatten()
    y_p = val_pred_np[i, 1:, :].flatten()
    var_y = np.sum((y_t - np.mean(y_t))**2)
    ss_res = np.sum((y_t - y_p)**2)
    
    r2 = 1.0 - ss_res / var_y if var_y > 1e-8 else 0.0
    r = pearsonr(y_t, y_p)[0] if np.std(y_t) > 1e-8 and np.std(y_p) > 1e-8 else 0.0
    
    traj_r2.append(r2)
    traj_r.append(r)
    traj_rmse.append(np.sqrt(np.mean((y_t - y_p)**2)))
    traj_mae.append(np.mean(np.abs(y_t - y_p)))
    
    # Final timepoint metrics (t = 60h)
    y_t_fin = val_true_np[i, -1, :]
    y_p_fin = val_pred_np[i, -1, :]
    final_bc_sim.append(1.0 - braycurtis(y_t_fin, y_p_fin))
    r_fin = pearsonr(y_t_fin, y_p_fin)[0] if np.std(y_t_fin) > 1e-8 and np.std(y_p_fin) > 1e-8 else 0.0
    final_r.append(r_fin)

df = pd.DataFrame({
    'Mean_Jaccard_Dist': jaccard_dists,
    'Mean_JSD': jsd_dists,
    'Mean_BrayCurtis_Dist': bc_dists,
    'Min_Jaccard_Dist': min_jaccard_dists,
    'Min_JSD': min_jsd_dists,
    'Min_BrayCurtis_Dist': min_bc_dists,
    'Richness': richness,
    'Shannon_Entropy': entropy,
    'Avg_Pairwise_Competition': avg_comp_unweighted,
    'Weighted_Competition': avg_comp_weighted,
    'Avg_Pianka_Overlap': avg_pianka_unweighted,
    'Trajectory_Pearson_r': traj_r,
    'Trajectory_R2': traj_r2,
    'Trajectory_RMSE': traj_rmse,
    'Trajectory_MAE': traj_mae,
    'Final_BC_Similarity': final_bc_sim,
    'Final_Pearson_r': final_r
})

csv_path = 'Data/Ophelia_Test_Community_Accuracy_Features.csv'
df.to_csv(csv_path, index=False)
print(f"Saved dataset with {len(df)} test communities to {csv_path}")

# ─── 7. Plot Publication-Quality 6-Panel Figure ───
print("\nGenerating 6-Panel Figure: Accuracy vs Candidate Features...")
fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.6), dpi=300)

plot_configs = [
    # Row 1: Mean Distance from Training Set
    (0, 0, 'Mean_Jaccard_Dist', 'Mean Jaccard Distance from Train', 'Trajectory Pearson r', 'A. Mean Jaccard Distance', '#1b6ca8'),
    (0, 1, 'Mean_JSD', 'Mean Jensen-Shannon Divergence (JSD)', 'Trajectory Pearson r', 'B. Mean Jensen-Shannon Divergence', '#2b5c8f'),
    (0, 2, 'Mean_BrayCurtis_Dist', 'Mean Bray-Curtis Dissimilarity', 'Trajectory Pearson r', 'C. Mean Bray-Curtis Dissimilarity', '#3d72a4'),
    # Row 2: Diversity & Competition
    (1, 0, 'Richness', 'Initial Species Richness ($S$)', 'Trajectory Pearson r', 'D. Species Richness', '#d9534f'),
    (1, 1, 'Shannon_Entropy', 'Initial Shannon Entropy ($H$)', 'Trajectory Pearson r', 'E. Community Shannon Entropy', '#e08b53'),
    (1, 2, 'Avg_Pairwise_Competition', r'Average Pairwise Competition ($\bar{A}_{ij}$)', 'Trajectory Pearson r', 'F. Average Pairwise Competition', '#27ae60'),
]

for row, col, feat_col, xlabel, ylabel, title, color in plot_configs:
    ax = axes[row, col]
    x_vals = df[feat_col].values
    y_vals = df['Trajectory_Pearson_r'].values
    
    # Scatter plot colored by richness
    scatter = ax.scatter(
        x_vals, y_vals,
        c=df['Richness'].values, cmap='plasma', s=55,
        edgecolors='black', linewidth=0.7, alpha=0.88, zorder=3
    )
    
    # Regression line
    slope, intercept, r_val, p_val, std_err = linregress(x_vals, y_vals)
    x_line = np.linspace(np.min(x_vals), np.max(x_vals), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='#2c3e50', linestyle='--', lw=2.0, zorder=4)
    
    # Spearman rho
    rho, sp_val = spearmanr(x_vals, y_vals)
    
    # Stat callout text
    stat_text = f"Pearson $r = {r_val:+.3f}$ ($p = {p_val:.3f}$)\nSpearman $\\rho = {rho:+.3f}$ ($p = {sp_val:.3f}$)"
    ax.text(0.05, 0.10, stat_text, transform=ax.transAxes, fontsize=10.0, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.92))
    
    ax.set_xlabel(xlabel, fontsize=11.5, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11.5, fontweight='bold')
    ax.set_title(title, fontsize=12.5, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_ylim(0.20, 1.03)

# Shared Colorbar for Richness
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.set_label('Species Richness ($S$)', fontsize=12.0, fontweight='bold')
cbar.set_ticks(range(1, 13, 2))
cbar.ax.tick_params(labelsize=10.5)

plt.subplots_adjust(left=0.07, right=0.90, top=0.94, bottom=0.08, wspace=0.28, hspace=0.32)

fig1_path = 'Poster/figures/Ophelia_Accuracy_vs_Features.png'
plt.savefig(fig1_path, dpi=300)
plt.close()

# ─── 8. Generate Companion 6-Panel Figure: Final Bray-Curtis Similarity ───
print("Generating Companion 6-Panel Figure: Final Bray-Curtis Similarity vs Features...")
fig_bc, axes_bc = plt.subplots(2, 3, figsize=(15.8, 9.6), dpi=300)

for row, col, feat_col, xlabel, _, title, color in plot_configs:
    ax = axes_bc[row, col]
    x_vals = df[feat_col].values
    y_vals = df['Final_BC_Similarity'].values
    
    scatter_bc = ax.scatter(
        x_vals, y_vals,
        c=df['Richness'].values, cmap='viridis', s=55,
        edgecolors='black', linewidth=0.7, alpha=0.88, zorder=3
    )
    
    slope, intercept, r_val, p_val, std_err = linregress(x_vals, y_vals)
    x_line = np.linspace(np.min(x_vals), np.max(x_vals), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='#2c3e50', linestyle='--', lw=2.0, zorder=4)
    
    rho, sp_val = spearmanr(x_vals, y_vals)
    stat_text = f"Pearson $r = {r_val:+.3f}$ ($p = {p_val:.3f}$)\nSpearman $\\rho = {rho:+.3f}$ ($p = {sp_val:.3f}$)"
    ax.text(0.05, 0.10, stat_text, transform=ax.transAxes, fontsize=10.0, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.92))
    
    ax.set_xlabel(xlabel, fontsize=11.5, fontweight='bold')
    ax.set_ylabel('Final Bray-Curtis Similarity ($1 - \\mathrm{BC}$)', fontsize=11.5, fontweight='bold')
    ax.set_title(title, fontsize=12.5, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_ylim(0.20, 1.03)

cbar_ax_bc = fig_bc.add_axes([0.92, 0.15, 0.015, 0.70])
cbar_bc = fig_bc.colorbar(scatter_bc, cax=cbar_ax_bc)
cbar_bc.set_label('Species Richness ($S$)', fontsize=12.0, fontweight='bold')
cbar_bc.set_ticks(range(1, 13, 2))
cbar_bc.ax.tick_params(labelsize=10.5)

plt.subplots_adjust(left=0.07, right=0.90, top=0.94, bottom=0.08, wspace=0.28, hspace=0.32)

fig2_path = 'Poster/figures/Ophelia_Accuracy_vs_Features_BrayCurtis.png'
plt.savefig(fig2_path, dpi=300)
plt.close()

# Copy to artifact directory
artifact_dir = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
shutil.copyfile(fig1_path, os.path.join(artifact_dir, 'Ophelia_Accuracy_vs_Features.png'))
shutil.copyfile(fig2_path, os.path.join(artifact_dir, 'Ophelia_Accuracy_vs_Features_BrayCurtis.png'))

print(f"\nSuccessfully generated and saved figures:")
print(f"  1. {fig1_path}")
print(f"  2. {fig2_path}")

import os, sys
sys.path.insert(0, '.')
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.io import loadmat
from scipy.stats import pearsonr
from torchdiffeq import odeint
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1. Load Data
test_mat = loadmat('Data/test_data.mat')['test_data']
test_valid = [np.nan_to_num(test_mat[i], nan=0.0) for i in range(test_mat.shape[0]) 
              if not np.isnan(test_mat[i][0]).any() and np.sum(test_mat[i][0]) > 1e-4]
val_tensor = torch.tensor(np.array(test_valid), dtype=torch.float32, device=device)

# 2. Model Definition
class OpheliaODEK(nn.Module):
    def __init__(self, K=7):
        super().__init__()
        self.N, self.K = 12, K
        self.a = nn.Parameter(torch.randn(1, K))
        self.theta_raw = nn.Parameter(torch.randn(12, K))
        self.w_raw = nn.Parameter(torch.randn(K, K))
        self.alpha_raw = nn.Parameter(torch.randn(12))
    def get_params(self):
        return F.softplus(self.theta_raw), F.softplus(self.w_raw), F.softplus(self.alpha_raw), F.softplus(self.a)
    def ode_deriv(self, t, S, Theta, W, alpha):
        N_t, R_t = torch.clamp(S[..., :12], min=0.0), torch.clamp(S[..., 12:], min=0.0)
        return torch.cat([
            N_t * torch.matmul(R_t, Theta.t()) - alpha * (N_t**2), 
            - R_t * torch.matmul(torch.matmul(N_t, Theta), W.t())
        ], dim=-1)

model = OpheliaODEK(K=7).to(device)
ckpt = torch.load('Data/Ophelia_Physical_ODE_12Strains.pth', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 3. Simulate on Held-Out Test Data
t_span = torch.linspace(0.0, 5.0, 6, device=device)
with torch.no_grad():
    Theta, W, alpha, R0_table = model.get_params()
    N0_val = val_tensor[:, 0, :]
    batch_size = N0_val.size(0)
    R0_batch = R0_table[0].unsqueeze(0).expand(batch_size, -1)
    S0_val = torch.cat([N0_val, R0_batch], dim=-1)
    
    S_t_val = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0_val, t_span, method='euler', options={'step_size': 0.1})
    val_pred = torch.clamp(S_t_val[..., :12].permute(1, 0, 2), min=0.0)

val_true_np = val_tensor.cpu().numpy()
val_pred_np = val_pred.cpu().numpy()

# 4. Create Six-Panel Figure Sized for Column Width with Extra Large Legend/Badge Fonts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 3, figsize=(15.0, 6.6), dpi=300)
axes = axes.flatten()

species_colors = plt.cm.tab20(np.linspace(0, 1, 12))
time_labels = ['t = 0h (Inoculum)', 't = 12h', 't = 24h', 't = 36h', 't = 48h', 't = 60h']

for t_idx in range(6):
    ax = axes[t_idx]
    o_t = val_true_np[:, t_idx, :]
    p_t = val_pred_np[:, t_idx, :]
    
    o_flat = o_t.flatten()
    p_flat = p_t.flatten()
    
    var_t = np.sum((o_flat - np.mean(o_flat))**2)
    r2_t = 1.0 - np.sum((o_flat - p_flat)**2) / var_t if var_t > 0 else 1.0
    r_t, _ = pearsonr(o_flat, p_flat) if np.std(p_flat) > 0 else (1.0, 0.0)
    
    # Larger scatter points
    for sp in range(12):
        ax.scatter(o_t[:, sp], p_t[:, sp], color=species_colors[sp], alpha=0.65, s=32, edgecolors='none')
        
    max_val = max(np.max(o_flat), np.max(p_flat)) * 1.08
    ax.plot([0, max_val], [0, max_val], 'k--', lw=2.0, alpha=0.8)
    ax.set_xlim(-0.02 * max_val, max_val)
    ax.set_ylim(-0.02 * max_val, max_val)
    
    # Extra large font size for badge / metrics (15pt, bold)
    if t_idx == 0:
        badge_text = f"$\\mathbf{{{time_labels[t_idx]}}}$\n$R^2 = 1.000$\n$r = 1.000$"
    else:
        badge_text = f"$\\mathbf{{{time_labels[t_idx]}}}$\n$R^2 = {r2_t:.3f}$\n$r = {r_t:.3f}$"
        
    props = dict(boxstyle='round,pad=0.5', facecolor='#F0F4F8', edgecolor='#0F2942', alpha=0.92, lw=1.6)
    ax.text(0.06, 0.94, badge_text, transform=ax.transAxes, fontsize=14.5, fontweight='bold', 
            verticalalignment='top', bbox=props)
    
    # Extra large axis labels and ticks
    if t_idx in [3, 4, 5]:
        ax.set_xlabel('Observed OD$_{600}$', fontsize=15.0, fontweight='bold', color='#1A365D')
    if t_idx in [0, 3]:
        ax.set_ylabel('Predicted OD$_{600}$', fontsize=15.0, fontweight='bold', color='#1A365D')
        
    ax.tick_params(labelsize=12.5)
    ax.grid(True, linestyle='--', alpha=0.5)

# 5. Prominent, Extra-Large Species Legend Across the Bottom
legend_handles = [
    mlines.Line2D([], [], color=species_colors[sp], marker='o', linestyle='None',
                  markersize=9, label=f'Sp {sp+1}')
    for sp in range(12)
]
fig.legend(handles=legend_handles, loc='lower center', ncol=6, fontsize=13.5, 
           frameon=True, facecolor='#FFFFFF', edgecolor='#0F2942', framealpha=0.95,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_path = 'Poster/figures/Ophelia_Test_Scatter.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved large font 6-panel Ophelia test figure with species legend to {out_path}")

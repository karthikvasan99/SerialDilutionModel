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
train_mat = loadmat('Data/train_data.mat')['train_data']
train_valid = [np.nan_to_num(train_mat[i], nan=0.0) for i in range(train_mat.shape[0]) 
              if not np.isnan(train_mat[i][0]).any() and np.sum(train_mat[i][0]) > 1e-4]
train_tensor = torch.tensor(np.array(train_valid), dtype=torch.float32, device=device)

test_mat = loadmat('Data/test_data.mat')['test_data']
test_valid = [np.nan_to_num(test_mat[i], nan=0.0) for i in range(test_mat.shape[0]) 
              if not np.isnan(test_mat[i][0]).any() and np.sum(test_mat[i][0]) > 1e-4]
val_tensor = torch.tensor(np.array(test_valid), dtype=torch.float32, device=device)

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

# 3. Simulate on Train and Held-Out Test Data
t_span = torch.linspace(0.0, 5.0, 6, device=device)
with torch.no_grad():
    Theta, W, alpha, R0_table, c = model.get_params()
    
    # Train rollout
    N0_train = train_tensor[:, 0, :]
    batch_size_tr = N0_train.size(0)
    R0_batch_tr = R0_table[0].unsqueeze(0).expand(batch_size_tr, -1)
    S0_train = torch.cat([N0_train, R0_batch_tr], dim=-1)
    S_t_train = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha, c), S0_train, t_span, method='euler', options={'step_size': 0.1})
    train_pred = torch.clamp(S_t_train[..., :12].permute(1, 0, 2), min=0.0)
    
    # Test rollout
    N0_val = val_tensor[:, 0, :]
    batch_size_val = N0_val.size(0)
    R0_batch_val = R0_table[0].unsqueeze(0).expand(batch_size_val, -1)
    S0_val = torch.cat([N0_val, R0_batch_val], dim=-1)
    S_t_val = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha, c), S0_val, t_span, method='euler', options={'step_size': 0.1})
    val_pred = torch.clamp(S_t_val[..., :12].permute(1, 0, 2), min=0.0)

train_true_np = train_tensor.cpu().numpy()
train_pred_np = train_pred.cpu().numpy()
val_true_np = val_tensor.cpu().numpy()
val_pred_np = val_pred.cpu().numpy()

# 4. Create Six-Panel Figure Sized for Column Width with Right-Side Legend
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.8
plt.rcParams['xtick.major.width'] = 1.6
plt.rcParams['ytick.major.width'] = 1.6
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

fig, axes = plt.subplots(2, 3, figsize=(15.2, 6.2), dpi=300)
axes = axes.flatten()

species_colors = plt.cm.tab20(np.linspace(0, 1, 12))

# --- Panel 0: Full Training Fit (Dynamic timepoints t = 12h to 60h) ---
ax0 = axes[0]
o_tr_dyn = train_true_np[:, 1:, :]  # (102, 5, 12)
p_tr_dyn = train_pred_np[:, 1:, :]

o_tr_flat = o_tr_dyn.flatten()
p_tr_flat = p_tr_dyn.flatten()
var_tr = np.sum((o_tr_flat - np.mean(o_tr_flat))**2)
r2_tr = 1.0 - np.sum((o_tr_flat - p_tr_flat)**2) / var_tr

for sp in range(12):
    o_sp = o_tr_dyn[:, :, sp].flatten()
    p_sp = p_tr_dyn[:, :, sp].flatten()
    ax0.scatter(o_sp, p_sp, color=species_colors[sp], alpha=0.35, s=22, edgecolors='none')

max_val0 = max(np.max(o_tr_flat), np.max(p_tr_flat)) * 1.08
ax0.plot([0, max_val0], [0, max_val0], 'k--', lw=2.0, alpha=0.8)
ax0.set_xlim(-0.02 * max_val0, max_val0)
ax0.set_ylim(-0.02 * max_val0, max_val0)

badge_tr = (
    r"$\mathbf{Training\ Fit}$"
    + "\n"
    + rf"$R^2 = {r2_tr:.3f}$"
)
props_tr = dict(boxstyle='round,pad=0.45', facecolor='#EBF3FA', edgecolor='#00356B', alpha=0.95, lw=1.6)
ax0.text(0.06, 0.94, badge_tr, transform=ax0.transAxes, fontsize=15.0, fontweight='bold',
         verticalalignment='top', bbox=props_tr)
ax0.tick_params(labelsize=14)
ax0.grid(True, linestyle='--', alpha=0.3)

# --- Panels 1-5: Dynamic Generalization (t = 12h to 60h) ---
test_time_labels = ['12', '24', '36', '48', '60']

for i, t_idx in enumerate(range(1, 6)):
    ax = axes[i + 1]
    o_t = val_true_np[:, t_idx, :]
    p_t = val_pred_np[:, t_idx, :]
    
    o_flat = o_t.flatten()
    p_flat = p_t.flatten()
    
    var_t = np.sum((o_flat - np.mean(o_flat))**2)
    r2_t = 1.0 - np.sum((o_flat - p_flat)**2) / var_t if var_t > 0 else 1.0
    
    for sp in range(12):
        ax.scatter(o_t[:, sp], p_t[:, sp], color=species_colors[sp], alpha=0.65, s=32, edgecolors='none')
        
    max_val = max(np.max(o_flat), np.max(p_flat)) * 1.08
    ax.plot([0, max_val], [0, max_val], 'k--', lw=2.0, alpha=0.8)
    ax.set_xlim(-0.02 * max_val, max_val)
    ax.set_ylim(-0.02 * max_val, max_val)
    
    badge_text = (
        rf"$\mathbf{{t = {test_time_labels[i]}\ h}}$"
        + "\n"
        + rf"$R^2 = {r2_t:.3f}$"
    )
    props = dict(boxstyle='round,pad=0.45', facecolor='#F0F4F8', edgecolor='#0F2942', alpha=0.92, lw=1.6)
    ax.text(0.06, 0.94, badge_text, transform=ax.transAxes, fontsize=15.0, fontweight='bold',
            verticalalignment='top', bbox=props)
        
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)

# 5. Prominent Species Legend Box on the Right
legend_handles = [
    mlines.Line2D([], [], color=species_colors[sp], marker='o', linestyle='None',
                  markersize=10, label=f'Sp {sp+1}')
    for sp in range(12)
]
fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.88, 0.5),
           ncol=1, fontsize=13.5, labelspacing=0.55,
           frameon=True, facecolor='#FFFFFF', edgecolor='black', framealpha=0.95,
           title=r'$\mathbf{Species}$', title_fontsize=14.5)

# 6. Single Shared X and Y Axis Labels
plt.tight_layout(rect=[0.035, 0.045, 0.88, 0.98])
fig.supxlabel('Observed OD', fontsize=18.0, fontweight='bold', color='black', y=0.015)
fig.supylabel('Predicted OD', fontsize=18.0, fontweight='bold', color='black', x=0.012)

out_path = 'Poster/figures/Ophelia_Test_Scatter.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved updated Ophelia figure to {out_path}")

"""
Train Ophelia (Venturelli et al., 12 Strains) with Physical Byproduct Secretion Consumer-Resource Neural ODE.
Equations:
  dN_i/dt = N_i * sum_k( Theta_ik * R_k ) - alpha_i * N_i^2
  flux_k = c * R_k * sum_j( N_j * Theta_jk )
  byproduct_k = sum_{l != k} flux_l * W_lk
  dR_k/dt = - flux_k + byproduct_k
Where:
  W >= 0, W_ll = 0, sum_k W_lk <= max_crossfeed < 1.0 (mass conservation).
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
from torchdiffeq import odeint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Training Ophelia Byproduct ODE on: {device}")

# 1. Load Data
train_mat = loadmat('Data/train_data.mat')['train_data']
train_valid = [np.nan_to_num(train_mat[i], nan=0.0) for i in range(train_mat.shape[0]) 
              if not np.isnan(train_mat[i][0]).any() and np.sum(train_mat[i][0]) > 1e-4]
train_tensor = torch.tensor(np.array(train_valid), dtype=torch.float32, device=device)

test_mat = loadmat('Data/test_data.mat')['test_data']
test_valid = [np.nan_to_num(test_mat[i], nan=0.0) for i in range(test_mat.shape[0]) 
              if not np.isnan(test_mat[i][0]).any() and np.sum(test_mat[i][0]) > 1e-4]
val_tensor = torch.tensor(np.array(test_valid), dtype=torch.float32, device=device)

N, K = 12, 7

class OpheliaByproductODE(nn.Module):
    def __init__(self, K=7, max_crossfeed=0.40):
        super().__init__()
        self.N, self.K = 12, K
        self.max_crossfeed = max_crossfeed
        self.a = nn.Parameter(torch.randn(1, K) * 0.2 + 1.5)
        self.theta_raw = nn.Parameter(torch.randn(12, K) * 0.1 + 0.5)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1 - 2.0)
        self.alpha_raw = nn.Parameter(torch.randn(12) * 0.05 - 0.5)
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
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        growth = torch.matmul(R_t, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        flux = c * R_t * torch.matmul(N_t, Theta)
        byproduct = torch.matmul(flux, W)
        dR = - flux + byproduct
        return torch.cat([dN, dR], dim=-1)

torch.manual_seed(42)
np.random.seed(42)
model = OpheliaByproductODE(K=7).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.015)
t_span = torch.linspace(0.0, 5.0, 6, device=device)

best_loss = float('inf')
best_weights = None

epochs = 600
for ep in range(1, epochs + 1):
    opt.zero_grad()
    Theta, W, alpha, R0_table, c = model.get_params()
    N0 = train_tensor[:, 0, :]
    R0 = R0_table[0].unsqueeze(0).expand(train_tensor.size(0), -1)
    S0 = torch.cat([N0, R0], dim=-1)
    
    S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha, c), S0, t_span, method='euler', options={'step_size': 0.1})
    N_pred = torch.clamp(S_t[..., :12].permute(1, 0, 2), min=0.0)
    
    loss = F.mse_loss(N_pred, train_tensor)
    loss.backward()
    opt.step()
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
    if ep % 100 == 0 or ep == epochs:
        print(f"Epoch {ep:3d} | Train MSE: {loss.item():.5f} | Best MSE: {best_loss:.5f}")

model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

# Evaluate on Held-Out Test Set
with torch.no_grad():
    Theta, W, alpha, R0_table, c = model.get_params()
    N0_val = val_tensor[:, 0, :]
    R0_val = R0_table[0].unsqueeze(0).expand(val_tensor.size(0), -1)
    S0_val = torch.cat([N0_val, R0_val], dim=-1)
    S_t_val = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha, c), S0_val, t_span, method='euler', options={'step_size': 0.1})
    val_pred = torch.clamp(S_t_val[..., :12].permute(1, 0, 2), min=0.0).cpu().numpy()

val_true = val_tensor.cpu().numpy()
print("\n" + "="*60)
print(f"OPHELIA HELD-OUT TEST EVALUATION (73 UNSEEN COMMUNITIES)")
print("="*60)
for t in range(6):
    o_flat = val_true[:, t, :].flatten()
    p_flat = val_pred[:, t, :].flatten()
    var = np.sum((o_flat - np.mean(o_flat))**2)
    r2 = 1.0 - np.sum((o_flat - p_flat)**2) / var if var > 0 else 1.0
    r, _ = pearsonr(o_flat, p_flat) if np.std(p_flat) > 0 else (1.0, 0.0)
    print(f"t = {t*12:2d}h: R2 = {r2:.4f}, Pearson r = {r:.4f}")

# Save locked checkpoint
save_path = 'Data/Ophelia_Physical_ODE_12Strains.pth'
torch.save({
    'model_state_dict': best_weights,
    'num_species': 12,
    'K': 7,
    'best_loss': best_loss,
    'architecture': 'byproduct_secretion_linear_mass_action'
}, save_path)
print(f"\nSaved Ophelia Byproduct Secretion model to {save_path}")

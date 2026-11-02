"""
Training Script: Clean Physical Consumer-Resource Neural ODE for Gore et al.
Guarantees:
- Pure physical biomass propagation across serial dilution transfers: N0 = N(tau)/30.0 + mu * mask
- Zero compounding yield multiplier (biomass is directly in OD units)
- Cross-feeding byproduct matrix W >= 0 with thermodynamic dissipation
- Monotonic environmental resource pools: R0(LN) < R0(MN) < R0(HN)
- Intra-specific quadratic self-limitation floor (alpha_i >= 0.02)
- Saves clean checkpoint to Data/Gore_Clean_Physical_ODE_K4.pth
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

# Path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load Gore processed dataset
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
data = torch.load(data_path, weights_only=False)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

# Prepare Batch 11 (11 timepoints: initial + 10 daily passages)
# and Batch 4 (4 timepoints: initial + days 7, 8, 10)
b11_N, b11_c, b11_m, b11_keys = [], [], [], []
b4_N, b4_c, b4_m, b4_keys = [], [], [], []

for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    c_idx = cond_to_idx[nutrient]
    mask = data['presence_masks'][(S, comm)]
    od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
    
    if traj_rel.shape[0] == 11:
        b11_N.append(traj_rel * od_mean.unsqueeze(-1))
        b11_c.append(c_idx)
        b11_m.append(mask)
        b11_keys.append((nutrient, S, comm))
    elif traj_rel.shape[0] == 4:
        od_4 = od_mean[[0, 7, 8, 10]]
        b4_N.append(traj_rel * od_4.unsqueeze(-1))
        b4_c.append(c_idx)
        b4_m.append(mask)
        b4_keys.append((nutrient, S, comm))

b11_N = torch.stack(b11_N).to(device) # (75, 11, 48)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)

b4_N = torch.stack(b4_N).to(device)   # (24, 4, 48)
b4_c = torch.tensor(b4_c, device=device)
b4_m = torch.stack(b4_m).to(device)

print(f"Loaded Gore Data: Batch 11 = {b11_N.size(0)} communities | Batch 4 = {b4_N.size(0)} communities")

class CleanGorePhysicalODE(nn.Module):
    def __init__(self, N=48, K=4, D=30.0, mu=1e-5):
        super().__init__()
        self.N = N
        self.K = K
        self.D = D
        self.mu = mu
        
        # Metabolic traits Theta >= 0
        init_scale = 3.6 / float(K)
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.1 + init_scale)
        # Byproduct matrix W >= 0 (metabolic cross-feeding)
        self.W_raw = nn.Parameter(torch.randn(K, K) * 0.1 - 0.5)
        # Intra-specific self-limitation
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.05 + 0.1)
        
        # Environmental supply R0: LN < MN < HN
        self.a_ln = nn.Parameter(torch.randn(K) * 0.1 + 1.2)
        self.delta_mn = nn.Parameter(torch.randn(K) * 0.1 + 0.8)
        self.delta_hn = nn.Parameter(torch.randn(K) * 0.1 + 1.5)
        
    def get_R0(self):
        r_ln = F.softplus(self.a_ln)
        r_mn = r_ln + F.softplus(self.delta_mn)
        r_hn = r_mn + F.softplus(self.delta_hn)
        return torch.stack([r_ln, r_mn, r_hn])
        
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        Theta = F.softplus(self.Theta_raw)
        # Thermodynamic dissipation: row sum < 0.85
        W = F.softmax(self.W_raw, dim=-1) * 0.85
        alpha = F.softplus(self.alpha_raw) + 0.02
        
        growth = torch.matmul(R_t, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        
        uptake = torch.matmul(N_t, Theta)
        byproduct = torch.matmul(uptake, W)
        dR = - R_t * uptake + byproduct
        return torch.cat([dN, dR], dim=-1)

def dixit_generalized_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    pred_safe = torch.clamp(y_pred, min=eps)
    y_for_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_for_log / pred_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + pred_safe) * mask
    return div.sum(dim=-1).mean()

def train_model(epochs=150, lr=0.02, delta_t=0.05):
    model = CleanGorePhysicalODE(N=48, K=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)
    t_span = torch.linspace(0.0, 1.0, int(1.0 / delta_t) + 1, device=device)
    
    best_loss = float('inf')
    best_state = None
    
    print(f"\nTraining CleanGorePhysicalODE for {epochs} epochs on {device}...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        R0_11 = model.get_R0()[b11_c]
        R0_4  = model.get_R0()[b4_c]
        
        # Scheduled sampling decay to 0.05 over 75% of training
        raw_tf = 1.0 - (epoch / (epochs * 0.75))
        tf_ratio = max(0.05, min(1.0, raw_tf))
        
        # Batch 11 rollout (all 10 passages)
        N_curr = b11_N[:, 0, :]
        loss11 = 0.0
        for p in range(1, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_11], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            loss11 += dixit_generalized_kl(b11_N[:, p, :], N_final, b11_m)
            N_curr = tf_ratio * b11_N[:, p, :] + (1.0 - tf_ratio) * N_final
            
        # Batch 4 rollout
        N_curr_4 = b4_N[:, 0, :]
        loss4 = 0.0
        d4_map = {7: 1, 8: 2, 10: 3}
        for day in range(1, 11):
            N_in_4 = torch.clamp(N_curr_4 / model.D + model.mu * b4_m, min=0.0)
            S0_4 = torch.cat([N_in_4, R0_4], dim=-1)
            S_t_4 = odeint(model.ode_deriv, S0_4, t_span, method='euler', options={'step_size': delta_t})
            N_final_4 = torch.clamp(S_t_4[-1, ..., :48] * b4_m, min=0.0)
            if day in d4_map:
                t_idx = d4_map[day]
                loss4 += dixit_generalized_kl(b4_N[:, t_idx, :], N_final_4, b4_m)
                N_curr_4 = tf_ratio * b4_N[:, t_idx, :] + (1.0 - tf_ratio) * N_final_4
            else:
                N_curr_4 = N_final_4
                
        # Biological trait L2 penalty
        reg_penalty = 0.001 * (model.Theta_raw.pow(2.0).sum() + model.W_raw.pow(2.0).sum())
        total_loss = (loss11 / 10.0) + (loss4 / 3.0) + reg_penalty
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 25 == 0 or epoch == epochs or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {total_loss.item():.4f} (TF: {tf_ratio:.2f}) | Best: {best_loss:.4f} | Time: {elapsed:.1f}s")
            
    # Save checkpoint
    save_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Clean_Physical_ODE_K4.pth')
    torch.save({
        'model_state_dict': best_state,
        'N': 48,
        'K': 4,
        'D': 30.0,
        'mu': 1e-5,
        'best_loss': best_loss
    }, save_path)
    print(f"\nSaved clean Gore model to: {save_path}")
    
    # Load best weights
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model

if __name__ == '__main__':
    model = train_model(epochs=150, lr=0.02, delta_t=0.05)

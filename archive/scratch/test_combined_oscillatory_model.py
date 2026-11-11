import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Gore processed data
data_path = 'Data/Gore_processed_data.pt'
data = torch.load(data_path, weights_only=False)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

b11_N, b11_c, b11_m, b11_names = [], [], [], []
for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    if traj_rel.shape[0] == 11:
        od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
        mask = data['presence_masks'][(S, comm)]
        b11_N.append(traj_rel * od_mean.unsqueeze(-1))
        b11_c.append(cond_to_idx[nutrient])
        b11_m.append(mask)
        b11_names.append((nutrient, S, comm))

b11_N = torch.stack(b11_N).to(device)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)

def dixit_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    p_safe = torch.clamp(y_pred, min=eps)
    y_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_log / p_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + p_safe) * mask
    return div.sum(dim=-1).mean()

# Model Architecture
class CombinedOscillatoryModel(nn.Module):
    def __init__(self, N=48, K=3, D=30.0, mu=1e-4, Km=0.05, alpha=0.0):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        self.Km = Km
        self.alpha = alpha
        
        # 1. Decoupled Theta and Phi
        # Initialized with niche specialization bias
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.3 + 1.2)
        self.Phi_raw = nn.Parameter(torch.randn(N, K) * 0.3 + 1.2)
        
        # 2. Fixed Non-Learnable Cyclic W with 0s and 0.99
        W_fixed = torch.tensor([
            [0.0, 0.99, 0.0],
            [0.0, 0.0, 0.99],
            [0.99, 0.0, 0.0]
        ], device=device)
        self.register_buffer('W', W_fixed)
        
        # 3. Nutrient pulses (Primary nutrient focused on R1)
        self.scale_ln = nn.Parameter(torch.tensor(0.4))
        self.scale_mn = nn.Parameter(torch.tensor(0.8))
        self.scale_hn = nn.Parameter(torch.tensor(1.6))
        # Initial bias to pulse predominantly into primary nutrient R1
        self.r_dist = nn.Parameter(torch.tensor([2.5, -1.0, -1.0]))

    def get_R0(self):
        s_ln = F.softplus(self.scale_ln)
        s_mn = F.softplus(self.scale_mn)
        s_hn = F.softplus(self.scale_hn)
        dist = F.softmax(self.r_dist, dim=-1)
        return torch.stack([dist * s_ln, dist * s_mn, dist * s_hn])

    def get_Theta(self):
        return F.softplus(self.Theta_raw)

    def get_Phi(self):
        return F.softplus(self.Phi_raw)

    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        Theta = self.get_Theta()
        Phi = self.get_Phi()
        
        monod = R_t / (R_t + self.Km)
        growth = torch.matmul(monod, Theta.t())
        dN = N_t * growth - self.alpha * (N_t ** 2)
        
        flux = monod * torch.matmul(N_t, Phi)
        byproduct = torch.matmul(flux, self.W)
        dR = - flux + byproduct
        return torch.cat([dN, dR], dim=-1)

# Training with:
# 1. 1-Step fit loss
# 2. Sparsity on Theta: lambda_theta * mean(Theta)
# 3. Sparsity on Phi: lambda_phi * mean(Phi)
# 4. Cosine similarity penalty between Theta and Phi: lambda_sim * CosineSimilarity(Theta, Phi)
def train_and_evaluate(lambda_sparse=0.05, lambda_sim=2.0, epochs=35):
    print("\n" + "="*80)
    print(f"TRAINING MODEL: lambda_sparse = {lambda_sparse}, lambda_sim = {lambda_sim}")
    print("="*80)
    torch.manual_seed(42)
    model = CombinedOscillatoryModel(N=48, K=3, D=30.0, alpha=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    t_span = torch.linspace(0.0, 1.0, 25, device=device)
    
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        R0_all = model.get_R0()[b11_c]
        loss_fit = 0.0
        for p in range(1, 11):
            N_in = torch.clamp(b11_N[:, p-1, :] / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_pred = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            loss_fit += dixit_kl(b11_N[:, p, :], N_pred, b11_m)
        loss_fit = loss_fit / 10.0
        
        Theta = model.get_Theta()
        Phi = model.get_Phi()
        
        # Sparsity penalty (L1 norm encouraging specialization)
        loss_sparse = lambda_sparse * (Theta.mean() + Phi.mean())
        
        # Cosine similarity penalty between Theta and Phi
        cos_sim = F.cosine_similarity(Theta.flatten(), Phi.flatten(), dim=0)
        loss_sim = lambda_sim * cos_sim
        
        total_loss = loss_fit + loss_sparse + loss_sim
        total_loss.backward()
        opt.step()
        
        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:2d}/{epochs} | Fit: {loss_fit.item():.4f} | Sparse: {loss_sparse.item():.4f} | CosSim: {cos_sim.item():.4f} | Loss: {total_loss.item():.4f}")
            
    # Simulate 80 passages under DOPRI5
    print("\nSimulating 80 autonomous passages under DOPRI5...")
    model.eval()
    with torch.no_grad():
        R0_hn = model.get_R0()[2]
        print(f"R0_HN allocation: {R0_hn.cpu().numpy()} (Total = {R0_hn.sum().item():.3f})")
        
        # Check Community 1 (species [21, 22, 23])
        N_init = b11_N[0, 0, :]
        S = torch.cat([N_init, R0_hn])
        traj = []
        for p in range(80):
            N_in = S[:48] / model.D + model.mu * b11_m[0]
            S_start = torch.cat([N_in, R0_hn])
            S_t = odeint(model.ode_deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
            S = torch.clamp(S_t[-1], min=0.0)
            traj.append(S[:48].cpu().numpy())
            
        traj = np.array(traj)
        
    sp_active = [21, 22, 23]
    last20 = traj[60:, sp_active]
    diffs = np.diff(last20, axis=0)
    reversals = np.sum(diffs[:-1] * diffs[1:] < -1e-6, axis=0)
    p2p = (last20.max(axis=0) - last20.min(axis=0)) / np.maximum(last20.mean(axis=0), 1e-4)
    print("\nResults for Active Species [21, 22, 23] in Passages 60-80:")
    for idx, sp in enumerate(sp_active):
        print(f"  Species {sp}: Mean = {last20[:, idx].mean():.3f}, P2P Amplitude = {p2p[idx]*100:.2f}%, Reversals = {reversals[idx]}")
    
    return model, traj, p2p, reversals

if __name__ == '__main__':
    # Test a grid of (lambda_sparse, lambda_sim)
    for l_sp in [0.02, 0.10]:
        for l_sim in [1.0, 4.0]:
            train_and_evaluate(lambda_sparse=l_sp, lambda_sim=l_sim, epochs=25)

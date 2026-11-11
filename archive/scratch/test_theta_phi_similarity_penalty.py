import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load data and metadata
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

class DecoupledCRModel(nn.Module):
    def __init__(self, N=48, K=3, D=30.0, mu=1e-4, alpha=0.0):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        self.Km = 0.05
        self.alpha = alpha
        
        # Initializations
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.2 + 1.5)
        self.Phi_raw = nn.Parameter(torch.randn(N, K) * 0.2 + 1.5)
        
        # Nutrient pulses
        self.scale_ln = nn.Parameter(torch.tensor(0.4))
        self.scale_mn = nn.Parameter(torch.tensor(0.8))
        self.scale_hn = nn.Parameter(torch.tensor(1.6))
        self.r_dist = nn.Parameter(torch.tensor([2.0, -1.0, -1.0]))
        
        # Cyclic W
        W_init = torch.tensor([
            [0.0, 0.99, 0.0],
            [0.0, 0.0, 0.99],
            [0.99, 0.0, 0.0]
        ], device=device)
        self.register_buffer('W', W_init)

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

# Run test across multiple lambda_sim
results = {}
for lambda_sim in [0.0, 1.0, 5.0]:
    print(f"\n" + "="*70)
    print(f"EXPERIMENT: lambda_sim = {lambda_sim} (Penalty on CosineSimilarity(Theta, Phi))")
    print("="*70)
    torch.manual_seed(42)
    model = DecoupledCRModel(N=48, K=3, D=30.0, alpha=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    t_span = torch.linspace(0.0, 1.0, 25, device=device)
    
    for ep in range(1, 26):
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
        cos_sim = F.cosine_similarity(Theta.flatten(), Phi.flatten(), dim=0)
        
        total_loss = loss_fit + lambda_sim * cos_sim
        total_loss.backward()
        opt.step()
        
        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep:2d} | Fit Loss: {loss_fit.item():.4f} | Cosine Sim(Theta, Phi): {cos_sim.item():.4f}")
            
    # Simulate 50 passages with dopri5
    print("Testing long-term dynamics under DOPRI5 (50 Passages)...")
    model.eval()
    with torch.no_grad():
        R0_hn = model.get_R0()[2]
        N_init = b11_N[0, 0, :]
        S = torch.cat([N_init, R0_hn])
        traj = []
        for p in range(50):
            N_in = S[:48] / model.D + model.mu * b11_m[0]
            S_start = torch.cat([N_in, R0_hn])
            S_t = odeint(model.ode_deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
            S = torch.clamp(S_t[-1], min=0.0)
            traj.append(S[:48].cpu().numpy())
            
        traj = np.array(traj)
        last_std = traj[30:].std(axis=0)
        last_mean = traj[30:].mean(axis=0)
        active = last_mean > 0.01
        cv = (last_std[active] / last_mean[active]).mean() if np.sum(active) > 0 else 0.0
        print(f"Late passages mean CV across active species: {cv:.4f}")
        if cv > 0.05:
            print(">>> STATUS: SUSTAINED OSCILLATIONS OBSERVED UNDER DOPRI5! <<<")
        else:
            print(">>> STATUS: DAMPED / FIXED POINT (NO SUSTAINED OSCILLATIONS) <<<")
            
    results[lambda_sim] = {'fit_loss': loss_fit.item(), 'cos_sim': cos_sim.item(), 'cv': cv}

print("\nSUMMARY OF RESULTS:")
for k, v in results.items():
    print(f"lambda_sim = {k}: Fit Loss = {v['fit_loss']:.4f}, Cosine Sim = {v['cos_sim']:.4f}, CV = {v['cv']:.4f}")

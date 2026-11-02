import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
data = torch.load(data_path, weights_only=False)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

b11_N, b11_c, b11_m, b11_keys = [], [], [], []
for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    if traj_rel.shape[0] == 11:
        od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
        mask = data['presence_masks'][(S, comm)]
        b11_N.append(traj_rel * od_mean.unsqueeze(-1))
        b11_c.append(cond_to_idx[nutrient])
        b11_m.append(mask)
        b11_keys.append((nutrient, S, comm))

b11_N = torch.stack(b11_N).to(device) # (75, 11, 48)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)

def dixit_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    p_safe = torch.clamp(y_pred, min=eps)
    y_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_log / p_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + p_safe) * mask
    return div.sum(dim=-1).mean()

class PersistentChaosODE(nn.Module):
    def __init__(self, N=48, K=5, D=30.0, mu=1e-5):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        
        # Niche-differentiated guild initialization
        theta_init = torch.randn(N, K) * 0.15 + 0.35
        for i in range(N):
            guild = i % K
            theta_init[i, guild] += 2.2  # strong primary guild affinity
        self.Theta_raw = nn.Parameter(theta_init)
        
        # Cross-feeding matrix W: cyclic feeding bias (guild k feeds guild k+1)
        w_init = torch.randn(K, K) * 0.1 - 0.5
        for k in range(K):
            w_init[k, (k + 1) % K] += 1.5
            w_init[k, k] = -5.0  # suppress self-feeding
        self.W_raw = nn.Parameter(w_init)
        
        # Strict low-alpha bound: alpha in [1e-5, 0.005]
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.5 - 1.0)
        
        # Km in [0.05, 0.5]
        self.Km_raw = nn.Parameter(torch.zeros(K) - 1.2)
        
        # R0: LN < MN < HN
        self.a_ln = nn.Parameter(torch.randn(K) * 0.1 + 0.5)
        self.delta_mn = nn.Parameter(torch.randn(K) * 0.1 + 0.6)
        self.delta_hn = nn.Parameter(torch.randn(K) * 0.1 + 1.4)
        
    def get_R0(self):
        r_ln = F.softplus(self.a_ln)
        r_mn = r_ln + F.softplus(self.delta_mn)
        r_hn = r_mn + F.softplus(self.delta_hn)
        return torch.stack([r_ln, r_mn, r_hn])
        
    def get_W(self):
        # Dissipative mass-conserving row normalization (sum <= 0.85)
        W_sp = F.softplus(self.W_raw)
        # zero diagonal
        W_sp = W_sp * (1.0 - torch.eye(self.K, device=W_sp.device))
        row_sums = W_sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(0.85 / torch.clamp(row_sums, min=1e-6), max=1.0)
        return W_sp * scale
        
    def get_alpha(self):
        # Strict bound: alpha in [1e-5, 0.005]
        return 0.005 * torch.sigmoid(self.alpha_raw) + 1e-5
        
    def get_Km(self):
        return F.softplus(self.Km_raw) + 0.05
        
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        Theta = F.softplus(self.Theta_raw)
        W = self.get_W()
        alpha = self.get_alpha()
        Km = self.get_Km()
        
        monod = R_t / (R_t + Km)
        growth = torch.matmul(monod, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        
        flux = monod * torch.matmul(N_t, Theta)
        byproduct = torch.matmul(flux, W)
        dR = - flux + byproduct
        return torch.cat([dN, dR], dim=-1)

print("PersistentChaosODE defined. Running test...")

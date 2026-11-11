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

# 1. Load Gore Data
data_path = 'Data/Gore_processed_data.pt'
data = torch.load(data_path, weights_only=False)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

# Select top fluctuating HN communities:
# ('HN', 24, 'Community1'), ('HN', 24, 'Community6'), ('HN', 6, 'Community1')
# Let's collect all HN fluctuating communities where CV > 0.40
fluct_N, fluct_m, fluct_names = [], [], []
for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    if traj_rel.shape[0] == 11 and nutrient == 'HN':
        od = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
        biomass = traj_rel * od.unsqueeze(-1)
        mask = data['presence_masks'][(S, comm)]
        active = torch.nonzero(mask).flatten().tolist()
        late_std = biomass[5:, active].std(dim=0)
        late_mean = biomass[5:, active].mean(dim=0)
        cv = (late_std / torch.clamp(late_mean, min=1e-4)).mean().item()
        if cv >= 0.40:
            fluct_N.append(biomass)
            fluct_m.append(mask)
            fluct_names.append((S, comm, cv))

print(f"Selected {len(fluct_N)} highly fluctuating HN communities (CV >= 0.40):")
for name in fluct_names:
    print(f"  S={name[0]}, {name[1]}: CV = {name[2]:.3f}")

fluct_N = torch.stack(fluct_N).to(device) # (B, 11, 48)
fluct_m = torch.stack(fluct_m).to(device) # (B, 48)
B = fluct_N.shape[0]

# Empirical variance in late passages (passages 6..10)
emp_std = fluct_N[:, 6:, :].std(dim=1) * fluct_m

def dixit_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    p_safe = torch.clamp(y_pred, min=eps)
    y_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_log / p_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + p_safe) * mask
    return div.sum(dim=-1).mean()

# Architecture
class OptimalOscillatoryModel(nn.Module):
    def __init__(self, N=48, K=3, D=30.0, mu=1e-4, Km=0.05, Kd=0.02):
        super().__init__()
        self.N, self.K, self.D, self.mu, self.Kd = N, K, D, mu, Kd
        self.Km = Km
        self.alpha = 0.0 # Strictly zero damping
        
        # Growth and Uptake
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.3 + 1.2)
        self.Phi_raw = nn.Parameter(torch.randn(N, K) * 0.3 + 1.2)
        
        # Starvation decay rate delta_i
        self.delta_raw = nn.Parameter(torch.zeros(N) + 0.5)
        
        # Cyclic W with 0s and 0.99
        W_fixed = torch.tensor([
            [0.0, 0.99, 0.0],
            [0.0, 0.0, 0.99],
            [0.99, 0.0, 0.0]
        ], device=device)
        self.register_buffer('W', W_fixed)
        
        # Nutrient pulse (dominant on R1)
        self.scale_hn = nn.Parameter(torch.tensor(1.6))
        self.r_dist = nn.Parameter(torch.tensor([2.5, -1.0, -1.0]))

    def get_R0(self):
        s_hn = F.softplus(self.scale_hn)
        dist = F.softmax(self.r_dist, dim=-1)
        return dist * s_hn

    def get_Theta(self): return F.softplus(self.Theta_raw)
    def get_Phi(self): return F.softplus(self.Phi_raw)
    def get_delta(self): return F.softplus(self.delta_raw) * 2.0 + 0.5

    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        Theta = self.get_Theta()
        Phi = self.get_Phi()
        delta = self.get_delta()
        
        monod = R_t / (R_t + self.Km)
        growth = torch.matmul(monod, Theta.t())
        
        # Starvation gate
        R_tot = R_t.sum(dim=-1, keepdim=True)
        starv = self.Kd / (R_tot + self.Kd)
        death = delta * starv
        dN = N_t * (growth - death)
        
        flux = monod * torch.matmul(N_t, Phi)
        byproduct = torch.matmul(flux, self.W)
        dR = - flux + byproduct
        return torch.cat([dN, dR], dim=-1)

# Training with the Optimal Set of Penalties:
# 1. 1-Step flow matching on chaotic communities
# 2. Multi-step late rollout variance matching (encourages non-static states)
# 3. Skew-symmetry maximization on A = Theta R0 Phi^T
# 4. Reciprocal competition penalty on A
def train_optimal_model(epochs=40, lr=0.015, lambda_var=10.0, lambda_skew=1.0, lambda_recip=0.5):
    torch.manual_seed(42)
    model = OptimalOscillatoryModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t_span = torch.linspace(0.0, 1.0, 25, device=device)
    triu = torch.triu_indices(model.N, model.N, offset=1)
    
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        R0_hn = model.get_R0()
        R0_batch = R0_hn.unsqueeze(0).expand(B, -1)
        
        # 1. 1-Step Loss
        loss_1step = 0.0
        for p in range(1, 11):
            N_in = torch.clamp(fluct_N[:, p-1, :] / model.D + model.mu * fluct_m, min=0.0)
            S0 = torch.cat([N_in, R0_batch], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_pred = torch.clamp(S_t[-1, ..., :48] * fluct_m, min=0.0)
            loss_1step += dixit_kl(fluct_N[:, p, :], N_pred, fluct_m)
        loss_1step = loss_1step / 10.0
        
        # 2. Multi-step late rollout variance loss (passages 6..10)
        N_curr = fluct_N[:, 6, :]
        rollout = [N_curr]
        for p in range(7, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * fluct_m, min=0.0)
            S0 = torch.cat([N_in, R0_batch], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_next = torch.clamp(S_t[-1, ..., :48] * fluct_m, min=0.0)
            rollout.append(N_next)
            N_curr = N_next
        rollout_stack = torch.stack(rollout, dim=1)
        model_std = rollout_stack.std(dim=1) * fluct_m
        loss_var = F.mse_loss(model_std, emp_std)
        
        # 3. Skew-Symmetry Maximization on A = Theta @ diag(R0) @ Phi^T
        Theta = model.get_Theta()
        Phi = model.get_Phi()
        A = torch.matmul(Theta * R0_hn.unsqueeze(0), Phi.t()) # (48, 48)
        A_sym = 0.5 * (A + A.t())
        A_skew = 0.5 * (A - A.t())
        skew_ratio = torch.norm(A_skew) / torch.clamp(torch.norm(A_sym), min=1e-6)
        loss_skew = - lambda_skew * skew_ratio
        
        # 4. Non-reciprocity penalty
        loss_recip = lambda_recip * (A[triu[0], triu[1]] * A[triu[1], triu[0]]).mean()
        
        total_loss = loss_1step + lambda_var * loss_var + loss_skew + loss_recip
        total_loss.backward()
        opt.step()
        
        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:2d}/{epochs} | 1-Step: {loss_1step.item():.4f} | VarLoss: {loss_var.item():.5f} | SkewRatio: {skew_ratio.item():.4f} | Total: {total_loss.item():.4f}")
            
    return model

print("\n" + "="*80)
print("TRAINING OPTIMAL OSCILLATORY MODEL ON HIGH-FLUCTUATION GORE COMMUNITIES")
print("="*80)
model = train_optimal_model(epochs=35, lr=0.015, lambda_var=10.0, lambda_skew=1.5, lambda_recip=1.0)

# Evaluate each chaotic community over 60 passages under DOPRI5
print("\n" + "="*80)
print("EVALUATING LONG-TERM DYNAMICS UNDER DOPRI5 ACROSS CHAOTIC COMMUNITIES")
print("="*80)
model.eval()
t_span = torch.linspace(0.0, 1.0, 25, device=device)
R0_hn = model.get_R0()
print(f"Learned HN pulse: R0 = {R0_hn.detach().cpu().numpy()}")

results = []
with torch.no_grad():
    for i, name in enumerate(fluct_names):
        N_init = fluct_N[i, 0, :]
        mask = fluct_m[i]
        active = torch.nonzero(mask).flatten().tolist()
        
        S = torch.cat([N_init, R0_hn])
        traj = []
        for p in range(60):
            N_in = S[:48] / model.D + model.mu * mask
            S_start = torch.cat([N_in, R0_hn])
            S_t = odeint(model.ode_deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
            S = torch.clamp(S_t[-1], min=0.0)
            traj.append(S[:48].cpu().numpy())
            
        traj = np.array(traj) # (60, 48)
        last20 = traj[40:, active] # (20, len(active))
        
        # Check reversals and amplitude
        diffs = np.diff(last20, axis=0)
        reversals = np.sum(diffs[:-1] * diffs[1:] < -1e-6, axis=0)
        p2p = (last20.max(axis=0) - last20.min(axis=0)) / np.maximum(last20.mean(axis=0), 1e-4)
        mean_bio = last20.mean(axis=0)
        surviving = mean_bio > 0.02
        
        num_surv = np.sum(surviving)
        max_flips = reversals[surviving].max() if num_surv > 0 else 0
        mean_p2p = p2p[surviving].mean() if num_surv > 0 else 0.0
        
        status = "SUSTAINED OSCILLATION / LIMIT CYCLE" if (max_flips >= 3 and mean_p2p > 0.15) else "Fixed Point / Damped"
        print(f"Community S={name[0]} {name[1]} (Empirical CV={name[2]:.2f}): Surviving={num_surv}/{len(active)}, Max Reversals={max_flips}, Mean P2P={mean_p2p*100:.1f}% -> {status}")
        results.append((name, num_surv, max_flips, mean_p2p, status, traj))


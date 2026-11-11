"""
Test Alpha Regularization on Authentic Data Fitting to Gore et al. (Science 2022)
We sweep the self-limitation scale / regularization to answer:
1. Does suppressing alpha allow the authentic data-fitted model to bifurcate into sustained non-equilibrium chaos over 200 passages?
2. What is the trade-off between empirical data accuracy and sustained chaos?
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load empirical data
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
data = torch.load(data_path, weights_only=False)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

b11_N, b11_c, b11_m = [], [], []
for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    if traj_rel.shape[0] == 11:
        od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
        mask = data['presence_masks'][(S, comm)]
        b11_N.append(traj_rel * od_mean.unsqueeze(-1))
        b11_c.append(cond_to_idx[nutrient])
        b11_m.append(mask)

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

class AlphaRegularizedGoreODE(nn.Module):
    def __init__(self, N=48, K=6, alpha_max=1.0, D=30.0, mu=1e-6):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        self.alpha_max = alpha_max
        
        # Random initialization without hand-tuned guild structure
        init_scale = 3.4 / (float(K) * 0.8)
        inv_scale = math.log(math.exp(init_scale) - 1.0)
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.15 + inv_scale)
        
        # Random cross-feeding without hand-tuned cyclic structure
        self.W_raw = nn.Parameter(torch.randn(K, K) * 0.2 - 0.5)
        
        # Self-limitation bound: alpha in [1e-5, alpha_max]
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.5 - 1.0)
        
        # Monod Km
        self.Km_raw = nn.Parameter(torch.zeros(K) - 1.0)
        
        # Environmental supply R0: LN < MN < HN (learned)
        self.a_ln = nn.Parameter(torch.randn(K) * 0.1 + 0.8)
        self.delta_mn = nn.Parameter(torch.randn(K) * 0.1 + 0.6)
        self.delta_hn = nn.Parameter(torch.randn(K) * 0.1 + 1.2)
        
    def get_R0(self):
        r_ln = F.softplus(self.a_ln)
        r_mn = r_ln + F.softplus(self.delta_mn)
        r_hn = r_mn + F.softplus(self.delta_hn)
        return torch.stack([r_ln, r_mn, r_hn])
        
    def get_W(self):
        W_sp = F.softplus(self.W_raw)
        W_sp = W_sp * (1.0 - torch.eye(self.K, device=W_sp.device))
        row_sums = W_sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(0.85 / torch.clamp(row_sums, min=1e-6), max=1.0)
        return W_sp * scale
        
    def get_alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_raw) + 1e-5
        
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

def train_and_test(alpha_max, epochs=80, lr=0.015):
    set_seed(42)
    print("\n" + "="*80)
    print(f"TRAINING WITH ALPHA BOUND: alpha in [1e-5, {alpha_max}]")
    print("="*80)
    
    model = AlphaRegularizedGoreODE(N=48, K=6, alpha_max=alpha_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)
    t_span = torch.linspace(0.0, 1.0, 21, device=device)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        R0_all = model.get_R0()[b11_c]
        tf_ratio = max(0.0, 1.0 - epoch / (epochs * 0.75))
        
        N_curr = b11_N[:, 0, :]
        tot_loss = 0.0
        
        for p in range(1, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_final = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            tot_loss += dixit_kl(b11_N[:, p, :], N_final, b11_m)
            N_curr = tf_ratio * b11_N[:, p, :] + (1.0 - tf_ratio) * N_final
            
        data_loss = tot_loss / 10.0
        loss = data_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch >= 30 and loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 20 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Data Loss: {data_loss.item():.4f} (TF: {tf_ratio:.2f}) | Alpha mean: {model.get_alpha().mean().item():.5f}")
            
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    
    # Evaluate 10-day accuracy
    with torch.no_grad():
        preds = []
        for p in range(11):
            if p == 0:
                N_c = b11_N[:, 0, :]
                preds.append(N_c)
            else:
                N_in = torch.clamp(N_c / model.D + model.mu * b11_m, min=0.0)
                S0 = torch.cat([N_in, model.get_R0()[b11_c]], dim=-1)
                S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
                N_c = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
                preds.append(N_c)
        preds = torch.stack(preds, dim=1)
        y_t = b11_N.cpu().numpy().flatten()
        y_p = preds.cpu().numpy().flatten()
        mask_f = b11_m.unsqueeze(1).expand(-1, 11, -1).cpu().numpy().flatten() > 0
        r_val, _ = pearsonr(y_t[mask_f], y_p[mask_f])
        
    print(f"Fitted 10-Day Pearson r = {r_val:.4f} | Mean predicted biomass = {y_p[mask_f].mean():.4f}")
    
    # Simulate 200 Passages on 48-species Community 1 in LN and HN
    def simulate_200(cond):
        c_idx = cond_to_idx[cond]
        mask = data['presence_masks'][(48, 'Community1')].float().to(device)
        rel = data['relative_abundance'][(cond, 48, 'Community1')].to(device)
        od = data['biomass_od'][(cond, 48, 'Community1')].mean(dim=0).to(device)
        N0 = (rel[0] * od[0]) * mask
        R0 = model.get_R0()[c_idx]
        
        N_curr = N0.clone()
        N_traj = [N_curr.cpu().numpy()]
        t_rk4 = torch.linspace(0.0, 1.0, 25, device=device)
        
        with torch.no_grad():
            for p in range(200):
                N_in = torch.clamp(N_curr / model.D + model.mu * mask, min=0.0)
                S0 = torch.cat([N_in, R0], dim=-1)
                S_t = odeint(model.ode_deriv, S0, t_rk4, method='rk4')
                N_curr = torch.clamp(S_t[-1, :48] * mask, min=0.0)
                N_traj.append(N_curr.cpu().numpy())
                
        return np.array(N_traj)
        
    traj_ln = simulate_200('LN')
    traj_hn = simulate_200('HN')
    
    # Measure late fluctuation in passages 100..200
    p2p_ln = (traj_ln[100:].max(axis=0) - traj_ln[100:].min(axis=0)).max()
    p2p_hn = (traj_hn[100:].max(axis=0) - traj_hn[100:].min(axis=0)).max()
    std_hn = np.std(traj_hn[100:], axis=0).max()
    
    # Compute Lyapunov on HN attractor
    active_idx = np.where(traj_hn[150] > 0.005)[0]
    lam_hn = 0.0
    if len(active_idx) > 0:
        eps = 1e-4
        torch.manual_seed(99)
        N_base = torch.tensor(traj_hn[150], device=device).float()
        pert = torch.zeros_like(N_base)
        pert[active_idx] = torch.randn(len(active_idx), device=device)
        pert = (pert / torch.norm(pert)) * eps
        
        N_cb = N_base.clone()
        N_cp = N_base + pert
        R0_hn = model.get_R0()[2]
        mask_48 = data['presence_masks'][(48, 'Community1')].float().to(device)
        t_rk4 = torch.linspace(0.0, 1.0, 25, device=device)
        
        lyap_steps = []
        with torch.no_grad():
            for _ in range(50):
                N_in_b = torch.clamp(N_cb / 30.0 + 1e-6 * mask_48, min=0.0)
                N_in_p = torch.clamp(N_cp / 30.0 + 1e-6 * mask_48, min=0.0)
                Sb = odeint(model.ode_deriv, torch.cat([N_in_b, R0_hn], dim=-1), t_rk4, method='rk4')[-1, :48]
                Sp = odeint(model.ode_deriv, torch.cat([N_in_p, R0_hn], dim=-1), t_rk4, method='rk4')[-1, :48]
                
                diff = (Sp - Sb)[active_idx]
                norm_d = torch.norm(diff).item()
                if norm_d > 1e-12:
                    lyap_steps.append(math.log(norm_d / eps))
                    new_pert = torch.zeros_like(N_base)
                    new_pert[active_idx] = (diff / norm_d) * eps
                    N_cp = Sb + new_pert
                N_cb = Sb
        lam_hn = float(np.mean(lyap_steps)) if len(lyap_steps) > 0 else 0.0
        
    print(f"\n200-Passage Dynamic Outcomes:")
    print(f"  LN Peak-to-Peak (p100-200): {p2p_ln:.6f}")
    print(f"  HN Peak-to-Peak (p100-200): {p2p_hn:.6f}")
    print(f"  HN Max Std     (p100-200): {std_hn:.6f}")
    print(f"  HN Maximal Lyapunov lambda: {lam_hn:+.4f} / passage ({len(active_idx)} active species)")
    
    return {
        'alpha_max': alpha_max,
        'r_val': r_val,
        'p2p_ln': p2p_ln,
        'p2p_hn': p2p_hn,
        'std_hn': std_hn,
        'lam_hn': lam_hn
    }

if __name__ == '__main__':
    # Test 3 distinct regimes of alpha regularization
    # 1. Mild bound (alpha <= 1.0)
    # 2. Moderate bound (alpha <= 0.1)
    # 3. Strong bound (alpha <= 0.01)
    results = []
    for a_max in [1.0, 0.1, 0.01]:
        res = train_and_test(alpha_max=a_max, epochs=60)
        results.append(res)
        
    print("\n" + "="*80)
    print("FINAL SUMMARY: IMPACT OF ALPHA REGULARIZATION ON DATA FIT & 200-PASSAGE CHAOS")
    print("="*80)
    print(f"{'Alpha Bound':<15} | {'10-Day Pearson r':<18} | {'HN P2P (p100-200)':<18} | {'Lyapunov lambda':<16} | {'Regime'}")
    print("-" * 85)
    for r in results:
        regime = "CHAOS" if r['lam_hn'] > 0.05 and r['p2p_hn'] > 0.1 else ("OSCILLATING" if r['p2p_hn'] > 0.01 else "FIXED POINT")
        print(f"alpha <= {r['alpha_max']:<7} | {r['r_val']:<18.4f} | {r['p2p_hn']:<18.4f} | {r['lam_hn']:<+16.4f} | {regime}")

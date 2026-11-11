"""
Systematic Sweep across K in [3, 4, 5, 6, 7] for Gore Physical Consumer-Resource ODE
Key modifications requested by user:
1. ZERO L2 regularization on W (cross-feeding matrix), allowing unrestrained off-diagonal cyclic couplings.
2. Uncompounded physical dilution continuity across transfers.
3. Autonomous 30-passage rollout evaluation: test whether fitted model produces perpetual oscillations in High Nutrients and stability in Low Nutrients.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load data
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

class PhysicalGoreODE(nn.Module):
    def __init__(self, N=48, K=4, D=30.0, mu=1e-5):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        init_scale = 3.6 / float(K)
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.1 + init_scale)
        # Unconstrained cross-feeding W_raw (no L2 penalty!)
        self.W_raw = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.05 + 0.1)
        
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
        # Thermodynamic row-sum dissipation: sum_k W_lk <= 0.85
        W = F.softmax(self.W_raw, dim=-1) * 0.85
        alpha = F.softplus(self.alpha_raw) + 0.02
        
        growth = torch.matmul(R_t, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        
        uptake = torch.matmul(N_t, Theta)
        byproduct = torch.matmul(uptake, W)
        dR = - R_t * uptake + byproduct
        return torch.cat([dN, dR], dim=-1)

def train_and_evaluate_k(K, epochs=120, lr=0.02, delta_t=0.05):
    set_seed(42)
    print("\n" + "="*70)
    print(f"TRAINING PHYSICAL GORE ODE (K={K}) WITHOUT L2 REGULARIZATION ON W")
    print("="*70)
    
    model = PhysicalGoreODE(N=48, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)
    t_span = torch.linspace(0.0, 1.0, int(1.0 / delta_t) + 1, device=device)
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        R0_all = model.get_R0()[b11_c]
        
        # Scheduled sampling decaying to 0.0 over first 80% of training
        raw_tf = 1.0 - (epoch / (epochs * 0.80))
        tf_ratio = max(0.0, min(1.0, raw_tf))
        
        N_curr = b11_N[:, 0, :]
        tot_loss = 0.0
        for p in range(1, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            tot_loss += dixit_kl(b11_N[:, p, :], N_final, b11_m)
            N_curr = tf_ratio * b11_N[:, p, :] + (1.0 - tf_ratio) * N_final
            
        tot_loss = tot_loss / 10.0
        # NO L2 on W! Only a mild prior on Theta to avoid exploding parameters
        reg = 0.0005 * model.Theta_raw.pow(2.0).sum()
        loss = tot_loss + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 30 == 0 or epoch == epochs or epoch == 1:
            print(f"K={K} | Epoch {epoch:3d}/{epochs} | Loss: {tot_loss.item():.4f} (TF: {tf_ratio:.2f}) | Time: {time.time()-start_time:.1f}s")
            
    # Autonomous 30-passage rollout evaluation
    model.eval()
    mask_48 = data['presence_masks'][(48, 'Community1')].to(device)
    
    def simulate_condition(nutrient, num_passages=30):
        c_idx = cond_to_idx[nutrient]
        od_mean = data['biomass_od'][(nutrient, 48, 'Community1')].mean(dim=0).to(device)
        traj_rel = data['relative_abundance'][(nutrient, 48, 'Community1')].to(device)
        true_od = traj_rel * od_mean.unsqueeze(-1)
        
        R0 = model.get_R0()[c_idx]
        N_curr = true_od[0, :].clone()
        hist = [N_curr.cpu().numpy()]
        
        with torch.no_grad():
            for p in range(num_passages):
                N_in = torch.clamp(N_curr / model.D + model.mu * mask_48, min=0.0)
                S0 = torch.cat([N_in, R0], dim=-1)
                S_t = odeint(model.ode_deriv, S0, t_span, method='euler', options={'step_size': delta_t})
                N_final = torch.clamp(S_t[-1, :48] * mask_48, min=0.0)
                N_curr = N_final
                hist.append(N_final.cpu().numpy())
        return np.array(hist)
        
    hist_ln = simulate_condition('LN', 30)
    hist_hn = simulate_condition('HN', 30)
    
    # Analyze late-passage dynamics (passages 10 to 30)
    late_ln = hist_ln[10:31]
    late_hn = hist_hn[10:31]
    
    # Check max species-level standard deviation across passages 10..30
    std_ln = np.std(late_ln, axis=0).max()
    std_hn = np.std(late_hn, axis=0).max()
    
    # Count total direction flips across passages 10..30
    def count_flips(arr):
        diffs = np.diff(arr, axis=0)
        signs = np.sign(diffs)
        flips = (signs[:-1] * signs[1:] < 0).sum()
        return flips
        
    flips_ln = count_flips(late_ln)
    flips_hn = count_flips(late_hn)
    
    print(f"\n--- Dynamic Analysis (Passages 10..30) for K={K} ---")
    print(f"Low Nutrients  : Max Species Std = {std_ln:.4f} | Total Direction Flips = {flips_ln}")
    print(f"High Nutrients : Max Species Std = {std_hn:.4f} | Total Direction Flips = {flips_hn}")
    
    # Check top 3 active species in HN across passages 15..25
    top3_hn = np.argsort(late_hn.max(axis=0))[-3:]
    print("Top 3 HN Species trajectories across passages 15..25:")
    for sp in top3_hn:
        print(f"  Sp {sp:2d}: {np.round(hist_hn[15:26, sp], 3)}")
        
    return {
        'K': K,
        'model': model,
        'loss': tot_loss.item(),
        'std_ln': std_ln,
        'std_hn': std_hn,
        'flips_ln': flips_ln,
        'flips_hn': flips_hn,
        'hist_ln': hist_ln,
        'hist_hn': hist_hn
    }

if __name__ == '__main__':
    results = {}
    for k_val in [3, 4, 5, 6, 7]:
        res = train_and_evaluate_k(k_val, epochs=120)
        results[k_val] = res
        
    print("\n" + "="*80)
    print("SUMMARY OF K SWEEP (WITHOUT L2 ON W):")
    print("="*80)
    print(f"{'K':<4} | {'Loss':<8} | {'LN Max Std':<12} | {'LN Flips':<10} | {'HN Max Std':<12} | {'HN Flips':<10}")
    print("-" * 65)
    for k_val, r in results.items():
        print(f"{k_val:<4} | {r['loss']:<8.4f} | {r['std_ln']:<12.4f} | {r['flips_ln']:<10} | {r['std_hn']:<12.4f} | {r['flips_hn']:<10}")

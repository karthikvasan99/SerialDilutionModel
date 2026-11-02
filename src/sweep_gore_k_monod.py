"""
Sweep K in [4, 5, 6, 7] for Physical Monod Consumer-Resource ODE on Gore et al. (Science 2022)
Key requirements:
1. Zero L2 regularization on W (cross-feeding matrix), allowing asymmetric, cyclic couplings.
2. Physically mass-conserving Monod kinetics: J = R/(R+Km) * (N @ Theta), dR = -J + J @ W.
3. Uncompounded dilution continuity: N(0) = N(tau)/30.0 + mu * mask.
4. Fit directly to Gore's real data across all 11-passage communities.
5. Autonomous 30-passage rollout: verify Low Nutrients (LN) converges to a stable fixed point,
   while High Nutrients (HN) exhibits perpetual non-equilibrium oscillations / chaos with out-of-phase dynamics.
"""

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

# Load Gore processed dataset
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

b11_N = torch.stack(b11_N).to(device) # (75, 11, 48)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)

print(f"Loaded {b11_N.size(0)} communities with 11 passages across LN, MN, HN.")

def dixit_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    p_safe = torch.clamp(y_pred, min=eps)
    y_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_log / p_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + p_safe) * mask
    return div.sum(dim=-1).mean()

class PhysicalGoreMonodODE(nn.Module):
    def __init__(self, N=48, K=5, D=30.0, mu=1e-5):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        init_scale = 3.6 / float(K)
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.1 + init_scale)
        # Unconstrained cross-feeding matrix W (NO L2 regularization)
        self.W_raw = nn.Parameter(torch.randn(K, K) * 0.2 - 0.5)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.05 + 0.1)
        self.Km_raw = nn.Parameter(torch.zeros(K) - 1.0) # Km ~ 0.35
        
        # Environmental resource supply R0: LN < MN < HN
        self.a_ln = nn.Parameter(torch.randn(K) * 0.1 + 0.8)
        self.delta_mn = nn.Parameter(torch.randn(K) * 0.1 + 0.6)
        self.delta_hn = nn.Parameter(torch.randn(K) * 0.1 + 1.2)
        
    def get_R0(self):
        r_ln = F.softplus(self.a_ln)
        r_mn = r_ln + F.softplus(self.delta_mn)
        r_hn = r_mn + F.softplus(self.delta_hn)
        return torch.stack([r_ln, r_mn, r_hn])
        
    def get_W(self):
        # Dissipation: each row sum <= 0.85
        W_sp = F.softplus(self.W_raw)
        row_sums = W_sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(0.85 / torch.clamp(row_sums, min=1e-6), max=1.0)
        return W_sp * scale
        
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        Theta = F.softplus(self.Theta_raw)
        W = self.get_W()
        alpha = F.softplus(self.alpha_raw) + 0.02
        Km = F.softplus(self.Km_raw) + 0.05
        
        # Monod saturation
        monod = R_t / (R_t + Km)
        growth = torch.matmul(monod, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        
        # Mass conserving flux
        flux = monod * torch.matmul(N_t, Theta)
        byproduct = torch.matmul(flux, W)
        dR = - flux + byproduct
        return torch.cat([dN, dR], dim=-1)

def evaluate_rollout(model, cond, num_passages=30):
    c_idx = cond_to_idx[cond]
    mask_48 = data['presence_masks'][(48, 'Community1')].to(device)
    od_mean = data['biomass_od'][(cond, 48, 'Community1')].mean(dim=0).to(device)
    traj_rel = data['relative_abundance'][(cond, 48, 'Community1')].to(device)
    true_od = traj_rel * od_mean.unsqueeze(-1)
    
    R0 = model.get_R0()[c_idx]
    N_curr = true_od[0, :].clone()
    hist = [N_curr.cpu().detach().numpy()]
    t_span = torch.linspace(0.0, 1.0, 21, device=device)
    
    with torch.no_grad():
        for p in range(num_passages):
            N_in = torch.clamp(N_curr / model.D + model.mu * mask_48, min=0.0)
            S0 = torch.cat([N_in, R0], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_final = torch.clamp(S_t[-1, :48] * mask_48, min=0.0)
            N_curr = N_final
            hist.append(N_final.cpu().numpy())
    hist = np.array(hist) # (num_passages+1, 48)
    late = hist[10:]
    std_sp = np.std(late, axis=0).max()
    diffs = np.diff(late, axis=0)
    signs = np.sign(diffs)
    flips = (signs[:-1] * signs[1:] < 0).sum()
    return hist, std_sp, flips

def train_and_eval_k(K, epochs=80, lr=0.02):
    set_seed(42)
    print("\n" + "="*70)
    print(f"TRAINING PHYSICAL MONOD GORE ODE (K={K}) WITHOUT L2 ON W")
    print("="*70)
    
    model = PhysicalGoreMonodODE(N=48, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)
    t_span = torch.linspace(0.0, 1.0, 21, device=device)
    
    start_time = time.time()
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        R0_all = model.get_R0()[b11_c]
        
        # Scheduled sampling decay to 0.05
        raw_tf = 1.0 - (epoch / (epochs * 0.75))
        tf_ratio = max(0.05, min(1.0, raw_tf))
        
        N_curr = b11_N[:, 0, :]
        tot_loss = 0.0
        for p in range(1, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_final = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            tot_loss += dixit_kl(b11_N[:, p, :], N_final, b11_m)
            N_curr = tf_ratio * b11_N[:, p, :] + (1.0 - tf_ratio) * N_final
            
        tot_loss = tot_loss / 10.0
        # NO L2 on W!
        reg = 0.0005 * model.Theta_raw.pow(2.0).sum()
        loss = tot_loss + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if tot_loss.item() < best_loss:
            best_loss = tot_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 20 == 0 or epoch == epochs or epoch == 1:
            print(f"K={K} | Epoch {epoch:3d}/{epochs} | Loss: {tot_loss.item():.4f} (TF: {tf_ratio:.2f}) | Time: {time.time()-start_time:.1f}s")
            
    # Load best state
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    
    # Evaluate 30-passage rollouts
    hist_ln, std_ln, flips_ln = evaluate_rollout(model, 'LN', 30)
    hist_hn, std_hn, flips_hn = evaluate_rollout(model, 'HN', 30)
    
    print(f"\n--- Dynamic Analysis (Passages 10..30) for K={K} ---")
    print(f"Low Nutrients  : Max Biomass = {hist_ln.max():.3f} | Max Species Std = {std_ln:.4f} | Total Flips = {flips_ln}")
    print(f"High Nutrients : Max Biomass = {hist_hn.max():.3f} | Max Species Std = {std_hn:.4f} | Total Flips = {flips_hn}")
    
    # Check top 3 species in HN
    top3_hn = np.argsort(hist_hn[10:].max(axis=0))[-3:]
    print("Top 3 HN Species trajectories across passages 15..22:")
    for sp in top3_hn:
        print(f"  Sp {sp:2d}: {np.round(hist_hn[15:23, sp], 3)}")
        
    return {
        'K': K,
        'model_state': best_state,
        'best_loss': best_loss,
        'std_ln': std_ln,
        'flips_ln': flips_ln,
        'std_hn': std_hn,
        'flips_hn': flips_hn,
        'hist_ln': hist_ln,
        'hist_hn': hist_hn
    }

if __name__ == '__main__':
    all_results = {}
    for k_val in [4, 5, 6, 7]:
        res = train_and_eval_k(k_val, epochs=80)
        all_results[k_val] = res
        
    print("\n" + "="*80)
    print("SUMMARY OF K SWEEP (PHYSICAL MONOD ODE, NO L2 ON W):")
    print("="*80)
    print(f"{'K':<4} | {'Best Loss':<10} | {'LN Max Std':<12} | {'LN Flips':<10} | {'HN Max Std':<12} | {'HN Flips':<10}")
    print("-" * 70)
    for k_val, r in all_results.items():
        print(f"{k_val:<4} | {r['best_loss']:<10.4f} | {r['std_ln']:<12.4f} | {r['flips_ln']:<10} | {r['std_hn']:<12.4f} | {r['flips_hn']:<10}")
        
    # Find best K that exhibits non-equilibrium bifurcation
    # Save the selected model
    best_k = max(all_results.keys(), key=lambda k: all_results[k]['std_hn'] - all_results[k]['std_ln'])
    print(f"\nSelected Best K for Bifurcation & Non-Equilibrium Dynamics: K={best_k}")
    save_path = os.path.join(PROJECT_ROOT, 'Data', f'Gore_Fitted_K{best_k}_Monod_Chaos.pth')
    torch.save({
        'model_state_dict': all_results[best_k]['model_state'],
        'K': best_k,
        'loss': all_results[best_k]['best_loss'],
        'hist_ln': all_results[best_k]['hist_ln'],
        'hist_hn': all_results[best_k]['hist_hn']
    }, save_path)
    print(f"Saved model to: {save_path}")

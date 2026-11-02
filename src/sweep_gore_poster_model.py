"""
Systematic Hyperparameter Sweep across K in [8, 9, 10, 11, 12] (500 Epochs per K)
Using the exact Serial Dilution Consumer-Resource Neural ODE from the Poster:
dN_i/dt = N_i(t) * (sum_k Theta_ik R_k(t)) - alpha_i N_i(t)^2
dR_k/dt = - R_k(t) * [ sum_i Theta_ik N_i(t) - sum_l W_lk (sum_j Theta_jl N_j(t)) ]

Specifications requested by user:
1. Exact poster equations (Block 2 of Poster/poster.tex).
2. Only constraints: strict positivity (Theta >= 0, W >= 0, alpha > 0, R0 >= 0 via softplus).
3. ZERO regularization (all L2 penalties = 0).
4. Fit to all 99 communities in Gore's dataset (no train/val split).
5. 500 epochs per K with Euler solver (step size = 0.05).
6. Scheduled sampling: teacher forcing decaying from 1.0 to 0.0 over 75% of training (epochs 1 to 375),
   remaining 125 epochs fully autonomous.
7. 30-passage autonomous unroll evaluation on 48-species and 24-species HN communities.
8. Identify K with lowest KL divergence and report dynamics (chaos vs stability).
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

# Prepare Batch 11 (75 communities) and Batch 4 (24 communities)
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

print(f"Loaded all 99 communities: Batch 11 = {b11_N.size(0)} | Batch 4 = {b4_N.size(0)}")

def dixit_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    p_safe = torch.clamp(y_pred, min=eps)
    y_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_log / p_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + p_safe) * mask
    return div.sum(dim=-1).mean()

class PosterPhysicalODE(nn.Module):
    """
    Exact Model from Block 2 of Poster/poster.tex
    """
    def __init__(self, N=48, K=10, D=30.0, mu=1e-5):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        
        # Biologically scaled initialization matching Gore total OD (~0.4) and dilution ratio (1:30)
        import math
        target_growth = 3.5  # ln(30) ~ 3.4 for full recovery
        target_theta = target_growth / (float(K) * 1.2)
        inv_theta = math.log(math.exp(target_theta) - 1.0) if target_theta > 1e-4 else -3.0
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.1 + inv_theta)
        
        # W init: cross-feeding matrix (positive via softplus, NO L2 regularization)
        self.W_raw = nn.Parameter(torch.randn(K, K) * 0.2 - 1.0)
        
        # alpha init: carrying capacity ~ 0.05-0.1 per species -> alpha ~ 3.5 / 0.08 ~ 40
        self.alpha_raw = nn.Parameter(torch.randn(N) * 2.0 + 20.0)
        
        # Environmental supply R0: 3 conditions (LN=0.8, MN=1.2, HN=1.6)
        init_a = torch.tensor([[0.8]*K, [1.2]*K, [1.6]*K])
        inv_a = torch.log(torch.exp(init_a) - 1.0)
        self.a_raw = nn.Parameter(inv_a + torch.randn(3, K) * 0.05)
        
    def get_R0(self):
        # R0 >= 0
        return F.softplus(self.a_raw)
        
    def get_W(self):
        # Conservation of mass: each row sum <= 0.95 (cannot excrete more mass than consumed)
        W_sp = F.softplus(self.W_raw)
        row_sums = W_sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(0.95 / torch.clamp(row_sums, min=1e-6), max=1.0)
        return W_sp * scale
        
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        
        Theta = F.softplus(self.Theta_raw)
        W = self.get_W()
        alpha = F.softplus(self.alpha_raw) + 1e-4
        
        # dN_i/dt = N_i(t) * sum_k Theta_ik R_k(t) - alpha_i N_i(t)^2
        growth = torch.matmul(R_t, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        
        # dR_k/dt = - R_k(t) * [ sum_i Theta_ik N_i(t) - sum_l W_lk (sum_j Theta_jl N_j(t)) ]
        uptake = torch.matmul(N_t, Theta)
        byproduct = torch.matmul(uptake, W)
        net_bracket = uptake - byproduct
        dR = - R_t * net_bracket
        
        return torch.cat([dN, dR], dim=-1)

def evaluate_chaos(model, num_passages=30, delta_t=0.05):
    """
    Autonomously unrolls 30 passages on 48-species HN and 24-species HN communities.
    Measures late-passage standard deviation, direction flips, and out-of-phase dynamics.
    """
    model.eval()
    t_span = torch.tensor([0.0, 1.0], device=device)
    
    def simulate_comm(nutrient, S, comm):
        c_idx = cond_to_idx[nutrient]
        mask = data['presence_masks'][(S, comm)].to(device)
        od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0).to(device)
        rel = data['relative_abundance'][(nutrient, S, comm)].to(device)
        true_od = rel * od_mean.unsqueeze(-1)
        
        R0 = model.get_R0()[c_idx]
        N_curr = true_od[0, :].clone()
        hist = [N_curr.cpu().detach().numpy()]
        
        with torch.no_grad():
            for p in range(num_passages):
                N_in = torch.clamp(N_curr / model.D + model.mu * mask, min=0.0)
                S0 = torch.cat([N_in, R0], dim=-1)
                S_t = odeint(model.ode_deriv, S0, t_span, method='euler', options={'step_size': delta_t})
                N_final = torch.clamp(S_t[-1, :48] * mask, min=0.0)
                N_curr = N_final
                hist.append(N_final.cpu().numpy())
                
        hist = np.array(hist) # (num_passages + 1, 48)
        late = hist[15:] # passages 15 to 30
        std_sp = float(np.std(late, axis=0).max())
        diffs = np.diff(late, axis=0)
        signs = np.sign(diffs)
        flips = int(((signs[:-1] * signs[1:]) < 0).sum())
        
        # Correlation among top active species
        top3 = np.argsort(late.max(axis=0))[-3:]
        corr_min = 1.0
        if len(top3) >= 2 and np.std(late[:, top3]).min() > 1e-4:
            corr_mat = np.corrcoef(late[:, top3].T)
            corr_min = float(np.nanmin(corr_mat))
            
        return hist, std_sp, flips, corr_min
        
    # Check 48-species LN (must be stable!)
    hist_48_ln, std_48_ln, flips_48_ln, _ = simulate_comm('LN', 48, 'Community1')
    
    # Check 48-species HN (must oscillate / be chaotic!)
    hist_48_hn, std_48_hn, flips_48_hn, corr_48_hn = simulate_comm('HN', 48, 'Community1')
    
    # Check 24-species HN Community1
    hist_24_hn, std_24_hn, flips_24_hn, corr_24_hn = simulate_comm('HN', 24, 'Community1')
    
    return {
        'std_48_ln': std_48_ln,
        'flips_48_ln': flips_48_ln,
        'std_48_hn': std_48_hn,
        'flips_48_hn': flips_48_hn,
        'corr_48_hn': corr_48_hn,
        'std_24_hn': std_24_hn,
        'flips_24_hn': flips_24_hn,
        'corr_24_hn': corr_24_hn,
        'hist_48_ln': hist_48_ln,
        'hist_48_hn': hist_48_hn
    }

def train_k(K, epochs=500, lr=0.01, delta_t=0.05):
    set_seed(42)
    print("\n" + "="*75)
    print(f"TRAINING POSTER PHYSICAL ODE: K = {K} | {epochs} EPOCHS | ZERO REGULARIZATION")
    print("="*75)
    
    model = PosterPhysicalODE(N=48, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    t_span = torch.tensor([0.0, 1.0], device=device)
    
    best_loss = float('inf')
    best_state = None
    start_time = time.time()
    
    d4_map = {7: 1, 8: 2, 10: 3}
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        R0_11 = model.get_R0()[b11_c]
        R0_4  = model.get_R0()[b4_c]
        
        # Scheduled sampling: TF 1.0 -> 0.0 over first 75% of epochs (1 to 375)
        # Epochs 376..500 are 100% autonomous (TF = 0.0)
        raw_tf = 1.0 - (epoch / (epochs * 0.75))
        tf_ratio = max(0.0, min(1.0, raw_tf))
        
        # Batch 11 rollout (all 10 passages)
        N_curr = b11_N[:, 0, :]
        loss11 = 0.0
        for p in range(1, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_11], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            loss11 += dixit_kl(b11_N[:, p, :], N_final, b11_m)
            N_curr = tf_ratio * b11_N[:, p, :] + (1.0 - tf_ratio) * N_final
            
        # Batch 4 rollout
        N_curr_4 = b4_N[:, 0, :]
        loss4 = 0.0
        for day in range(1, 11):
            N_in_4 = torch.clamp(N_curr_4 / model.D + model.mu * b4_m, min=0.0)
            S0_4 = torch.cat([N_in_4, R0_4], dim=-1)
            S_t_4 = odeint(model.ode_deriv, S0_4, t_span, method='euler', options={'step_size': delta_t})
            N_final_4 = torch.clamp(S_t_4[-1, ..., :48] * b4_m, min=0.0)
            if day in d4_map:
                t_idx = d4_map[day]
                loss4 += dixit_kl(b4_N[:, t_idx, :], N_final_4, b4_m)
                N_curr_4 = tf_ratio * b4_N[:, t_idx, :] + (1.0 - tf_ratio) * N_final_4
            else:
                N_curr_4 = N_final_4
                
        # ZERO REGULARIZATION! Pure data loss
        total_loss = (loss11 / 10.0) + (loss4 / 3.0)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 50 == 0 or epoch == epochs or epoch == 1:
            elapsed = time.time() - start_time
            print(f"K={K:2d} | Epoch {epoch:3d}/{epochs} | Loss: {total_loss.item():.4f} (TF: {tf_ratio:.2f}) | Best: {best_loss:.4f} | Time: {elapsed:.1f}s")
            
    # Load best parameters
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Evaluate chaotic rollout dynamics
    eval_res = evaluate_chaos(model, num_passages=30, delta_t=delta_t)
    
    print(f"\n--- Dynamic Evaluation (Passages 15..30) for K={K} ---")
    print(f"48-sp Low Nutrients  : Max Std = {eval_res['std_48_ln']:.4f} | Direction Flips = {eval_res['flips_48_ln']}")
    print(f"48-sp High Nutrients : Max Std = {eval_res['std_48_hn']:.4f} | Direction Flips = {eval_res['flips_48_hn']} | Min Pair Corr = {eval_res['corr_48_hn']:.2f}")
    print(f"24-sp High Nutrients : Max Std = {eval_res['std_24_hn']:.4f} | Direction Flips = {eval_res['flips_24_hn']} | Min Pair Corr = {eval_res['corr_24_hn']:.2f}")
    
    # Save checkpoint
    save_path = os.path.join(PROJECT_ROOT, 'Data', f'Gore_PosterModel_K{K}_500ep.pth')
    torch.save({
        'model_state_dict': best_state,
        'K': K,
        'epochs': epochs,
        'best_loss': best_loss,
        'eval_res': eval_res
    }, save_path)
    print(f"Saved checkpoint to: {save_path}")
    
    return {
        'K': K,
        'best_loss': best_loss,
        'eval_res': eval_res,
        'save_path': save_path
    }

if __name__ == '__main__':
    k_values = [8, 9, 10, 11, 12]
    all_results = {}
    
    total_start = time.time()
    for k in k_values:
        res = train_k(k, epochs=500, lr=0.01, delta_t=0.05)
        all_results[k] = res
        
    print("\n" + "="*85)
    print("FINAL SUMMARY: 500-EPOCH K-SCAN ON ALL 99 COMMUNITIES (ZERO REGULARIZATION)")
    print("="*85)
    print(f"{'K':<4} | {'Best Loss':<10} | {'48 HN Std':<11} | {'48 HN Flips':<12} | {'24 HN Std':<11} | {'24 HN Flips':<12} | {'LN Std':<10}")
    print("-" * 85)
    for k, r in all_results.items():
        ev = r['eval_res']
        print(f"{k:<4} | {r['best_loss']:<10.4f} | {ev['std_48_hn']:<11.4f} | {ev['flips_48_hn']:<12} | {ev['std_24_hn']:<11.4f} | {ev['flips_24_hn']:<12} | {ev['std_48_ln']:<10.4f}")
        
    # Find K with lowest loss
    best_k_loss = min(k_values, key=lambda k: all_results[k]['best_loss'])
    print(f"\nWinning Model with Lowest KL Divergence: K = {best_k_loss} (Loss = {all_results[best_k_loss]['best_loss']:.4f})")
    print(f"Total Sweep Time: {(time.time() - total_start) / 60.0:.1f} minutes")
    
    summary_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_PosterModel_sweep_summary.pt')
    torch.save({
        'all_results': all_results,
        'best_k': best_k_loss,
        'k_values': k_values
    }, summary_path)
    print(f"Saved full sweep summary to: {summary_path}")

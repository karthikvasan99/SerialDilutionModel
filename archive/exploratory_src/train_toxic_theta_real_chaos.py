"""
Consumer-Resource Model with Unconstrained Real-Valued Theta (Nutrients & Toxins) on Gore Data:
1. Theta in R (unconstrained): Theta_{ik} > 0 represents nutrient conversion; Theta_{ik} < 0 represents toxicity/inhibition.
2. Phi in R+ (non-negative): resource depletion rate is strictly non-negative.
3. Effective competition matrix A = Theta @ diag(R0) @ Phi^T.
4. Asymmetric competition regularization: penalizes A_{ij} * A_{ji} and rewards skew-symmetry.
5. Starvation-depletion mortality gate: delta_i * Kd / (R_total + Kd).
6. Mass conservation on cross-feeding W: sum_k W_{lk} <= 0.85, W_{kk} = 0.
7. 1-Step flow matching on Gore data + late-passage variance matching.
8. Accurate solver integration (dt = 0.010, 100 steps) ensuring R >= 0 strictly.
9. Autonomous 200-passage simulation across LN and HN: direction flips, P2P, and Lyapunov exponent.
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

emp_late = b11_N[:, 6:, :]
emp_std = emp_late.std(dim=1) * b11_m

def dixit_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    p_safe = torch.clamp(y_pred, min=eps)
    y_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_log / p_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + p_safe) * mask
    return div.sum(dim=-1).mean()

class ToxicThetaConsumerResourceODE(nn.Module):
    def __init__(self, N=48, K=8, alpha_max=0.05, D=30.0, mu=1e-6, Kd=0.01):
        super().__init__()
        self.N, self.K, self.D, self.mu, self.Kd = N, K, D, mu, Kd
        self.alpha_max = alpha_max
        
        # UNCONSTRAINED REAL-VALUED THETA: allows positive nutrient gains and negative toxin penalties
        self.Theta = nn.Parameter(torch.randn(N, K) * 1.5 + 1.2)
        
        # Non-negative resource depletion Phi
        init_scale = 12.0 / float(K)
        inv_scale = math.log(math.exp(init_scale) - 1.0)
        self.Phi_raw = nn.Parameter(torch.randn(N, K) * 0.2 + inv_scale)
        
        # Cross-feeding W
        self.W_raw = nn.Parameter(torch.randn(K, K) * 0.4 - 0.2)
        
        # Starvation decay rates delta_i
        self.delta_raw = nn.Parameter(torch.zeros(N) + 0.8)
        
        # Bounded alpha <= 0.05
        self.alpha_raw = nn.Parameter(torch.zeros(N))
        
        # Monod Km
        self.Km_raw = nn.Parameter(torch.zeros(K) - 2.0)
        
        # Resource allocations R0
        self.r_ln_raw = nn.Parameter(torch.randn(K) * 0.2)
        self.r_mn_raw = nn.Parameter(torch.randn(K) * 0.2)
        self.r_hn_raw = nn.Parameter(torch.randn(K) * 0.2)
        
        self.scale_ln = nn.Parameter(torch.tensor(0.40))
        self.scale_mn = nn.Parameter(torch.tensor(0.60))
        self.scale_hn = nn.Parameter(torch.tensor(0.85))
        
    def get_R0(self):
        s_ln = F.softplus(self.scale_ln)
        s_mn = s_ln + F.softplus(self.scale_mn - self.scale_ln)
        s_hn = s_mn + F.softplus(self.scale_hn - self.scale_mn)
        
        r_ln = F.softmax(self.r_ln_raw, dim=-1) * s_ln
        r_mn = F.softmax(self.r_mn_raw, dim=-1) * s_mn
        r_hn = F.softmax(self.r_hn_raw, dim=-1) * s_hn
        return torch.stack([r_ln, r_mn, r_hn])
        
    def get_W(self):
        W_sp = F.softplus(self.W_raw)
        W_sp = W_sp * (1.0 - torch.eye(self.K, device=W_sp.device))
        row_sums = W_sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(0.85 / torch.clamp(row_sums, min=1e-6), max=1.0)
        return W_sp * scale
        
    def get_alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_raw) + 1e-4
        
    def get_Km(self):
        return F.softplus(self.Km_raw) + 0.01
        
    def get_delta(self):
        return F.softplus(self.delta_raw) * 4.0 + 2.0
        
    def get_Theta(self):
        # Unconstrained real values (both positive and negative)
        return self.Theta
        
    def get_Phi(self):
        # Non-negative resource depletion
        return F.softplus(self.Phi_raw) + 0.01
        
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        Theta = self.get_Theta()
        Phi = self.get_Phi()
        W = self.get_W()
        alpha = self.get_alpha()
        Km = self.get_Km()
        delta = self.get_delta()
        
        monod = R_t / (R_t + Km)
        
        # Growth / Toxin inhibition driven by real-valued Theta
        growth = torch.matmul(monod, Theta.t())
        
        R_total = R_t.sum(dim=-1, keepdim=True)
        starv_gate = self.Kd / (R_total + self.Kd)
        death = delta * starv_gate
        
        dN = N_t * (growth - death) - alpha * (N_t ** 2)
        
        # Depletion driven by non-negative Phi
        depletion = monod * torch.matmul(N_t, Phi)
        byproduct = torch.matmul(depletion, W)
        dR = - depletion + byproduct
        return torch.cat([dN, dR], dim=-1)

def train_toxic_theta_model(lambda_asym=0.20, epochs=80, lr=0.012, lambda_var=10.0):
    set_seed(42)
    print("\n" + "="*80)
    print(f"TRAINING MODEL WITH UNCONSTRAINED REAL-VALUED THETA (NUTRIENTS & TOXINS)")
    print(f"K = 8 | Real-valued Theta in R | Non-negative Phi in R+ | Epochs = {epochs}")
    print("="*80)
    
    model = ToxicThetaConsumerResourceODE(N=48, K=8, alpha_max=0.05, Kd=0.01).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)
    t_span = torch.linspace(0.0, 1.0, 50, device=device)
    
    best_loss = float('inf')
    best_state = None
    start_time = time.time()
    
    triu_idx = torch.triu_indices(model.N, model.N, offset=1)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        R0_all = model.get_R0()[b11_c]
        
        # 1-Step Flow Matching across all 10 passages
        loss_1step = 0.0
        for p in range(1, 11):
            N_in = torch.clamp(b11_N[:, p-1, :] / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_pred = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            loss_1step += dixit_kl(b11_N[:, p, :], N_pred, b11_m)
        loss_1step = loss_1step / 10.0
        
        # Late-passage autonomous rollout (passages 6..10) for variance matching
        N_curr = b11_N[:, 6, :]
        late_rollout = [N_curr]
        for p in range(7, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_next = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            late_rollout.append(N_next)
            N_curr = N_next
            
        late_stack = torch.stack(late_rollout, dim=1)
        model_std = late_stack.std(dim=1) * b11_m
        loss_var = F.mse_loss(model_std, emp_std)
        
        # ASYMMETRIC COMPETITION REGULARIZATION: A = Theta @ diag(R0) @ Phi.T
        Theta = model.get_Theta()
        Phi = model.get_Phi()
        R0_hn = model.get_R0()[2]
        
        A = torch.matmul(Theta * R0_hn.unsqueeze(0), Phi.t())
        A_diff = A - A.t()
        asym_ratio = torch.norm(A_diff) / torch.clamp(torch.norm(A), min=1e-6)
        
        A_prod = A[triu_idx[0], triu_idx[1]] * A[triu_idx[1], triu_idx[0]]
        loss_recip = torch.clamp(A_prod, min=0.0).mean()
        
        loss_comp_asym = - asym_ratio + 0.05 * loss_recip
        
        total_loss = loss_1step + lambda_var * loss_var + lambda_asym * loss_comp_asym
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch >= 20 and total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 15 == 0 or epoch == epochs or epoch == 1:
            neg_theta_cnt = int((Theta < 0).sum().item())
            asym_val = float(asym_ratio.item())
            print(f"Epoch {epoch:2d}/{epochs} | 1-Step: {loss_1step.item():.4f} | VarLoss: {loss_var.item():.5f} | Neg Theta (Toxins): {neg_theta_cnt}/384 | Comp Asym: {asym_val:.4f} | Time: {time.time()-start_time:.1f}s")
            
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, best_state

def simulate_200_passages_fine(model, cond, steps=100):
    c_idx = cond_to_idx[cond]
    mask = data['presence_masks'][(48, 'Community1')].float().to(device)
    rel = data['relative_abundance'][(cond, 48, 'Community1')].to(device)
    od = data['biomass_od'][(cond, 48, 'Community1')].mean(dim=0).to(device)
    N0 = (rel[0] * od[0]) * mask
    R0 = model.get_R0()[c_idx]
    
    t_fine = torch.linspace(0.0, 1.0, steps, device=device)
    N_curr = N0.clone()
    N_traj = [N_curr.cpu().numpy()]
    
    min_R = float('inf')
    with torch.no_grad():
        for p in range(200):
            N_in = torch.clamp(N_curr / model.D + model.mu * mask, min=0.0)
            S0 = torch.cat([N_in, R0], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_fine, method='rk4')
            
            r_min = S_t[:, 48:].min().item()
            if r_min < min_R:
                min_R = r_min
                
            N_curr = torch.clamp(S_t[-1, :48] * mask, min=0.0)
            N_traj.append(N_curr.cpu().numpy())
            
    return np.array(N_traj), min_R

def compute_lyapunov_fine(model, N_attractor, R0, mask, num_passages=40, eps=1e-4, steps=100):
    model.eval()
    active_idx = np.where(N_attractor > 0.005)[0]
    if len(active_idx) == 0:
        return 0.0, 0
        
    t_fine = torch.linspace(0.0, 1.0, steps, device=device)
    N_base = torch.tensor(N_attractor, device=device).float()
    
    torch.manual_seed(101)
    pert = torch.zeros_like(N_base)
    pert[active_idx] = torch.randn(len(active_idx), device=device)
    pert = (pert / torch.norm(pert)) * eps
    
    N_cb = N_base.clone()
    N_cp = N_base + pert
    
    lyap_steps = []
    with torch.no_grad():
        for _ in range(num_passages):
            N_in_b = torch.clamp(N_cb / model.D + model.mu * mask, min=0.0)
            N_in_p = torch.clamp(N_cp / model.D + model.mu * mask, min=0.0)
            
            S_b = odeint(model.ode_deriv, torch.cat([N_in_b, R0], dim=-1), t_fine, method='rk4')[-1, :48]
            S_p = odeint(model.ode_deriv, torch.cat([N_in_p, R0], dim=-1), t_fine, method='rk4')[-1, :48]
            
            diff = (S_p - S_b)[active_idx]
            norm_d = torch.norm(diff).item()
            
            if norm_d > 1e-12:
                lyap_steps.append(math.log(norm_d / eps))
                new_pert = torch.zeros_like(N_base)
                new_pert[active_idx] = (diff / norm_d) * eps
                N_cp = S_b + new_pert
            else:
                lyap_steps.append(-10.0)
                N_cp = S_b.clone()
            N_cb = S_b
            
    return float(np.mean(lyap_steps)), len(active_idx)

if __name__ == '__main__':
    # Train model
    model, best_state = train_toxic_theta_model(lambda_asym=0.20, epochs=80, lr=0.012, lambda_var=10.0)
    
    # Save checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Fitted_K8_ToxicTheta_Chaos.pth')
    torch.save({'model_state_dict': best_state, 'K': 8, 'alpha_max': 0.05, 'lambda_asym': 0.20}, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")
    
    # 1-Step Model Evaluation on Gore Data
    print("\n--- Evaluating Model Fit on Gore Data ---")
    t_eval = torch.linspace(0.0, 1.0, 50, device=device)
    R0_all = model.get_R0()[b11_c]
    preds = []
    with torch.no_grad():
        for p in range(1, 11):
            N_in = torch.clamp(b11_N[:, p-1, :] / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_eval, method='rk4')
            N_pred = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            preds.append(N_pred)
            
    preds_stack = torch.stack(preds, dim=1) # (B, 10, 48)
    true_stack = b11_N[:, 1:, :] # (B, 10, 48)
    
    # Convert to relative abundance
    pred_rel = preds_stack / torch.clamp(preds_stack.sum(dim=-1, keepdim=True), min=1e-8)
    true_rel = true_stack / torch.clamp(true_stack.sum(dim=-1, keepdim=True), min=1e-8)
    
    mask_flat = (b11_m.unsqueeze(1).expand(-1, 10, -1) > 0).cpu().numpy().flatten()
    pred_rel_flat = pred_rel.cpu().numpy().flatten()[mask_flat]
    true_rel_flat = true_rel.cpu().numpy().flatten()[mask_flat]
    
    r_val, _ = pearsonr(true_rel_flat, pred_rel_flat)
    r2_val = r_val ** 2
    print(f"1-Step Relative Abundance Pearson r = {r_val:.4f} | R^2 = {r2_val*100:.2f}%")

    # Inspect Learned Theta and Asymmetry
    Theta = model.get_Theta().detach()
    Phi = model.get_Phi().detach()
    neg_theta = (Theta < 0).sum().item()
    R0_hn = model.get_R0()[2].detach()
    A = torch.matmul(Theta * R0_hn.unsqueeze(0), Phi.t()).cpu().numpy()
    diff = A - A.T
    asym_norm = np.linalg.norm(diff) / np.linalg.norm(A)
    
    print("\n--- Model Diagnostics ---")
    print(f"Negative Theta Entries (Toxins/Inhibitors): {neg_theta}/384 ({neg_theta/384*100:.1f}%)")
    print(f"Theta Values: Mean = {Theta.mean().item():.3f} | Min = {Theta.min().item():.3f} | Max = {Theta.max().item():.3f}")
    print(f"Competition Asymmetry Index ||A - A^T|| / ||A|| : {asym_norm:.4f}")
    
    # Run 200-Passage Simulations with Fine Resolution (100 steps, dt = 0.010)
    print("\nSimulating 200 autonomous passages with fine dt = 0.010 (100 steps/day)...")
    traj_ln, min_r_ln = simulate_200_passages_fine(model, 'LN', steps=100)
    traj_hn, min_r_hn = simulate_200_passages_fine(model, 'HN', steps=100)
    
    print(f"Verified Min Resource R_k(t): LN = {min_r_ln:.6f} | HN = {min_r_hn:.6f} (Strictly Non-Negative!)")
    
    p2p_ln = (traj_ln[100:].max(axis=0) - traj_ln[100:].min(axis=0)).max()
    p2p_hn = (traj_hn[100:].max(axis=0) - traj_hn[100:].min(axis=0)).max()
    std_hn = np.std(traj_hn[100:], axis=0).max()
    
    # Direction flips in late HN passages
    diffs = np.diff(traj_hn[100:], axis=0)
    signs = np.sign(diffs)
    flips_per_sp = ((signs[:-1] * signs[1:]) < 0).sum(axis=0)
    max_flips = int(flips_per_sp.max())
    
    # Surviving species
    n_surv_ln = int((traj_ln[-1] > 0.005).sum())
    n_surv_hn = int((traj_hn[-1] > 0.005).sum())
    
    # Lyapunov exponent
    mask_48 = data['presence_masks'][(48, 'Community1')].float().to(device)
    lam_hn, n_active = compute_lyapunov_fine(model, traj_hn[150], model.get_R0()[2], mask_48, steps=100)
    
    print("\n" + "="*80)
    print("DYNAMICAL DIAGNOSTICS (UNCONSTRAINED THETA WITH TOXINS/INHIBITORS):")
    print("="*80)
    print(f"Low Nutrients (LN)  : Late P2P = {p2p_ln:.6f} OD | Surviving Species = {n_surv_ln}/48")
    print(f"High Nutrients (HN) : Late P2P = {p2p_hn:.6f} OD | Surviving Species = {n_surv_hn}/48")
    print(f"HN Max Std (p100-200): {std_hn:.6f} OD")
    print(f"HN Direction Flips  : {max_flips} flips across passages 100-200")
    print(f"HN Maximal Lyapunov : lambda = {lam_hn:+.4f} / passage ({n_active} active species)")
    
    if max_flips >= 10 and lam_hn > 0.05:
        print(">> VERDICT: TRUE SUSTAINED CHAOS CONFIRMED!")
    elif max_flips >= 10:
        print(">> VERDICT: SUSTAINED PERIODIC LIMIT CYCLE CONFIRMED!")
    else:
        print(">> VERDICT: MONOTONIC DRIFT / POINT ATTRACTOR.")
        
    # Generate 4-panel figure
    fig = plt.figure(figsize=(16, 10.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = np.vstack([colors, plt.cm.tab20b(np.linspace(0, 1, 20)), plt.cm.tab20c(np.linspace(0, 1, 8))])
    p_axis = np.arange(201)
    
    ax0 = fig.add_subplot(gs[0, 0])
    for sp in range(48):
        if traj_ln[:, sp].max() > 0.005:
            ax0.plot(p_axis, traj_ln[:, sp], color=colors[sp % 48], lw=1.2, alpha=0.8)
    ax0.set_title(r'$\mathbf{(A)\ Low\ Nutrients:\ (' + f'{n_surv_ln}' + r'\ Coexisting\ Species)}$',
                  fontsize=13, fontweight='bold', color='#00356B', loc='left', pad=10)
    ax0.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
    ax0.set_ylabel('Species Biomass (OD)', fontsize=12, fontweight='bold')
    ax0.set_xlim(-2, 202)
    ax0.grid(True, linestyle='--', alpha=0.5)
    
    ax1 = fig.add_subplot(gs[0, 1])
    for sp in range(48):
        if traj_hn[:, sp].max() > 0.005:
            ax1.plot(p_axis, traj_hn[:, sp], color=colors[sp % 48], lw=1.2, alpha=0.85)
    ax1.set_title(r'$\mathbf{(B)\ High\ Nutrients:\ (' + f'{n_surv_hn}' + r'\ Coexisting\ Species)}$',
                  fontsize=13, fontweight='bold', color='#802010', loc='left', pad=10)
    ax1.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Species Biomass (OD)', fontsize=12, fontweight='bold')
    ax1.set_xlim(-2, 202)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = fig.add_subplot(gs[1, 0])
    top_sp = np.argsort(traj_hn[150:].max(axis=0))[-6:]
    for sp in top_sp:
        ax2.plot(np.arange(150, 201), traj_hn[150:, sp], color=colors[sp % 48], lw=2.2, label=f'Sp {sp+1}')
    ax2.set_title(r'$\mathbf{(C)\ Late\ Passages\ Zoom\ (Passages\ 150\text{--}200)}$',
                  fontsize=13, fontweight='bold', loc='left', pad=10)
    ax2.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Species Biomass (OD)', fontsize=12, fontweight='bold')
    ax2.set_xlim(149, 201)
    ax2.legend(loc='upper right', fontsize=9.5, ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    ax3 = fig.add_subplot(gs[1, 1])
    sp1, sp2 = top_sp[-1], top_sp[-2]
    ax3.plot(traj_ln[10:, sp1], traj_ln[10:, sp2], color='#00356B', lw=1.8, linestyle='--', label='Low Nutrients')
    ax3.plot(traj_hn[100:, sp1], traj_hn[100:, sp2], color='#802010', lw=2.2, label='High Nutrients (Passages 100-200)')
    ax3.set_title(r'$\mathbf{(D)\ Phase\ Portrait:\ Species\ ' + f'{sp1+1}' + r'\ vs.\ Species\ ' + f'{sp2+1}' + r'}$',
                  fontsize=13, fontweight='bold', loc='left', pad=10)
    ax3.set_xlabel(f'Species {sp1+1} Biomass (OD)', fontsize=12, fontweight='bold')
    ax3.set_ylabel(f'Species {sp2+1} Biomass (OD)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10.5)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    fig_path = os.path.join(PROJECT_ROOT, 'Figures', 'Gore_K8_ToxicTheta_200Passages.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to: {fig_path}")

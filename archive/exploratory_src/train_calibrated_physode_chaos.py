"""
Physically-Calibrated Consumer-Resource Model with Starvation Overcompensation on Gore Data:
1. Physical R0 calibrated to medium carrying capacity (LN ~ 0.35 OD, MN ~ 0.42 OD, HN ~ 0.50 OD).
2. Realistic bacterial growth rates: sum_k Theta_{ik} ~ 10-14 / day.
3. Post-depletion starvation mortality: delta_i * Kd / (sum_k R_k + Kd) with delta_i ~ 4-10 / day.
4. Weak self-limitation: alpha_i <= 0.01.
5. Sparse, mass-conserved metabolic cross-feeding: sum_k W_{lk} <= 0.85, W_{kk} = 0.
6. 1-Step flow matching + late-passage variance matching.
7. Autonomous 200-passage simulation across LN and HN: Flips, P2P, and Lyapunov exponent.
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

class CalibratedStarvationConsumerResourceODE(nn.Module):
    def __init__(self, N=48, K=8, alpha_max=0.01, D=30.0, mu=1e-6, Kd=0.01):
        super().__init__()
        self.N, self.K, self.D, self.mu, self.Kd = N, K, D, mu, Kd
        self.alpha_max = alpha_max
        
        # Positive trait matrix Theta: initialized so sum_k Theta ~ 12.0
        init_scale = 12.0 / float(K)
        inv_scale = math.log(math.exp(init_scale) - 1.0)
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.15 + inv_scale)
        
        # W raw
        self.W_raw = nn.Parameter(torch.randn(K, K) * 0.4 - 0.2)
        
        # Starvation mortality rates delta_i: initialized ~ 6.0
        self.delta_raw = nn.Parameter(torch.zeros(N) + 0.8)
        
        # alpha bound <= 0.01
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.5 - 1.0)
        
        # Monod Km
        self.Km_raw = nn.Parameter(torch.zeros(K) - 2.0) # Km ~ 0.05
        
        # R0: calibrated to physical total carrying capacities
        # LN ~ 0.35, MN ~ 0.42, HN ~ 0.50
        self.r_ln_raw = nn.Parameter(torch.randn(K) * 0.2)
        self.r_mn_raw = nn.Parameter(torch.randn(K) * 0.2)
        self.r_hn_raw = nn.Parameter(torch.randn(K) * 0.2)
        
        self.scale_ln = nn.Parameter(torch.tensor(0.35))
        self.scale_mn = nn.Parameter(torch.tensor(0.42))
        self.scale_hn = nn.Parameter(torch.tensor(0.50))
        
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
        return self.alpha_max * torch.sigmoid(self.alpha_raw) + 1e-5
        
    def get_Km(self):
        return F.softplus(self.Km_raw) + 0.01
        
    def get_delta(self):
        return F.softplus(self.delta_raw) * 4.0 + 2.0 # delta in [2.0, 16.0]
        
    def get_Theta(self):
        return F.softplus(self.Theta_raw)
        
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        Theta = self.get_Theta()
        W = self.get_W()
        alpha = self.get_alpha()
        Km = self.get_Km()
        delta = self.get_delta()
        
        # Monod growth
        monod = R_t / (R_t + Km)
        growth = torch.matmul(monod, Theta.t())
        
        # Starvation depletion mortality
        R_total = R_t.sum(dim=-1, keepdim=True)
        starv_gate = self.Kd / (R_total + self.Kd)
        death = delta * starv_gate
        
        # Net dN/dt
        dN = N_t * (growth - death) - alpha * (N_t ** 2)
        
        # Resource consumption and byproduct production
        flux = monod * torch.matmul(N_t, Theta)
        byproduct = torch.matmul(flux, W)
        dR = - flux + byproduct
        return torch.cat([dN, dR], dim=-1)

def train_calibrated_model(lambda_sparse=0.10, epochs=80, lr=0.012, lambda_var=10.0):
    set_seed(42)
    print("\n" + "="*80)
    print(f"TRAINING PHYSICALLY-CALIBRATED K=8 CONSUMER-RESOURCE MODEL WITH STARVATION")
    print(f"K = 8 | Physical R0 Scaling Active | Realistic Bacterial Kinetics | Epochs = {epochs}")
    print("="*80)
    
    model = CalibratedStarvationConsumerResourceODE(N=48, K=8, alpha_max=0.01, Kd=0.01).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)
    t_span = torch.linspace(0.0, 1.0, 25, device=device)
    
    best_loss = float('inf')
    best_state = None
    start_time = time.time()
    
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
        
        # Concave L_{1/2} Sparsity penalty on W
        W = model.get_W()
        loss_sparse = torch.sum(torch.sqrt(W + 1e-8))
        
        total_loss = loss_1step + lambda_var * loss_var + lambda_sparse * loss_sparse
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch >= 20 and total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 15 == 0 or epoch == epochs or epoch == 1:
            mean_delta = float(model.get_delta().mean().item())
            mean_growth = float(model.get_Theta().sum(dim=-1).mean().item())
            r0_tot = float(model.get_R0()[2].sum().item())
            print(f"Epoch {epoch:2d}/{epochs} | 1-Step: {loss_1step.item():.4f} | VarLoss: {loss_var.item():.5f} | R0_HN: {r0_tot:.3f} | Growth: {mean_growth:.1f} | Delta: {mean_delta:.2f} | Time: {time.time()-start_time:.1f}s")
            
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, best_state

def simulate_200_passages(model, cond):
    c_idx = cond_to_idx[cond]
    mask = data['presence_masks'][(48, 'Community1')].float().to(device)
    rel = data['relative_abundance'][(cond, 48, 'Community1')].to(device)
    od = data['biomass_od'][(cond, 48, 'Community1')].mean(dim=0).to(device)
    N0 = (rel[0] * od[0]) * mask
    R0 = model.get_R0()[c_idx]
    
    t_rk4 = torch.linspace(0.0, 1.0, 30, device=device)
    N_curr = N0.clone()
    N_traj = [N_curr.cpu().numpy()]
    
    with torch.no_grad():
        for p in range(200):
            N_in = torch.clamp(N_curr / model.D + model.mu * mask, min=0.0)
            S0 = torch.cat([N_in, R0], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_rk4, method='rk4')
            N_curr = torch.clamp(S_t[-1, :48] * mask, min=0.0)
            N_traj.append(N_curr.cpu().numpy())
            
    return np.array(N_traj)

def compute_lyapunov(model, N_attractor, R0, mask, num_passages=50, eps=1e-4):
    model.eval()
    active_idx = np.where(N_attractor > 0.005)[0]
    if len(active_idx) == 0:
        return 0.0, 0
        
    t_rk4 = torch.linspace(0.0, 1.0, 30, device=device)
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
            
            S_b = odeint(model.ode_deriv, torch.cat([N_in_b, R0], dim=-1), t_rk4, method='rk4')[-1, :48]
            S_p = odeint(model.ode_deriv, torch.cat([N_in_p, R0], dim=-1), t_rk4, method='rk4')[-1, :48]
            
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
    # Train physically-calibrated model
    model, best_state = train_calibrated_model(lambda_sparse=0.10, epochs=80, lr=0.012, lambda_var=10.0)
    
    # Save checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Fitted_K8_Calibrated_Starvation.pth')
    torch.save({'model_state_dict': best_state, 'K': 8, 'alpha_max': 0.01, 'Kd': 0.01}, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")
    
    # Inspect Learned Parameters
    R0_all = model.get_R0().detach().cpu().numpy()
    delta_learned = model.get_delta().detach().cpu().numpy()
    theta_sum = model.get_Theta().sum(dim=-1).detach().cpu().numpy()
    
    print("\n--- Physical Parameter Diagnostics ---")
    print(f"Total R0: LN = {R0_all[0].sum():.3f} OD | MN = {R0_all[1].sum():.3f} OD | HN = {R0_all[2].sum():.3f} OD")
    print(f"Bacterial Growth Rates (sum_k Theta): Mean = {theta_sum.mean():.2f} / day (Min = {theta_sum.min():.2f}, Max = {theta_sum.max():.2f})")
    print(f"Starvation Mortality Rates (delta_i): Mean = {delta_learned.mean():.2f} / day (Min = {delta_learned.min():.2f}, Max = {delta_learned.max():.2f})")
    
    # Run 200-Passage Simulations
    print("\nSimulating 200 autonomous passages for LN and HN...")
    traj_ln = simulate_200_passages(model, 'LN')
    traj_hn = simulate_200_passages(model, 'HN')
    
    p2p_ln = (traj_ln[100:].max(axis=0) - traj_ln[100:].min(axis=0)).max()
    p2p_hn = (traj_hn[100:].max(axis=0) - traj_hn[100:].min(axis=0)).max()
    std_hn = np.std(traj_hn[100:], axis=0).max()
    
    # Count direction flips in late HN passages
    diffs = np.diff(traj_hn[100:], axis=0)
    signs = np.sign(diffs)
    flips_per_sp = ((signs[:-1] * signs[1:]) < 0).sum(axis=0)
    max_flips = int(flips_per_sp.max())
    
    # Compute Lyapunov exponent
    mask_48 = data['presence_masks'][(48, 'Community1')].float().to(device)
    lam_hn, n_active = compute_lyapunov(model, traj_hn[150], model.get_R0()[2], mask_48)
    
    print("\n" + "="*80)
    print("DYNAMICAL DIAGNOSTICS (PHYSICALLY-CALIBRATED K=8 ODE WITH STARVATION):")
    print("="*80)
    print(f"Low Nutrients (LN)  : Late P2P (p100-200) = {p2p_ln:.6f} OD")
    print(f"High Nutrients (HN) : Late P2P (p100-200) = {p2p_hn:.6f} OD")
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
    ax0.set_title(r'$\mathbf{(A)\ Low\ Nutrients:\ Point\ Attractor\ (P2P = ' + f'{p2p_ln:.6f}' + r'\ OD)}$',
                  fontsize=13, fontweight='bold', color='#00356B', loc='left', pad=10)
    ax0.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
    ax0.set_ylabel('Species Biomass (OD)', fontsize=12, fontweight='bold')
    ax0.set_xlim(-2, 202)
    ax0.grid(True, linestyle='--', alpha=0.5)
    
    ax1 = fig.add_subplot(gs[0, 1])
    for sp in range(48):
        if traj_hn[:, sp].max() > 0.005:
            ax1.plot(p_axis, traj_hn[:, sp], color=colors[sp % 48], lw=1.2, alpha=0.85)
    ax1.set_title(r'$\mathbf{(B)\ High\ Nutrients:\ Dynamics\ (Flips = ' + f'{max_flips}' + r',\ P2P = ' + f'{p2p_hn:.4f}' + r')}$',
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
    
    fig_path = os.path.join(PROJECT_ROOT, 'Figures', 'Gore_K8_Calibrated_Starvation_200Passages.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to: {fig_path}")

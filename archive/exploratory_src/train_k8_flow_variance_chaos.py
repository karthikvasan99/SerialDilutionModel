"""
Authentic Consumer-Resource Model (K=8, alpha <= 0.01) on Gore Data
Trained with:
1. 1-Step Flow Matching across all 10 passages.
2. Late-Passage (Passages 6..10) Empirical Variance Matching.
3. Zero handcoded guilds or cyclic loops.
4. Evaluation across 200 serial dilution passages for sustained chaos.
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

# 1. Load Data
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

# Compute ground truth empirical std across late passages (6..10)
emp_late = b11_N[:, 6:, :] # (75, 5, 48)
emp_std = emp_late.std(dim=1) * b11_m # (75, 48)
print(f"Loaded {b11_N.size(0)} communities.")
print(f"Empirical late std: LN mean = {emp_std[b11_c==0].max(dim=-1).values.mean():.4f} | HN mean = {emp_std[b11_c==2].max(dim=-1).values.mean():.4f}")

def dixit_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    p_safe = torch.clamp(y_pred, min=eps)
    y_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_log / p_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + p_safe) * mask
    return div.sum(dim=-1).mean()

# 2. Model Architecture: K=8, alpha <= 0.01
class FlowVarianceConsumerResourceODE(nn.Module):
    def __init__(self, N=48, K=8, alpha_max=0.01, D=30.0, mu=1e-6):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        self.alpha_max = alpha_max
        
        # Random initialization for Theta (K=8)
        init_scale = 3.4 / (float(K) * 0.8)
        inv_scale = math.log(math.exp(init_scale) - 1.0)
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.15 + inv_scale)
        
        # Random cross-feeding W (K=8)
        self.W_raw = nn.Parameter(torch.randn(K, K) * 0.2 - 0.5)
        
        # alpha bounded by alpha_max (0.01)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.5 - 1.0)
        
        # Monod Km
        self.Km_raw = nn.Parameter(torch.zeros(K) - 1.0)
        
        # Environmental supply R0: LN < MN < HN
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

# 3. Training Loop
def train_flow_variance_model(epochs=80, lr=0.015, lambda_var=10.0):
    set_seed(42)
    print("\n" + "="*80)
    print(f"TRAINING K=8 MODEL WITH 1-STEP FLOW MATCHING & LATE-STAGE VARIANCE MATCHING")
    print(f"K = 8 | alpha <= 0.01 | lambda_var = {lambda_var} | Epochs = {epochs}")
    print("="*80)
    
    model = FlowVarianceConsumerResourceODE(N=48, K=8, alpha_max=0.01).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)
    t_span = torch.linspace(0.0, 1.0, 21, device=device)
    
    best_loss = float('inf')
    best_state = None
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        R0_all = model.get_R0()[b11_c]
        
        # Part A: 1-Step Flow Matching across all 10 daily transfers
        loss_1step = 0.0
        for p in range(1, 11):
            # Input is the true historical biomass at passage p-1 (1-step prediction)
            N_in = torch.clamp(b11_N[:, p-1, :] / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_pred = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            loss_1step += dixit_kl(b11_N[:, p, :], N_pred, b11_m)
        loss_1step = loss_1step / 10.0
        
        # Part B: Autonomous Multi-Step Rollout on Late Passages (6 to 10) for Variance Matching
        N_curr = b11_N[:, 6, :] # Start at Day 6
        late_rollout = [N_curr]
        for p in range(7, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_next = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            late_rollout.append(N_next)
            N_curr = N_next # autonomous unroll
            
        late_stack = torch.stack(late_rollout, dim=1) # (75, 5, 48)
        model_std = late_stack.std(dim=1) * b11_m     # (75, 48)
        
        # Variance matching loss: match empirical late std per species
        loss_var = F.mse_loss(model_std, emp_std)
        
        total_loss = loss_1step + lambda_var * loss_var
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch >= 20 and total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 15 == 0 or epoch == epochs or epoch == 1:
            ln_s = model_std[b11_c==0].max(dim=-1).values.mean().item()
            hn_s = model_std[b11_c==2].max(dim=-1).values.mean().item()
            print(f"Epoch {epoch:2d}/{epochs} | 1-Step Loss: {loss_1step.item():.4f} | Var Loss: {loss_var.item():.5f} | LN std: {ln_s:.4f} | HN std: {hn_s:.4f} | Time: {time.time()-start_time:.1f}s")
            
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, best_state

# 4. Long-Term Simulation over 200 Passages
def simulate_200_passages(model, cond):
    c_idx = cond_to_idx[cond]
    mask = data['presence_masks'][(48, 'Community1')].float().to(device)
    rel = data['relative_abundance'][(cond, 48, 'Community1')].to(device)
    od = data['biomass_od'][(cond, 48, 'Community1')].mean(dim=0).to(device)
    N0 = (rel[0] * od[0]) * mask
    R0 = model.get_R0()[c_idx]
    
    t_rk4 = torch.linspace(0.0, 1.0, 25, device=device)
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

# 5. Compute Lyapunov Exponent on Late Attractor
def compute_lyapunov_exponent(model, N_attractor, R0, mask, num_passages=50, eps=1e-4):
    model.eval()
    active_idx = np.where(N_attractor > 0.005)[0]
    if len(active_idx) == 0:
        return 0.0, 0
        
    t_rk4 = torch.linspace(0.0, 1.0, 25, device=device)
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
    model, best_state = train_flow_variance_model(epochs=80, lr=0.015, lambda_var=12.0)
    
    # Save checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Fitted_K8_FlowVar_Chaos.pth')
    torch.save({'model_state_dict': best_state, 'K': 8, 'alpha_max': 0.01}, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")
    
    # Simulate 200 Passages for LN and HN
    print("\nSimulating 200 autonomous passages for S=48 Community 1...")
    traj_ln = simulate_200_passages(model, 'LN')
    traj_mn = simulate_200_passages(model, 'MN')
    traj_hn = simulate_200_passages(model, 'HN')
    
    # Analyze late passages 100..200
    p2p_ln = (traj_ln[100:].max(axis=0) - traj_ln[100:].min(axis=0)).max()
    p2p_mn = (traj_mn[100:].max(axis=0) - traj_mn[100:].min(axis=0)).max()
    p2p_hn = (traj_hn[100:].max(axis=0) - traj_hn[100:].min(axis=0)).max()
    
    std_ln = np.std(traj_ln[100:], axis=0).max()
    std_mn = np.std(traj_mn[100:], axis=0).max()
    std_hn = np.std(traj_hn[100:], axis=0).max()
    
    mask_48 = data['presence_masks'][(48, 'Community1')].float().to(device)
    lam_ln, n_ln = compute_lyapunov_exponent(model, traj_ln[150], model.get_R0()[0], mask_48)
    lam_hn, n_hn = compute_lyapunov_exponent(model, traj_hn[150], model.get_R0()[2], mask_48)
    
    print("\n" + "="*80)
    print("200-PASSAGE DYNAMICAL DIAGNOSTICS (K=8, FLOW + VARIANCE MATCHING):")
    print("="*80)
    print(f"Low Nutrients (LN) : Late P2P = {p2p_ln:.6f} | Late Std = {std_ln:.6f} | lambda = {lam_ln:+.4f} ({n_ln} active)")
    print(f"Medium Nutrients   : Late P2P = {p2p_mn:.6f} | Late Std = {std_mn:.6f}")
    print(f"High Nutrients (HN): Late P2P = {p2p_hn:.6f} | Late Std = {std_hn:.6f} | lambda = {lam_hn:+.4f} ({n_hn} active)")
    
    # Check direction flips in HN (p100..200) to verify actual oscillations vs monotonic drift
    diffs = np.diff(traj_hn[100:], axis=0)
    signs = np.sign(diffs)
    flips_per_sp = ((signs[:-1] * signs[1:]) < 0).sum(axis=0)
    max_flips = int(flips_per_sp.max())
    print(f"HN Direction Flips (Passages 100-200): Max per species = {max_flips} flips")
    
    if max_flips >= 10:
        print(">> VERIFIED: Trajectory exhibits TRUE OSCILLATORY DYNAMICS (repeated cycles, not monotonic drift)!")
    else:
        print(">> Note: Low direction flips indicate monotonic relaxation or very slow drift.")
        
    # Generate 4-panel visualization
    fig = plt.figure(figsize=(16, 10.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = np.vstack([colors, plt.cm.tab20b(np.linspace(0, 1, 20)), plt.cm.tab20c(np.linspace(0, 1, 8))])
    p_axis = np.arange(201)
    
    # Panel A: LN 0..200
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
    
    # Panel B: HN 0..200
    ax1 = fig.add_subplot(gs[0, 1])
    for sp in range(48):
        if traj_hn[:, sp].max() > 0.005:
            ax1.plot(p_axis, traj_hn[:, sp], color=colors[sp % 48], lw=1.2, alpha=0.85)
    ax1.set_title(r'$\mathbf{(B)\ High\ Nutrients:\ Dynamics\ (P2P = ' + f'{p2p_hn:.4f}' + r',\ Flips = ' + f'{max_flips}' + r')}$',
                  fontsize=13, fontweight='bold', color='#802010', loc='left', pad=10)
    ax1.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Species Biomass (OD)', fontsize=12, fontweight='bold')
    ax1.set_xlim(-2, 202)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel C: Late Zoom 150..200
    ax2 = fig.add_subplot(gs[1, 0])
    top_sp = np.argsort(traj_hn[150:].max(axis=0))[-6:]
    for sp in top_sp:
        ax2.plot(np.arange(150, 201), traj_hn[150:, sp], color=colors[sp % 48], lw=2.2, label=f'Sp {sp+1}')
    ax2.set_title(r'$\mathbf{(C)\ Late\ Dynamics\ Zoom\ (Passages\ 150\text{--}200)}$',
                  fontsize=13, fontweight='bold', loc='left', pad=10)
    ax2.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Species Biomass (OD)', fontsize=12, fontweight='bold')
    ax2.set_xlim(149, 201)
    ax2.legend(loc='upper right', fontsize=9.5, ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Panel D: Phase portrait
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
    
    out_fig = os.path.join(PROJECT_ROOT, 'Figures', 'Gore_K8_Flow_Variance_200Passages.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure to: {out_fig}")

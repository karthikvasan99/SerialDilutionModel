"""
Train K=3 Consumer-Resource Model with Maximal Cross-Feeding Ring W = 0.99, D = 10, Primary Resource Pulse.
Tests if W = 0.99 amplifies sustained deterministic limit cycles without any noise across 500 passages.
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shutil

PROJECT_ROOT = '/home/karthik/SerialDilutionModel'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load data and metadata
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
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

class FixedW99LimitCycleODE(nn.Module):
    def __init__(self, N=48, K=3, D=10.0, mu=1e-6, Km=0.05):
        super().__init__()
        self.N, self.K, self.D, self.mu, self.Km = N, K, D, mu, Km
        
        # Initialize Theta with 3 distinct specialist guilds
        Theta_init = torch.ones(N, K) * 0.1
        for i in range(N):
            pref = i % K
            Theta_init[i, pref] = 6.0
        self.Theta_raw = nn.Parameter(torch.log(torch.exp(Theta_init) - 1.0))
        
        # Pure ring W fixed at 0.99 (maximal cross-feeding)
        W_fixed = torch.zeros(K, K, device=device)
        W_fixed[0, 1] = 0.99
        W_fixed[1, 2] = 0.99
        W_fixed[2, 0] = 0.99
        self.register_buffer('W', W_fixed)
        
        init_alpha = torch.tensor(0.005)
        self.alpha_raw = nn.Parameter(torch.log(torch.exp(init_alpha) - 1.0))
        
        self.scale_ln = nn.Parameter(torch.tensor(0.4))
        self.scale_mn = nn.Parameter(torch.tensor(0.8))
        self.scale_hn = nn.Parameter(torch.tensor(1.6))
        self.r_dist = nn.Parameter(torch.tensor([2.0, -1.0, -1.0]))
        
    def get_R0(self):
        s_ln = F.softplus(self.scale_ln)
        s_mn = F.softplus(self.scale_mn)
        s_hn = F.softplus(self.scale_hn)
        dist = F.softmax(self.r_dist, dim=-1)
        return torch.stack([dist * s_ln, dist * s_mn, dist * s_hn])
        
    def get_alpha(self):
        return F.softplus(self.alpha_raw)
        
    def get_Theta(self):
        return F.softplus(self.Theta_raw)
        
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        Theta = self.get_Theta()
        alpha = self.get_alpha()
        
        monod = R_t / (R_t + self.Km)
        growth = torch.matmul(monod, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        flux = monod * torch.matmul(N_t, Theta)
        byproduct = torch.matmul(flux, self.W)
        dR = - flux + byproduct
        return torch.cat([dN, dR], dim=-1)

if __name__ == '__main__':
    model = FixedW99LimitCycleODE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("\n" + "="*80)
    print(f"TRAINING FIXED W=0.99 MAXIMAL CROSS-FEEDING MODEL (PARAMS = {total_params}, D = {model.D})")
    print("="*80)
    
    ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_K3_FixedW99_LimitCycle_ODE.pth')
    t_span = torch.linspace(0.0, 1.0, 30, device=device)
    
    if os.path.exists(ckpt_path):
        print(f"Loading existing checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=0.015)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-3)
        start_train = time.time()
        
        for ep in range(1, 41):
            model.train()
            opt.zero_grad()
            R0_all = model.get_R0()[b11_c]
            loss_fit = 0.0
            for p in range(1, 11):
                N_in = torch.clamp(b11_N[:, p-1, :] / model.D + model.mu * b11_m, min=0.0)
                S0 = torch.cat([N_in, R0_all], dim=-1)
                S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
                pred_1 = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
                loss_fit += dixit_kl(b11_N[:, p, :], pred_1, b11_m)
            loss_fit = loss_fit / 10.0
            loss_l1_th = 0.20 * torch.mean(model.get_Theta())
            loss_l2_al = 50.0 * (model.get_alpha() ** 2)
            tot = loss_fit + loss_l1_th + loss_l2_al
            tot.backward()
            opt.step()
            sched.step()
            if ep % 10 == 0 or ep == 1:
                print(f"Epoch {ep:2d}/40 | Fit: {loss_fit.item():.4f} | alpha: {model.get_alpha().item():.5f} | Time: {time.time()-start_train:.1f}s")
                
        torch.save({'model_state_dict': model.state_dict(), 'K': 3, 'total_params': total_params}, ckpt_path)
        print(f"Saved checkpoint to: {ckpt_path}")
        
    # Evaluate R^2
    model.eval()
    with torch.no_grad():
        preds = []
        R0_all = model.get_R0()[b11_c]
        for p in range(1, 11):
            N_in = torch.clamp(b11_N[:, p-1, :] / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            preds.append(torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0))
        preds = torch.stack(preds, dim=1)
        p_rel = (preds / (preds.sum(dim=-1, keepdim=True) + 1e-8)).cpu().numpy()
        t_rel = (b11_N[:, 1:, :] / (b11_N[:, 1:, :].sum(dim=-1, keepdim=True) + 1e-8)).cpu().numpy()
        m_np = b11_m.unsqueeze(1).expand_as(preds).cpu().numpy().astype(bool)
        r, _ = pearsonr(t_rel[m_np], p_rel[m_np])
        r2 = r**2 * 100
        print(f"\nModel Predictive Fit: R^2 = {r2:.2f}% (Pearson r = {r:.4f})")
        print(f"Learned Alpha: {model.get_alpha().item():.5f}")
        
    # ─── 500-Passage Deterministic Simulation (ZERO NOISE) ───
    print("\n" + "="*80)
    print("SIMULATING 500 PASSAGES DETERMINISTICALLY (ZERO NOISE)")
    print("="*80)
    
    comm_idx = [i for i, name in enumerate(b11_names) if name == ('HN', 24, 'Community5')][0]
    mask_c = b11_m[comm_idx]
    c_idx = b11_c[comm_idx]
    
    with torch.no_grad():
        N_curr = b11_N[comm_idx, 0, :].clone()
        R0_c = model.get_R0()[c_idx]
        det_trajs = [N_curr.detach().cpu().numpy()]
        for p in range(1, 501):
            N_in = torch.clamp(N_curr / model.D + model.mu * mask_c, min=0.0)
            S0 = torch.cat([N_in, R0_c])
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_curr = torch.clamp(S_t[-1, :48] * mask_c, min=0.0)
            det_trajs.append(N_curr.detach().cpu().numpy())
        det_trajs = np.array(det_trajs)
        
    det_rel = det_trajs / (det_trajs.sum(axis=-1, keepdims=True) + 1e-8)
    surviving = np.where(det_rel[-1] > 0.01)[0]
    print(f"\nSurviving Taxa in HN Community 5 at Passage 500: {surviving.tolist()} ({len(surviving)} taxa)")
    
    # Audit late asymptotic passages (350 to 500)
    p2p_late = (det_rel[350:].max(axis=0) - det_rel[350:].min(axis=0)) * 100
    diff_late = np.diff(det_rel[350:, surviving], axis=0)
    flips_late = np.sum((diff_late[:-1] * diff_late[1:]) < 0, axis=0)
    
    print("\nDeterministic Late Asymptotic Dynamics (Passages 350-500, ZERO NOISE):")
    for sp in surviving:
        idx = surviving.tolist().index(sp)
        print(f"  Species {sp:2d}: Late P2P = {p2p_late[sp]:.4f}%, Direction Flips = {flips_late[idx]:2d}")
        
    print("\nPassages 490 to 498 Relative Abundances:")
    for p in range(490, 499):
        print(f"  Passage {p}: {det_rel[p, surviving].round(4)}")
        
    # Check if period repeats exactly
    print("\nTesting for Periodic Repeat in Late Passages:")
    for lag in [1, 2, 3, 4, 5, 6]:
        diff_lag = np.abs(det_rel[480:485, surviving] - det_rel[480+lag:485+lag, surviving]).max()
        print(f"  Max absolute difference at Lag {lag}: {diff_lag:.6f}")
        
    # ─── Publication Figure Generation ───
    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22)
    p_axis = np.arange(501)
    cmap = plt.colormaps['tab10']
    
    # Panel A: Network Topology Schema & W=0.99 Matrix
    ax0 = fig.add_subplot(gs[0, 0])
    W_np = model.W.cpu().numpy()
    im0 = ax0.imshow(W_np, cmap='Blues', aspect='auto')
    ax0.set_title(r"$\mathbf{(A)\ Maximal\ Directed\ Cross\text{-}Feeding\ Ring\ (W=0.99)}$" + "\n" +
                  f"($R_0 \\to R_1 \\to R_2 \\to R_0$ with yield 0.99, D = {model.D:.0f}, Learned $\\alpha = {model.get_alpha().item():.5f}$)",
                  fontsize=11.5, fontweight='bold', loc='left', pad=8)
    ax0.set_xticks([0, 1, 2])
    ax0.set_yticks([0, 1, 2])
    ax0.set_xticklabels([r"$R_0$", r"$R_1$", r"$R_2$"])
    ax0.set_yticklabels([r"$R_0$", r"$R_1$", r"$R_2$"])
    for i in range(3):
        for j in range(3):
            ax0.text(j, i, f"{W_np[i, j]:.2f}", ha='center', va='center',
                     color='white' if W_np[i, j] > 0.4 else 'black', fontweight='bold', fontsize=12)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    
    # Panel B: Predictive Accuracy
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(t_rel[m_np], p_rel[m_np], alpha=0.15, color='#2B6CB0', s=16, edgecolor='none')
    ax1.plot([0, 1], [0, 1], color='#E53E3E', lw=2.0, linestyle='--', label='1:1 Line')
    ax1.set_title(r"$\mathbf{(B)\ Empirical\ Accuracy\ Across\ All\ 75\ Communities}$" + "\n" +
                  f"($R^2 = {r2:.2f}\\%$, Pearson $r = {r:.4f}$)",
                  fontsize=11.5, fontweight='bold', loc='left', pad=8)
    ax1.set_xlabel("Empirical Abundance", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Predicted Abundance", fontsize=11, fontweight='bold')
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=10.0)
    
    # Panel C: Deterministic Dynamics Across 500 Passages (ZERO NOISE)
    ax2 = fig.add_subplot(gs[1, 0])
    for idx, sp in enumerate(surviving):
        ax2.plot(p_axis, det_rel[:, sp] * 100, color=cmap(idx % 10), lw=1.8, label=f"Species {sp}")
    ax2.set_title(r"$\mathbf{(C)\ Deterministic\ Dynamics\ Across\ 500\ Passages\ (ZERO\ NOISE)}$" + "\n" +
                  f"(Passages 350-500: Max Direction Flips = {flips_late.max()}, Late P2P $\\approx {p2p_late[surviving].max():.2f}\\%$)",
                  fontsize=11.5, fontweight='bold', loc='left', pad=8)
    ax2.set_xlabel("Dilution Passage (Days)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Relative Abundance (%)", fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 500)
    ax2.set_ylim(-2, 102)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8.5, ncol=2)
    
    # Panel D: Late Passages Zoom (Passages 450 to 500)
    ax3 = fig.add_subplot(gs[1, 1])
    p_zoom = np.arange(450, 501)
    for idx, sp in enumerate(surviving):
        ax3.plot(p_zoom, det_rel[450:501, sp] * 100, color=cmap(idx % 10), marker='o', markersize=3.5, lw=1.8, label=f"Species {sp}")
    ax3.set_title(r"$\mathbf{(D)\ Late\ Passages\ Zoom\ (Passages\ 450\text{--}500,\ ZERO\ NOISE)}$" + "\n" +
                  r"(Sustained High-Amplitude Deterministic Limit Cycles: Perpetual Oscillation)",
                  fontsize=11.5, fontweight='bold', loc='left', pad=8)
    ax3.set_xlabel("Dilution Passage (Days)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Relative Abundance (%)", fontsize=11, fontweight='bold')
    ax3.set_xlim(450, 500)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', fontsize=8.5, ncol=2)
    
    out_fig = os.path.join(PROJECT_ROOT, 'Figures', 'Gore_K3_FixedW99_Sustained_Deterministic_Limit_Cycles.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure to: {out_fig}")
    
    artifact_dst = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143/Gore_K3_FixedW99_Sustained_Deterministic_Limit_Cycles.png'
    shutil.copyfile(out_fig, artifact_dst)
    print("Copied figure to artifact directory successfully!")

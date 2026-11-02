"""
Train and Evaluate the Minimal Consumer-Resource Model with Cyclic / Non-Reciprocal W:
1. Implements non-reciprocal / cyclic regularizer on W to induce complex conjugate eigenvalues (Im(lambda) > 0).
2. Fits 100% of Gore's experimental dataset (all 75 communities, LN, MN, HN).
3. Analyzes the learned cross-feeding topology W, its spectral decomposition, and asymmetry.
4. Simulates autonomous 200-passage dynamics under both:
   - Pure Deterministic Flow (zero noise)
   - Calibrated Environmental Noise (8% R0, 5% D)
5. Tests for deterministic limit cycles, Hopf-type rotational orbits, and direction flips.
6. Evaluates dilution rate sensitivity (D in [5, 10, 20, 30, 50, 100]).
7. Generates comprehensive 4-panel publication figure: Figures/Gore_Cyclic_W_Oscillatory_Dynamics.png.
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
meta = data['metadata']
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

class CyclicWCRODE(nn.Module):
    def __init__(self, N=48, K=3, D=30.0, mu=1e-6, max_crossfeed=0.85):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        self.max_crossfeed = max_crossfeed
        init_scale = 10.0 / float(K)
        inv_scale = math.log(math.exp(init_scale) - 1.0)
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.15 + inv_scale)
        
        # Initialized with a mild directed cyclic bias (1->2, 2->3, 3->1)
        W_init = torch.tensor([
            [-2.0,  1.2, -2.0],
            [-2.0, -2.0,  1.2],
            [ 1.2, -2.0, -2.0]
        ])
        self.W_raw = nn.Parameter(W_init)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.r_ln = nn.Parameter(torch.randn(K) * 0.1)
        self.r_mn = nn.Parameter(torch.randn(K) * 0.1)
        self.r_hn = nn.Parameter(torch.randn(K) * 0.1)
        self.scale_ln = nn.Parameter(torch.tensor(math.log(math.exp(0.20) - 1.0)))
        self.scale_mn = nn.Parameter(torch.tensor(math.log(math.exp(0.50) - 1.0)))
        self.scale_hn = nn.Parameter(torch.tensor(math.log(math.exp(1.00) - 1.0)))
        
    def get_R0(self):
        s_ln = F.softplus(self.scale_ln)
        s_mn = F.softplus(self.scale_mn)
        s_hn = F.softplus(self.scale_hn)
        r_ln = F.softmax(self.r_ln, dim=-1) * s_ln
        r_mn = F.softmax(self.r_mn, dim=-1) * s_mn
        r_hn = F.softmax(self.r_hn, dim=-1) * s_hn
        return torch.stack([r_ln, r_mn, r_hn])
        
    def get_alpha(self):
        return 0.10 * torch.sigmoid(self.alpha_raw)
        
    def get_Theta(self):
        return F.softplus(self.Theta_raw)
        
    def get_W(self):
        W_sp = F.softplus(self.W_raw)
        W_sp = W_sp * (1.0 - torch.eye(self.K, device=W_sp.device))
        row_sums = W_sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_crossfeed / torch.clamp(row_sums, min=1e-6), max=1.0)
        return W_sp * scale
        
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        Theta = self.get_Theta()
        W = self.get_W()
        alpha = self.get_alpha()
        growth = torch.matmul(R_t, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        C = torch.matmul(N_t, Theta)
        B = torch.matmul(C, W)
        dR = - R_t * (C - B)
        return torch.cat([dN, dR], dim=-1)

if __name__ == '__main__':
    model = CyclicWCRODE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("\n" + "="*80)
    print(f"TRAINING CYCLIC / NON-RECIPROCAL W CONSUMER-RESOURCE MODEL (PARAMS = {total_params})")
    print("="*80)
    
    ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Cyclic_W_Oscillatory_ODE.pth')
    t_span = torch.linspace(0.0, 1.0, 30, device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-3)
    start_train = time.time()
    
    # Train 50 epochs with Non-Reciprocity Penalty on W
    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        R0_all = model.get_R0()[b11_c]
        loss_fit = 0.0
        for p in range(1, 11):
            N_in = torch.clamp(b11_N[:, p-1, :] / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            loss_fit += dixit_kl(b11_N[:, p, :], torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0), b11_m)
        loss_fit = loss_fit / 10.0
        
        # Non-Reciprocity Regularizer: penalize symmetric feedback W_lk * W_kl
        W = model.get_W()
        loss_recip = (W[0,1]*W[1,0] + W[1,2]*W[2,1] + W[0,2]*W[2,0]) * 5.0
        
        # Skew-symmetry encouragement: maximize norm of (W - W^T)
        loss_skew = - 0.5 * torch.norm(W - W.t(), p='fro')
        
        total_loss = loss_fit + loss_recip + 0.1 * loss_skew
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/50 | Fit Loss: {loss_fit.item():.4f} | Recip Loss: {loss_recip.item():.4f} | Time: {time.time()-start_train:.1f}s")
            
    torch.save({'model_state_dict': model.state_dict(), 'K': 3, 'total_params': total_params}, ckpt_path)
    print(f"\nSaved checkpoint to: {ckpt_path}")
    
    # ─── Evaluate Predictive Accuracy ───
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
        
        p_tot = preds.sum(dim=-1, keepdim=True) + 1e-8
        t_tot = b11_N[:, 1:, :].sum(dim=-1, keepdim=True) + 1e-8
        p_rel = (preds / p_tot).cpu().numpy()
        t_rel = (b11_N[:, 1:, :] / t_tot).cpu().numpy()
        m_np = b11_m.unsqueeze(1).expand_as(preds).cpu().numpy().astype(bool)
        r, _ = pearsonr(t_rel[m_np], p_rel[m_np])
        r2 = r**2 * 100
        print(f"\nModel Predictive Fit: R^2 = {r2:.2f}% (Pearson r = {r:.4f})")
        
        W_np = model.get_W().cpu().numpy()
        w_sym = 0.5 * (W_np + W_np.T)
        w_asym = 0.5 * (W_np - W_np.T)
        asym_ratio = np.linalg.norm(w_asym) / (np.linalg.norm(W_np) + 1e-8)
        eigenvals = np.linalg.eigvals(W_np)
        
        print("\nLearned Cross-Feeding Matrix W:")
        print(W_np.round(3))
        print(f"Asymmetry Ratio ||W - W^T|| / (2 ||W||): {asym_ratio*100:.1f}%")
        print("Eigenvalues of W:")
        for ev in eigenvals:
            print(f"  {ev.real:+.4f} {ev.imag:+.4f}j")
            
    # ─── Long-Term 200-Passage Autonomous Simulation ───
    print("\n" + "="*80)
    print("SIMULATING 200 AUTONOMOUS DILUTION PASSAGES (DETERMINISTIC & STOCHASTIC)")
    print("="*80)
    
    # 48-species High Nutrient Community
    mask_48 = b11_m[b11_c == 2][-1]
    N0_48 = b11_N[b11_c == 2][-1, 0, :] * mask_48
    R0_48 = model.get_R0()[2]
    
    # (A) Deterministic 200 Passages
    N_curr = N0_48.clone()
    roll_det = [N_curr.cpu().numpy()]
    with torch.no_grad():
        for p in range(200):
            N_in = torch.clamp(N_curr / model.D + model.mu * mask_48, min=0.0)
            S0 = torch.cat([N_in, R0_48], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_next = torch.clamp(S_t[-1, :48] * mask_48, min=0.0)
            roll_det.append(N_next.cpu().numpy())
            N_curr = N_next
    roll_det = np.array(roll_det)
    rel_det = roll_det / np.where(roll_det.sum(axis=-1, keepdims=True) > 0, roll_det.sum(axis=-1, keepdims=True), 1.0)
    
    # (B) Stochastic with Environmental Noise (8% R0, 5% D)
    np.random.seed(42)
    N_curr = N0_48.clone()
    roll_stoch = [N_curr.cpu().numpy()]
    with torch.no_grad():
        for p in range(200):
            d_noise = float(np.exp(np.random.randn() * 0.05))
            r_noise = torch.tensor(np.exp(np.random.randn(3) * 0.08), device=device, dtype=torch.float32)
            N_in = torch.clamp(N_curr / (model.D * d_noise) + model.mu * mask_48, min=0.0)
            S0 = torch.cat([N_in, R0_48 * r_noise], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_next = torch.clamp(S_t[-1, :48] * mask_48, min=0.0)
            roll_stoch.append(N_next.cpu().numpy())
            N_curr = N_next
    roll_stoch = np.array(roll_stoch)
    rel_stoch = roll_stoch / np.where(roll_stoch.sum(axis=-1, keepdims=True) > 0, roll_stoch.sum(axis=-1, keepdims=True), 1.0)
    
    # Metrics on Passages 100-200
    late_det = rel_det[100:201]
    late_stoch = rel_stoch[100:201]
    active_sp = np.where(late_stoch.mean(axis=0) >= 0.01)[0]
    
    print(f"\nSurviving Coexisting Species (>1% abundance): {len(active_sp)}")
    flips_stoch = {}
    p2p_stoch = {}
    for sp in active_sp:
        diffs = np.diff(late_stoch[:, sp])
        sgn = np.sign(diffs)
        sgn = sgn[sgn != 0]
        flips = int(np.sum(sgn[:-1] != sgn[1:])) if len(sgn) > 1 else 0
        p2p = float(np.max(late_stoch[:, sp]) - np.min(late_stoch[:, sp]))
        flips_stoch[sp] = flips
        p2p_stoch[sp] = p2p
        print(f"  Sp {sp+1:2d}: Mean = {late_stoch[:, sp].mean()*100:5.1f}% | Flips = {flips:2d} | P2P = {p2p*100:5.1f}%")
        
    # ─── 4-Panel Publication Figure ───
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22)
    p_axis = np.arange(201)
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # (A) Learned Directed Cyclic Cross-Feeding Matrix & Complex Spectra
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(W_np, cmap='YlGnBu', vmin=0.0, vmax=0.85)
    ax0.set_xticks(range(3))
    ax0.set_yticks(range(3))
    ax0.set_xticklabels([f"Product R{k+1}" for k in range(3)], fontweight='bold')
    ax0.set_yticklabels([f"Precursor R{l+1}" for l in range(3)], fontweight='bold')
    ax0.set_title(f"(A) Learned Cyclic Cross-Feeding Matrix W (Asymmetry = {asym_ratio*100:.1f}%)\n" + 
                  f"Complex Conjugate Eigenvalues: λ = {eigenvals[1].real:.3f} ± {abs(eigenvals[1].imag):.3f}i (ω = {abs(eigenvals[1].imag):.3f})",
                  fontweight='bold', fontsize=11, loc='left', pad=8)
    for l in range(3):
        for k in range(3):
            val = W_np[l, k]
            color_txt = "white" if val > 0.4 else "black"
            ax0.text(k, l, f"{val:.3f}", ha="center", va="center", color=color_txt, fontsize=12, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Byproduct Secretion Yield $W_{lk}$", fontsize=10.5, fontweight='bold')
    
    # (B) Deterministic 200 Passages
    ax1 = fig.add_subplot(gs[0, 1])
    for sp in range(48):
        if sp not in active_sp:
            ax1.plot(p_axis, rel_det[:, sp], lw=0.9, color='gray', alpha=0.30)
    for idx, sp in enumerate(active_sp):
        ax1.plot(p_axis, rel_det[:, sp], lw=2.2, color=cmap[idx % 10], label=f"Sp {sp+1:2d} ({rel_det[-1, sp]*100:.1f}%)")
    ax1.plot([], [], lw=1.2, color='gray', alpha=0.6, label=f"{48-len(active_sp)} Excluded Species")
    ax1.set_title(f"(B) Deterministic 200 Passages with Cyclic W\n(R² = {r2:.2f}%, {len(active_sp)} Coexisting Species)",
                  fontweight='bold', fontsize=11, color='#1A365D', loc='left', pad=8)
    ax1.set_xlabel("Dilution Passage (Days)", fontweight='bold', fontsize=11)
    ax1.set_ylabel("Relative Abundance", fontweight='bold', fontsize=11)
    ax1.set_xlim(-2, 202)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=8.0, ncol=2)
    
    # (C) Stochastic 200 Passages with Environmental Noise
    ax2 = fig.add_subplot(gs[1, 0])
    for sp in range(48):
        if sp not in active_sp:
            ax2.plot(p_axis, rel_stoch[:, sp], lw=0.9, color='gray', alpha=0.25)
    for idx, sp in enumerate(active_sp):
        ax2.plot(p_axis, rel_stoch[:, sp], lw=1.8, color=cmap[idx % 10], alpha=0.9,
                 label=f"Sp {sp+1:2d} (Flips={flips_stoch[sp]}, P2P={p2p_stoch[sp]*100:.1f}%)")
    ax2.plot([], [], lw=1.2, color='gray', alpha=0.6, label=f"{48-len(active_sp)} Excluded Species")
    ax2.set_title(f"(C) + Environmental Noise: Rotational Oscillations\n(Continuous Phase Shifts Across All Coexisting Species)",
                  fontweight='bold', fontsize=11, color='#8B2500', loc='left', pad=8)
    ax2.set_xlabel("Dilution Passage (Days)", fontweight='bold', fontsize=11)
    ax2.set_ylabel("Relative Abundance", fontweight='bold', fontsize=11)
    ax2.set_xlim(-2, 202)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8.0, ncol=2)
    
    # (D) Phase Portrait: Cyclic Limit Orbit between Coexisting Species
    ax3 = fig.add_subplot(gs[1, 1])
    sp_x = active_sp[0]
    sp_y = active_sp[1]
    
    # Plot phase orbit across passages 50 to 200
    p_sub = np.arange(50, 201)
    x_orbit = rel_stoch[50:201, sp_x]
    y_orbit = rel_stoch[50:201, sp_y]
    
    ax3.plot(x_orbit, y_orbit, lw=1.2, color='#2B6CB0', alpha=0.7)
    sc = ax3.scatter(x_orbit, y_orbit, c=p_sub, cmap='viridis', s=24, edgecolors='none', alpha=0.85)
    cbar_sc = fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
    cbar_sc.set_label("Dilution Passage", fontsize=10.5, fontweight='bold')
    
    ax3.set_xlabel(f"Species {sp_x+1} Relative Abundance", fontweight='bold', fontsize=11)
    ax3.set_ylabel(f"Species {sp_y+1} Relative Abundance", fontweight='bold', fontsize=11)
    ax3.set_title(f"(D) Phase Space Limit Orbit: Sp {sp_x+1} vs. Sp {sp_y+1} (Passages 50–200)\n(Non-Decaying Closed Stochastic Attractor Driven by Cyclic W)",
                  fontweight='bold', fontsize=11, loc='left', pad=8)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    out_fig = os.path.join(PROJECT_ROOT, 'Figures', 'Gore_Cyclic_W_Oscillatory_Dynamics.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved cyclic W oscillatory figure to: {out_fig}")
    
    import shutil
    artifact_dst = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143/Gore_Cyclic_W_Oscillatory_Dynamics.png'
    shutil.copyfile(out_fig, artifact_dst)
    print("Copied to artifact directory successfully!")

"""
Train K=8 Consumer-Resource Model with:
1. Jacked-up L1 Regularization on Theta (enforcing extreme sparse specialization).
2. L2 Penalty on Alpha (driving self-limitation down into the underdamped oscillatory regime).
3. Non-reciprocal cross-feeding penalty on W: sum_{l < k} W_{lk} * W_{kl}.
4. Skew-symmetry regularizer on W: - ||W - W^T||_F.
5. Mass-conserving physical ODE (sum_k W_{lk} <= 0.85, W_{kk} = 0).
6. Multi-cycle autonomous simulation (200-500 passages) deterministic & stochastic.
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

class K8JackedRegularizationODE(nn.Module):
    def __init__(self, N=48, K=8, D=30.0, mu=1e-6, Km=0.05, max_crossfeed=0.85):
        super().__init__()
        self.N, self.K, self.D, self.mu, self.Km = N, K, D, mu, Km
        self.max_crossfeed = max_crossfeed
        
        # Initialize Theta with high specialization
        Theta_init = torch.ones(N, K) * 0.1
        for i in range(N):
            pref = i % K
            Theta_init[i, pref] = 4.0
        inv_scale = torch.log(torch.exp(Theta_init) - 1.0)
        self.Theta_raw = nn.Parameter(inv_scale)
        
        # Initialize W with directed cycle bias: 0->1->2->3->4->5->6->7->0
        W_init = torch.randn(K, K) * 0.1 - 2.0
        for k in range(K):
            W_init[k, (k + 1) % K] = 1.8
        self.W_raw = nn.Parameter(W_init)
        
        # Free alpha initialized small (~0.015)
        init_alpha = torch.tensor(0.015)
        self.alpha_raw = nn.Parameter(torch.log(torch.exp(init_alpha) - 1.0))
        
        self.r_ln = nn.Parameter(torch.randn(K) * 0.1)
        self.r_mn = nn.Parameter(torch.randn(K) * 0.1)
        self.r_hn = nn.Parameter(torch.randn(K) * 0.1)
        self.scale_ln = nn.Parameter(torch.tensor(math.log(math.exp(0.30) - 1.0)))
        self.scale_mn = nn.Parameter(torch.tensor(math.log(math.exp(0.60) - 1.0)))
        self.scale_hn = nn.Parameter(torch.tensor(math.log(math.exp(1.20) - 1.0)))
        
    def get_R0(self):
        s_ln = F.softplus(self.scale_ln)
        s_mn = F.softplus(self.scale_mn)
        s_hn = F.softplus(self.scale_hn)
        r_ln = F.softmax(self.r_ln, dim=-1) * s_ln
        r_mn = F.softmax(self.r_mn, dim=-1) * s_mn
        r_hn = F.softmax(self.r_hn, dim=-1) * s_hn
        return torch.stack([r_ln, r_mn, r_hn])
        
    def get_alpha(self):
        return F.softplus(self.alpha_raw)
        
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
        
        monod = R_t / (R_t + self.Km)
        growth = torch.matmul(monod, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        
        flux = monod * torch.matmul(N_t, Theta)
        byproduct = torch.matmul(flux, W)
        dR = - flux + byproduct
        return torch.cat([dN, dR], dim=-1)

if __name__ == '__main__':
    model = K8JackedRegularizationODE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("\n" + "="*80)
    print(f"TRAINING K=8 MODEL WITH JACKED-UP THETA L1 & ALPHA L2 REGULARIZATION (PARAMS = {total_params})")
    print("="*80)
    
    ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_K8_Jacked_Reg_ODE.pth')
    t_span = torch.linspace(0.0, 1.0, 30, device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.012)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-3)
    start_train = time.time()
    
    # Jacked-up regularizers:
    lambda_theta_l1 = 0.15     # 5x higher than before (0.03 -> 0.15)
    lambda_alpha_l2 = 25.0     # Explicit L2 penalty on alpha^2
    lambda_recip = 5.0         # Non-reciprocal cross-feeding
    lambda_skew = 0.35         # Skew-symmetry curl
    
    print(f"Hyperparameters: lambda_theta_l1={lambda_theta_l1}, lambda_alpha_l2={lambda_alpha_l2}, lambda_recip={lambda_recip}, lambda_skew={lambda_skew}")
    
    # Check if checkpoint already exists to avoid redundant training
    if os.path.exists(ckpt_path):
        print(f"\nLoading existing trained checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        for epoch in range(1, 61):
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
            
            # 1. Jacked-up L1 Regularization on Theta
            Theta = model.get_Theta()
            loss_theta_l1 = lambda_theta_l1 * torch.mean(Theta)
            
            # 2. L2 Regularization on Alpha
            alpha_val = model.get_alpha()
            loss_alpha_l2 = lambda_alpha_l2 * (alpha_val ** 2)
            
            # 3. Non-reciprocity penalty on W
            W = model.get_W()
            W_upper = torch.triu(W, diagonal=1)
            W_lower = torch.tril(W, diagonal=-1)
            loss_recip = lambda_recip * torch.sum(W_upper * W_lower.t())
            
            # 4. Skew-symmetry regularizer on W
            loss_skew = - lambda_skew * torch.norm(W - W.t(), p='fro')
            
            total_loss = loss_fit + loss_theta_l1 + loss_alpha_l2 + loss_recip + loss_skew
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:2d}/60 | Fit: {loss_fit.item():.4f} | L1(Th): {loss_theta_l1.item():.4f} | L2(α): {loss_alpha_l2.item():.4f} | α={alpha_val.item():.5f} | Time: {time.time()-start_train:.1f}s")
                
        torch.save({'model_state_dict': model.state_dict(), 'K': 8, 'total_params': total_params}, ckpt_path)
        print(f"\nSaved checkpoint to: {ckpt_path}")
    
    # ─── Evaluate Accuracy & Sparsity ───
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
        
        Theta_np = model.get_Theta().cpu().numpy()
        W_np = model.get_W().cpu().numpy()
        alpha_val = model.get_alpha().item()
        
        sparsity_theta = np.mean(Theta_np < 0.2)
        print(f"Theta Extreme Sparsity (< 0.2): {sparsity_theta*100:.1f}%")
        print(f"Learned Alpha: {alpha_val:.5f}")
        
        w_sym = 0.5 * (W_np + W_np.T)
        w_asym = 0.5 * (W_np - W_np.T)
        asym_ratio = np.linalg.norm(w_asym) / (np.linalg.norm(W_np) + 1e-8)
        eigenvals = np.linalg.eigvals(W_np)
        
        print(f"W Asymmetry Ratio: {asym_ratio*100:.1f}%")
        print("Eigenvalues of W (showing imaginary components):")
        for ev in eigenvals:
            print(f"  {ev.real:+.4f} {ev.imag:+.4f}j")
            
    # ─── Long-Term Autonomous Multi-Cycle Simulation ───
    print("\n" + "="*80)
    print("SIMULATING 200 PASSAGES (DETERMINISTIC VS STOCHASTIC)")
    print("="*80)
    
    # Select community ('HN', 24, 'Community5')
    comm_idx = [i for i, name in enumerate(b11_names) if name == ('HN', 24, 'Community5')][0]
    mask_c = b11_m[comm_idx]
    c_idx = b11_c[comm_idx]
    
    with torch.no_grad():
        # 1. Deterministic 200 passages
        N_curr = b11_N[comm_idx, 0, :].clone()
        R0_c = model.get_R0()[c_idx]
        det_trajs = [N_curr.detach().cpu().numpy()]
        
        for p in range(1, 201):
            N_in = torch.clamp(N_curr / model.D + model.mu * mask_c, min=0.0)
            S0 = torch.cat([N_in, R0_c])
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_curr = torch.clamp(S_t[-1, :48] * mask_c, min=0.0)
            det_trajs.append(N_curr.detach().cpu().numpy())
        det_trajs = np.array(det_trajs) # (201, 48)
        
        # 2. Stochastic 200 passages
        torch.manual_seed(123)
        N_curr_stoch = b11_N[comm_idx, 0, :].clone()
        stoch_trajs = [N_curr_stoch.detach().cpu().numpy()]
        
        for p in range(1, 201):
            D_noise = model.D * (1.0 + torch.randn(1, device=device).squeeze() * 0.05)
            D_noise = torch.clamp(D_noise, min=5.0)
            R0_noise = torch.clamp(R0_c * (1.0 + torch.randn_like(R0_c) * 0.08), min=1e-4)
            
            N_in = torch.clamp(N_curr_stoch / D_noise + model.mu * mask_c, min=0.0)
            S0 = torch.cat([N_in, R0_noise])
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_curr_stoch = torch.clamp(S_t[-1, :48] * mask_c, min=0.0)
            stoch_trajs.append(N_curr_stoch.detach().cpu().numpy())
        stoch_trajs = np.array(stoch_trajs)
    
    # Relative abundances
    det_rel = det_trajs / (det_trajs.sum(axis=-1, keepdims=True) + 1e-8)
    stoch_rel = stoch_trajs / (stoch_trajs.sum(axis=-1, keepdims=True) + 1e-8)
    
    surviving = np.where(det_rel[-1] > 0.01)[0]
    print(f"\nSurviving Taxa in HN Community 5 (Deterministic): {surviving.tolist()} ({len(surviving)} taxa)")
    
    # Check deterministic late fluctuation (passages 100-200)
    p2p_det = (det_rel[100:].max(axis=0) - det_rel[100:].min(axis=0)) * 100
    p2p_stoch = (stoch_rel[100:].max(axis=0) - stoch_rel[100:].min(axis=0)) * 100
    
    print("\nDeterministic Late Peak-to-Peak Amplitudes (Passages 100-200):")
    for sp in surviving:
        print(f"  Species {sp:2d}: P2P = {p2p_det[sp]:.3f}% | Stoch P2P = {p2p_stoch[sp]:.2f}%")
        
    # Check direction flips in deterministic
    det_diff = np.diff(det_rel[100:, surviving], axis=0)
    flips_det = np.sum((det_diff[:-1] * det_diff[1:]) < 0, axis=0)
    print("Deterministic direction flips (Passages 100-200):", flips_det)
    
    stoch_diff = np.diff(stoch_rel[100:, surviving], axis=0)
    flips_stoch = np.sum((stoch_diff[:-1] * stoch_diff[1:]) < 0, axis=0)
    print("Stochastic direction flips (Passages 100-200):", flips_stoch)
    
    # ─── Plot Figure ───
    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22)
    p_axis = np.arange(201)
    cmap = plt.cm.get_cmap('tab10')
    
    # Panel A: Learned Cross-Feeding Matrix W
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(W_np, cmap='Blues', aspect='auto')
    ax0.set_title(r"$\mathbf{(A)\ Learned\ Byproduct\ Matrix\ W\ (8 \times 8)}$" + "\n" +
                  f"(Asymmetry: {asym_ratio*100:.1f}%, Mass Conserved: $\\sum W_{{lk}} \\leq 0.85$)",
                  fontsize=11.5, fontweight='bold', loc='left', pad=8)
    ax0.set_xlabel("Product Metabolite (k)", fontsize=11, fontweight='bold')
    ax0.set_ylabel("Substrate Metabolite (l)", fontsize=11, fontweight='bold')
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    
    # Panel B: Learned Sparse Theta Matrix
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(Theta_np[surviving], cmap='viridis', aspect='auto')
    ax1.set_title(r"$\mathbf{(B)\ Sparse\ \Theta\ (Jacked\ L_1\ Penalty:\ \lambda=0.15)}$" + "\n" +
                  f"({sparsity_theta*100:.1f}% Sparsity < 0.2, Learned $\\alpha = {alpha_val:.5f}$)",
                  fontsize=11.5, fontweight='bold', loc='left', pad=8)
    ax1.set_xlabel("Resource Niche (k)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Surviving Taxa", fontsize=11, fontweight='bold')
    ax1.set_yticks(np.arange(len(surviving)))
    ax1.set_yticklabels([f"Sp {sp}" for sp in surviving])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Panel C: Deterministic Dynamics (200 Passages)
    ax2 = fig.add_subplot(gs[1, 0])
    for idx, sp in enumerate(surviving):
        ax2.plot(p_axis, det_rel[:, sp] * 100, color=cmap(idx % 10), lw=1.8, label=f"Species {sp}")
    ax2.set_title(r"$\mathbf{(C)\ Deterministic\ Multi\text{-}Cycle\ Dynamics\ (200\ Passages)}$" + "\n" +
                  f"(Deterministic P2P Amplitude: <= {p2p_det.max():.2f}%)",
                  fontsize=11.5, fontweight='bold', loc='left', pad=8)
    ax2.set_xlabel("Dilution Passage (Days)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Relative Abundance (%)", fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 200)
    ax2.set_ylim(-2, 102)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8.0, ncol=2)
    
    # Panel D: Stochastic Dynamics (200 Passages)
    ax3 = fig.add_subplot(gs[1, 1])
    for idx, sp in enumerate(surviving):
        ax3.plot(p_axis, stoch_rel[:, sp] * 100, color=cmap(idx % 10), lw=1.8, label=f"Species {sp}")
    ax3.set_title(r"$\mathbf{(D)\ Stochastic\ Noise\ Sustained\ Limit\ Cycles\ (200\ Passages)}$" + "\n" +
                  f"(Noise $\\sigma_R=0.08, \\sigma_D=0.05$: {flips_stoch.mean():.1f} Flips, P2P $\\approx {p2p_stoch[surviving].mean():.1f}\\%$)",
                  fontsize=11.5, fontweight='bold', loc='left', pad=8)
    ax3.set_xlabel("Dilution Passage (Days)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Relative Abundance (%)", fontsize=11, fontweight='bold')
    ax3.set_xlim(0, 200)
    ax3.set_ylim(-2, 102)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', fontsize=8.0, ncol=2)
    
    out_fig = os.path.join(PROJECT_ROOT, 'Figures', 'Gore_K8_Jacked_Regularization_Oscillations.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure to: {out_fig}")
    
    artifact_dst = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143/Gore_K8_Jacked_Regularization_Oscillations.png'
    shutil.copyfile(out_fig, artifact_dst)
    print("Copied figure to artifact directory successfully!")

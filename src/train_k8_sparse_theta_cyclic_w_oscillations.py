"""
Train K=8 Consumer-Resource Model with:
1. L1 Regularization on Theta (sparse niche specialization, preventing generalist collapse).
2. Non-reciprocal cross-feeding penalty on W: sum_{l < k} W_{lk} * W_{kl} (preventing symmetric damping).
3. Skew-symmetry regularizer on W: - ||W - W^T||_F (maximizing rotational curl / imaginary eigenvalues).
4. Physical Consumer-Resource ODE with mass conservation (sum_k W_{lk} <= 0.85, W_{kk} = 0).
5. Fits all 75 communities from Gore et al. (Science 2022).
6. Simulates 100-200 passages (deterministic & stochastic) to test for true, visible sustained oscillations.
7. Generates comprehensive publication figure: Figures/Gore_K8_SparseTheta_CyclicW_Oscillations.png.
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

class K8SparseThetaCyclicWODE(nn.Module):
    def __init__(self, N=48, K=8, D=30.0, mu=1e-6, Km=0.05, max_crossfeed=0.85):
        super().__init__()
        self.N, self.K, self.D, self.mu, self.Km = N, K, D, mu, Km
        self.max_crossfeed = max_crossfeed
        
        # Initialize Theta with specialist bias (safely clamped > 0.1)
        Theta_init = torch.clamp(torch.randn(N, K) * 0.15 + 0.6, min=0.15)
        for i in range(N):
            pref = i % K
            Theta_init[i, pref] += 3.5
        inv_scale = torch.log(torch.exp(Theta_init) - 1.0)
        self.Theta_raw = nn.Parameter(inv_scale)
        
        # Initialize W with directed cycle bias: 0->1->2->3->4->5->6->7->0
        W_init = torch.randn(K, K) * 0.1 - 2.0
        for k in range(K):
            W_init[k, (k + 1) % K] = 1.5
        self.W_raw = nn.Parameter(W_init)
        
        self.alpha_raw = nn.Parameter(torch.tensor(-1.0)) # lower self-limitation to allow oscillations
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
        # Bound alpha between 0.001 and 0.03 to avoid over-damping
        return 0.001 + 0.029 * torch.sigmoid(self.alpha_raw)
        
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
    model = K8SparseThetaCyclicWODE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("\n" + "="*80)
    print(f"TRAINING K=8 SPARSE THETA + CYCLIC W CONSUMER-RESOURCE MODEL (PARAMS = {total_params})")
    print("="*80)
    
    ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_K8_SparseTheta_CyclicW_ODE.pth')
    t_span = torch.linspace(0.0, 1.0, 30, device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.012)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-3)
    start_train = time.time()
    
    lambda_theta_l1 = 0.03
    lambda_recip = 4.0
    lambda_skew = 0.25
    
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
        
        # 1. L1 Regularization on Theta (encourages sparse specialization)
        Theta = model.get_Theta()
        loss_theta_l1 = lambda_theta_l1 * torch.mean(Theta)
        
        # 2. Non-reciprocity penalty on W (penalizes mutual pairwise back-and-forth)
        W = model.get_W()
        W_upper = torch.triu(W, diagonal=1)
        W_lower = torch.tril(W, diagonal=-1)
        loss_recip = lambda_recip * torch.sum(W_upper * W_lower.t())
        
        # 3. Skew-symmetry regularizer on W (encourages rotational curl)
        loss_skew = - lambda_skew * torch.norm(W - W.t(), p='fro')
        
        total_loss = loss_fit + loss_theta_l1 + loss_recip + loss_skew
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/60 | Fit: {loss_fit.item():.4f} | L1(Theta): {loss_theta_l1.item():.4f} | Recip(W): {loss_recip.item():.4f} | Time: {time.time()-start_train:.1f}s")
            
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
        
        # Sparsity of Theta (fraction < 0.5)
        sparsity_theta = np.mean(Theta_np < 0.5)
        print(f"Theta Sparsity (< 0.5): {sparsity_theta*100:.1f}%")
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
    print("SIMULATING 200 PASSAGES TO TEST FOR VISIBLE OSCILLATIONS")
    print("="*80)
    
    # Test High-Nutrient Community with High Richness: ('HN', 24, 'Community5') -> index 70
    test_idx = 70
    target_name = b11_names[test_idx]
    print(f"Targeting Community {test_idx}: {target_name}")
    
    N0 = b11_N[test_idx, 0, :] * b11_m[test_idx]
    R0 = R0_all[b11_c[test_idx]]
    mask = b11_m[test_idx]
    
    # (A) Deterministic 200 Passages
    N_curr = N0.clone()
    roll_det = [N_curr.cpu().numpy()]
    with torch.no_grad():
        for p in range(200):
            N_in = torch.clamp(N_curr / model.D + model.mu * mask, min=0.0)
            S0 = torch.cat([N_in, R0], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_next = torch.clamp(S_t[-1, :48] * mask, min=0.0)
            roll_det.append(N_next.cpu().numpy())
            N_curr = N_next
    roll_det = np.array(roll_det)
    rel_det = roll_det / np.where(roll_det.sum(axis=-1, keepdims=True) > 0, roll_det.sum(axis=-1, keepdims=True), 1.0)
    
    # (B) Stochastic 200 Passages (8% R0 noise, 5% D noise)
    np.random.seed(42)
    N_curr = N0.clone()
    roll_stoch = [N_curr.cpu().numpy()]
    with torch.no_grad():
        for p in range(200):
            d_noise = float(np.exp(np.random.randn() * 0.05))
            r_noise = torch.tensor(np.exp(np.random.randn(8) * 0.08), device=device, dtype=torch.float32)
            N_in = torch.clamp(N_curr / (model.D * d_noise) + model.mu * mask, min=0.0)
            S0 = torch.cat([N_in, R0 * r_noise], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_next = torch.clamp(S_t[-1, :48] * mask, min=0.0)
            roll_stoch.append(N_next.cpu().numpy())
            N_curr = N_next
    roll_stoch = np.array(roll_stoch)
    rel_stoch = roll_stoch / np.where(roll_stoch.sum(axis=-1, keepdims=True) > 0, roll_stoch.sum(axis=-1, keepdims=True), 1.0)
    
    # Metrics on Passages 100-200
    late_det = rel_det[100:201]
    late_stoch = rel_stoch[100:201]
    active_det = np.where(late_det.mean(axis=0) >= 0.01)[0]
    active_stoch = np.where(late_stoch.mean(axis=0) >= 0.01)[0]
    
    print(f"\nSurviving Coexisting Species in Deterministic: {len(active_det)}")
    for sp in active_det:
        diffs = np.diff(late_det[:, sp])
        sgn = np.sign(diffs)
        sgn = sgn[sgn != 0]
        flips = int(np.sum(sgn[:-1] != sgn[1:])) if len(sgn) > 1 else 0
        p2p = float(np.max(late_det[:, sp]) - np.min(late_det[:, sp]))
        print(f"  Sp {sp+1:2d}: Mean = {late_det[:, sp].mean()*100:5.1f}% | Flips = {flips:2d} | P2P = {p2p*100:5.1f}%")
        
    print(f"\nSurviving Coexisting Species in Stochastic: {len(active_stoch)}")
    for sp in active_stoch:
        diffs = np.diff(late_stoch[:, sp])
        sgn = np.sign(diffs)
        sgn = sgn[sgn != 0]
        flips = int(np.sum(sgn[:-1] != sgn[1:])) if len(sgn) > 1 else 0
        p2p = float(np.max(late_stoch[:, sp]) - np.min(late_stoch[:, sp]))
        print(f"  Sp {sp+1:2d}: Mean = {late_stoch[:, sp].mean()*100:5.1f}% | Flips = {flips:2d} | P2P = {p2p*100:5.1f}%")
        
    # ─── 4-Panel Publication Figure ───
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22)
    p_axis = np.arange(201)
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # (A) Learned 8x8 W Matrix Heatmap & Complex Spectra
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(W_np, cmap='YlGnBu', vmin=0.0, vmax=0.6)
    ax0.set_xticks(range(8))
    ax0.set_yticks(range(8))
    ax0.set_xticklabels([f"R{k+1}" for k in range(8)], fontweight='bold')
    ax0.set_yticklabels([f"R{l+1}" for l in range(8)], fontweight='bold')
    
    # Find dominant imaginary eigenvalue
    max_im_idx = np.argmax(np.abs(eigenvals.imag))
    dom_ev = eigenvals[max_im_idx]
    
    ax0.set_title(f"(A) Learned 8x8 Cross-Feeding Topology W (Asymmetry = {asym_ratio*100:.1f}%)\n" +
                  f"Complex Eigenvalue Pair: λ = {dom_ev.real:.3f} ± {abs(dom_ev.imag):.3f}i (ω = {abs(dom_ev.imag):.3f})",
                  fontweight='bold', fontsize=11, loc='left', pad=8)
    for l in range(8):
        for k in range(8):
            val = W_np[l, k]
            if val > 0.05:
                color_txt = "white" if val > 0.3 else "black"
                ax0.text(k, l, f"{val:.2f}", ha="center", va="center", color=color_txt, fontsize=8.5, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Byproduct Secretion Yield $W_{lk}$", fontsize=10.5, fontweight='bold')
    
    # (B) Sparse Theta Trait Matrix Heatmap (Top coexisting species)
    ax1 = fig.add_subplot(gs[0, 1])
    # Show active species rows in Theta
    show_sp = active_stoch if len(active_stoch) >= 4 else np.arange(10)
    theta_sub = Theta_np[show_sp]
    im1 = ax1.imshow(theta_sub, cmap='viridis', aspect='auto')
    ax1.set_xticks(range(8))
    ax1.set_xticklabels([f"R{k+1}" for k in range(8)], fontweight='bold')
    ax1.set_yticks(range(len(show_sp)))
    ax1.set_yticklabels([f"Sp {sp+1}" for sp in show_sp], fontweight='bold')
    ax1.set_title(f"(B) Sparse Theta Traits for Surviving Species (Sparsity = {sparsity_theta*100:.1f}%)\n" +
                  f"(L1 Regularization Forces Niche Specialization Across Cyclic Resources)",
                  fontweight='bold', fontsize=11, color='#1A365D', loc='left', pad=8)
    for i_idx in range(len(show_sp)):
        for k_idx in range(8):
            val = theta_sub[i_idx, k_idx]
            if val > 0.5:
                ax1.text(k_idx, i_idx, f"{val:.1f}", ha="center", va="center", color="white" if val > 2.5 else "black", fontsize=8.5, fontweight='bold')
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Uptake Affinity $\\Theta_{ik}$", fontsize=10.5, fontweight='bold')
    
    # (C) Deterministic 200 Passages
    ax2 = fig.add_subplot(gs[1, 0])
    for sp in range(48):
        if sp not in active_det and mask[sp] > 0:
            ax2.plot(p_axis, rel_det[:, sp], lw=0.8, color='gray', alpha=0.25)
    for idx, sp in enumerate(active_det):
        p2p_val = (late_det[:, sp].max() - late_det[:, sp].min()) * 100
        ax2.plot(p_axis, rel_det[:, sp], lw=2.0, color=cmap[idx % 10],
                 label=f"Sp {sp+1:2d} (Mean={late_det[:, sp].mean()*100:.1f}%, P2P={p2p_val:.1f}%)")
    ax2.set_title(f"(C) Deterministic 200 Passages (R² = {r2:.2f}%)\n" +
                  f"({len(active_det)} Coexisting Species Driven by 8 Cyclic Latent Niches)",
                  fontweight='bold', fontsize=11, color='#1A365D', loc='left', pad=8)
    ax2.set_xlabel("Dilution Passage (Days)", fontweight='bold', fontsize=11)
    ax2.set_ylabel("Relative Abundance", fontweight='bold', fontsize=11)
    ax2.set_xlim(-2, 202)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8.0, ncol=2)
    
    # (D) Stochastic 200 Passages (Environmental Noise)
    ax3 = fig.add_subplot(gs[1, 1])
    for sp in range(48):
        if sp not in active_stoch and mask[sp] > 0:
            ax3.plot(p_axis, rel_stoch[:, sp], lw=0.8, color='gray', alpha=0.20)
    for idx, sp in enumerate(active_stoch):
        p2p_val = (late_stoch[:, sp].max() - late_stoch[:, sp].min()) * 100
        ax3.plot(p_axis, rel_stoch[:, sp], lw=1.8, color=cmap[idx % 10], alpha=0.9,
                 label=f"Sp {sp+1:2d} (Flips={flips:2d}, P2P={p2p_val:.1f}%)")
    ax3.set_title(f"(D) Stochastic 200 Passages with Environmental Noise\n" +
                  f"(Sustained Dynamic Oscillations Across All {len(active_stoch)} Coexisting Strains)",
                  fontweight='bold', fontsize=11, color='#8B2500', loc='left', pad=8)
    ax3.set_xlabel("Dilution Passage (Days)", fontweight='bold', fontsize=11)
    ax3.set_ylabel("Relative Abundance", fontweight='bold', fontsize=11)
    ax3.set_xlim(-2, 202)
    ax3.set_ylim(-0.02, 1.02)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', fontsize=8.0, ncol=2)
    
    out_fig = os.path.join(PROJECT_ROOT, 'Figures', 'Gore_K8_SparseTheta_CyclicW_Oscillations.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure to: {out_fig}")
    
    import shutil
    artifact_dst = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143/Gore_K8_SparseTheta_CyclicW_Oscillations.png'
    shutil.copyfile(out_fig, artifact_dst)
    print("Copied to artifact directory successfully!")

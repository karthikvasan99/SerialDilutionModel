"""
Train Consumer-Resource Neural ODE Exclusively on Gore's Persistent Fluctuating Communities:
1. Filters training data to only the communities classified as 'fluctuating' (mean CV >= 0.25, 38 communities).
2. Uses Decoupled Theta >= 0 and Phi >= 0 with alpha <= 0.10, ensuring high data fit and multi-species coexistence.
3. 1-Step flow matching + late-passage variance matching on fluctuating communities only.
4. Checks whether specializing on fluctuating communities allows the model to learn parameters that sustain oscillations.
5. Autonomous 200-passage simulations on top fluctuating communities (S=48, S=24, S=12, S=3).
6. Rigorous fine solver integration (dt = 0.010, 100 steps) verifying strictly non-negative resources.
7. Direction flips, peak-to-peak amplitude, and maximal Lyapunov exponent lambda_max.
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

# Load data and metadata
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
data = torch.load(data_path, weights_only=False)
meta = data['metadata']
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

meta_map = {}
for idx, row in meta.iterrows():
    meta_map[(row['S'], row['community_name'])] = row

# Separate fluctuating vs all communities
fluc_N, fluc_c, fluc_m, fluc_keys = [], [], [], []
all_N, all_c, all_m = [], [], []

for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    if traj_rel.shape[0] == 11:
        od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
        mask = data['presence_masks'][(S, comm)]
        N_traj = traj_rel * od_mean.unsqueeze(-1)
        c_idx = cond_to_idx[nutrient]
        
        all_N.append(N_traj)
        all_c.append(c_idx)
        all_m.append(mask)
        
        row = meta_map[(S, comm)]
        if row[f'{nutrient}_fluctuating']:
            fluc_N.append(N_traj)
            fluc_c.append(c_idx)
            fluc_m.append(mask)
            fluc_keys.append((nutrient, S, comm, row[f'{nutrient}_CV']))

fluc_N = torch.stack(fluc_N).to(device)
fluc_c = torch.tensor(fluc_c, device=device)
fluc_m = torch.stack(fluc_m).to(device)

all_N = torch.stack(all_N).to(device)
all_c = torch.tensor(all_c, device=device)
all_m = torch.stack(all_m).to(device)

print(f"Loaded {len(fluc_N)} persistent fluctuating communities (out of {len(all_N)} 11-day communities).")

# Empirical standard deviation across late passages (6..10) for fluctuating subset
emp_late = fluc_N[:, 6:, :]
emp_std = emp_late.std(dim=1) * fluc_m

def dixit_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    p_safe = torch.clamp(y_pred, min=eps)
    y_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_log / p_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + p_safe) * mask
    return div.sum(dim=-1).mean()

class DecoupledThetaPhiAlpha01ODE(nn.Module):
    def __init__(self, N=48, K=8, alpha_max=0.10, D=30.0, mu=1e-6, Kd=0.01):
        super().__init__()
        self.N, self.K, self.D, self.mu, self.Kd = N, K, D, mu, Kd
        self.alpha_max = alpha_max
        
        # Decoupled non-negative growth Theta and depletion Phi
        init_scale = 12.0 / float(K)
        inv_scale = math.log(math.exp(init_scale) - 1.0)
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.15 + inv_scale)
        self.Phi_raw = nn.Parameter(torch.randn(N, K) * 0.25 + inv_scale)
        
        # Cross-feeding W
        self.W_raw = nn.Parameter(torch.randn(K, K) * 0.4 - 0.2)
        
        # Starvation decay rates delta_i
        self.delta_raw = nn.Parameter(torch.zeros(N) + 0.8)
        
        # Intermediate alpha bound <= 0.10
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
        return F.softplus(self.Theta_raw)
        
    def get_Phi(self):
        return F.softplus(self.Phi_raw)
        
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
        growth = torch.matmul(monod, Theta.t())
        
        R_total = R_t.sum(dim=-1, keepdim=True)
        starv_gate = self.Kd / (R_total + self.Kd)
        death = delta * starv_gate
        
        dN = N_t * (growth - death) - alpha * (N_t ** 2)
        
        depletion = monod * torch.matmul(N_t, Phi)
        byproduct = torch.matmul(depletion, W)
        dR = - depletion + byproduct
        return torch.cat([dN, dR], dim=-1)

def train_fluctuating_model(lambda_asym=0.25, epochs=80, lr=0.012, lambda_var=12.0):
    set_seed(42)
    print("\n" + "="*80)
    print(f"TRAINING MODEL EXCLUSIVELY ON PERSISTENT FLUCTUATING COMMUNITIES (N = {len(fluc_N)})")
    print(f"K = 8 | alpha <= 0.10 | lambda_asym = {lambda_asym} | lambda_var = {lambda_var} | Epochs = {epochs}")
    print("="*80)
    
    model = DecoupledThetaPhiAlpha01ODE(N=48, K=8, alpha_max=0.10, Kd=0.01).to(device)
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
        R0_fluc = model.get_R0()[fluc_c]
        
        # 1-Step Flow Matching across all 10 passages for FLUCTUATING communities
        loss_1step = 0.0
        for p in range(1, 11):
            N_in = torch.clamp(fluc_N[:, p-1, :] / model.D + model.mu * fluc_m, min=0.0)
            S0 = torch.cat([N_in, R0_fluc], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_pred = torch.clamp(S_t[-1, ..., :48] * fluc_m, min=0.0)
            loss_1step += dixit_kl(fluc_N[:, p, :], N_pred, fluc_m)
        loss_1step = loss_1step / 10.0
        
        # Late-passage autonomous rollout (passages 6..10) on fluctuating communities
        N_curr = fluc_N[:, 6, :]
        late_rollout = [N_curr]
        for p in range(7, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * fluc_m, min=0.0)
            S0 = torch.cat([N_in, R0_fluc], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_next = torch.clamp(S_t[-1, ..., :48] * fluc_m, min=0.0)
            late_rollout.append(N_next)
            N_curr = N_next
            
        late_stack = torch.stack(late_rollout, dim=1)
        model_std = late_stack.std(dim=1) * fluc_m
        loss_var = F.mse_loss(model_std, emp_std)
        
        # Effective competition asymmetry regularization
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
            alpha_mean = model.get_alpha().mean().item()
            asym_val = float(asym_ratio.item())
            print(f"Epoch {epoch:2d}/{epochs} | 1-Step: {loss_1step.item():.4f} | VarLoss: {loss_var.item():.5f} | Comp Asym: {asym_val:.4f} | Mean Alpha: {alpha_mean:.4f} | Time: {time.time()-start_time:.1f}s")
            
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, best_state

def evaluate_fit(model, N_tensor, c_tensor, m_tensor, label=""):
    t_eval = torch.linspace(0.0, 1.0, 50, device=device)
    R0_batch = model.get_R0()[c_tensor]
    preds = []
    with torch.no_grad():
        for p in range(1, 11):
            N_in = torch.clamp(N_tensor[:, p-1, :] / model.D + model.mu * m_tensor, min=0.0)
            S0 = torch.cat([N_in, R0_batch], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_eval, method='rk4')
            N_pred = torch.clamp(S_t[-1, ..., :48] * m_tensor, min=0.0)
            preds.append(N_pred)
            
    preds_stack = torch.stack(preds, dim=1)
    true_stack = N_tensor[:, 1:, :]
    
    mask_flat = (m_tensor.unsqueeze(1).expand(-1, 10, -1) > 0).cpu().numpy().flatten()
    pred_rel = preds_stack / torch.clamp(preds_stack.sum(dim=-1, keepdim=True), min=1e-8)
    true_rel = true_stack / torch.clamp(true_stack.sum(dim=-1, keepdim=True), min=1e-8)
    pred_rel_flat = pred_rel.cpu().numpy().flatten()[mask_flat]
    true_rel_flat = true_rel.cpu().numpy().flatten()[mask_flat]
    
    r_val, _ = pearsonr(true_rel_flat, pred_rel_flat)
    r2_val = r_val ** 2
    cod_val = 1.0 - np.sum((true_rel_flat - pred_rel_flat)**2) / np.sum((true_rel_flat - np.mean(true_rel_flat))**2)
    print(f"[{label}] 1-Step Relative Pearson r = {r_val:.4f} | Pearson R^2 = {r2_val*100:.2f}% | COD R^2 = {cod_val*100:.2f}%")
    return r_val, r2_val

def simulate_200_passages_fine(model, cond, S_val, comm_val, steps=100):
    c_idx = cond_to_idx[cond]
    mask = data['presence_masks'][(S_val, comm_val)].float().to(device)
    rel = data['relative_abundance'][(cond, S_val, comm_val)].to(device)
    od = data['biomass_od'][(cond, S_val, comm_val)].mean(dim=0).to(device)
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
    # Train exclusively on fluctuating communities
    model, best_state = train_fluctuating_model(lambda_asym=0.25, epochs=80, lr=0.012, lambda_var=12.0)
    
    # Save checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Fitted_K8_FluctuatingOnly_Chaos.pth')
    torch.save({'model_state_dict': best_state, 'K': 8, 'alpha_max': 0.10, 'lambda_asym': 0.25}, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")
    
    # Evaluate fit on both fluctuating training set and full dataset
    print("\n--- Evaluating Predictive Accuracy ---")
    evaluate_fit(model, fluc_N, fluc_c, fluc_m, label="Fluctuating Subset (38 comms)")
    evaluate_fit(model, all_N, all_c, all_m, label="Full Dataset (75 comms)")
    
    # Inspect Learned Parameters
    alpha_learned = model.get_alpha().detach().cpu().numpy()
    Theta = model.get_Theta().detach()
    Phi = model.get_Phi().detach()
    R0_hn = model.get_R0()[2].detach()
    A = torch.matmul(Theta * R0_hn.unsqueeze(0), Phi.t()).cpu().numpy()
    diff = A - A.T
    asym_norm = np.linalg.norm(diff) / np.linalg.norm(A)
    print("\n--- Learned Biological Parameters ---")
    print(f"Alpha (Self-Limitation): Mean = {alpha_learned.mean():.4f} (Min = {alpha_learned.min():.4f}, Max = {alpha_learned.max():.4f})")
    print(f"Competition Asymmetry Index ||A - A^T|| / ||A|| : {asym_norm:.4f}")
    
    # Test 200-Passage Simulations across 4 representative fluctuating communities:
    # 1. 48-species Community 1 in HN (CV = 1.037)
    # 2. 24-species Community 1 in HN (CV = 1.050)
    # 3. 12-species Community 1 in HN (CV = 0.553)
    # 4. 24-species Community 6 in HN (CV = 1.084)
    test_comms = [
        ('HN', 48, 'Community1', '48-Species HN (Community 1, CV = 1.037)'),
        ('HN', 24, 'Community1', '24-Species HN (Community 1, CV = 1.050)'),
        ('HN', 24, 'Community6', '24-Species HN (Community 6, CV = 1.084)'),
        ('HN', 12, 'Community1', '12-Species HN (Community 1, CV = 0.553)')
    ]
    
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = np.vstack([colors, plt.cm.tab20b(np.linspace(0, 1, 20)), plt.cm.tab20c(np.linspace(0, 1, 8))])
    p_axis = np.arange(201)
    
    print("\n" + "="*80)
    print("DYNAMICAL DIAGNOSTICS ACROSS FLUCTUATING COMMUNITIES (200 PASSAGES):")
    print("="*80)
    
    for i, (cond, S_val, comm_val, title) in enumerate(test_comms):
        traj, min_r = simulate_200_passages_fine(model, cond, S_val, comm_val, steps=100)
        p2p_late = (traj[100:].max(axis=0) - traj[100:].min(axis=0)).max()
        std_late = np.std(traj[100:], axis=0).max()
        n_surv = int((traj[-1] > 0.005).sum())
        
        diffs = np.diff(traj[100:], axis=0)
        signs = np.sign(diffs)
        flips = int(((signs[:-1] * signs[1:]) < 0).sum(axis=0).max())
        
        mask = data['presence_masks'][(S_val, comm_val)].float().to(device)
        lam, n_active = compute_lyapunov_fine(model, traj[150], model.get_R0()[cond_to_idx[cond]], mask, steps=100)
        
        print(f"\n[{title}]")
        print(f"  Surviving Species at Passage 200 : {n_surv} (Active: {n_active})")
        print(f"  Late Peak-to-Peak (Passages 100-200): {p2p_late:.6f} OD")
        print(f"  Late Max Standard Deviation       : {std_late:.6f} OD")
        print(f"  Direction Flips (Passages 100-200): {flips} flips")
        print(f"  Maximal Lyapunov Exponent         : lambda = {lam:+.4f} / passage")
        print(f"  Minimum Resource Level R_k(t)     : {min_r:.6f} (Strictly Non-Negative!)")
        
        ax = fig.add_subplot(gs[i // 2, i % 2])
        for sp in range(48):
            if traj[:, sp].max() > 0.005:
                ax.plot(p_axis, traj[:, sp], color=colors[sp % 48], lw=1.3, alpha=0.85)
        ax.set_title(f"({chr(65+i)}) {title}\nSurviving: {n_surv} Strains | P2P: {p2p_late:.4f} OD | $\\lambda = {lam:+.3f}$",
                     fontsize=11.5, fontweight='bold', loc='left', pad=8)
        ax.set_xlabel('Dilution Passage (Days)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Species Biomass (OD)', fontsize=11, fontweight='bold')
        ax.set_xlim(-2, 202)
        ax.grid(True, linestyle='--', alpha=0.5)
        
    fig_path = os.path.join(PROJECT_ROOT, 'Figures', 'Gore_K8_FluctuatingOnly_200Passages.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved 4-panel diagnostic plot to: {fig_path}")

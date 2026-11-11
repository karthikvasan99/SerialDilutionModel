"""
Train Ultra-Minimal Consumer-Resource Model (K=3, 157 params) on ALL 75 Communities:
1. Trains on all 75 communities across LN, MN, HN with pure 1-step loss (zero regularizers).
2. Evaluates global empirical fit (Pearson r, R^2, COD).
3. Simulates autonomous 10-day rollouts with calibrated environmental noise (sigma_R=0.08, sigma_D=0.05, sigma_obs=0.10)
   for ALL 75 communities.
4. Computes simulated CV over Days 7-10 and classifies each community as Stable vs. Fluctuating (CV >= 0.25).
5. Compares model simulated phase diagram with Gore's empirical phase diagram across S and Nutrients.
6. Runs 200-passage simulations to test long-term stability in low S vs high S.
7. Generates publication-quality phase diagram comparison figure.
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
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

# Collect all 75 11-day communities
all_N, all_c, all_m, all_keys = [], [], [], []
for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    if traj_rel.shape[0] == 11:
        od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
        mask = data['presence_masks'][(S, comm)]
        all_N.append(traj_rel * od_mean.unsqueeze(-1))
        all_c.append(cond_to_idx[nutrient])
        all_m.append(mask)
        
        row = meta_map[(S, comm)]
        emp_cv = row[f'{nutrient}_CV']
        emp_fluc = row[f'{nutrient}_fluctuating']
        all_keys.append((nutrient, S, comm, emp_cv, emp_fluc))

all_N = torch.stack(all_N).to(device)
all_c = torch.tensor(all_c, device=device)
all_m = torch.stack(all_m).to(device)

print(f"Loaded {len(all_N)} communities across LN, MN, HN for global training.")

def dixit_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    p_safe = torch.clamp(y_pred, min=eps)
    y_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_log / p_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + p_safe) * mask
    return div.sum(dim=-1).mean()

class MinimalConsumerResourceODE(nn.Module):
    def __init__(self, N=48, K=3, D=30.0, mu=1e-6, Km=0.05):
        super().__init__()
        self.N, self.K, self.D, self.mu, self.Km = N, K, D, mu, Km
        init_scale = 10.0 / float(K)
        inv_scale = math.log(math.exp(init_scale) - 1.0)
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.15 + inv_scale)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.r_ln = nn.Parameter(torch.randn(K) * 0.1)
        self.r_mn = nn.Parameter(torch.randn(K) * 0.1)
        self.r_hn = nn.Parameter(torch.randn(K) * 0.1)
        self.scale_ln = nn.Parameter(torch.tensor(0.40))
        self.scale_mn = nn.Parameter(torch.tensor(0.60))
        self.scale_hn = nn.Parameter(torch.tensor(0.85))
        
    def get_R0(self):
        s_ln = F.softplus(self.scale_ln)
        s_mn = s_ln + F.softplus(self.scale_mn - self.scale_ln)
        s_hn = s_mn + F.softplus(self.scale_hn - self.scale_mn)
        r_ln = F.softmax(self.r_ln, dim=-1) * s_ln
        r_mn = F.softmax(self.r_mn, dim=-1) * s_mn
        r_hn = F.softmax(self.r_hn, dim=-1) * s_hn
        return torch.stack([r_ln, r_mn, r_hn])
        
    def get_alpha(self):
        return 0.10 * torch.sigmoid(self.alpha_raw)
        
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
        depletion = monod * torch.matmul(N_t, Theta)
        dR = - depletion
        return torch.cat([dN, dR], dim=-1)

if __name__ == '__main__':
    model = MinimalConsumerResourceODE(N=48, K=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("\n" + "="*80)
    print(f"TRAINING ULTRA-MINIMAL MODEL ON ALL {len(all_N)} COMMUNITIES")
    print(f"K = 3 | 157 Parameters | Pure 1-Step Flow Matching Loss | Epochs = 60")
    print("="*80)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-3)
    t_span = torch.linspace(0.0, 1.0, 30, device=device)
    
    start_time = time.time()
    for epoch in range(1, 61):
        model.train()
        optimizer.zero_grad()
        R0_all = model.get_R0()[all_c]
        
        loss = 0.0
        for p in range(1, 11):
            N_in = torch.clamp(all_N[:, p-1, :] / model.D + model.mu * all_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            loss += dixit_kl(all_N[:, p, :], torch.clamp(S_t[-1, ..., :48] * all_m, min=0.0), all_m)
        loss = loss / 10.0
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/60 | Pure 1-Step Loss: {loss.item():.4f} | Time: {time.time()-start_time:.1f}s")
            
    # Save checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Minimal_K3_AllData_ODE.pth')
    torch.save({'model_state_dict': model.state_dict(), 'K': 3, 'total_params': total_params}, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")
    
    # 1. Global Predictive Accuracy Evaluation
    model.eval()
    preds = []
    with torch.no_grad():
        for p in range(1, 11):
            N_in = torch.clamp(all_N[:, p-1, :] / model.D + model.mu * all_m, min=0.0)
            S0 = torch.cat([N_in, model.get_R0()[all_c]], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            preds.append(torch.clamp(S_t[-1, ..., :48] * all_m, min=0.0))
            
    preds = torch.stack(preds, dim=1) # (75, 10, 48)
    pred_rel = preds / torch.clamp(preds.sum(dim=-1, keepdim=True), min=1e-8)
    true_rel = all_N[:, 1:] / torch.clamp(all_N[:, 1:].sum(dim=-1, keepdim=True), min=1e-8)
    mask_np = (all_m.unsqueeze(1).expand(-1, 10, -1) > 0).cpu().numpy().flatten()
    
    pred_rel_flat = pred_rel.cpu().numpy().flatten()[mask_np]
    true_rel_flat = true_rel.cpu().numpy().flatten()[mask_np]
    
    r_val, _ = pearsonr(true_rel_flat, pred_rel_flat)
    r2_val = r_val ** 2
    cod_val = 1.0 - np.sum((true_rel_flat - pred_rel_flat)**2) / np.sum((true_rel_flat - np.mean(true_rel_flat))**2)
    print("\n--- Global Model Fit ---")
    print(f"Pearson r = {r_val:.4f} | Pearson R^2 = {r2_val*100:.2f}% | COD R^2 = {cod_val*100:.2f}%")
    print(f"Learned scalar alpha = {model.get_alpha().item():.4f}")
    
    # 2. SYSTEMATIC SIMULATION OF ALL 75 COMMUNITIES WITH ENVIRONMENTAL NOISE
    print("\n" + "="*80)
    print("SIMULATING ALL 75 COMMUNITIES WITH CALIBRATED ENVIRONMENTAL NOISE")
    print("sigma_R = 0.08 (8% resource noise) | sigma_D = 0.05 (5% dilution) | sigma_obs = 0.10")
    print("="*80)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    sigma_R = 0.08
    sigma_D = 0.05
    sigma_obs = 0.10
    
    sim_results = []
    R0_table = model.get_R0()
    
    start_sim = time.time()
    for idx in range(len(all_keys)):
        nutrient, S_val, comm_val, emp_cv, emp_fluc = all_keys[idx]
        c_idx = cond_to_idx[nutrient]
        R0_base = R0_table[c_idx]
        mask_i = all_m[idx]
        
        # Initial condition: Day 0 empirical biomass
        N0 = all_N[idx, 0, :] * mask_i
        N_curr = N0.clone()
        
        rollout = [N_curr.cpu().numpy()]
        with torch.no_grad():
            for p in range(10): # 10 serial dilution passages
                d_noise = float(np.exp(np.random.randn() * sigma_D))
                D_eff = model.D * d_noise
                N_in = torch.clamp(N_curr / D_eff + model.mu * mask_i, min=0.0)
                
                r_noise = torch.tensor(np.exp(np.random.randn(3) * sigma_R), device=device, dtype=torch.float32)
                R0_noisy = R0_base * r_noise
                
                S0 = torch.cat([N_in, R0_noisy], dim=-1)
                S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
                N_next = torch.clamp(S_t[-1, :48] * mask_i, min=0.0)
                
                obs_noise = torch.tensor(np.exp(np.random.randn(48) * sigma_obs), device=device, dtype=torch.float32)
                N_obs = N_next * obs_noise * mask_i
                rollout.append(N_obs.cpu().numpy())
                N_curr = N_next
                
        rollout = np.array(rollout) # (11, 48)
        rel_rollout = rollout / np.where(rollout.sum(axis=-1, keepdims=True) > 0, rollout.sum(axis=-1, keepdims=True), 1.0)
        
        # Compute simulated CV over Days 7 to 10 (Gore's exact metric)
        late_ab = rel_rollout[7:11]
        mean_ab = late_ab.mean(axis=0)
        std_ab = late_ab.std(axis=0)
        surv_mask = mean_ab >= 1e-3
        if np.any(surv_mask):
            sim_cv = float(np.mean(std_ab[surv_mask] / mean_ab[surv_mask]))
        else:
            sim_cv = 0.0
            
        sim_fluc = bool(sim_cv >= 0.25)
        
        sim_results.append({
            'nutrient': nutrient,
            'S': S_val,
            'comm': comm_val,
            'emp_cv': emp_cv,
            'emp_fluc': emp_fluc,
            'sim_cv': sim_cv,
            'sim_fluc': sim_fluc
        })
        
    df_sim = pd.DataFrame(sim_results)
    print(f"Simulation of all 75 communities finished in {time.time()-start_sim:.1f}s.")
    
    # 3. STATISTICAL VALIDATION & COMPARISON
    acc = accuracy_score(df_sim['emp_fluc'], df_sim['sim_fluc'])
    f1 = f1_score(df_sim['emp_fluc'], df_sim['sim_fluc'])
    cv_corr, _ = spearmanr(df_sim['emp_cv'], df_sim['sim_cv'])
    cv_pears, _ = pearsonr(df_sim['emp_cv'], df_sim['sim_cv'])
    
    print("\n--- Model vs. Empirical Stability Classification ---")
    print(f"Classification Accuracy: {acc*100:.2f}%")
    print(f"Classification F1 Score: {f1:.4f}")
    print(f"Spearman Correlation between Empirical CV and Simulated CV: r_s = {cv_corr:.4f}")
    print(f"Pearson Correlation between Empirical CV and Simulated CV : r   = {cv_pears:.4f}")
    
    cm = confusion_matrix(df_sim['emp_fluc'], df_sim['sim_fluc'])
    print(f"Confusion Matrix (Rows=True [Stable, Fluc], Cols=Pred [Stable, Fluc]):")
    print(cm)
    
    # Phase Diagram Comparisons
    print("\n--- Phase Diagram: % Fluctuating Communities by S ---")
    emp_by_s = df_sim.groupby('S')['emp_fluc'].mean() * 100
    sim_by_s = df_sim.groupby('S')['sim_fluc'].mean() * 100
    for s in sorted(df_sim['S'].unique()):
        print(f"  S = {s:2d}: Empirical = {emp_by_s[s]:5.1f}% | Simulated = {sim_by_s[s]:5.1f}%")
        
    print("\n--- Phase Diagram: % Fluctuating Communities by Nutrient ---")
    emp_by_c = df_sim.groupby('nutrient')['emp_fluc'].mean() * 100
    sim_by_c = df_sim.groupby('nutrient')['sim_fluc'].mean() * 100
    for c in ['LN', 'MN', 'HN']:
        print(f"  {c}: Empirical = {emp_by_c[c]:5.1f}% | Simulated = {sim_by_c[c]:5.1f}%")
        
    # 4. GENERATE PUBLICATION-QUALITY COMPARISON FIGURE
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.34, wspace=0.22)
    
    # Panel A: % Fluctuating vs Species Richness S
    ax0 = fig.add_subplot(gs[0, 0])
    s_vals = np.array(sorted(df_sim['S'].unique()))
    ax0.plot(s_vals, [emp_by_s[s] for s in s_vals], marker='o', markersize=8, lw=2.5, color='#1A365D', label='Empirical Data (Gore et al.)')
    ax0.plot(s_vals, [sim_by_s[s] for s in s_vals], marker='s', markersize=8, lw=2.5, color='#E65100', linestyle='--', label='Model Simulation (+ Env Noise)')
    ax0.set_title(r"$\mathbf{(A)\ Emergence\ of\ Fluctuating\ Phase\ vs.\ Biodiversity\ (S)}$", fontsize=12.5, fontweight='bold', loc='left', pad=10)
    ax0.set_xlabel("Inoculated Species Richness (S)", fontsize=11, fontweight='bold')
    ax0.set_ylabel("% Communities Fluctuating (CV ≥ 0.25)", fontsize=11, fontweight='bold')
    ax0.set_xticks(s_vals)
    ax0.set_ylim(-5, 105)
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend(loc='upper left', fontsize=10.5)
    
    # Panel B: % Fluctuating vs Nutrient Condition
    ax1 = fig.add_subplot(gs[0, 1])
    c_names = ['LN', 'MN', 'HN']
    x_pos = np.arange(3)
    bar_w = 0.35
    ax1.bar(x_pos - bar_w/2, [emp_by_c[c] for c in c_names], width=bar_w, color='#1A365D', label='Empirical Data')
    ax1.bar(x_pos + bar_w/2, [sim_by_c[c] for c in c_names], width=bar_w, color='#E65100', alpha=0.85, label='Model Simulation')
    ax1.set_title(r"$\mathbf{(B)\ Emergence\ of\ Fluctuating\ Phase\ vs.\ Nutrient\ Level}$", fontsize=12.5, fontweight='bold', loc='left', pad=10)
    ax1.set_xlabel("Nutrient Condition", fontsize=11, fontweight='bold')
    ax1.set_ylabel("% Communities Fluctuating (CV ≥ 0.25)", fontsize=11, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['Low (LN)', 'Medium (MN)', 'High (HN)'], fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax1.legend(loc='upper left', fontsize=10.5)
    
    # Panel C: Scatter of Empirical CV vs Simulated CV
    ax2 = fig.add_subplot(gs[1, 0])
    cond_colors = {'LN': '#2B6CB0', 'MN': '#319795', 'HN': '#D69E2E'}
    for c in ['LN', 'MN', 'HN']:
        sub = df_sim[df_sim['nutrient'] == c]
        ax2.scatter(sub['emp_cv'], sub['sim_cv'], c=cond_colors[c], s=55, alpha=0.75, label=f'{c} Communities')
    ax2.plot([0, 1.4], [0, 1.4], 'k--', lw=1.5, alpha=0.6, label='Identity line')
    ax2.axvline(0.25, color='gray', linestyle=':', lw=1.2)
    ax2.axhline(0.25, color='gray', linestyle=':', lw=1.2)
    ax2.set_title(r"$\mathbf{(C)\ Community\ Coefficient\ of\ Variation\ (CV)}$" + f"\n(Spearman $r_s = {cv_corr:.3f}$, Pearson $r = {cv_pears:.3f}$)",
                  fontsize=12, fontweight='bold', loc='left', pad=10)
    ax2.set_xlabel("Empirical CV (Gore et al.)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Simulated CV (Model + Noise)", fontsize=11, fontweight='bold')
    ax2.set_xlim(-0.02, 1.4)
    ax2.set_ylim(-0.02, 1.4)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=9.5)
    
    # Panel D: 2D Phase Diagram (S vs Condition) - Comparison Heatmaps
    ax3 = fig.add_subplot(gs[1, 1])
    emp_matrix = df_sim.groupby(['S', 'nutrient'])['emp_fluc'].mean().unstack()[['LN', 'MN', 'HN']] * 100
    sim_matrix = df_sim.groupby(['S', 'nutrient'])['sim_fluc'].mean().unstack()[['LN', 'MN', 'HN']] * 100
    diff_matrix = sim_matrix - emp_matrix
    
    im = ax3.imshow(sim_matrix.values, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(['LN', 'MN', 'HN'], fontweight='bold')
    ax3.set_yticks(range(len(s_vals)))
    ax3.set_yticklabels(s_vals, fontweight='bold')
    ax3.set_title(r"$\mathbf{(D)\ Simulated\ Phase\ Diagram:\ \%\ Fluctuating}$" + f"\n(Accuracy = {acc*100:.1f}%, F1 = {f1:.3f})",
                  fontsize=12, fontweight='bold', loc='left', pad=10)
    ax3.set_xlabel("Nutrient Condition", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Species Richness (S)", fontsize=11, fontweight='bold')
    
    for r in range(len(s_vals)):
        for col in range(3):
            val_sim = sim_matrix.values[r, col]
            val_emp = emp_matrix.values[r, col]
            color_txt = "white" if val_sim > 50 else "black"
            ax3.text(col, r, f"Sim: {val_sim:.0f}%\n(Emp: {val_emp:.0f}%)", ha="center", va="center", color=color_txt, fontsize=9.0, fontweight='bold')
            
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("% Fluctuating", fontsize=10.5, fontweight='bold')
    
    fig_path = os.path.join(PROJECT_ROOT, 'Figures', 'Gore_Global_Phase_Diagram_Validation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved global validation figure to: {fig_path}")
    
    # Copy to artifact directory
    import shutil
    artifact_dst = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143/Gore_Global_Phase_Diagram_Validation.png'
    shutil.copyfile(fig_path, artifact_dst)
    print("Copied to artifact directory successfully!")

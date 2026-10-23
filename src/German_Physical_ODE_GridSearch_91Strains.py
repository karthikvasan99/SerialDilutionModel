#!/usr/bin/env python3
"""
Fine-Grained Hyperparameter Grid Search for the Physical Consumer-Resource Neural ODE
on the German et al. Rumen Enrichment Microbiome Dataset (N = 91 Strains, Peak >= 1%).

Searches over:
  - Latent Resource Niches K in increments of 1: K in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  - Learning rates: [0.01, 0.02, 0.03]
  - L2 Regularization penalties on Theta and W: [0.0, 1e-4, 1e-3]
  
Evaluates:
  - Global Trajectory R^2
  - Pearson Correlation r
  - Simplex KL Divergence Loss
  - Mean Squared Error (MSE)
  - Information Criterion (AIC & BIC) to penalize overparameterization
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'Figures')
CSV_OUT = os.path.join(RESULTS_DIR, 'German_GridSearch_91Strains_Results.csv')
BEST_CKPT_OUT = os.path.join(PROJECT_ROOT, 'Data', 'German_Physical_ODE_91Strains_Best.pth')
PLOT_OUT = os.path.join(FIGURES_DIR, 'German_GridSearch_K_Sweep_91Strains.png')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing Grid Search on Device: {device}")

# ─── 1. Load and Filter Data (Peak >= 1%) ───
raw_data = torch.load(DATA_PATH, weights_only=False)
raw_strains = raw_data['strains']
trajectories_dict = raw_data['trajectories']

# Stack raw trajectories: 12 conditions, 6 transfers, 751 strains
raw_traj = torch.stack(list(trajectories_dict.values()), dim=0) # (12, 6, 751)
peaks = raw_traj.amax(dim=(0, 1))
mask_1pct = peaks >= 0.01

strains_91 = [raw_strains[i] for i in range(len(raw_strains)) if mask_1pct[i]]
data_91 = raw_traj[:, :, mask_1pct].to(device)
# Simplex re-normalization
data_91 = data_91 / (data_91.sum(dim=-1, keepdim=True) + 1e-12)

num_conditions, num_passages, num_species = data_91.shape
methano_mask = [any(m in s.lower() for m in ['methan', 'archaea']) for s in strains_91]

print(f"Dataset prepared: {num_species} strains ({sum(methano_mask)} methanogens, {num_species - sum(methano_mask)} bacteria)")
print(f"Conditions: {num_conditions}, Passages: {num_passages}")

# Condition mapping (6 unique media/pressure conditions across 2 hosts)
# BRRS: conditions 0-5; OSU: conditions 6-11
cond_ids = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=torch.long, device=device)
num_unique_envs = 6

# Host masks
brrs_mask = (data_91[:6].sum(dim=(0, 1)) > 0).float()
osu_mask = (data_91[6:].sum(dim=(0, 1)) > 0).float()
host_masks = torch.cat([brrs_mask.unsqueeze(0).repeat(6, 1), osu_mask.unsqueeze(0).repeat(6, 1)], dim=0).to(device)

# Seed T1 with sequencing detection limit floor for species in host pool
x0 = data_91[:, 0, :]
seed_t1 = torch.where(x0 > 0, x0, torch.tensor(1e-4, device=device)) * host_masks
seed_t1 = seed_t1 / seed_t1.sum(dim=-1, keepdim=True)


# ─── 2. Model Architecture ───
class GermanPurePhysicalODE(nn.Module):
    def __init__(self, N, K, num_conditions=6):
        super().__init__()
        self.N = N
        self.K = K
        self.num_conditions = num_conditions
        self.r0_raw = nn.Parameter(torch.randn(num_conditions, K) * 0.1)
        self.theta_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.05 - 1.0)
        self.b0_raw = nn.Parameter(torch.tensor(0.0))

    def get_params(self):
        Theta = F.softplus(self.theta_raw)
        W = F.softplus(self.w_raw)
        alpha = F.softplus(self.alpha_raw)
        R0 = F.softplus(self.r0_raw)
        B0 = F.softplus(self.b0_raw) + 0.1
        return Theta, W, alpha, R0, B0

    def ode_deriv(self, t, S, Theta, W, alpha):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        growth = torch.matmul(R_t, Theta.t())
        dN_dt = N_t * growth - alpha * (N_t ** 2)
        flux = torch.matmul(torch.matmul(N_t, Theta), W.t())
        dR_dt = - R_t * flux
        return torch.cat([dN_dt, dR_dt], dim=-1)


# ─── 3. Grid Search Space ───
K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # In increments of 1
LR_VALUES = [0.01, 0.02, 0.03]
L2_VALUES = [0.0, 1e-4, 1e-3]
EPOCHS = 300
T_SPAN = torch.linspace(0.0, 1.0, 11, device=device)

total_combinations = len(K_VALUES) * len(LR_VALUES) * len(L2_VALUES)
print(f"Total Grid Combinations: {total_combinations} (K: {len(K_VALUES)}, LR: {len(LR_VALUES)}, L2: {len(L2_VALUES)})")
print(f"Targeting ~35-45 minutes total wall-clock runtime.\n")

results = []
best_global_r2 = -float('inf')
best_checkpoint_data = None

start_time_all = time.time()
combo_count = 0

for K in K_VALUES:
    for lr in LR_VALUES:
        for l2 in L2_VALUES:
            combo_count += 1
            t0 = time.time()
            
            torch.manual_seed(42)
            np.random.seed(42)
            
            model = GermanPurePhysicalODE(N=num_species, K=K, num_conditions=num_unique_envs).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            best_epoch_loss = float('inf')
            best_model_weights = None
            
            for epoch in range(1, EPOCHS + 1):
                optimizer.zero_grad()
                Theta, W, alpha, R0_table, B0 = model.get_params()
                N_curr = seed_t1 * B0
                R0 = R0_table[cond_ids]
                
                loss = 0.0
                preds = [seed_t1]
                
                for p in range(1, num_passages):
                    N0 = torch.clamp((N_curr / 100.0) * host_masks, min=0.0)
                    S0 = torch.cat([N0, R0], dim=-1)
                    
                    def f(t, S):
                        return model.ode_deriv(t, S, Theta, W, alpha)
                        
                    S_t = odeint(f, S0, T_SPAN, method='euler', options={'step_size': 0.1})
                    N_final = torch.clamp(S_t[-1, :, :num_species] * host_masks, min=0.0)
                    x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
                    preds.append(x_pred)
                    
                    target = data_91[:, p, :]
                    kl = (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean()
                    loss = loss + kl
                    N_curr = N_final
                    
                if l2 > 0:
                    l2_pen = l2 * (torch.sum(Theta ** 2) + torch.sum(W ** 2))
                    loss = loss + l2_pen
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                
                if loss.item() < best_epoch_loss:
                    best_epoch_loss = loss.item()
                    best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    
            dt = time.time() - t0
            
            # Evaluate best model weights on full rollout
            model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})
            model.eval()
            
            with torch.no_grad():
                Theta, W, alpha, R0_table, B0 = model.get_params()
                N_curr = seed_t1 * B0
                R0 = R0_table[cond_ids]
                eval_preds = [seed_t1]
                eval_loss = 0.0
                
                for p in range(1, num_passages):
                    N0 = torch.clamp((N_curr / 100.0) * host_masks, min=0.0)
                    S0 = torch.cat([N0, R0], dim=-1)
                    def f(t, S):
                        return model.ode_deriv(t, S, Theta, W, alpha)
                    S_t = odeint(f, S0, T_SPAN, method='euler', options={'step_size': 0.1})
                    N_final = torch.clamp(S_t[-1, :, :num_species] * host_masks, min=0.0)
                    x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
                    eval_preds.append(x_pred)
                    target = data_91[:, p, :]
                    eval_loss += (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean().item()
                    N_curr = N_final
                    
                pred_tensor = torch.stack(eval_preds, dim=1) # (12, 6, N)
                pred_np = pred_tensor.cpu().numpy()
                true_np = data_91.cpu().numpy()
                
                ss_res = np.sum((true_np - pred_np) ** 2)
                ss_tot = np.sum((true_np - np.mean(true_np)) ** 2)
                r2 = float(1.0 - ss_res / ss_tot)
                r = float(np.corrcoef(true_np.flatten(), pred_np.flatten())[0, 1])
                mse = float(np.mean((true_np - pred_np) ** 2))
                
                # Model complexity & Information Criteria
                # Parameters: Theta (N*K), W (K*K), alpha (N), R0 (6*K), B0 (1)
                num_params = num_species * K + K * K + num_species + num_unique_envs * K + 1
                n_data_points = 12 * 6 * num_species
                aic = float(2 * num_params + n_data_points * np.log(max(mse, 1e-12)))
                bic = float(num_params * np.log(n_data_points) + n_data_points * np.log(max(mse, 1e-12)))

            row = {
                'K': K,
                'lr': lr,
                'l2': l2,
                'epochs': EPOCHS,
                'kl_loss': eval_loss,
                'r2': r2,
                'pearson_r': r,
                'mse': mse,
                'num_params': num_params,
                'aic': aic,
                'bic': bic,
                'runtime_s': dt
            }
            results.append(row)
            
            # Progress saving to CSV
            pd.DataFrame(results).to_csv(CSV_OUT, index=False)
            
            elapsed = time.time() - start_time_all
            avg_per_combo = elapsed / combo_count
            eta_min = (total_combinations - combo_count) * avg_per_combo / 60.0
            
            print(f"[{combo_count:2d}/{total_combinations}] K={K:2d} | lr={lr:5.3f} | l2={l2:6.4f} => "
                  f"R2: {r2:.4f} | r: {r:.4f} | KL: {eval_loss:.4f} | Time: {dt:.1f}s | ETA: {eta_min:.1f} min")
                  
            if r2 > best_global_r2:
                best_global_r2 = r2
                best_checkpoint_data = {
                    'model_state_dict': best_model_weights,
                    'strains': strains_91,
                    'num_species': num_species,
                    'K': K,
                    'lr': lr,
                    'l2': l2,
                    'r2': r2,
                    'pearson_r': r,
                    'kl_loss': eval_loss,
                    'cond_ids': cond_ids.cpu(),
                    'num_unique_envs': num_unique_envs
                }
                torch.save(best_checkpoint_data, BEST_CKPT_OUT)
                print(f"  ==> New Champion Saved to {BEST_CKPT_OUT} (K={K}, R2={r2:.4f}, r={r:.4f})")

# ─── 4. Summary & Plotting ───
df_res = pd.DataFrame(results)
print("\n" + "="*80)
print(f"GRID SEARCH COMPLETE! Total Runtime: {(time.time() - start_time_all)/60.0:.2f} minutes")
print("="*80)

print("\nTop 10 Configurations Sorted by R^2:")
top10 = df_res.sort_values(by='r2', ascending=False).head(10)
print(top10[['K', 'lr', 'l2', 'r2', 'pearson_r', 'kl_loss', 'bic']].to_string(index=False))

# Group by K to see the marginal effect of K
k_summary = df_res.groupby('K').agg({
    'r2': ['mean', 'max'],
    'pearson_r': ['mean', 'max'],
    'kl_loss': ['mean', 'min'],
    'bic': ['mean', 'min']
})
print("\nMarginal Summary by Latent Niches K (in increments of 1):")
print(k_summary)

# Plotting K-Sweep
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. R^2 vs K
for l2_val in L2_VALUES:
    sub = df_res[df_res['l2'] == l2_val]
    for lr_val in LR_VALUES:
        sub_lr = sub[sub['lr'] == lr_val].sort_values('K')
        axes[0, 0].plot(sub_lr['K'], sub_lr['r2'], marker='o', alpha=0.6,
                        label=f"lr={lr_val}, l2={l2_val}" if l2_val == 0.0 else None)
max_r2_by_k = df_res.groupby('K')['r2'].max()
axes[0, 0].plot(max_r2_by_k.index, max_r2_by_k.values, color='red', linewidth=3, marker='s', label='Best R^2 per K')
axes[0, 0].set_title('Predictive $R^2$ vs. Latent Niches ($K$)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Number of Latents ($K$)', fontsize=11)
axes[0, 0].set_ylabel('$R^2$', fontsize=11)
axes[0, 0].grid(True, linestyle='--', alpha=0.5)
axes[0, 0].legend(fontsize=8)

# 2. Pearson r vs K
max_r_by_k = df_res.groupby('K')['pearson_r'].max()
mean_r_by_k = df_res.groupby('K')['pearson_r'].mean()
axes[0, 1].plot(max_r_by_k.index, max_r_by_k.values, color='darkgreen', linewidth=2.5, marker='^', label='Max Pearson r')
axes[0, 1].plot(mean_r_by_k.index, mean_r_by_k.values, color='teal', linestyle='--', marker='o', label='Mean Pearson r')
axes[0, 1].set_title('Pearson Correlation ($r$) vs. Latents ($K$)', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Number of Latents ($K$)', fontsize=11)
axes[0, 1].set_ylabel('Pearson $r$', fontsize=11)
axes[0, 1].grid(True, linestyle='--', alpha=0.5)
axes[0, 1].legend(fontsize=10)

# 3. KL Loss vs K
min_kl_by_k = df_res.groupby('K')['kl_loss'].min()
axes[1, 0].plot(min_kl_by_k.index, min_kl_by_k.values, color='purple', linewidth=2.5, marker='d', label='Min KL Loss')
axes[1, 0].set_title('Minimum KL Divergence vs. Latents ($K$)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Number of Latents ($K$)', fontsize=11)
axes[1, 0].set_ylabel('Validation KL Loss', fontsize=11)
axes[1, 0].grid(True, linestyle='--', alpha=0.5)
axes[1, 0].legend(fontsize=10)

# 4. BIC vs K (Model complexity tradeoff)
min_bic_by_k = df_res.groupby('K')['bic'].min()
axes[1, 1].plot(min_bic_by_k.index, min_bic_by_k.values, color='darkorange', linewidth=2.5, marker='p', label='Min BIC')
axes[1, 1].set_title('Bayesian Information Criterion (BIC) vs. $K$', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Number of Latents ($K$)', fontsize=11)
axes[1, 1].set_ylabel('BIC (Lower is Better)', fontsize=11)
axes[1, 1].grid(True, linestyle='--', alpha=0.5)
axes[1, 1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=300)
plt.close()
print(f"Diagnostic plot saved to: {PLOT_OUT}")

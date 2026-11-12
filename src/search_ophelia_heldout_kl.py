"""
Systematic 5-Fold Cross-Validation for Ophelia (Venturelli et al.) Dataset
Strictly evaluated within Data/train_data.mat (102 communities).
Test data (Data/test_data.mat) is strictly locked and untouched.

Primary Selection Metric: Discrete Simplex Held-Out KL Divergence (nats).
Also tracks Generalized KL (GKL), Absolute/Relative R2, MAE, and Pearson r.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.io import loadmat
from torchdiffeq import odeint
from scipy.stats import pearsonr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running Ophelia 5-Fold CV on: {device}")

# 1. Load Training Data Only (102 communities)
train_mat = loadmat('Data/train_data.mat')['train_data']
train_valid = [np.nan_to_num(train_mat[i], nan=0.0) for i in range(train_mat.shape[0]) 
              if not np.isnan(train_mat[i][0]).any() and np.sum(train_mat[i][0]) > 1e-4]
train_tensor = torch.tensor(np.array(train_valid), dtype=torch.float32, device=device) # (102, 6, 12)
N_comms, N_timepoints, S = train_tensor.shape
print(f"Loaded {N_comms} training communities (S={S} species, T={N_timepoints} timepoints).")

# 2. Partition into 5 Deterministic Folds
N_FOLDS = 5
np.random.seed(42)
shuffled_indices = np.random.permutation(N_comms)
folds = np.array_split(shuffled_indices, N_FOLDS)

print(f"5-Fold Partition sizes: {[len(f) for f in folds]}")

# 3. Model Definition
class OpheliaByproductODE(nn.Module):
    def __init__(self, K=5, max_crossfeed=0.40):
        super().__init__()
        self.N, self.K = 12, K
        self.max_crossfeed = max_crossfeed
        self.a = nn.Parameter(torch.randn(1, K) * 0.2 + 1.5)
        self.theta_raw = nn.Parameter(torch.randn(12, K) * 0.1 + 0.5)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1 - 2.0)
        self.alpha_raw = nn.Parameter(torch.randn(12) * 0.05 - 0.5)
        self.log_c = nn.Parameter(torch.tensor(2.0))
        
    def get_params(self):
        Theta = F.softplus(self.theta_raw)
        alpha = F.softplus(self.alpha_raw)
        R0 = F.softplus(self.a)
        c = torch.exp(self.log_c)
        
        W_sp = F.softplus(self.w_raw)
        W_sp = W_sp * (1.0 - torch.eye(self.K, device=W_sp.device))
        row_sums = W_sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_crossfeed / torch.clamp(row_sums, min=1e-6), max=1.0)
        W = W_sp * scale
        return Theta, W, alpha, R0, c

    def ode_deriv(self, t, S, Theta, W, alpha, c):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        growth = torch.matmul(R_t, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        flux = c * R_t * torch.matmul(N_t, Theta)
        byproduct = torch.matmul(flux, W)
        dR = - flux + byproduct
        return torch.cat([dN, dR], dim=-1)

# 4. Evaluation Function
def evaluate_fold(model, val_data, t_span):
    model.eval()
    with torch.no_grad():
        Theta, W, alpha, R0_table, c = model.get_params()
        N0_val = val_data[:, 0, :]
        R0_val = R0_table[0].unsqueeze(0).expand(val_data.size(0), -1)
        S0_val = torch.cat([N0_val, R0_val], dim=-1)
        
        S_t_val = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha, c), 
                         S0_val, t_span, method='euler', options={'step_size': 0.1})
        val_pred = torch.clamp(S_t_val[..., :12].permute(1, 0, 2), min=0.0).cpu().numpy()
        val_true = val_data.cpu().numpy()
        
        # Absolute metrics across dynamic timepoints t=1..5 (12h to 60h)
        vt_flat = val_true[:, 1:, :].flatten()
        vp_flat = val_pred[:, 1:, :].flatten()
        ss_tot = np.sum((vt_flat - np.mean(vt_flat))**2)
        r2_abs = 1.0 - np.sum((vt_flat - vp_flat)**2) / ss_tot if ss_tot > 0 else 1.0
        r_abs, _ = pearsonr(vt_flat, vp_flat) if np.std(vp_flat) > 0 else (0.0, 1.0)
        mae_abs = np.mean(np.abs(vt_flat - vp_flat))
        
        # Relative abundances across dynamic timepoints t=1..5
        tot_true = val_true.sum(axis=-1, keepdims=True) + 1e-8
        tot_pred = val_pred.sum(axis=-1, keepdims=True) + 1e-8
        rel_true = val_true / tot_true
        rel_pred = val_pred / tot_pred
        
        rt_flat = rel_true[:, 1:, :].flatten()
        rp_flat = rel_pred[:, 1:, :].flatten()
        ss_tot_rel = np.sum((rt_flat - np.mean(rt_flat))**2)
        r2_rel = 1.0 - np.sum((rt_flat - rp_flat)**2) / ss_tot_rel if ss_tot_rel > 0 else 1.0
        r_rel, _ = pearsonr(rt_flat, rp_flat) if np.std(rp_flat) > 0 else (0.0, 1.0)
        mae_rel = np.mean(np.abs(rt_flat - rp_flat))
        
        # Discrete Simplex KL divergence: D_KL(P || Q) = sum_i P_i log(P_i / Q_i)
        q_safe = np.clip(rel_pred[:, 1:, :], 1e-6, 1.0)
        p_safe = rel_true[:, 1:, :]
        kl_matrix = np.where(p_safe > 1e-8, p_safe * np.log(np.clip(p_safe, 1e-8, 1.0) / q_safe), 0.0)
        mean_kl_simplex = float(np.mean(np.sum(kl_matrix, axis=-1)))
        
        # Generalized KL divergence on absolute OD:
        eps = 1e-6
        y = val_true[:, 1:, :]
        y_hat = val_pred[:, 1:, :]
        gkl_matrix = y * np.log((y + eps) / (y_hat + eps)) - y + y_hat
        mean_gkl = float(np.mean(np.sum(gkl_matrix, axis=-1)))
        
        return {
            'kl_simplex': mean_kl_simplex,
            'gkl': mean_gkl,
            'r2_abs': float(r2_abs),
            'pearson_abs': float(r_abs),
            'mae_abs': float(mae_abs),
            'r2_rel': float(r2_rel),
            'pearson_rel': float(r_rel),
            'mae_rel': float(mae_rel)
        }

# 5. Grid Specifications
K_LIST = [2, 3, 4, 5, 6, 7, 8]
CW_LIST = [0.20, 0.40, 0.60, 0.80]
LR = 0.015
EPOCHS = 250
t_span = torch.linspace(0.0, 5.0, 6, device=device)

results = []
total_configs = len(K_LIST) * len(CW_LIST)
print(f"Starting 5-Fold Cross-Validation: {total_configs} configurations x 5 folds = {total_configs * N_FOLDS} models.")

config_idx = 0
start_time_all = time.time()

for K in K_LIST:
    for cw in CW_LIST:
        config_idx += 1
        t0_cfg = time.time()
        fold_metrics = []
        
        for fold_idx in range(N_FOLDS):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(N_FOLDS) if i != fold_idx])
            
            fold_train = train_tensor[train_idx]
            fold_val = train_tensor[val_idx]
            
            # Seed each fold deterministically
            torch.manual_seed(42 + fold_idx * 10)
            np.random.seed(42 + fold_idx * 10)
            
            model = OpheliaByproductODE(K=K, max_crossfeed=cw).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            
            best_train_loss = float('inf')
            best_weights = None
            
            for ep in range(1, EPOCHS + 1):
                opt.zero_grad()
                Theta, W, alpha, R0_table, c = model.get_params()
                N0 = fold_train[:, 0, :]
                R0 = R0_table[0].unsqueeze(0).expand(fold_train.size(0), -1)
                S0 = torch.cat([N0, R0], dim=-1)
                
                S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha, c), 
                             S0, t_span, method='euler', options={'step_size': 0.1})
                N_pred = torch.clamp(S_t[..., :12].permute(1, 0, 2), min=0.0)
                
                loss = F.mse_loss(N_pred, fold_train)
                loss.backward()
                opt.step()
                
                if loss.item() < best_train_loss:
                    best_train_loss = loss.item()
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Load best weights
            model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
            
            # Evaluate on held-out fold
            m = evaluate_fold(model, fold_val, t_span)
            fold_metrics.append(m)
            
        elapsed_cfg = time.time() - t0_cfg
        
        # Aggregate across folds
        kl_sim_list = [m['kl_simplex'] for m in fold_metrics]
        gkl_list = [m['gkl'] for m in fold_metrics]
        r2_abs_list = [m['r2_abs'] for m in fold_metrics]
        r_abs_list = [m['pearson_abs'] for m in fold_metrics]
        mae_abs_list = [m['mae_abs'] for m in fold_metrics]
        r2_rel_list = [m['r2_rel'] for m in fold_metrics]
        r_rel_list = [m['pearson_rel'] for m in fold_metrics]
        mae_rel_list = [m['mae_rel'] for m in fold_metrics]
        
        res_row = {
            'K': K,
            'c_w': cw,
            'Mean_HeldOut_KL_Simplex': np.mean(kl_sim_list),
            'Std_HeldOut_KL_Simplex': np.std(kl_sim_list),
            'SEM_HeldOut_KL_Simplex': np.std(kl_sim_list) / np.sqrt(N_FOLDS),
            'Mean_HeldOut_GKL': np.mean(gkl_list),
            'Std_HeldOut_GKL': np.std(gkl_list),
            'Mean_HeldOut_R2_Abs': np.mean(r2_abs_list),
            'Std_HeldOut_R2_Abs': np.std(r2_abs_list),
            'Mean_HeldOut_Pearson_Abs': np.mean(r_abs_list),
            'Mean_HeldOut_MAE_Abs': np.mean(mae_abs_list),
            'Mean_HeldOut_R2_Rel': np.mean(r2_rel_list),
            'Std_HeldOut_R2_Rel': np.std(r2_rel_list),
            'Mean_HeldOut_Pearson_Rel': np.mean(r_rel_list),
            'Mean_HeldOut_MAE_Rel': np.mean(mae_rel_list),
            'Fold1_KL': kl_sim_list[0],
            'Fold2_KL': kl_sim_list[1],
            'Fold3_KL': kl_sim_list[2],
            'Fold4_KL': kl_sim_list[3],
            'Fold5_KL': kl_sim_list[4],
            'Fold1_R2_Abs': r2_abs_list[0],
            'Fold2_R2_Abs': r2_abs_list[1],
            'Fold3_R2_Abs': r2_abs_list[2],
            'Fold4_R2_Abs': r2_abs_list[3],
            'Fold5_R2_Abs': r2_abs_list[4],
        }
        results.append(res_row)
        
        print(f"[{config_idx:2d}/{total_configs}] K={K:d}, c_w={cw:.2f} | Held-Out Simplex KL: {res_row['Mean_HeldOut_KL_Simplex']:.4f} +/- {res_row['SEM_HeldOut_KL_Simplex']:.4f} | Abs R2: {res_row['Mean_HeldOut_R2_Abs']:.4f} | Rel R2: {res_row['Mean_HeldOut_R2_Rel']:.4f} ({elapsed_cfg:.1f}s)")

total_elapsed = time.time() - start_time_all
print(f"\nAll {total_configs} configurations finished in {total_elapsed/60:.2f} minutes.")

# 6. Save and Rank Results
df = pd.DataFrame(results)
df = df.sort_values(by='Mean_HeldOut_KL_Simplex', ascending=True).reset_index(drop=True)

out_csv = 'Data/Ophelia_HeldOut_KL_5Fold_Search_Results.csv'
df.to_csv(out_csv, index=False)
print(f"Saved complete 5-fold CV results to {out_csv}")

print("\n" + "="*80)
print("TOP 10 CONFIGURATIONS RANKED BY HELD-OUT SIMPLEX KL DIVERGENCE")
print("="*80)
cols_show = ['K', 'c_w', 'Mean_HeldOut_KL_Simplex', 'SEM_HeldOut_KL_Simplex', 'Mean_HeldOut_GKL', 'Mean_HeldOut_R2_Abs', 'Mean_HeldOut_R2_Rel', 'Mean_HeldOut_MAE_Abs']
print(df[cols_show].head(10).to_string(index=True))

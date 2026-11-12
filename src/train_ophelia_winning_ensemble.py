"""
Train Ophelia Winning Architecture (K=3, c_w=0.20) Multi-Seed Ensemble (N=10 Seeds)
Trained strictly on Data/train_data.mat (102 communities).
Evaluates null models and performs one-time out-of-sample evaluation on Data/test_data.mat (73 communities).
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
from scipy.stats import pearsonr, spearmanr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Training Ophelia Winning Model on: {device}")

# 1. Load Data
train_mat = loadmat('Data/train_data.mat')['train_data']
train_valid = [np.nan_to_num(train_mat[i], nan=0.0) for i in range(train_mat.shape[0]) 
              if not np.isnan(train_mat[i][0]).any() and np.sum(train_mat[i][0]) > 1e-4]
train_tensor = torch.tensor(np.array(train_valid), dtype=torch.float32, device=device) # (102, 6, 12)

test_mat = loadmat('Data/test_data.mat')['test_data']
test_valid = [np.nan_to_num(test_mat[i], nan=0.0) for i in range(test_mat.shape[0]) 
             if not np.isnan(test_mat[i][0]).any() and np.sum(test_mat[i][0]) > 1e-4]
test_tensor = torch.tensor(np.array(test_valid), dtype=torch.float32, device=device)   # (73, 6, 12)

N_train, N_time, S = train_tensor.shape
N_test = test_tensor.shape[0]
print(f"Dataset sizes: Train = {N_train} communities, Test = {N_test} communities (S={S} species).")

K_WINNER = 3
CW_WINNER = 0.20
LR = 0.015
EPOCHS = 400
t_span = torch.linspace(0.0, 5.0, 6, device=device)

# 2. Model Class
class OpheliaByproductODE(nn.Module):
    def __init__(self, K=3, max_crossfeed=0.20):
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

def evaluate_tensor(model, data_tensor):
    model.eval()
    with torch.no_grad():
        Theta, W, alpha, R0_table, c = model.get_params()
        N0 = data_tensor[:, 0, :]
        R0 = R0_table[0].unsqueeze(0).expand(data_tensor.size(0), -1)
        S0 = torch.cat([N0, R0], dim=-1)
        
        S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha, c), S0, t_span, method='euler', options={'step_size': 0.1})
        pred = torch.clamp(S_t[..., :12].permute(1, 0, 2), min=0.0).cpu().numpy()
        true = data_tensor.cpu().numpy()
        
        # Absolute metrics
        vt_flat = true[:, 1:, :].flatten()
        vp_flat = pred[:, 1:, :].flatten()
        ss_tot = np.sum((vt_flat - np.mean(vt_flat))**2)
        r2_abs = 1.0 - np.sum((vt_flat - vp_flat)**2) / ss_tot if ss_tot > 0 else 1.0
        r_abs, _ = pearsonr(vt_flat, vp_flat) if np.std(vp_flat) > 0 else (0.0, 1.0)
        mae_abs = np.mean(np.abs(vt_flat - vp_flat))
        mse_abs = np.mean((vt_flat - vp_flat)**2)
        
        # Relative metrics
        tot_true = true.sum(axis=-1, keepdims=True) + 1e-8
        tot_pred = pred.sum(axis=-1, keepdims=True) + 1e-8
        rel_true = true / tot_true
        rel_pred = pred / tot_pred
        
        rt_flat = rel_true[:, 1:, :].flatten()
        rp_flat = rel_pred[:, 1:, :].flatten()
        ss_tot_rel = np.sum((rt_flat - np.mean(rt_flat))**2)
        r2_rel = 1.0 - np.sum((rt_flat - rp_flat)**2) / ss_tot_rel if ss_tot_rel > 0 else 1.0
        r_rel, _ = pearsonr(rt_flat, rp_flat) if np.std(rp_flat) > 0 else (0.0, 1.0)
        mae_rel = np.mean(np.abs(rt_flat - rp_flat))
        
        # Simplex KL divergence
        q_safe = np.clip(rel_pred[:, 1:, :], 1e-6, 1.0)
        p_safe = rel_true[:, 1:, :]
        kl_matrix = np.where(p_safe > 1e-8, p_safe * np.log(np.clip(p_safe, 1e-8, 1.0) / q_safe), 0.0)
        mean_kl = float(np.mean(np.sum(kl_matrix, axis=-1)))
        
        # Per-timepoint metrics
        time_metrics = {}
        for t in range(1, 6):
            t_true = true[:, t, :].flatten()
            t_pred = pred[:, t, :].flatten()
            t_ss = np.sum((t_true - np.mean(t_true))**2)
            t_r2 = 1.0 - np.sum((t_true - t_pred)**2) / t_ss if t_ss > 0 else 1.0
            t_r, _ = pearsonr(t_true, t_pred) if np.std(t_pred) > 0 else (0.0, 1.0)
            time_metrics[f't={t*12}h'] = {'r2': float(t_r2), 'pearson_r': float(t_r)}
            
        return {
            'r2_abs': float(r2_abs),
            'pearson_abs': float(r_abs),
            'mae_abs': float(mae_abs),
            'mse_abs': float(mse_abs),
            'r2_rel': float(r2_rel),
            'pearson_rel': float(r_rel),
            'mae_rel': float(mae_rel),
            'kl_simplex': mean_kl,
            'time_metrics': time_metrics,
            'pred': pred,
            'true': true,
            'rel_pred': rel_pred,
            'rel_true': rel_true
        }

# 3. Train Multi-Seed Ensemble (N=10)
SEEDS = [42, 101, 102, 103, 104, 105, 106, 107, 108, 109]
runs_data = []

best_global_r2 = -float('inf')
best_seed = None
best_model_state = None

print("\n" + "="*80)
print(f"TRAINING 10-SEED ENSEMBLE (K={K_WINNER}, c_w={CW_WINNER:.2f}) ON 102 TRAINING COMMUNITIES")
print("="*80)

for s in SEEDS:
    torch.manual_seed(s)
    np.random.seed(s)
    
    model = OpheliaByproductODE(K=K_WINNER, max_crossfeed=CW_WINNER).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_loss = float('inf')
    best_weights = None
    
    for ep in range(1, EPOCHS + 1):
        opt.zero_grad()
        Theta, W, alpha, R0_table, c = model.get_params()
        N0 = train_tensor[:, 0, :]
        R0 = R0_table[0].unsqueeze(0).expand(train_tensor.size(0), -1)
        S0 = torch.cat([N0, R0], dim=-1)
        
        S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha, c), S0, t_span, method='euler', options={'step_size': 0.1})
        N_pred = torch.clamp(S_t[..., :12].permute(1, 0, 2), min=0.0)
        
        loss = F.mse_loss(N_pred, train_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    
    # Evaluate Train
    res_train = evaluate_tensor(model, train_tensor)
    # Evaluate Test (One-time blind test)
    res_test = evaluate_tensor(model, test_tensor)
    
    runs_data.append({
        'seed': s,
        'fit_loss': best_loss,
        'res_train': res_train,
        'res_test': res_test,
        'weights': best_weights
    })
    
    if res_train['r2_abs'] > best_global_r2:
        best_global_r2 = res_train['r2_abs']
        best_seed = s
        best_model_state = best_weights
        
    print(f"Seed {s:3d} | Train Abs R2: {res_train['r2_abs']:.4f}, Rel R2: {res_train['r2_rel']:.4f} | Test Abs R2: {res_test['r2_abs']:.4f}, Test Pearson r: {res_test['pearson_abs']:.4f}, Test Rel r: {res_test['pearson_rel']:.4f}")

# Ensemble Averages
train_r2_abs_all = [r['res_train']['r2_abs'] for r in runs_data]
train_r2_rel_all = [r['res_train']['r2_rel'] for r in runs_data]
test_r2_abs_all = [r['res_test']['r2_abs'] for r in runs_data]
test_r_abs_all = [r['res_test']['pearson_abs'] for r in runs_data]
test_r2_rel_all = [r['res_test']['r2_rel'] for r in runs_data]
test_r_rel_all = [r['res_test']['pearson_rel'] for r in runs_data]
test_kl_all = [r['res_test']['kl_simplex'] for r in runs_data]

print("\n" + "="*80)
print(f"ENSEMBLE SUMMARY (N=10 Initializations, K={K_WINNER}, c_w={CW_WINNER})")
print(f"Train Abs R2:  {np.mean(train_r2_abs_all):.4f} +/- {np.std(train_r2_abs_all):.4f}")
print(f"Train Rel R2:  {np.mean(train_r2_rel_all):.4f} +/- {np.std(train_r2_rel_all):.4f}")
print(f"Test Abs R2:   {np.mean(test_r2_abs_all):.4f} +/- {np.std(test_r2_abs_all):.4f}")
print(f"Test Abs r:    {np.mean(test_r_abs_all):.4f} +/- {np.std(test_r_abs_all):.4f}")
print(f"Test Rel R2:   {np.mean(test_r2_rel_all):.4f} +/- {np.std(test_r2_rel_all):.4f}")
print(f"Test Rel r:    {np.mean(test_r_rel_all):.4f} +/- {np.std(test_r_rel_all):.4f}")
print(f"Test Simplex KL:{np.mean(test_kl_all):.4f} +/- {np.std(test_kl_all):.4f} nats")
print(f"Best Seed:     {best_seed} (Train Abs R2 = {best_global_r2:.4f})")
print("="*80)

# 4. Null Model Benchmarks for Best Seed Model
best_model = OpheliaByproductODE(K=K_WINNER, max_crossfeed=CW_WINNER).to(device)
best_model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

print("\nEvaluating Null Benchmarks on Best Seed Model (Seed", best_seed, "):")

# A) Persistence Null: t_k = t_{k-1}
true_test_np = test_tensor.cpu().numpy()
persist_pred = np.zeros_like(true_test_np)
persist_pred[:, 0, :] = true_test_np[:, 0, :]
for t in range(1, 6):
    persist_pred[:, t, :] = true_test_np[:, t-1, :]
pt_flat = true_test_np[:, 1:, :].flatten()
pp_flat = persist_pred[:, 1:, :].flatten()
persist_r2 = 1.0 - np.sum((pt_flat - pp_flat)**2) / np.sum((pt_flat - np.mean(pt_flat))**2)
persist_r, _ = pearsonr(pt_flat, pp_flat)
print(f"1. Persistence Baseline (t_k = t_k-1): Test Abs R2 = {persist_r2:.4f}, r = {persist_r:.4f}")

# B) Static Initial Abundance Baseline: t_k = t_0
init_pred = np.repeat(true_test_np[:, 0:1, :], 6, axis=1)
ip_flat = init_pred[:, 1:, :].flatten()
init_r2 = 1.0 - np.sum((pt_flat - ip_flat)**2) / np.sum((pt_flat - np.mean(pt_flat))**2)
print(f"2. Static Baseline (t_k = t_0): Test Abs R2 = {init_r2:.4f}")

# C) Shuffled Initial Resources R0 (N=50 random draws)
r0_r2s = []
Theta, W, alpha, R0_table, c = best_model.get_params()
for _ in range(50):
    shuff_R0 = R0_table[0, torch.randperm(K_WINNER)].unsqueeze(0).expand(test_tensor.size(0), -1)
    S0 = torch.cat([test_tensor[:, 0, :], shuff_R0], dim=-1)
    with torch.no_grad():
        S_t = odeint(lambda t, S: best_model.ode_deriv(t, S, Theta, W, alpha, c), S0, t_span, method='euler', options={'step_size': 0.1})
        pred_shuff = torch.clamp(S_t[..., :12].permute(1, 0, 2), min=0.0).cpu().numpy()
        sp_flat = pred_shuff[:, 1:, :].flatten()
        r2_shuff = 1.0 - np.sum((pt_flat - sp_flat)**2) / np.sum((pt_flat - np.mean(pt_flat))**2)
        r0_r2s.append(r2_shuff)
print(f"3. Scrambled Initial Resources R0: Test Abs R2 = {np.mean(r0_r2s):.4f} +/- {np.std(r0_r2s):.4f}")

# D) Shuffled Metabolic Affinities Theta (N=50 random draws)
th_r2s = []
for _ in range(50):
    shuff_Theta = Theta[torch.randperm(12)]
    R0 = R0_table[0].unsqueeze(0).expand(test_tensor.size(0), -1)
    S0 = torch.cat([test_tensor[:, 0, :], R0], dim=-1)
    with torch.no_grad():
        S_t = odeint(lambda t, S: best_model.ode_deriv(t, S, shuff_Theta, W, alpha, c), S0, t_span, method='euler', options={'step_size': 0.1})
        pred_shuff = torch.clamp(S_t[..., :12].permute(1, 0, 2), min=0.0).cpu().numpy()
        sp_flat = pred_shuff[:, 1:, :].flatten()
        r2_shuff = 1.0 - np.sum((pt_flat - sp_flat)**2) / np.sum((pt_flat - np.mean(pt_flat))**2)
        th_r2s.append(r2_shuff)
print(f"4. Shuffled Affinities Theta: Test Abs R2 = {np.mean(th_r2s):.4f} +/- {np.std(th_r2s):.4f}")

# E) Shuffled Crowding alpha (N=50 random draws)
alpha_r2s = []
for _ in range(50):
    shuff_alpha = alpha[torch.randperm(12)]
    R0 = R0_table[0].unsqueeze(0).expand(test_tensor.size(0), -1)
    S0 = torch.cat([test_tensor[:, 0, :], R0], dim=-1)
    with torch.no_grad():
        S_t = odeint(lambda t, S: best_model.ode_deriv(t, S, Theta, W, shuff_alpha, c), S0, t_span, method='euler', options={'step_size': 0.1})
        pred_shuff = torch.clamp(S_t[..., :12].permute(1, 0, 2), min=0.0).cpu().numpy()
        sp_flat = pred_shuff[:, 1:, :].flatten()
        r2_shuff = 1.0 - np.sum((pt_flat - sp_flat)**2) / np.sum((pt_flat - np.mean(pt_flat))**2)
        alpha_r2s.append(r2_shuff)
print(f"5. Shuffled Crowding alpha: Test Abs R2 = {np.mean(alpha_r2s):.4f} +/- {np.std(alpha_r2s):.4f}")

# F) Retrained Ablation W = 0 (No Byproduct Cross-Feeding)
print("6. Retraining W=0 Ablation Model on 102 Training Communities...")
torch.manual_seed(best_seed)
model_no_w = OpheliaByproductODE(K=K_WINNER, max_crossfeed=0.0).to(device)
opt_no_w = torch.optim.Adam(model_no_w.parameters(), lr=LR)
for ep in range(1, EPOCHS + 1):
    opt_no_w.zero_grad()
    Theta_nw, W_nw, alpha_nw, R0_nw, c_nw = model_no_w.get_params()
    W_zero = torch.zeros_like(W_nw)
    N0 = train_tensor[:, 0, :]
    R0 = R0_nw[0].unsqueeze(0).expand(train_tensor.size(0), -1)
    S0 = torch.cat([N0, R0], dim=-1)
    S_t = odeint(lambda t, S: model_no_w.ode_deriv(t, S, Theta_nw, W_zero, alpha_nw, c_nw), S0, t_span, method='euler', options={'step_size': 0.1})
    N_pred = torch.clamp(S_t[..., :12].permute(1, 0, 2), min=0.0)
    loss = F.mse_loss(N_pred, train_tensor)
    loss.backward()
    opt_no_w.step()

res_test_now = evaluate_tensor(model_no_w, test_tensor)
print(f"Retrained W=0 Ablation Model: Test Abs R2 = {res_test_now['r2_abs']:.4f}, Test Pearson r = {res_test_now['pearson_abs']:.4f}")

# 5. Invariance Analysis on Test Set for Best Seed Model
res_best_test = evaluate_tensor(best_model, test_tensor)
pred_test = res_best_test['pred']
true_test = res_best_test['true']

# Calculate per-community Pearson r
comm_pearsons = []
comm_shannons = []
comm_richness = []
comm_bc_dists = []

# Training distribution centroid for Bray-Curtis
train_np = train_tensor.cpu().numpy()
train_mean_t0 = train_np[:, 0, :].mean(axis=0)

for c_i in range(N_test):
    c_true = true_test[c_i, 1:, :].flatten()
    c_pred = pred_test[c_i, 1:, :].flatten()
    if np.std(c_pred) > 0 and np.std(c_true) > 0:
        r_c, _ = pearsonr(c_true, c_pred)
    else:
        r_c = 1.0
    comm_pearsons.append(r_c)
    
    # Initial Shannon Diversity H and Richness S
    init_c = true_test[c_i, 0, :]
    init_rel = init_c / (np.sum(init_c) + 1e-8)
    h_c = -np.sum(np.where(init_rel > 1e-8, init_rel * np.log(init_rel + 1e-12), 0.0))
    s_c = np.sum(init_c > 1e-4)
    comm_shannons.append(h_c)
    comm_richness.append(s_c)
    
    # Mean Bray-Curtis distance to train set communities at t=0
    bc_to_train = []
    for tr_i in range(N_train):
        t_tr = train_np[tr_i, 0, :]
        bc = np.sum(np.abs(init_c - t_tr)) / (np.sum(init_c + t_tr) + 1e-8)
        bc_to_train.append(bc)
    comm_bc_dists.append(np.mean(bc_to_train))

comm_pearsons = np.array(comm_pearsons)
comm_shannons = np.array(comm_shannons)
comm_richness = np.array(comm_richness)
comm_bc_dists = np.array(comm_bc_dists)

r_shannon, p_shannon = pearsonr(comm_shannons, comm_pearsons)
rho_shannon, p_rho_shannon = spearmanr(comm_shannons, comm_pearsons)

r_richness, p_richness = pearsonr(comm_richness, comm_pearsons)
r_bc, p_bc = pearsonr(comm_bc_dists, comm_pearsons)
rho_bc, p_rho_bc = spearmanr(comm_bc_dists, comm_pearsons)

print("\n" + "="*80)
print("INVARIANCE STRESS-TEST RESULTS ON UNTOUCHED TEST SET")
print(f"Accuracy vs. Shannon Diversity: Pearson r = {r_shannon:.4f} (p = {p_shannon:.4f}), Spearman rho = {rho_shannon:.4f} (p = {p_rho_shannon:.4f})")
print(f"Accuracy vs. Species Richness:   Pearson r = {r_richness:.4f} (p = {p_richness:.4f})")
print(f"Accuracy vs. Mean Bray-Curtis:   Pearson r = {r_bc:.4f} (p = {p_bc:.4f}), Spearman rho = {rho_bc:.4f} (p = {p_rho_bc:.4f})")
print("="*80)

# Save test community features
df_features = pd.DataFrame({
    'Community': [f'Test_Comm_{i+1}' for i in range(N_test)],
    'Trajectory_Pearson_r': comm_pearsons,
    'Initial_Shannon_H': comm_shannons,
    'Initial_Richness_S': comm_richness,
    'Mean_Bray_Curtis': comm_bc_dists
})
df_features.to_csv('Data/Ophelia_Test_Community_Accuracy_Features_K3.csv', index=False)
print("Saved Data/Ophelia_Test_Community_Accuracy_Features_K3.csv")

# 6. Save Winning Model Checkpoints
save_path_pth = 'Data/Ophelia_Physical_ODE_Winning_K3.pth'
torch.save({
    'model_state_dict': best_model_state,
    'num_species': 12,
    'K': K_WINNER,
    'c_w': CW_WINNER,
    'best_seed': best_seed,
    'res_train': runs_data[SEEDS.index(best_seed)]['res_train'],
    'res_test': res_best_test,
    'null_persist_r2': persist_r2,
    'null_shuff_r0_r2': float(np.mean(r0_r2s)),
    'null_shuff_r0_std': float(np.std(r0_r2s)),
    'null_shuff_theta_r2': float(np.mean(th_r2s)),
    'null_shuff_theta_std': float(np.std(th_r2s)),
    'null_shuff_alpha_r2': float(np.mean(alpha_r2s)),
    'null_shuff_alpha_std': float(np.std(alpha_r2s)),
    'res_test_now': res_test_now,
    'invariance': {
        'r_shannon': r_shannon, 'p_shannon': p_shannon,
        'r_richness': r_richness, 'p_richness': p_richness,
        'r_bc': r_bc, 'p_bc': p_bc
    }
}, save_path_pth)
print(f"Saved winning checkpoint to {save_path_pth}")

save_path_multi = 'Data/Ophelia_DualTheta_MultiRun_10Seeds_K3.pt'
torch.save({
    'seeds': SEEDS,
    'K': K_WINNER,
    'c_w': CW_WINNER,
    'runs_data': runs_data,
    'best_seed': best_seed,
    'best_model_state': best_model_state
}, save_path_multi)
print(f"Saved 10-seed multi-run data to {save_path_multi}")

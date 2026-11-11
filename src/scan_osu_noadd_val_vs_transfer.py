import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchdiffeq import odeint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing OSU No-Additives Validation vs. Cross-Host Transfer on: {device}")

# 1. Load Data & Shared 81 Species
data_path = 'Data/German_processed_data.pt'
raw_data = torch.load(data_path, weights_only=False)
raw_strains = raw_data['strains']
trajectories_dict = raw_data['trajectories']
raw_traj = torch.stack(list(trajectories_dict.values()), dim=0)

peaks = raw_traj.amax(dim=(0, 1))
mask_1pct = peaks >= 0.01
strains_91 = [raw_strains[i] for i in range(len(raw_strains)) if mask_1pct[i]]
data_91 = raw_traj[:, :, mask_1pct]

brrs_present = (data_91[:6].sum(dim=(0, 1)) > 0)
osu_present = (data_91[6:].sum(dim=(0, 1)) > 0)
shared_mask = brrs_present & osu_present

strains_81 = [strains_91[i] for i in range(len(strains_91)) if shared_mask[i]]
data_81 = data_91[:, :, shared_mask].to(device)
data_81 = data_81 / (data_81.sum(dim=-1, keepdim=True) + 1e-12)

num_conditions, num_passages, num_species = data_81.shape
cond_env_ids = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=torch.long, device=device)

x0 = data_81[:, 0, :]
seed_t1 = torch.where(x0 > 0, x0, torch.tensor(1e-4, device=device))
seed_t1 = seed_t1 / seed_t1.sum(dim=-1, keepdim=True)

# Define Physical ODE Model
class PurePhysicalODE(nn.Module):
    def __init__(self, N, K):
        super().__init__()
        self.N, self.K = N, K
        self.a = nn.Parameter(torch.randn(6, K) * 0.5 + 2.0)
        self.theta_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.05 - 1.0)
        self.b0_raw = nn.Parameter(torch.tensor(0.0))

    def get_params(self):
        Theta = F.softplus(self.theta_raw)
        W = F.softplus(self.w_raw)
        alpha = F.softplus(self.alpha_raw)
        R0 = F.softplus(self.a)
        B0 = F.softplus(self.b0_raw) + 0.1
        return Theta, W, alpha, R0, B0

    def ode_deriv(self, t, S, Theta, W, alpha):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        dN_dt = N_t * torch.matmul(R_t, Theta.t()) - alpha * (N_t ** 2)
        flux = torch.matmul(torch.matmul(N_t, Theta), W.t())
        dR_dt = - R_t * flux
        return torch.cat([dN_dt, dR_dt], dim=-1)

# Training on OSU [7, 8, 9, 10, 11], Validating on OSU 6 (No Additive 5 bar), Testing on BRRS [2, 4]
def train_and_eval(K=3, lr=0.015, l2_alpha=0.24, tf_decay=0.85, epochs=250, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = PurePhysicalODE(num_species, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_indices = [7, 8, 9, 10, 11] # OSU H2 30b, MeOH 5b & 30b, TMA 5b & 30b
    train_data = data_81[train_indices]
    train_seed = seed_t1[train_indices]
    train_envs = cond_env_ids[train_indices]

    best_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        Theta, W, alpha, R0_table, B0 = model.get_params()
        N_curr = train_seed * B0
        R0 = R0_table[train_envs]
        loss = 0.0
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * tf_decay)))

        for p in range(1, num_passages):
            N0 = torch.clamp(N_curr / 100.0, min=0.0)
            S0 = torch.cat([N0, R0], dim=-1)
            t_span = torch.tensor([0.0, 1.0], device=device)
            S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, t_span, method='euler', options={'step_size': 0.1})
            N_final = torch.clamp(S_t[-1, :, :num_species], min=0.0)
            tot_bio = N_final.sum(dim=-1, keepdim=True) + 1e-12
            x_pred = N_final / tot_bio
            target = train_data[:, p, :]
            loss = loss + (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean()
            
            if tf_ratio > 0:
                x_mix = tf_ratio * target + (1.0 - tf_ratio) * x_pred
                N_curr = x_mix * tot_bio
            else:
                N_curr = N_final

        if l2_alpha > 0:
            loss = loss + l2_alpha * torch.sum(alpha ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()

    # 1. Validation on Held-Out OSU Condition 6 (No Additive 5 bar)
    val_data = data_81[[6]]
    val_seed = seed_t1[[6]]
    with torch.no_grad():
        Theta, W, alpha, R0_table, B0 = model.get_params()
        # Use baseline H2 nutrient vector (env 1 from cond 7: OSU H2 30b)
        r_val = R0_table[[1]]
        val_preds = [val_seed]
        N_curr = val_seed * B0
        val_kl = 0.0
        for p in range(1, num_passages):
            N0 = torch.clamp(N_curr / 100.0, min=0.0)
            S0 = torch.cat([N0, r_val], dim=-1)
            t_span = torch.tensor([0.0, 1.0], device=device)
            S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, t_span, method='euler', options={'step_size': 0.1})
            N_final = torch.clamp(S_t[-1, :, :num_species], min=0.0)
            x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
            val_preds.append(x_pred)
            N_curr = N_final
            target = val_data[:, p, :]
            val_kl += (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean().item()

    pred_val_np = torch.stack(val_preds, dim=1).cpu().numpy()
    true_val_np = val_data.cpu().numpy()
    ss_res_val = np.sum((true_val_np - pred_val_np) ** 2)
    ss_tot_val = np.sum((true_val_np - np.mean(true_val_np)) ** 2)
    val_r2 = float(1.0 - ss_res_val / ss_tot_val)
    val_methano_t6 = float(pred_val_np[0, -1, 0] * 100)
    val_kl_mean = float(val_kl / (num_passages - 1))

    # 2. Test on Held-Out Cow BRRS 5 bar: cond 0 (no add), cond 2 (+MeOH), cond 4 (+TMA)
    test_indices = [0, 2, 4]
    test_data = data_81[test_indices]
    test_seed = seed_t1[test_indices]
    test_envs = cond_env_ids[test_indices] # [0, 2, 4] -> Note env 0 is mapped to env 1 in test if env 0 not trained
    # To be strictly fair, for test envs:
    # env 0 (H2 5b) uses env 1 (H2 30b)
    # env 2 (MeOH 5b) uses env 2
    # env 4 (TMA 5b) uses env 4
    test_env_mapped = torch.tensor([1, 2, 4], dtype=torch.long, device=device)

    with torch.no_grad():
        test_preds = [test_seed]
        Theta, W, alpha, R0_table, B0 = model.get_params()
        r_test = R0_table[test_env_mapped]
        N_curr = test_seed * B0
        for p in range(1, num_passages):
            N0 = torch.clamp(N_curr / 100.0, min=0.0)
            S0 = torch.cat([N0, r_test], dim=-1)
            t_span = torch.tensor([0.0, 1.0], device=device)
            S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, t_span, method='euler', options={'step_size': 0.1})
            N_final = torch.clamp(S_t[-1, :, :num_species], min=0.0)
            x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
            test_preds.append(x_pred)
            N_curr = N_final

    pred_test_np = torch.stack(test_preds, dim=1).cpu().numpy()
    true_test_np = test_data.cpu().numpy()
    ss_res_test = np.sum((true_test_np - pred_test_np) ** 2)
    ss_tot_test = np.sum((true_test_np - np.mean(true_test_np)) ** 2)
    test_r2 = float(1.0 - ss_res_test / ss_tot_test)
    
    brrs_noadd_pred = float(pred_test_np[0, -1, 0] * 100)
    brrs_meoh_pred = float(pred_test_np[1, -1, 0] * 100)
    brrs_tma_pred = float(pred_test_np[2, -1, 0] * 100)
    ratio = float(brrs_tma_pred / (brrs_meoh_pred + 1e-6))

    return val_r2, val_kl_mean, val_methano_t6, brrs_noadd_pred, brrs_meoh_pred, brrs_tma_pred, ratio, test_r2

# Hyperparameter Sweep Grid
k_vals = [2, 3, 4]
lr_vals = [0.013, 0.015, 0.017]
tf_vals = [0.75, 0.85]
l2_vals = [0.08, 0.16, 0.20, 0.24]

results = []
idx = 0
total_runs = len(k_vals) * len(lr_vals) * len(tf_vals) * len(l2_vals)

print(f"Starting Grid Sweep across {total_runs} Configurations...")
print("="*105)
print(f"{'Idx':3s} | {'K':1s} | {'lr':5s} | {'tf_d':4s} | {'l2_a':4s} | {'Val R2':7s} | {'Val KL':6s} | {'Val Meth':8s} | {'BRRS NoAdd':10s} | {'BRRS MeOH':9s} | {'BRRS TMA':8s} | {'Ratio':6s} | {'BRRS R2':7s}")
print("-"*105)

t0 = time.time()
for K in k_vals:
    for lr in lr_vals:
        for tf_d in tf_vals:
            for l2_a in l2_vals:
                idx += 1
                v_r2, v_kl, v_meth, b_no, b_me, b_tm, rat, t_r2 = train_and_eval(
                    K=K, lr=lr, l2_alpha=l2_a, tf_decay=tf_d, epochs=250, seed=42
                )
                results.append({
                    'idx': idx, 'K': K, 'lr': lr, 'tf_decay': tf_d, 'l2_alpha': l2_a,
                    'val_osu_noadd_r2': v_r2, 'val_osu_noadd_kl': v_kl, 'val_osu_methano_t6': v_meth,
                    'brrs_noadd_pred': b_no, 'brrs_meoh_pred': b_me, 'brrs_tma_pred': b_tm,
                    'enrichment_ratio': rat, 'brrs_test_r2': t_r2
                })
                print(f"{idx:3d} | {K:1d} | {lr:0.3f} | {tf_d:0.2f} | {l2_a:0.2f} | {v_r2:7.4f} | {v_kl:6.3f} | {v_meth:7.2f}% | {b_no:9.2f}% | {b_me:8.2f}% | {b_tm:7.2f}% | {rat:5.1f}x | {t_r2:7.4f}")

elapsed = time.time() - t0
print("="*105)
print(f"Completed {total_runs} configurations in {elapsed:0.1f} seconds!")

df = pd.DataFrame(results)
out_csv = 'archive/intermediate_results/Regime2_OSU_NoAdd_Validation_Sweep.csv'
df.to_csv(out_csv, index=False)
print(f"Saved complete results to: {out_csv}")

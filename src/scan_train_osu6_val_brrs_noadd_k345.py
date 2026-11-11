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
print(f"Executing: Train All 6 OSU -> Validate on BRRS No Additives -> Test Interventions on: {device}")

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

def train_and_eval(K=3, lr=0.015, l2_alpha=0.05, tf_decay=0.75, epochs=300, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = PurePhysicalODE(num_species, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train strictly on all 6 Cow OSU conditions: [6, 7, 8, 9, 10, 11]
    train_indices = [6, 7, 8, 9, 10, 11]
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

    # 1. Validation Set: Cow BRRS No Additives 5 bar (Condition 0)
    val_data = data_81[[0]]
    val_seed = seed_t1[[0]]
    val_envs = cond_env_ids[[0]] # Env 0: H2:CO2 5 bar
    with torch.no_grad():
        Theta, W, alpha, R0_table, B0 = model.get_params()
        r_val = R0_table[val_envs]
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
    brrs_noadd_r2 = float(1.0 - ss_res_val / ss_tot_val)
    brrs_noadd_kl = float(val_kl / (num_passages - 1))
    val_brrs_noadd_methano = float(pred_val_np[0, -1, 0] * 100)

    # 2. Test Interventions: Cow BRRS +MeOH (Cond 2) & Cow BRRS +TMA (Cond 4)
    test_indices = [2, 4]
    test_data = data_81[test_indices]
    test_seed = seed_t1[test_indices]
    test_envs = cond_env_ids[test_indices] # [2, 4] -> Env 2: MeOH 5b, Env 4: TMA 5b

    with torch.no_grad():
        test_preds = [test_seed]
        Theta, W, alpha, R0_table, B0 = model.get_params()
        r_test = R0_table[test_envs]
        N_curr = test_seed * B0
        test_kl = 0.0
        for p in range(1, num_passages):
            N0 = torch.clamp(N_curr / 100.0, min=0.0)
            S0 = torch.cat([N0, r_test], dim=-1)
            t_span = torch.tensor([0.0, 1.0], device=device)
            S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, t_span, method='euler', options={'step_size': 0.1})
            N_final = torch.clamp(S_t[-1, :, :num_species], min=0.0)
            x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
            test_preds.append(x_pred)
            N_curr = N_final
            target = test_data[:, p, :]
            test_kl += (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean().item()

    pred_test_np = torch.stack(test_preds, dim=1).cpu().numpy()
    true_test_np = test_data.cpu().numpy()
    ss_res_test = np.sum((true_test_np - pred_test_np) ** 2)
    ss_tot_test = np.sum((true_test_np - np.mean(true_test_np)) ** 2)
    test_r2 = float(1.0 - ss_res_test / ss_tot_test)
    test_kl_mean = float(test_kl / (num_passages - 1))

    brrs_meoh_pred = float(pred_test_np[0, -1, 0] * 100)
    brrs_tma_pred = float(pred_test_np[1, -1, 0] * 100)
    ratio = float(brrs_tma_pred / (brrs_meoh_pred + 1e-6))

    return brrs_noadd_kl, brrs_noadd_r2, val_brrs_noadd_methano, brrs_meoh_pred, brrs_tma_pred, ratio, test_r2, test_kl_mean

# 1. Load Existing K=3 and K=4 results
prev_csv = 'archive/intermediate_results/Train_OSU6_Val_BRRS_NoAdd_Sweep.csv'
if os.path.exists(prev_csv):
    df_prev = pd.read_csv(prev_csv)
    df_k34 = df_prev[df_prev['K'].isin([3, 4])].copy()
    print(f"Loaded {len(df_k34)} existing runs for K=3 and K=4 from {prev_csv}")
else:
    df_k34 = pd.DataFrame()

# 2. Run Sweep over K=5
lr_vals = [0.015, 0.020]
ep_vals = [250, 350]
tf_vals = [0.75, 0.85]
l2_vals = [0.05, 0.16, 0.24]

print(f"\nSweeping Grid for K=5 (24 configurations)...")
print("="*115)
print(f"{'Idx':3s} | {'K':1s} | {'lr':5s} | {'ep':3s} | {'tf_d':4s} | {'l2_a':4s} | {'Val KL (BRRS 0)':15s} | {'Val R2':7s} | {'BRRS NoAdd':10s} | {'BRRS MeOH':9s} | {'BRRS TMA':8s} | {'Ratio':6s} | {'Test R2':7s}")
print("-"*115)

k5_results = []
idx = 0
for lr in lr_vals:
    for ep in ep_vals:
        for tf_d in tf_vals:
            for l2_a in l2_vals:
                idx += 1
                v_kl, v_r2, v_meth, b_me, b_tm, rat, t_r2, t_kl = train_and_eval(
                    K=5, lr=lr, l2_alpha=l2_a, tf_decay=tf_d, epochs=ep, seed=42
                )
                k5_results.append({
                    'idx': idx, 'K': 5, 'lr': lr, 'epochs': ep, 'tf_decay': tf_d, 'l2_alpha': l2_a,
                    'val_brrs_noadd_kl': v_kl, 'val_brrs_noadd_r2': v_r2, 'val_brrs_noadd_methano': v_meth,
                    'brrs_meoh_pred': b_me, 'brrs_tma_pred': b_tm, 'enrichment_ratio': rat,
                    'test_interventions_r2': t_r2, 'test_interventions_kl': t_kl
                })
                print(f"{idx:3d} | 5 | {lr:0.3f} | {ep:3d} | {tf_d:0.2f} | {l2_a:0.2f} | {v_kl:15.4f} | {v_r2:7.4f} | {v_meth:9.2f}% | {b_me:8.2f}% | {b_tm:7.2f}% | {rat:5.1f}x | {t_r2:7.4f}")

df_k5 = pd.DataFrame(k5_results)

# 3. Combine K=3, K=4, K=5
df_all = pd.concat([df_k34, df_k5], ignore_index=True)
df_all['idx'] = np.arange(1, len(df_all) + 1)

out_csv = 'archive/intermediate_results/Train_OSU6_Val_BRRS_NoAdd_Sweep_K345.csv'
df_all.to_csv(out_csv, index=False)
print("="*115)
print(f"Complete! Saved merged K in [3, 4, 5] results ({len(df_all)} runs) to: {out_csv}")

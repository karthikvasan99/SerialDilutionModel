import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchdiffeq import odeint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing: Pseudocount Floor (+1e-4 on all passages) Benchmark on: {device}")

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
raw_data_81 = data_91[:, :, shared_mask].to(device)

# Standard Normalized Data (zeros preserved)
std_data_81 = raw_data_81 / (raw_data_81.sum(dim=-1, keepdim=True) + 1e-12)

# Floored and Renormalized Data (+1e-4 added to every bacteria in every passage)
floored_raw = raw_data_81 + 1e-4
floored_data_81 = floored_raw / floored_raw.sum(dim=-1, keepdim=True)

num_conditions, num_passages, num_species = std_data_81.shape
cond_env_ids = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=torch.long, device=device)

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

def train_and_eval(dataset_name, dataset, K=3, lr=0.015, l2_alpha=0.05, tf_decay=0.85, epochs=350, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Inoculum seed
    x0 = dataset[:, 0, :]
    seed_t1 = torch.where(x0 > 0, x0, torch.tensor(1e-4, device=device))
    seed_t1 = seed_t1 / seed_t1.sum(dim=-1, keepdim=True)

    model = PurePhysicalODE(num_species, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train strictly on all 6 Cow OSU conditions: [6, 7, 8, 9, 10, 11]
    train_indices = [6, 7, 8, 9, 10, 11]
    train_data = dataset[train_indices]
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
    val_data = dataset[[0]]
    val_seed = seed_t1[[0]]
    val_envs = cond_env_ids[[0]] # Env 0: H2:CO2 5 bar
    val_preds = [val_seed]
    with torch.no_grad():
        Theta, W, alpha, R0_table, B0 = model.get_params()
        r_val = R0_table[val_envs]
        N_curr = val_seed * B0
        val_kl_list = []
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
            kl_p = (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean().item()
            val_kl_list.append(kl_p)

    pred_val_np = torch.stack(val_preds, dim=1).cpu().numpy()
    true_val_np = val_data.cpu().numpy()

    # T6 Metrics
    val_t6_kl = float(val_kl_list[-1])
    ss_res_t6 = np.sum((true_val_np[0, -1, :] - pred_val_np[0, -1, :]) ** 2)
    ss_tot_t6 = np.sum((true_val_np[0, -1, :] - np.mean(true_val_np[0, -1, :])) ** 2)
    val_t6_r2 = float(1.0 - ss_res_t6 / ss_tot_t6)
    val_t6_methano = float(pred_val_np[0, -1, 0] * 100)

    # Full trajectory metrics
    val_mean_kl = float(np.mean(val_kl_list))
    ss_res_all = np.sum((true_val_np - pred_val_np) ** 2)
    ss_tot_all = np.sum((true_val_np - np.mean(true_val_np)) ** 2)
    val_mean_r2 = float(1.0 - ss_res_all / ss_tot_all)

    # 2. Test Interventions: Cow BRRS +MeOH (Cond 2) & Cow BRRS +TMA (Cond 4)
    test_indices = [2, 4]
    test_data = dataset[test_indices]
    test_seed = seed_t1[test_indices]
    test_envs = cond_env_ids[test_indices] # [2, 4] -> Env 2: MeOH 5b, Env 4: TMA 5b

    test_preds = [test_seed]
    with torch.no_grad():
        r_test = R0_table[test_envs]
        N_curr = test_seed * B0
        test_kl_list = []
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
            kl_p = (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean().item()
            test_kl_list.append(kl_p)

    pred_test_np = torch.stack(test_preds, dim=1).cpu().numpy()
    true_test_np = test_data.cpu().numpy()

    test_t6_kl = float(test_kl_list[-1])
    ss_res_test_t6 = np.sum((true_test_np[:, -1, :] - pred_test_np[:, -1, :]) ** 2)
    ss_tot_test_t6 = np.sum((true_test_np[:, -1, :] - np.mean(true_test_np[:, -1, :])) ** 2)
    test_t6_r2 = float(1.0 - ss_res_test_t6 / ss_tot_test_t6)

    # Full trajectory test R2
    ss_res_test_all = np.sum((true_test_np - pred_test_np) ** 2)
    ss_tot_test_all = np.sum((true_test_np - np.mean(true_test_np)) ** 2)
    test_all_r2 = float(1.0 - ss_res_test_all / ss_tot_test_all)

    brrs_meoh_pred = float(pred_test_np[0, -1, 0] * 100)
    brrs_tma_pred = float(pred_test_np[1, -1, 0] * 100)
    ratio = float(brrs_tma_pred / (brrs_meoh_pred + 1e-6))

    return {
        'dataset': dataset_name, 'K': K, 'lr': lr, 'epochs': epochs,
        'val_t6_kl': val_t6_kl, 'val_t6_r2': val_t6_r2, 'val_t6_methano': val_t6_methano,
        'val_mean_kl': val_mean_kl, 'val_mean_r2': val_mean_r2,
        'brrs_meoh_pred': brrs_meoh_pred, 'brrs_tma_pred': brrs_tma_pred, 'enrichment_ratio': ratio,
        'test_t6_r2': test_t6_r2, 'test_all_r2': test_all_r2
    }

# Run Experiment on Both Datasets across our top hyperparameters
test_configs = [
    # (lr, l2_alpha, epochs)
    (0.015, 0.05, 350), # #1 Champ in full-trajectory sweep
    (0.020, 0.05, 350), # #1 Champ in T6 sweep (>10% TMA)
    (0.015, 0.16, 350), # Stable regularizer
    (0.020, 0.16, 350), # Stable regularizer
]

results = []
print("\n" + "="*125)
print(f"BENCHMARK: STANDARD DATA (ZEROS PRESERVED) vs. FLOORED DATA (+1e-4 TO ALL PASSAGES)")
print("="*125)
print(f"{'Dataset':22s} | {'lr':5s} | {'l2':4s} | {'Val T6 KL':9s} | {'Val T6 R2':9s} | {'Val Mean KL':11s} | {'MeOH 5b':7s} | {'TMA 5b':7s} | {'Ratio':6s} | {'Test T6 R2':10s}")
print("-"*125)

for (lr, l2_a, ep) in test_configs:
    # 1. Standard Dataset
    res_std = train_and_eval("Standard (Zeros)", std_data_81, K=3, lr=lr, l2_alpha=l2_a, epochs=ep)
    results.append(res_std)
    print(f"{res_std['dataset']:22s} | {lr:0.3f} | {l2_a:0.2f} | {res_std['val_t6_kl']:9.4f} | {res_std['val_t6_r2']:9.4f} | {res_std['val_mean_kl']:11.4f} | {res_std['brrs_meoh_pred']:6.2f}% | {res_std['brrs_tma_pred']:6.2f}% | {res_std['enrichment_ratio']:5.1f}x | {res_std['test_t6_r2']:10.4f}")

    # 2. Floored Dataset (+1e-4)
    res_flr = train_and_eval("Floored (+1e-4)", floored_data_81, K=3, lr=lr, l2_alpha=l2_a, epochs=ep)
    results.append(res_flr)
    print(f"{res_flr['dataset']:22s} | {lr:0.3f} | {l2_a:0.2f} | {res_flr['val_t6_kl']:9.4f} | {res_flr['val_t6_r2']:9.4f} | {res_flr['val_mean_kl']:11.4f} | {res_flr['brrs_meoh_pred']:6.2f}% | {res_flr['brrs_tma_pred']:6.2f}% | {res_flr['enrichment_ratio']:5.1f}x | {res_flr['test_t6_r2']:10.4f}")
    print("-"*125)

df = pd.DataFrame(results)
out_csv = 'archive/intermediate_results/Pseudocount_Floor_Experiment_Results.csv'
df.to_csv(out_csv, index=False)
print("="*125)
print(f"Results saved to: {out_csv}")

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchdiffeq import odeint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing: Bidirectional Transfer Benchmark with Pseudocount Floor (+1e-4) on: {device}")

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

# Floored and Renormalized Data (+1e-4 added to every bacteria in every passage)
floored_raw = raw_data_81 + 1e-4
floored_data_81 = floored_raw / floored_raw.sum(dim=-1, keepdim=True)

num_conditions, num_passages, num_species = floored_data_81.shape
cond_env_ids = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=torch.long, device=device)

target_idx = 0
target_name = strains_81[target_idx].split('|')[-1]

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

def run_transfer(train_indices, test_indices, K=3, lr=0.015, l2_alpha=0.05, tf_decay=0.85, epochs=350, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Inoculum seed from floored data
    x0 = floored_data_81[:, 0, :]
    seed_t1 = torch.where(x0 > 0, x0, torch.tensor(1e-4, device=device))
    seed_t1 = seed_t1 / seed_t1.sum(dim=-1, keepdim=True)

    model = PurePhysicalODE(num_species, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = floored_data_81[train_indices]
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

    # Evaluate on held-out test host conditions (5 bar: index 0=NoAdd, 1=MeOH, 2=TMA)
    test_data = floored_data_81[test_indices]
    test_seed = seed_t1[test_indices]
    test_envs = cond_env_ids[test_indices]

    test_preds = [test_seed]
    with torch.no_grad():
        Theta, W, alpha, R0_table, B0 = model.get_params()
        r_test = R0_table[test_envs]
        N_curr = test_seed * B0
        kl_list = []
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
            kl_list.append(kl_p)

    pred_np = torch.stack(test_preds, dim=1).cpu().numpy() # (3, 6, 81)
    true_np = test_data.cpu().numpy()

    # Community R2 at 5 bar
    ss_res_all = np.sum((true_np - pred_np) ** 2)
    ss_tot_all = np.sum((true_np - np.mean(true_np)) ** 2)
    test_r2 = float(1.0 - ss_res_all / ss_tot_all)
    mean_kl = float(np.mean(kl_list))

    # Methanogen predictions at T6
    p_noadd = float(pred_np[0, -1, target_idx] * 100)
    p_meoh = float(pred_np[1, -1, target_idx] * 100)
    p_tma = float(pred_np[2, -1, target_idx] * 100)

    o_noadd = float(true_np[0, -1, target_idx] * 100)
    o_meoh = float(true_np[1, -1, target_idx] * 100)
    o_tma = float(true_np[2, -1, target_idx] * 100)

    return test_r2, mean_kl, p_noadd, p_meoh, p_tma, o_noadd, o_meoh, o_tma

configs = [
    (0.015, 0.05, 350),
    (0.020, 0.05, 350),
    (0.015, 0.16, 350),
    (0.020, 0.16, 350),
]

print("\n" + "="*135)
print(f"BIDIRECTIONAL EVALUATION WITH PSEUDOCOUNT FLOOR (+1e-4 everywhere)")
print("="*135)

# 1. Forward: Train OSU all 6 -> Test BRRS 5 bar [0, 2, 4]
print("\n--- FORWARD DIRECTION: TRAIN ON COW OSU (all 6) -> PREDICT COW BRRS (5 bar) ---")
print(f"{'lr':5s} | {'l2':4s} | {'Test R2':8s} | {'Mean KL':8s} | {'Pred NoAdd':10s} | {'Pred MeOH':9s} | {'Pred TMA':8s} | {'Ratio':6s} | {'Observed (NoAdd / MeOH / TMA)':32s}")
print("-"*135)
for (lr, l2_a, ep) in configs:
    r2, kl, p_no, p_me, p_tm, o_no, o_me, o_tm = run_transfer(
        train_indices=[6, 7, 8, 9, 10, 11], test_indices=[0, 2, 4],
        K=3, lr=lr, l2_alpha=l2_a, epochs=ep
    )
    rat = p_tm / (p_me + 1e-6)
    print(f"{lr:0.3f} | {l2_a:0.2f} | {r2:8.4f} | {kl:8.4f} | {p_no:9.2f}% | {p_me:8.2f}% | {p_tm:7.2f}% | {rat:5.1f}x | {o_no:5.1f}% / {o_me:5.1f}% / {o_tm:5.1f}%")

# 2. Reverse: Train BRRS all 6 -> Test OSU 5 bar [6, 8, 10]
print("\n--- REVERSE DIRECTION: TRAIN ON COW BRRS (all 6) -> PREDICT COW OSU (5 bar) ---")
print(f"{'lr':5s} | {'l2':4s} | {'Test R2':8s} | {'Mean KL':8s} | {'Pred NoAdd':10s} | {'Pred MeOH':9s} | {'Pred TMA':8s} | {'Ratio (Me/Tm)':13s} | {'Observed (NoAdd / MeOH / TMA)':32s}")
print("-"*135)
for (lr, l2_a, ep) in configs:
    r2, kl, p_no, p_me, p_tm, o_no, o_me, o_tm = run_transfer(
        train_indices=[0, 1, 2, 3, 4, 5], test_indices=[6, 8, 10],
        K=3, lr=lr, l2_alpha=l2_a, epochs=ep
    )
    # In OSU, MeOH should be greater than TMA! (Obs: 41.4% vs 3.6%)
    rat = p_me / (p_tm + 1e-6)
    print(f"{lr:0.3f} | {l2_a:0.2f} | {r2:8.4f} | {kl:8.4f} | {p_no:9.2f}% | {p_me:8.2f}% | {p_tm:7.2f}% | {rat:11.1f}x | {o_no:5.1f}% / {o_me:5.1f}% / {o_tm:5.1f}%")
print("="*135)

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchdiffeq import odeint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing: Train on BRRS (all 6) -> Predict on OSU (all 6 / 5 bar) on: {device}")

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

def train_and_eval(K=3, lr=0.015, l2_alpha=0.05, tf_decay=0.75, epochs=300, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = PurePhysicalODE(num_species, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TRAIN on all 6 Cow BRRS conditions: [0, 1, 2, 3, 4, 5]
    train_indices = [0, 1, 2, 3, 4, 5]
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

    # TEST on Cow OSU 5 bar conditions: [6 (NoAdd), 8 (MeOH), 10 (TMA)]
    # and all 6 OSU conditions [6, 7, 8, 9, 10, 11]
    test_5b_indices = [6, 8, 10]
    test_all_indices = [6, 7, 8, 9, 10, 11]

    with torch.no_grad():
        Theta, W, alpha, R0_table, B0 = model.get_params()

        # Evaluate on all 6 OSU
        osu_seed = seed_t1[test_all_indices]
        osu_envs = cond_env_ids[test_all_indices]
        osu_data = data_81[test_all_indices]

        osu_preds = [osu_seed]
        N_curr = osu_seed * B0
        r_osu = R0_table[osu_envs]
        kl_sum = 0.0
        for p in range(1, num_passages):
            N0 = torch.clamp(N_curr / 100.0, min=0.0)
            S0 = torch.cat([N0, r_osu], dim=-1)
            t_span = torch.tensor([0.0, 1.0], device=device)
            S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, t_span, method='euler', options={'step_size': 0.1})
            N_final = torch.clamp(S_t[-1, :, :num_species], min=0.0)
            x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
            osu_preds.append(x_pred)
            N_curr = N_final
            target = osu_data[:, p, :]
            kl_sum += (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean().item()

    pred_osu_np = torch.stack(osu_preds, dim=1).cpu().numpy() # (6, 6, 81)
    true_osu_np = osu_data.cpu().numpy()

    # Community R2 on all 6 OSU
    ss_res_all = np.sum((true_osu_np - pred_osu_np) ** 2)
    ss_tot_all = np.sum((true_osu_np - np.mean(true_osu_np)) ** 2)
    osu_all_r2 = float(1.0 - ss_res_all / ss_tot_all)
    osu_all_kl = float(kl_sum / (num_passages - 1))

    # Community R2 on OSU 5 bar only (indices 0, 2, 4 corresponding to 6, 8, 10)
    pred_5b = pred_osu_np[[0, 2, 4]]
    true_5b = true_osu_np[[0, 2, 4]]
    ss_res_5b = np.sum((true_5b - pred_5b) ** 2)
    ss_tot_5b = np.sum((true_5b - np.mean(true_5b)) ** 2)
    osu_5b_r2 = float(1.0 - ss_res_5b / ss_tot_5b)

    # Methanogen predictions at T6 in Cow OSU 5 bar:
    # index 0 is No Additive (cond 6)
    # index 2 is +MeOH (cond 8)
    # index 4 is +TMA (cond 10)
    pred_osu_noadd = float(pred_osu_np[0, -1, target_idx] * 100)
    pred_osu_meoh = float(pred_osu_np[2, -1, target_idx] * 100)
    pred_osu_tma = float(pred_osu_np[4, -1, target_idx] * 100)

    # Observed in Cow OSU 5 bar:
    obs_osu_noadd = float(true_osu_np[0, -1, target_idx] * 100)
    obs_osu_meoh = float(true_osu_np[2, -1, target_idx] * 100)
    obs_osu_tma = float(true_osu_np[4, -1, target_idx] * 100)

    return osu_all_kl, osu_all_r2, osu_5b_r2, pred_osu_noadd, pred_osu_meoh, pred_osu_tma, obs_osu_noadd, obs_osu_meoh, obs_osu_tma

# Test a representative grid
configs = [
    # (K, lr, ep, tf_d, l2_a)
    (3, 0.015, 250, 0.85, 0.05),
    (3, 0.015, 350, 0.85, 0.05),
    (3, 0.020, 350, 0.85, 0.05),
    (3, 0.015, 250, 0.75, 0.24),
    (3, 0.015, 350, 0.75, 0.05),
    (2, 0.015, 250, 0.75, 0.05),
    (2, 0.015, 350, 0.85, 0.05),
    (4, 0.015, 350, 0.85, 0.05),
    (5, 0.020, 250, 0.75, 0.24),
]

print("\n" + "="*120)
print(f"EXPERIMENT: TRAIN ON COW BRRS -> PREDICT ON COW OSU (REVERSE TRANSFER)")
print("="*120)
print(f"{'K':1s} | {'lr':5s} | {'ep':3s} | {'tf_d':4s} | {'l2_a':4s} | {'OSU All KL':10s} | {'OSU 6 R2':9s} | {'OSU 5b R2':9s} | {'Pred NoAdd':10s} | {'Pred MeOH':9s} | {'Pred TMA':8s} | {'Observed (NoAdd/MeOH/TMA)':26s}")
print("-"*120)

results = []
for (K, lr, ep, tf_d, l2_a) in configs:
    kl, r2_6, r2_5b, p_no, p_me, p_tm, o_no, o_me, o_tm = train_and_eval(
        K=K, lr=lr, epochs=ep, tf_decay=tf_d, l2_alpha=l2_a, seed=42
    )
    results.append({
        'K': K, 'lr': lr, 'epochs': ep, 'tf_decay': tf_d, 'l2_alpha': l2_a,
        'osu_all_kl': kl, 'osu_all_r2': r2_6, 'osu_5b_r2': r2_5b,
        'pred_osu_noadd': p_no, 'pred_osu_meoh': p_me, 'pred_osu_tma': p_tm,
        'obs_osu_noadd': o_no, 'obs_osu_meoh': o_me, 'obs_osu_tma': o_tm
    })
    print(f"{K:1d} | {lr:0.3f} | {ep:3d} | {tf_d:0.2f} | {l2_a:0.2f} | {kl:10.4f} | {r2_6:9.4f} | {r2_5b:9.4f} | {p_no:9.2f}% | {p_me:8.2f}% | {p_tm:7.2f}% | {o_no:5.1f}% / {o_me:5.1f}% / {o_tm:5.1f}%")

df = pd.DataFrame(results)
out_csv = 'archive/intermediate_results/Train_BRRS_Predict_OSU_Results.csv'
df.to_csv(out_csv, index=False)
print("="*120)
print(f"Results saved to: {out_csv}")

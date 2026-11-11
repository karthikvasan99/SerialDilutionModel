import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchdiffeq import odeint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing: Epoch Scaling & Contrast Benchmark on: {device}")

# 1. Load Data
raw_data = torch.load('Data/German_processed_data.pt', weights_only=False)
raw_strains = raw_data['strains']
raw_traj = torch.stack(list(raw_data['trajectories'].values()), dim=0)

peaks = raw_traj.amax(dim=(0, 1))
mask_1pct = peaks >= 0.01
strains_91 = [raw_strains[i] for i in range(len(raw_strains)) if mask_1pct[i]]
data_91 = raw_traj[:, :, mask_1pct]

shared_mask = (data_91[:6].sum(dim=(0, 1)) > 0) & (data_91[6:].sum(dim=(0, 1)) > 0)
strains_81 = [strains_91[i] for i in range(len(strains_91)) if shared_mask[i]]

# Uniform +1e-4 floor
floored_raw = data_91[:, :, shared_mask].to(device) + 1e-4
floored_data = floored_raw / floored_raw.sum(dim=-1, keepdim=True)

N, K = len(strains_81), 3
cond_env_ids = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=torch.long, device=device)
target_idx = 0 # Methanomicrobium sp.

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

def run_experiment(lr=0.015, l2_alpha=0.05, tf_decay=0.85, epochs=350, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PurePhysicalODE(N, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_indices = [6, 7, 8, 9, 10, 11] # Train OSU all 6
    train_data = floored_data[train_indices]
    train_seed = train_data[:, 0, :]
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

        for p in range(1, 6):
            N0 = torch.clamp(N_curr / 100.0, min=0.0)
            S0 = torch.cat([N0, R0], dim=-1)
            S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, torch.tensor([0.0, 1.0], device=device), method='euler', options={'step_size': 0.1})
            N_final = torch.clamp(S_t[-1, :, :N], min=0.0)
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

    # 1. Validation on BRRS NoAdd 5bar (cond 0)
    val_indices = [0]
    val_data = floored_data[val_indices] # (1, 6 passages, 81)
    val_seed = val_data[:, 0, :]
    val_envs = cond_env_ids[val_indices]

    with torch.no_grad():
        Theta, W, alpha, R0_table, B0 = model.get_params()
        r_val = R0_table[val_envs]
        N_curr = val_seed * B0
        kl_val_list = []
        for p in range(1, 6):
            N0 = torch.clamp(N_curr / 100.0, min=0.0)
            S0 = torch.cat([N0, r_val], dim=-1)
            S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, torch.tensor([0.0, 1.0], device=device), method='euler', options={'step_size': 0.1})
            N_final = torch.clamp(S_t[-1, :, :N], min=0.0)
            x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
            N_curr = N_final
            target = val_data[:, p, :]
            kl_p = (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean().item()
            kl_val_list.append(kl_p)
    val_noadd_kl = float(np.mean(kl_val_list))

    # 2. Test predictions on Cow BRRS (5 bar: 0=NoAdd, 2=MeOH, 4=TMA)
    test_indices = [0, 2, 4]
    test_data = floored_data[test_indices]
    test_seed = test_data[:, 0, :]
    test_envs = cond_env_ids[test_indices]

    test_preds = [test_seed]
    with torch.no_grad():
        Theta, W, alpha, R0_table, B0 = model.get_params()
        r_test = R0_table[test_envs]
        N_curr = test_seed * B0
        kl_test_list = []
        for p in range(1, 6):
            N0 = torch.clamp(N_curr / 100.0, min=0.0)
            S0 = torch.cat([N0, r_test], dim=-1)
            S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, torch.tensor([0.0, 1.0], device=device), method='euler', options={'step_size': 0.1})
            N_final = torch.clamp(S_t[-1, :, :N], min=0.0)
            x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
            test_preds.append(x_pred)
            N_curr = N_final
            target = test_data[:, p, :]
            kl_p = (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean().item()
            kl_test_list.append(kl_p)

    pred_np = torch.stack(test_preds, dim=1).cpu().numpy() # (3, 6, 81)
    true_np = test_data.cpu().numpy()

    ss_res = np.sum((true_np - pred_np) ** 2)
    ss_tot = np.sum((true_np - np.mean(true_np)) ** 2)
    test_r2 = float(1.0 - ss_res / ss_tot)
    test_mean_kl = float(np.mean(kl_test_list))

    pred_noadd = float(pred_np[0, -1, target_idx] * 100)
    pred_meoh = float(pred_np[1, -1, target_idx] * 100)
    pred_tma = float(pred_np[2, -1, target_idx] * 100)
    contrast_ratio = pred_tma / (pred_meoh + 1e-6)
    contrast_diff = pred_tma - pred_meoh

    return {
        'lr': lr,
        'l2_alpha': l2_alpha,
        'epochs': epochs,
        'train_loss': best_loss,
        'val_noadd_kl': val_noadd_kl,
        'test_r2': test_r2,
        'test_mean_kl': test_mean_kl,
        'pred_noadd': pred_noadd,
        'pred_meoh': pred_meoh,
        'pred_tma': pred_tma,
        'contrast_ratio': contrast_ratio,
        'contrast_diff': contrast_diff
    }

epoch_levels = [150, 250, 350, 500, 700, 1000]
configs = [
    (0.015, 0.05),
    (0.020, 0.05),
    (0.015, 0.16),
]

all_results = []
print("\n" + "="*115)
print(f"{'lr':5s} | {'l2':4s} | {'Epochs':6s} | {'Val KL':7s} | {'Test R2':7s} | {'Pred NoAdd':10s} | {'Pred MeOH':9s} | {'Pred TMA':8s} | {'Ratio':6s} | {'Diff':6s}")
print("="*115)

for lr, l2_a in configs:
    for ep in epoch_levels:
        res = run_experiment(lr=lr, l2_alpha=l2_a, tf_decay=0.85, epochs=ep)
        all_results.append(res)
        print(f"{res['lr']:0.3f} | {res['l2_alpha']:0.2f} | {res['epochs']:6d} | {res['val_noadd_kl']:7.4f} | {res['test_r2']:7.4f} | {res['pred_noadd']:9.2f}% | {res['pred_meoh']:8.2f}% | {res['pred_tma']:7.2f}% | {res['contrast_ratio']:5.1f}x | {res['contrast_diff']:+5.2f}%")
    print("-" * 115)

df = pd.DataFrame(all_results)
df.to_csv('archive/intermediate_results/Epoch_Scaling_Contrast_Results.csv', index=False)
print(f"Results saved to archive/intermediate_results/Epoch_Scaling_Contrast_Results.csv")

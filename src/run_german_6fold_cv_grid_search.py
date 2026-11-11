import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"=================================================================")
print(f"  EXHAUSTIVE 6-FOLD CV HYPERPARAMETER TUNING ON FULL DATASET")
print(f"  Device: {device}")
print(f"=================================================================")

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

floored_raw = data_91[:, :, shared_mask].to(device) + 1e-4
floored_data = floored_raw / floored_raw.sum(dim=-1, keepdim=True)

N = len(strains_81)
cond_env_ids = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=torch.long, device=device)
target_idx = 0 # Methanomicrobium sp.

# 6 Folds: Leave out the same media in both pressure conditions (5b & 30b) in one host
folds = [
    ('BRRS_NoAdd', [0, 1]),
    ('BRRS_MeOH',  [2, 3]),
    ('BRRS_TMA',   [4, 5]),
    ('OSU_NoAdd',  [6, 7]),
    ('OSU_MeOH',   [8, 9]),
    ('OSU_TMA',    [10, 11])
]

# Output CSV setup
os.makedirs('archive/intermediate_results', exist_ok=True)
csv_path = 'archive/intermediate_results/German_6Fold_CV_Hyperparameter_Search.csv'

# Fast exact Euler integration (10 steps per passage, dt=0.1)
def ode_step(S, Theta, W, alpha, N, dt=0.1):
    for _ in range(10):
        N_t = torch.clamp(S[..., :N], min=0.0)
        R_t = torch.clamp(S[..., N:], min=0.0)
        dN_dt = N_t * torch.matmul(R_t, Theta.t()) - alpha * (N_t ** 2)
        flux = torch.matmul(torch.matmul(N_t, Theta), W.t())
        dR_dt = - R_t * flux
        dS = torch.cat([dN_dt, dR_dt], dim=-1)
        S = S + dt * dS
    return S

def evaluate_fold_cv(K, lr, l2_alpha, epochs=300):
    fold_kls = []
    fold_r2s = []
    fold_pearsons = []
    target_preds = {}

    for fold_idx, (fold_name, test_indices) in enumerate(folds):
        torch.manual_seed(42 + fold_idx)
        np.random.seed(42 + fold_idx)

        train_indices = [i for i in range(12) if i not in test_indices]
        train_data = floored_data[train_indices]
        train_seed = train_data[:, 0, :]
        train_envs = cond_env_ids[train_indices]

        test_data = floored_data[test_indices]
        test_seed = test_data[:, 0, :]
        test_envs = cond_env_ids[test_indices]

        # Model Parameters
        a = nn.Parameter(torch.randn(6, K, device=device) * 0.5 + 2.0)
        theta_raw = nn.Parameter(torch.randn(N, K, device=device) * 0.1)
        w_raw = nn.Parameter(torch.randn(K, K, device=device) * 0.1)
        alpha_raw = nn.Parameter(torch.randn(N, device=device) * 0.05 - 1.0)
        b0_raw = nn.Parameter(torch.tensor(0.0, device=device))

        opt = torch.optim.Adam([a, theta_raw, w_raw, alpha_raw, b0_raw], lr=lr)

        best_loss = float('inf')
        best_params = None

        for ep in range(1, epochs + 1):
            opt.zero_grad()
            Theta = F.softplus(theta_raw)
            W = F.softplus(w_raw)
            alpha = F.softplus(alpha_raw)
            R0_table = F.softplus(a)
            B0 = F.softplus(b0_raw) + 0.1

            N_curr = train_seed * B0
            R0 = R0_table[train_envs]
            loss = 0.0
            tf_ratio = max(0.0, 1.0 - (ep / (epochs * 0.85)))

            for p in range(1, 6):
                N0 = torch.clamp(N_curr / 100.0, min=0.0)
                S0 = torch.cat([N0, R0], dim=-1)
                S_fin = ode_step(S0, Theta, W, alpha, N)
                N_final = torch.clamp(S_fin[..., :N], min=0.0)
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
            nn.utils.clip_grad_norm_([a, theta_raw, w_raw, alpha_raw, b0_raw], 5.0)
            opt.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = (a.detach().clone(), theta_raw.detach().clone(), w_raw.detach().clone(), alpha_raw.detach().clone(), b0_raw.detach().clone())

        # Evaluate Out-of-Sample on Test Fold
        with torch.no_grad():
            b_a, b_th, b_w, b_al, b_b0 = best_params
            Theta = F.softplus(b_th)
            W = F.softplus(b_w)
            alpha = F.softplus(b_al)
            R0_table = F.softplus(b_a)
            B0 = F.softplus(b_b0) + 0.1

            N_curr = test_seed * B0
            R0 = R0_table[test_envs]
            test_preds_list = []
            kl_list = []

            for p in range(1, 6):
                N0 = torch.clamp(N_curr / 100.0, min=0.0)
                S0 = torch.cat([N0, R0], dim=-1)
                S_fin = ode_step(S0, Theta, W, alpha, N)
                N_final = torch.clamp(S_fin[..., :N], min=0.0)
                tot_bio = N_final.sum(dim=-1, keepdim=True) + 1e-12
                x_pred = N_final / tot_bio
                test_preds_list.append(x_pred)
                N_curr = N_final

                target = test_data[:, p, :]
                kl_p = (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean().item()
                kl_list.append(kl_p)

            pred_all = torch.stack(test_preds_list, dim=1).cpu().numpy() # (2, 5, 81)
            true_all = test_data[:, 1:].cpu().numpy() # (2, 5, 81)

            ss_res = np.sum((true_all - pred_all) ** 2)
            ss_tot = np.sum((true_all - np.mean(true_all)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot)
            r_val, _ = pearsonr(true_all.flatten(), pred_all.flatten())

            fold_kls.append(float(np.mean(kl_list)))
            fold_r2s.append(r2)
            fold_pearsons.append(float(r_val))

            # Store T6 prediction for the 5 bar condition in this fold
            target_preds[fold_name] = float(pred_all[0, -1, target_idx] * 100)

    return fold_kls, fold_r2s, fold_pearsons, target_preds

# Grid definition
K_list = [3, 4, 5, 6]
lr_list = [0.01, 0.015, 0.02]
l2_alpha_list = [0.01, 0.05, 0.15, 0.30]
epochs = 300

total_combos = len(K_list) * len(lr_list) * len(l2_alpha_list)
print(f"Total configurations to evaluate: {total_combos}")
print(f"Folds per configuration: 6")
print(f"Total individual fold runs: {total_combos * 6}\n")

# Resume check
done_keys = set()
if os.path.exists(csv_path):
    try:
        existing_df = pd.read_csv(csv_path)
        for _, row in existing_df.iterrows():
            done_keys.add((int(row['K']), float(row['lr']), float(row['l2_alpha'])))
        print(f"Resuming search: {len(done_keys)} configurations already completed.")
    except Exception as e:
        print(f"Could not load existing CSV: {e}")

combo_idx = 0
start_time_all = time.time()

for K in K_list:
    for lr in lr_list:
        for l2_alpha in l2_alpha_list:
            combo_idx += 1
            if (K, lr, l2_alpha) in done_keys:
                print(f"[{combo_idx}/{total_combos}] Skipping already finished: K={K}, lr={lr}, l2_alpha={l2_alpha}")
                continue

            t_start = time.time()
            print(f"[{combo_idx}/{total_combos}] Running K={K}, lr={lr}, l2_alpha={l2_alpha}, epochs={epochs} ...", flush=True)

            fold_kls, fold_r2s, fold_pearsons, target_preds = evaluate_fold_cv(K, lr, l2_alpha, epochs=epochs)
            t_elapsed = time.time() - t_start

            mean_kl = float(np.mean(fold_kls))
            std_kl = float(np.std(fold_kls))
            mean_r2 = float(np.mean(fold_r2s))
            mean_pearson = float(np.mean(fold_pearsons))

            # Target predictions
            p_brrs_noadd = target_preds.get('BRRS_NoAdd', np.nan)
            p_brrs_meoh  = target_preds.get('BRRS_MeOH', np.nan)
            p_brrs_tma   = target_preds.get('BRRS_TMA', np.nan)
            p_osu_noadd  = target_preds.get('OSU_NoAdd', np.nan)
            p_osu_meoh   = target_preds.get('OSU_MeOH', np.nan)
            p_osu_tma    = target_preds.get('OSU_TMA', np.nan)

            contrast_brrs = p_brrs_tma / (p_brrs_meoh + 1e-6)
            contrast_osu  = p_osu_meoh / (p_osu_tma + 1e-6)

            # ETA calculation
            elapsed_total = time.time() - start_time_all
            combos_left = total_combos - combo_idx
            time_per_combo = elapsed_total / max(1, (combo_idx - len(done_keys)))
            eta_mins = (combos_left * time_per_combo) / 60.0

            print(f"  -> Done in {t_elapsed:.1f}s | Mean Test KL: {mean_kl:.4f} | R2: {mean_r2:.4f} | Pearson: {mean_pearson:.4f} | ETA: {eta_mins:.1f} min", flush=True)
            print(f"     BRRS TMA/MeOH: {p_brrs_tma:.1f}% vs {p_brrs_meoh:.1f}% (Ratio {contrast_brrs:.1f}x) | OSU MeOH/TMA: {p_osu_meoh:.1f}% vs {p_osu_tma:.1f}% (Ratio {contrast_osu:.1f}x)", flush=True)

            # Record
            record = {
                'combo_id': combo_idx,
                'K': K,
                'lr': lr,
                'l2_alpha': l2_alpha,
                'epochs': epochs,
                'mean_test_kl': mean_kl,
                'std_test_kl': std_kl,
                'mean_test_r2': mean_r2,
                'mean_test_pearson': mean_pearson,
                'fold0_brrs_noadd_kl': fold_kls[0],
                'fold1_brrs_meoh_kl': fold_kls[1],
                'fold2_brrs_tma_kl': fold_kls[2],
                'fold3_osu_noadd_kl': fold_kls[3],
                'fold4_osu_meoh_kl': fold_kls[4],
                'fold5_osu_tma_kl': fold_kls[5],
                'pred_brrs_noadd_t6': p_brrs_noadd,
                'pred_brrs_meoh_t6': p_brrs_meoh,
                'pred_brrs_tma_t6': p_brrs_tma,
                'pred_osu_noadd_t6': p_osu_noadd,
                'pred_osu_meoh_t6': p_osu_meoh,
                'pred_osu_tma_t6': p_osu_tma,
                'brrs_contrast_ratio': contrast_brrs,
                'osu_contrast_ratio': contrast_osu,
                'runtime_sec': t_elapsed
            }

            # Save to CSV
            row_df = pd.DataFrame([record])
            if not os.path.exists(csv_path):
                row_df.to_csv(csv_path, index=False)
            else:
                row_df.to_csv(csv_path, mode='a', header=False, index=False)

print("\n=================================================================")
print("  EXHAUSTIVE 6-FOLD CV GRID SEARCH COMPLETE!")
print(f"  Results saved to: {csv_path}")
print("=================================================================")

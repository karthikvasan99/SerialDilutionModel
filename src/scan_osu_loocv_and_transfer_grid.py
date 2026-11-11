"""
Comprehensive Hyperparameter Optimization for Cross-Host Transfer and OSU Generalization.

Scans:
  - Learning rate lr in [0.013, 0.015, 0.017]
  - Teacher forcing decay curriculum tf_decay in [0.60, 0.75, 0.85]
  - Self-limitation penalty l2_alpha in [0.16, 0.20, 0.24]
  - Model: Pure Physical Consumer-Resource Neural ODE (K=3, unpenalized Theta and W)
  - Training Set: Pure Cow OSU (all 6 conditions, zero BRRS data)
  - Target: Maximize BRRS Enrichment Ratio (TMA / MeOH), keep MeOH <= 1.0%,
            and maximize internal OSU fit & LOOCV cross-validation.
"""

import os
import sys
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing Hyperparameter Grid Search on Device: {device}")

# ─── 1. Load Data & Filter to 81 Shared Species ───
data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
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
print(f"Target Species (idx {target_idx}): {target_name}")

class PhysicalODELearnableB0(nn.Module):
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

pairs = {6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10}

def train_and_eval_full(K=3, lr=0.015, l2_alpha=0.20, tf_decay=0.75, epochs=250, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PhysicalODELearnableB0(num_species, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            kl = (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean()
            loss = loss + kl
            
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

    # 1. OSU Fit Metrics (T1 -> T6)
    with torch.no_grad():
        Theta, W, alpha, R0_table, B0 = model.get_params()
        eval_preds = [train_seed]
        N_curr = train_seed * B0
        R0 = R0_table[train_envs]
        for p in range(1, num_passages):
            N0 = torch.clamp(N_curr / 100.0, min=0.0)
            S0 = torch.cat([N0, R0], dim=-1)
            t_span = torch.tensor([0.0, 1.0], device=device)
            S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, t_span, method='euler', options={'step_size': 0.1})
            N_final = torch.clamp(S_t[-1, :, :num_species], min=0.0)
            x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
            eval_preds.append(x_pred)
            N_curr = N_final

    osu_pred = torch.stack(eval_preds, dim=1).cpu().numpy()
    osu_true = train_data.cpu().numpy()
    ss_res_osu = np.sum((osu_true - osu_pred) ** 2)
    ss_tot_osu = np.sum((osu_true - np.mean(osu_true)) ** 2)
    osu_r2 = float(1.0 - ss_res_osu / ss_tot_osu)
    osu_kl = float(best_loss)

    # Methanogen R2 in OSU
    target_true_osu = osu_true[:, :, target_idx].flatten()
    target_pred_osu = osu_pred[:, :, target_idx].flatten()
    target_r2_osu = float(1.0 - np.sum((target_true_osu - target_pred_osu)**2) / np.sum((target_true_osu - np.mean(target_true_osu))**2))

    # 2. Held-Out BRRS 5 Bar Evaluation (Conditions 0, 2, 4)
    test_indices = [0, 2, 4]
    test_seed = seed_t1[test_indices]
    test_envs = cond_env_ids[test_indices]
    test_data = data_81[test_indices]
    true_test_np = test_data.cpu().numpy()

    with torch.no_grad():
        eval_brrs = [test_seed]
        N_curr = test_seed * B0
        R0 = R0_table[test_envs]
        for p in range(1, num_passages):
            N0 = torch.clamp(N_curr / 100.0, min=0.0)
            S0 = torch.cat([N0, R0], dim=-1)
            t_span = torch.tensor([0.0, 1.0], device=device)
            S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, t_span, method='euler', options={'step_size': 0.1})
            N_final = torch.clamp(S_t[-1, :, :num_species], min=0.0)
            x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
            eval_brrs.append(x_pred)
            N_curr = N_final

    pred_test_np = torch.stack(eval_brrs, dim=1).cpu().numpy()
    ss_res_brrs = np.sum((true_test_np - pred_test_np) ** 2)
    ss_tot_brrs = np.sum((true_test_np - np.mean(true_test_np)) ** 2)
    brrs_r2 = float(1.0 - ss_res_brrs / ss_tot_brrs)

    meoh_pred = float(pred_test_np[1, -1, target_idx] * 100)
    tma_pred = float(pred_test_np[2, -1, target_idx] * 100)
    ratio = tma_pred / (meoh_pred + 1e-6)

    return osu_r2, osu_kl, target_r2_osu, meoh_pred, tma_pred, ratio, brrs_r2, B0.item()

def eval_6fold_osu_loocv(K=3, lr=0.015, l2_alpha=0.20, tf_decay=0.75, epochs=250, seed=42):
    osu_indices = [6, 7, 8, 9, 10, 11]
    val_r2s = []
    val_kls = []

    for hold_out in osu_indices:
        partner = pairs[hold_out]
        train_indices = [i for i in osu_indices if i != hold_out]

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PhysicalODELearnableB0(num_species, K).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
                kl = (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean()
                loss = loss + kl
                
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

        val_data = data_81[[hold_out]]
        val_seed = seed_t1[[hold_out]]
        h_env = cond_env_ids[hold_out].item()
        p_env = cond_env_ids[partner].item()

        with torch.no_grad():
            Theta, W, alpha, R0_table, B0 = model.get_params()
            norm_5b = torch.norm(R0_table[0], p=1)
            norm_30b = torch.norm(R0_table[1], p=1)
            scalar_scale = norm_5b / (norm_30b + 1e-6)
            
            if h_env in [0, 2, 4]:
                r_pred = R0_table[p_env] * scalar_scale
            else:
                r_pred = R0_table[p_env] / (scalar_scale + 1e-6)
                
            eval_preds = [val_seed]
            N_curr = val_seed * B0
            R0 = r_pred.unsqueeze(0)
            val_kl = 0.0
            for p in range(1, num_passages):
                N0 = torch.clamp(N_curr / 100.0, min=0.0)
                S0 = torch.cat([N0, R0], dim=-1)
                t_span = torch.tensor([0.0, 1.0], device=device)
                S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, t_span, method='euler', options={'step_size': 0.1})
                N_final = torch.clamp(S_t[-1, :, :num_species], min=0.0)
                x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
                eval_preds.append(x_pred)
                N_curr = N_final
                target = val_data[:, p, :]
                val_kl += (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean().item()

        pred_val_np = torch.stack(eval_preds, dim=1).cpu().numpy()
        true_val_np = val_data.cpu().numpy()
        ss_res = np.sum((true_val_np - pred_val_np) ** 2)
        ss_tot = np.sum((true_val_np - np.mean(true_val_np)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot)
        val_r2s.append(r2)
        val_kls.append(val_kl / (num_passages - 1))

    return float(np.mean(val_r2s)), float(np.mean(val_kls))

# ─── Execute Grid Search ───
lr_grid = [0.013, 0.015, 0.017]
tf_decay_grid = [0.60, 0.75, 0.85]
l2_alpha_grid = [0.16, 0.20, 0.24]

total_configs = len(lr_grid) * len(tf_decay_grid) * len(l2_alpha_grid)
print(f"\nStarting Grid Search over {total_configs} Hyperparameter Configurations...")
print(f"Parameters: lr in {lr_grid}, tf_decay in {tf_decay_grid}, l2_alpha in {l2_alpha_grid}\n")

results = []
idx = 0

print("idx | lr    | tf_dec | l2_a | OSU R2  | OSU KL  | BRRS MeOH | BRRS TMA | Ratio   | BRRS R2 | B0")
print("-" * 92)

for lr in lr_grid:
    for tf_d in tf_decay_grid:
        for l2_a in l2_alpha_grid:
            idx += 1
            t0 = time.time()
            osu_r2, osu_kl, meth_r2, meoh, tma, ratio, brrs_r2, b0 = train_and_eval_full(
                K=3, lr=lr, l2_alpha=l2_a, tf_decay=tf_d, epochs=250
            )
            elapsed = time.time() - t0
            print(f"{idx:2d}  | {lr:0.3f} | {tf_d:0.2f}   | {l2_a:0.2f} | {osu_r2:6.4f}  | {osu_kl:6.4f}  | {meoh:5.2f}%    | {tma:5.2f}%    | {ratio:6.2f}x | {brrs_r2:6.4f}  | {b0:.2f}")
            results.append({
                'idx': idx,
                'K': 3,
                'lr': lr,
                'tf_decay': tf_d,
                'l2_alpha': l2_a,
                'osu_r2': osu_r2,
                'osu_kl': osu_kl,
                'osu_target_r2': meth_r2,
                'brrs_meoh': meoh,
                'brrs_tma': tma,
                'enrichment_ratio': ratio,
                'brrs_r2': brrs_r2,
                'b0': b0
            })

# Sort by enrichment ratio descending
sorted_by_ratio = sorted(results, key=lambda x: x['enrichment_ratio'], reverse=True)

# Select top 5 candidates for 6-Fold OSU LOOCV validation
print("\n" + "="*80)
print("TOP 5 CONFIGURATIONS BY ENRICHMENT RATIO (TMA / MeOH):")
print("="*80)
for rank, c in enumerate(sorted_by_ratio[:5], 1):
    print(f"Rank {rank}: lr={c['lr']}, tf_decay={c['tf_decay']}, l2_alpha={c['l2_alpha']} -> Ratio={c['enrichment_ratio']:.2f}x (MeOH={c['brrs_meoh']:.2f}%, TMA={c['brrs_tma']:.2f}%), BRRS R2={c['brrs_r2']:.4f}, OSU R2={c['osu_r2']:.4f}")

print("\n" + "="*80)
print("RUNNING STRICT 6-FOLD OSU LOOCV ON TOP CANDIDATES...")
print("="*80)

for rank, c in enumerate(sorted_by_ratio[:5], 1):
    t0 = time.time()
    loocv_r2, loocv_kl = eval_6fold_osu_loocv(
        K=3, lr=c['lr'], l2_alpha=c['l2_alpha'], tf_decay=c['tf_decay'], epochs=250
    )
    elapsed = time.time() - t0
    c['osu_loocv_r2'] = loocv_r2
    c['osu_loocv_kl'] = loocv_kl
    print(f"Candidate {rank} (lr={c['lr']}, tf_decay={c['tf_decay']}, l2_a={c['l2_alpha']}):")
    print(f"  -> OSU LOOCV R^2: {loocv_r2:.4f} | OSU LOOCV KL: {loocv_kl:.4f} (Completed in {elapsed:.1f}s)")

# Save full results to CSV
out_csv = os.path.join(PROJECT_ROOT, 'archive', 'intermediate_results', 'Regime2_OSU_LOOCV_Grid_Results.csv')
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'idx', 'K', 'lr', 'tf_decay', 'l2_alpha', 'osu_r2', 'osu_kl', 'osu_target_r2',
        'brrs_meoh', 'brrs_tma', 'enrichment_ratio', 'brrs_r2', 'b0', 'osu_loocv_r2', 'osu_loocv_kl'
    ])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\nSaved full grid search results to: {out_csv}")
print("Optimization Complete.")

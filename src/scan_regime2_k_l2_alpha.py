import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchdiffeq import odeint

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing Regime 2 Systematic Hyperparameter Scan on Device: {device}")

# ─── 1. Load Data & Filter to 81 Mutually Shared Species ───
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
print(f"Target Species: {target_name} (idx {target_idx})")

# ─── 2. Model Definition with Learnable B0 ───
class PhysicalODELearnableB0(nn.Module):
    def __init__(self, N, K):
        super().__init__()
        self.N, self.K = N, K
        self.a = nn.Parameter(torch.randn(6, K) * 0.5 + 2.0)
        self.theta_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.05 - 1.0)
        self.b0_raw = nn.Parameter(torch.tensor(0.0)) # Learnable B0!

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

# ─── 3. Grid Search Space ───
K_VALUES = [2, 3, 4]
LR = 0.01
L2_THETA_W_VALUES = [0.0, 1e-4, 1e-3, 1e-2, 0.1]
L2_ALPHA_VALUES = [0.0, 1e-4, 1e-3, 1e-2, 0.1]
EPOCHS = 250

total_combos = len(K_VALUES) * len(L2_THETA_W_VALUES) * len(L2_ALPHA_VALUES)
print(f"Total Grid Combinations: {total_combos} (K: {K_VALUES}, LR: {LR}, L2_TW: {L2_THETA_W_VALUES}, L2_Alpha: {L2_ALPHA_VALUES})")

train_indices = [6, 7, 8, 9, 10, 11] # Pure OSU
test_indices = [0, 2, 4]             # BRRS 5 bar (Control 5b, MeOH 5b, TMA 5b)

train_data = data_81[train_indices]
train_seed = seed_t1[train_indices]
train_envs = cond_env_ids[train_indices]

test_data = data_81[test_indices]
test_seed = seed_t1[test_indices]
test_envs = cond_env_ids[test_indices]
true_test_np = test_data.cpu().numpy()

results = []
t0_start = time.time()
count = 0

for K in K_VALUES:
    for l2_tw in L2_THETA_W_VALUES:
        for l2_alpha in L2_ALPHA_VALUES:
            count += 1
            t_run = time.time()
            
            torch.manual_seed(42)
            np.random.seed(42)
            
            model = PhysicalODELearnableB0(num_species, K).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            
            best_train_loss = float('inf')
            best_weights = None
            
            for epoch in range(1, EPOCHS + 1):
                optimizer.zero_grad()
                Theta, W, alpha, R0_table, B0 = model.get_params()
                N_curr = train_seed * B0
                R0 = R0_table[train_envs]
                loss = 0.0
                
                # Scheduled teacher forcing curriculum
                tf_ratio = max(0.0, 1.0 - (epoch / (EPOCHS * 0.75)))
                
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
                        
                # Separate L2 regularizations
                if l2_tw > 0:
                    loss = loss + l2_tw * (torch.sum(Theta ** 2) + torch.sum(W ** 2))
                if l2_alpha > 0:
                    loss = loss + l2_alpha * torch.sum(alpha ** 2)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                
                if loss.item() < best_train_loss:
                    best_train_loss = loss.item()
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    
            # Evaluate best model zero-shot on held-out BRRS 5 bar
            model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
            model.eval()
            
            with torch.no_grad():
                Theta, W, alpha, R0_table, B0 = model.get_params()
                learned_b0 = B0.item()
                mean_alpha = alpha.mean().item()
                
                eval_preds = [test_seed]
                N_curr = test_seed * B0
                R0 = R0_table[test_envs]
                
                for p in range(1, num_passages):
                    N0 = torch.clamp(N_curr / 100.0, min=0.0)
                    S0 = torch.cat([N0, R0], dim=-1)
                    t_span = torch.tensor([0.0, 1.0], device=device)
                    S_t = odeint(lambda t, S: model.ode_deriv(t, S, Theta, W, alpha), S0, t_span, method='euler', options={'step_size': 0.1})
                    N_final = torch.clamp(S_t[-1, :, :num_species], min=0.0)
                    x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-12)
                    eval_preds.append(x_pred)
                    N_curr = N_final
                    
            preds_tensor = torch.stack(eval_preds, dim=1)
            pred_test_np = preds_tensor.cpu().numpy()
            
            # Metrics
            ss_res = np.sum((true_test_np - pred_test_np) ** 2)
            ss_tot = np.sum((true_test_np - np.mean(true_test_np)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot)
            r = float(np.corrcoef(true_test_np.flatten(), pred_test_np.flatten())[0, 1])
            
            # Target Methanomicrobium sp. (idx 0)
            # test[1] = BRRS +MeOH 5b (observed: 0.00%)
            # test[2] = BRRS +TMA 5b  (observed: 23.15%)
            meoh_pred = float(pred_test_np[1, -1, target_idx] * 100)
            tma_pred = float(pred_test_np[2, -1, target_idx] * 100)
            ratio = tma_pred / (meoh_pred + 1e-6)
            
            dt = time.time() - t_run
            print(f"[{count:2d}/{total_combos}] K={K} | L2_TW={l2_tw:.4f} | L2_A={l2_alpha:.4f} -> MeOH: {meoh_pred:5.2f}% | TMA: {tma_pred:5.2f}% | Ratio: {ratio:5.1f}x | R2: {r2:6.4f} | B0: {learned_b0:.3f} ({dt:.1f}s)")
            
            results.append({
                'K': K,
                'lr': LR,
                'l2_theta_w': l2_tw,
                'l2_alpha': l2_alpha,
                'best_train_loss': best_train_loss,
                'learned_b0': learned_b0,
                'mean_alpha': mean_alpha,
                'meoh_pred_t6': meoh_pred,
                'tma_pred_t6': tma_pred,
                'ratio_tma_to_meoh': ratio,
                'r2_brrs_5b': r2,
                'pearson_r_brrs_5b': r
            })

df = pd.DataFrame(results)
csv_out = os.path.join(PROJECT_ROOT, 'archive', 'intermediate_results', 'Regime2_GridSearch_K_L2Alpha_Results.csv')
os.makedirs(os.path.dirname(csv_out), exist_ok=True)
df.to_csv(csv_out, index=False)
print(f"\nSaved complete results ({len(df)} rows) to: {csv_out}")

# Top 10 by TMA enrichment ratio (and suppressing MeOH)
print("\n" + "="*80)
print("TOP 10 CONFIGURATIONS (Sorted by TMA/MeOH Enrichment Ratio with MeOH <= 3.0%):")
print("="*80)
valid_df = df[df['meoh_pred_t6'] <= 3.0].sort_values(by='tma_pred_t6', ascending=False)
print(valid_df[['K', 'l2_theta_w', 'l2_alpha', 'meoh_pred_t6', 'tma_pred_t6', 'ratio_tma_to_meoh', 'r2_brrs_5b', 'learned_b0']].head(10).to_string(index=False))

"""
Cache all numerical data needed for every poster plot to disk as .npz files.
Each figure has its own cache file under Data/Poster_Cache_*.npz.
Re-rendering any figure then requires only a cheap np.load() — no retraining.

Panels:
  1  (Ophelia)  → Data/Poster_Cache_Ophelia.npz
  2  (Pollard)  → Data/Poster_Cache_Pollard.npz
  3  (German)   → Data/Poster_Cache_German.npz   (NOTE: stored per-condition)
  6  (Gore)     → Data/Gore_Comm5_Oscillation_Cache.npz  (already exists)
"""
import sys, os
sys.path.insert(0, '.')
import torch, torch.nn as nn, torch.nn.functional as F
from torchdiffeq import odeint
from scipy.io import loadmat
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1: OPHELIA (Venturelli 2018)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("CACHING PANEL 1: OPHELIA")
print("=" * 70)

train_mat = loadmat('Data/train_data.mat')['train_data']
train_valid = [np.nan_to_num(train_mat[i], nan=0.0)
               for i in range(train_mat.shape[0])
               if not np.isnan(train_mat[i][0]).any() and np.sum(train_mat[i][0]) > 1e-4]
train_tensor = torch.tensor(np.array(train_valid), dtype=torch.float32, device=device)

test_mat = loadmat('Data/test_data.mat')['test_data']
test_valid = [np.nan_to_num(test_mat[i], nan=0.0)
              for i in range(test_mat.shape[0])
              if not np.isnan(test_mat[i][0]).any() and np.sum(test_mat[i][0]) > 1e-4]
val_tensor = torch.tensor(np.array(test_valid), dtype=torch.float32, device=device)

class OpheliaODEK(nn.Module):
    def __init__(self, K=7):
        super().__init__()
        self.N, self.K = 12, K
        self.a = nn.Parameter(torch.randn(1, K))
        self.theta_raw = nn.Parameter(torch.randn(12, K))
        self.w_raw = nn.Parameter(torch.randn(K, K))
        self.alpha_raw = nn.Parameter(torch.randn(12))
    def get_params(self):
        return F.softplus(self.theta_raw), F.softplus(self.w_raw), F.softplus(self.alpha_raw), F.softplus(self.a)
    def ode_deriv(self, t, S, Theta, W, alpha):
        N_t = torch.clamp(S[..., :12], min=0.0)
        R_t = torch.clamp(S[..., 12:], min=0.0)
        return torch.cat([
            N_t * torch.matmul(R_t, Theta.t()) - alpha * (N_t**2),
            - R_t * torch.matmul(torch.matmul(N_t, Theta), W.t())
        ], dim=-1)

model_oph = OpheliaODEK(K=7).to(device)
ckpt = torch.load('Data/Ophelia_Physical_ODE_12Strains.pth', weights_only=False)
model_oph.load_state_dict(ckpt['model_state_dict'])
model_oph.eval()

t_span = torch.linspace(0.0, 5.0, 6, device=device)
with torch.no_grad():
    Theta, W, alpha, R0_table = model_oph.get_params()
    # Train
    N0_train = train_tensor[:, 0, :]
    R0_batch_tr = R0_table[0].unsqueeze(0).expand(N0_train.size(0), -1)
    S0_train = torch.cat([N0_train, R0_batch_tr], dim=-1)
    S_t_train = odeint(lambda t, S: model_oph.ode_deriv(t, S, Theta, W, alpha),
                       S0_train, t_span, method='euler', options={'step_size': 0.1})
    train_pred = torch.clamp(S_t_train[..., :12].permute(1, 0, 2), min=0.0)
    # Test
    N0_val = val_tensor[:, 0, :]
    R0_batch_val = R0_table[0].unsqueeze(0).expand(N0_val.size(0), -1)
    S0_val = torch.cat([N0_val, R0_batch_val], dim=-1)
    S_t_val = odeint(lambda t, S: model_oph.ode_deriv(t, S, Theta, W, alpha),
                     S0_val, t_span, method='euler', options={'step_size': 0.1})
    val_pred = torch.clamp(S_t_val[..., :12].permute(1, 0, 2), min=0.0)

np.savez('Data/Poster_Cache_Ophelia.npz',
    train_true = train_tensor.cpu().numpy(),   # (N_tr, 6, 12)
    train_pred = train_pred.cpu().numpy(),     # (N_tr, 6, 12)
    val_true   = val_tensor.cpu().numpy(),     # (N_val, 6, 12)
    val_pred   = val_pred.cpu().numpy(),       # (N_val, 6, 12)
    Theta = Theta.cpu().numpy(),
    W = W.cpu().numpy(),
    alpha = alpha.cpu().numpy(),
)
print("  Saved: Data/Poster_Cache_Ophelia.npz")
for k, v in np.load('Data/Poster_Cache_Ophelia.npz').items():
    print(f"    {k:12s}: {v.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2: POLLARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CACHING PANEL 2: POLLARD")
print("=" * 70)

from src.Pollard_Train_Physical_ODE import PurePhysicalPollardODE, prepare_data

data_path_p = 'Data/SupplementaryDataS2.csv'
trajectories_p, conditions_p, unique_envs_p, num_species_p = prepare_data(data_path_p)

model_full = PurePhysicalPollardODE(N=num_species_p, K=5, num_conditions=len(unique_envs_p)).to(device)
ckpt_full = torch.load('Data/Pollard_Physical_ODE_50Strains.pth', weights_only=False)
model_full.load_state_dict(ckpt_full['model_state_dict'])
model_full.eval()

class NoCrossFeedingPollardODE(nn.Module):
    def __init__(self, N, K, num_conditions=5):
        super().__init__()
        self.N, self.K = N, K
        self.r0_raw = nn.Parameter(torch.randn(num_conditions, K))
        self.theta_raw = nn.Parameter(torch.randn(N, K))
        self.alpha_raw = nn.Parameter(torch.randn(N))
        self.b0_raw = nn.Parameter(torch.tensor(0.0))
    def get_params(self):
        return F.softplus(self.theta_raw), F.softplus(self.alpha_raw), F.softplus(self.r0_raw), F.softplus(self.b0_raw) + 0.1
    def ode_deriv(self, t, S, Theta, alpha):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        dN = N_t * torch.matmul(R_t, Theta.t()) - alpha * (N_t**2)
        dR = - R_t * torch.matmul(N_t, Theta)
        return torch.cat([dN, dR], dim=-1)

model_no_w = NoCrossFeedingPollardODE(num_species_p, 5, 5).to(device)
ckpt_now = torch.load('Data/Pollard_Physical_ODE_50Strains_NoW.pth', weights_only=False)
model_no_w.load_state_dict(ckpt_now['model_state_dict'])
model_no_w.eval()

delta_t = 0.10
tau_steps = int(1.0 / delta_t) + 1
t_span_p = torch.linspace(0.0, 1.0, tau_steps, device=device)
np.random.seed(42); torch.manual_seed(42)

with torch.no_grad():
    Theta_p, W_p, alpha_p, R0_table_p, B0_p = model_full.get_params()
    Theta_np = Theta_p.cpu().numpy()
    W_np = W_p.cpu().numpy()

    def eval_full_p(Th, W_m, al, R0_tab):
        r2s, true_all, pred_all = [], [], []
        for c in range(5):
            mask = (conditions_p == c)
            t_c = trajectories_p[mask, 1:, :].cpu().numpy().flatten()
            R0_sub = R0_tab[c:c+1].expand(mask.sum(), -1)
            N_curr = trajectories_p[mask, 0, :] * B0_p
            preds_c = []
            for p in range(1, 6):
                N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
                S0 = torch.cat([N0, R0_sub], dim=-1)
                S_t = odeint(lambda t, S: model_full.ode_deriv(t, S, Th, W_m, al),
                             S0, t_span_p, method='euler', options={'step_size': delta_t})
                N_final = torch.clamp(S_t[-1, :, :num_species_p], min=0.0)
                N_curr = N_final
                preds_c.append(N_final)
            pt = torch.stack(preds_c, dim=1)
            pred_rel = (pt / (pt.sum(-1, keepdim=True) + 1e-8)).cpu().numpy().flatten()
            var_t = np.sum((t_c - np.mean(t_c))**2)
            r2 = 1.0 - np.sum((t_c - pred_rel)**2) / var_t if var_t > 0 else 0.0
            r2s.append(r2); true_all.append(t_c); pred_all.append(pred_rel)
        return np.array(r2s), true_all, pred_all

    full_r2, full_true, full_pred = eval_full_p(Theta_p, W_p, alpha_p, R0_table_p)
    print(f"  Full model R2 per env: {full_r2}")

    # No-W model
    Theta_nw, alpha_nw, R0_nw, B0_nw = model_no_w.get_params()
    r2_nw = []
    for c in range(5):
        mask = (conditions_p == c)
        t_c = trajectories_p[mask, 1:, :].cpu().numpy().flatten()
        R0_sub = R0_nw[c:c+1].expand(mask.sum(), -1)
        N_curr = trajectories_p[mask, 0, :] * B0_nw
        preds_c = []
        for p in range(1, 6):
            N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
            S0 = torch.cat([N0, R0_sub], dim=-1)
            S_t = odeint(lambda t, S: model_no_w.ode_deriv(t, S, Theta_nw, alpha_nw),
                         S0, t_span_p, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :num_species_p], min=0.0)
            N_curr = N_final
            preds_c.append(N_final)
        pt = torch.stack(preds_c, dim=1)
        pred_rel = (pt / (pt.sum(-1, keepdim=True) + 1e-8)).cpu().numpy().flatten()
        var_t = np.sum((t_c - np.mean(t_c))**2)
        r2_nw.append(1.0 - np.sum((t_c - pred_rel)**2) / var_t if var_t > 0 else 0.0)
    no_w_r2 = np.array(r2_nw)

    # Persistence null
    persist_r2 = []
    for c in range(5):
        mask = (conditions_p == c)
        true_v = trajectories_p[mask, 1:, :].cpu().numpy().flatten()
        pers_v = trajectories_p[mask, 0:1, :].expand(-1, 5, -1).cpu().numpy().flatten()
        var_t = np.sum((true_v - np.mean(true_v))**2)
        persist_r2.append(1.0 - np.sum((true_v - pers_v)**2) / var_t if var_t > 0 else 0.0)
    persist_r2 = np.array(persist_r2)

    # Shuffled environment (25 reps)
    shuff_a_runs = []
    for _ in range(25):
        tmp = []
        for c in range(5):
            mask = (conditions_p == c)
            t_c = trajectories_p[mask, 1:, :].cpu().numpy().flatten()
            perm = torch.randperm(5)
            R0_sub = R0_table_p[c:c+1, perm].expand(mask.sum(), -1)
            N_curr = trajectories_p[mask, 0, :] * B0_p
            preds_c = []
            for p in range(1, 6):
                N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
                S0 = torch.cat([N0, R0_sub], dim=-1)
                S_t = odeint(lambda t, S: model_full.ode_deriv(t, S, Theta_p, W_p, alpha_p),
                             S0, t_span_p, method='euler', options={'step_size': delta_t})
                N_final = torch.clamp(S_t[-1, :, :num_species_p], min=0.0)
                N_curr = N_final
                preds_c.append(N_final)
            pt = torch.stack(preds_c, dim=1)
            pred_rel = (pt / (pt.sum(-1, keepdim=True) + 1e-8)).cpu().numpy().flatten()
            var_t = np.sum((t_c - np.mean(t_c))**2)
            tmp.append(1.0 - np.sum((t_c - pred_rel)**2) / var_t if var_t > 0 else 0.0)
        shuff_a_runs.append(tmp)
    shuff_a_r2 = np.mean(shuff_a_runs, axis=0)

    # Shuffled Theta (25 reps)
    shuff_th_runs = []
    for _ in range(25):
        perm = torch.randperm(num_species_p)
        r2_sh, _, _ = eval_full_p(Theta_p[perm], W_p, alpha_p, R0_table_p)
        shuff_th_runs.append(r2_sh)
    shuff_th_r2 = np.mean(shuff_th_runs, axis=0)

np.savez('Data/Poster_Cache_Pollard.npz',
    Theta = Theta_np,                     # (50, 5)
    W = W_np,                             # (5, 5)
    full_r2 = full_r2,                    # (5,)
    no_w_r2 = no_w_r2,                   # (5,)
    persist_r2 = persist_r2,             # (5,)
    shuff_a_r2 = shuff_a_r2,             # (5,)
    shuff_th_r2 = shuff_th_r2,           # (5,)
    env_names = np.array(['Mucin Carrier','Mucin Supern.','No Carrier (Liq.)','Plain Carrier','Plain Supern.']),
)
print("  Saved: Data/Poster_Cache_Pollard.npz")
for k, v in np.load('Data/Poster_Cache_Pollard.npz', allow_pickle=True).items():
    print(f"    {k:14s}: {v.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3: GERMAN BIOREACTOR (prescriptive search)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CACHING PANEL 3: GERMAN")
print("=" * 70)

from src.poster_fig3_prescriptive_search import (
    BRRS_5bar_conditions, OSU_conditions,
    model as german_model,
    obs_by_cond, pred_by_cond,
    sorted_bar_conds, pred_compositions,
    enrichment_actual, enrichment_pred,
    strain_names,
)
np.savez('Data/Poster_Cache_German.npz',
    obs_by_cond = np.array(list(obs_by_cond.values()), dtype=object),
    pred_by_cond = np.array(list(pred_by_cond.values()), dtype=object),
    cond_keys = np.array(list(obs_by_cond.keys())),
    enrichment_actual = np.array(enrichment_actual),
    enrichment_pred   = np.array(enrichment_pred),
    strain_names = np.array(strain_names),
)
print("  Saved: Data/Poster_Cache_German.npz")

print("\nAll poster caches complete!")

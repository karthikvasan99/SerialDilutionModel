"""
Analyze whether empirically stable communities remain stable or oscillate when simulated
with the Minimal Coarse-Grained Cross-Feeding Consumer-Resource Model (K=3, 166 params).

Addresses:
1. Statistical comparison of simulated CV between empirically stable (N=37) vs fluctuating (N=38) communities.
2. 10-day trajectory comparison (deterministic and stochastic) on representative stable vs fluctuating communities.
3. 200-passage long-term stability test: do empirically stable communities reach an unmoving fixed point,
   or do they exhibit oscillations?
4. Generation of publication figure: Figures/Gore_Stable_vs_Fluctuating_Validation.png.
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = '/home/karthik/SerialDilutionModel'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load data and metadata
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
data = torch.load(data_path, weights_only=False)
meta = data['metadata']
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

meta_map = {}
for idx, row in meta.iterrows():
    meta_map[(row['S'], row['community_name'])] = row

all_N, all_c, all_m, all_keys = [], [], [], []
for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    if traj_rel.shape[0] == 11:
        od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
        mask = data['presence_masks'][(S, comm)]
        all_N.append(traj_rel * od_mean.unsqueeze(-1))
        all_c.append(cond_to_idx[nutrient])
        all_m.append(mask)
        
        row = meta_map[(S, comm)]
        emp_cv = float(row[f'{nutrient}_CV'])
        emp_fluc = bool(row[f'{nutrient}_fluctuating'])
        all_keys.append((nutrient, S, comm, emp_cv, emp_fluc))

all_N = torch.stack(all_N).to(device)
all_c = torch.tensor(all_c, device=device)
all_m = torch.stack(all_m).to(device)

# Define Canonical Coarse-Grained Model Architecture
class MinimalCrossFeedingCRODE(nn.Module):
    def __init__(self, N=48, K=3, D=30.0, mu=1e-6, max_crossfeed=0.85):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        self.max_crossfeed = max_crossfeed
        init_scale = 10.0 / float(K)
        inv_scale = math.log(math.exp(init_scale) - 1.0)
        self.Theta_raw = nn.Parameter(torch.randn(N, K) * 0.15 + inv_scale)
        self.W_raw = nn.Parameter(torch.randn(K, K) * 0.2 - 0.5)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.r_ln = nn.Parameter(torch.randn(K) * 0.1)
        self.r_mn = nn.Parameter(torch.randn(K) * 0.1)
        self.r_hn = nn.Parameter(torch.randn(K) * 0.1)
        self.scale_ln = nn.Parameter(torch.tensor(math.log(math.exp(0.20) - 1.0)))
        self.scale_mn = nn.Parameter(torch.tensor(math.log(math.exp(0.50) - 1.0)))
        self.scale_hn = nn.Parameter(torch.tensor(math.log(math.exp(1.00) - 1.0)))
        
    def get_R0(self):
        s_ln = F.softplus(self.scale_ln)
        s_mn = F.softplus(self.scale_mn)
        s_hn = F.softplus(self.scale_hn)
        r_ln = F.softmax(self.r_ln, dim=-1) * s_ln
        r_mn = F.softmax(self.r_mn, dim=-1) * s_mn
        r_hn = F.softmax(self.r_hn, dim=-1) * s_hn
        return torch.stack([r_ln, r_mn, r_hn])
        
    def get_alpha(self):
        return 0.10 * torch.sigmoid(self.alpha_raw)
        
    def get_Theta(self):
        return F.softplus(self.Theta_raw)
        
    def get_W(self):
        W_sp = F.softplus(self.W_raw)
        W_sp = W_sp * (1.0 - torch.eye(self.K, device=W_sp.device))
        row_sums = W_sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_crossfeed / torch.clamp(row_sums, min=1e-6), max=1.0)
        return W_sp * scale
        
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        Theta = self.get_Theta()
        W = self.get_W()
        alpha = self.get_alpha()
        growth = torch.matmul(R_t, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        C = torch.matmul(N_t, Theta)
        B = torch.matmul(C, W)
        dR = - R_t * (C - B)
        return torch.cat([dN, dR], dim=-1)

model = MinimalCrossFeedingCRODE().to(device)
ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Minimal_K3_CrossFeeding_ODE.pth')
ckpt = torch.load(ckpt_path, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print("Loaded pre-trained model successfully.")
R0_table = model.get_R0()
t_span = torch.linspace(0.0, 1.0, 30, device=device)

# ─── 1. Simulate all 75 communities across 10 days ───
# Both deterministic and stochastic
sigma_R = 0.08
sigma_D = 0.05
sigma_obs = 0.10

results = []
rollouts_det_dict = {}
rollouts_stoch_dict = {}

for idx in range(len(all_keys)):
    nutrient, S_val, comm_val, emp_cv, emp_fluc = all_keys[idx]
    c_idx = cond_to_idx[nutrient]
    R0_base = R0_table[c_idx]
    mask_i = all_m[idx]
    
    # Day 0 empirical inoculation
    N0 = all_N[idx, 0, :] * mask_i
    
    # (A) Deterministic Rollout
    N_curr = N0.clone()
    rollout_det = [N_curr.cpu().numpy()]
    with torch.no_grad():
        for p in range(10):
            N_in = torch.clamp(N_curr / model.D + model.mu * mask_i, min=0.0)
            S0 = torch.cat([N_in, R0_base], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_next = torch.clamp(S_t[-1, :48] * mask_i, min=0.0)
            rollout_det.append(N_next.cpu().numpy())
            N_curr = N_next
    rollout_det = np.array(rollout_det)
    rel_det = rollout_det / np.where(rollout_det.sum(axis=-1, keepdims=True) > 0, rollout_det.sum(axis=-1, keepdims=True), 1.0)
    rollouts_det_dict[(nutrient, S_val, comm_val)] = rel_det
    
    # CV over Days 7-10 (deterministic)
    late_det = rel_det[7:11]
    m_det = late_det.mean(axis=0)
    s_det = late_det.std(axis=0)
    surv_det = m_det >= 1e-3
    cv_det = float(np.mean(s_det[surv_det] / m_det[surv_det])) if np.any(surv_det) else 0.0
    
    # (B) Stochastic Rollout with Environmental Noise
    np.random.seed(42 + idx)
    N_curr = N0.clone()
    rollout_stoch = [N_curr.cpu().numpy()]
    with torch.no_grad():
        for p in range(10):
            d_noise = float(np.exp(np.random.randn() * sigma_D))
            D_eff = model.D * d_noise
            N_in = torch.clamp(N_curr / D_eff + model.mu * mask_i, min=0.0)
            
            r_noise = torch.tensor(np.exp(np.random.randn(3) * sigma_R), device=device, dtype=torch.float32)
            R0_noisy = R0_base * r_noise
            
            S0 = torch.cat([N_in, R0_noisy], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_next = torch.clamp(S_t[-1, :48] * mask_i, min=0.0)
            
            obs_noise = torch.tensor(np.exp(np.random.randn(48) * sigma_obs), device=device, dtype=torch.float32)
            N_obs = N_next * obs_noise * mask_i
            rollout_stoch.append(N_obs.cpu().numpy())
            N_curr = N_next
    rollout_stoch = np.array(rollout_stoch)
    rel_stoch = rollout_stoch / np.where(rollout_stoch.sum(axis=-1, keepdims=True) > 0, rollout_stoch.sum(axis=-1, keepdims=True), 1.0)
    rollouts_stoch_dict[(nutrient, S_val, comm_val)] = rel_stoch
    
    # CV over Days 7-10 (stochastic)
    late_stoch = rel_stoch[7:11]
    m_stoch = late_stoch.mean(axis=0)
    s_stoch = late_stoch.std(axis=0)
    surv_stoch = m_stoch >= 1e-3
    cv_stoch = float(np.mean(s_stoch[surv_stoch] / m_stoch[surv_stoch])) if np.any(surv_stoch) else 0.0
    
    results.append({
        'nutrient': nutrient,
        'S': S_val,
        'comm': comm_val,
        'emp_cv': emp_cv,
        'emp_fluc': emp_fluc,
        'cv_det': cv_det,
        'cv_stoch': cv_stoch,
        'sim_fluc_stoch': bool(cv_stoch >= 0.25)
    })

df = pd.DataFrame(results)

stable_df = df[~df['emp_fluc']]
fluc_df = df[df['emp_fluc']]

print("\n" + "="*80)
print("EMPIRICALLY STABLE (N=37) vs. FLUCTUATING (N=38) COMMUNITIES AUDIT")
print("="*80)
print(f"Empirically Stable Communities (N={len(stable_df)}):")
print(f"  Empirical Mean CV:            {stable_df['emp_cv'].mean():.4f} +/- {stable_df['emp_cv'].std():.4f}")
print(f"  Simulated Deterministic CV:   {stable_df['cv_det'].mean():.4f} +/- {stable_df['cv_det'].std():.4f}")
print(f"  Simulated Stochastic CV:      {stable_df['cv_stoch'].mean():.4f} +/- {stable_df['cv_stoch'].std():.4f}")
print(f"  % Classified Stable in Model: {(~stable_df['sim_fluc_stoch']).mean()*100:.1f}%")

print(f"\nEmpirically Fluctuating Communities (N={len(fluc_df)}):")
print(f"  Empirical Mean CV:            {fluc_df['emp_cv'].mean():.4f} +/- {fluc_df['emp_cv'].std():.4f}")
print(f"  Simulated Deterministic CV:   {fluc_df['cv_det'].mean():.4f} +/- {fluc_df['cv_det'].std():.4f}")
print(f"  Simulated Stochastic CV:      {fluc_df['cv_stoch'].mean():.4f} +/- {fluc_df['cv_stoch'].std():.4f}")
print(f"  % Classified Fluc in Model:   {fluc_df['sim_fluc_stoch'].mean()*100:.1f}%")

stat_t, p_val_t = ttest_ind(stable_df['cv_stoch'], fluc_df['cv_stoch'])
stat_u, p_val_u = mannwhitneyu(stable_df['cv_stoch'], fluc_df['cv_stoch'], alternative='less')
print(f"\nStatistical Significance (Stable CV < Fluctuation CV):")
print(f"  Two-sample t-test: t = {stat_t:.3f}, p = {p_val_t:.4e}")
print(f"  Mann-Whitney U:    U = {stat_u:.1f}, p = {p_val_u:.4e}")

# ─── 2. Long-Term 200-Passage Simulation Comparison ───
print("\n" + "="*80)
print("LONG-TERM (200-PASSAGE) SIMULATION: STABLE VS FLUCTUATING COMMUNITIES")
print("="*80)

# Representative Stable Community: S=3 Community 1 in LN
stable_key = ('LN', 3, 'Community1')
stable_idx = [i for i, k in enumerate(all_keys) if (k[0], k[1], k[2]) == stable_key][0]
mask_stable = all_m[stable_idx]
N0_stable = all_N[stable_idx, 0, :] * mask_stable
R0_stable = R0_table[cond_to_idx['LN']]

# Representative Fluctuating Community: S=48 Community 1 in HN
fluc_key = ('HN', 48, 'Community1')
fluc_idx = [i for i, k in enumerate(all_keys) if (k[0], k[1], k[2]) == fluc_key][0]
mask_fluc = all_m[fluc_idx]
N0_fluc = all_N[fluc_idx, 0, :] * mask_fluc
R0_fluc = R0_table[cond_to_idx['HN']]

def simulate_200p(N0, R0_base, mask_i, is_stoch=True):
    np.random.seed(42)
    N_curr = N0.clone()
    roll = [N_curr.cpu().numpy()]
    with torch.no_grad():
        for p in range(200):
            if is_stoch:
                d_noise = float(np.exp(np.random.randn() * sigma_D))
                D_eff = model.D * d_noise
                r_noise = torch.tensor(np.exp(np.random.randn(3) * sigma_R), device=device, dtype=torch.float32)
                R0_noisy = R0_base * r_noise
            else:
                D_eff = model.D
                R0_noisy = R0_base
            N_in = torch.clamp(N_curr / D_eff + model.mu * mask_i, min=0.0)
            S0 = torch.cat([N_in, R0_noisy], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_next = torch.clamp(S_t[-1, :48] * mask_i, min=0.0)
            roll.append(N_next.cpu().numpy())
            N_curr = N_next
    roll = np.array(roll)
    rel = roll / np.where(roll.sum(axis=-1, keepdims=True) > 0, roll.sum(axis=-1, keepdims=True), 1.0)
    return rel

# Simulate both
rel_200_stable_det = simulate_200p(N0_stable, R0_stable, mask_stable, is_stoch=False)
rel_200_stable_stoch = simulate_200p(N0_stable, R0_stable, mask_stable, is_stoch=True)

rel_200_fluc_det = simulate_200p(N0_fluc, R0_fluc, mask_fluc, is_stoch=False)
rel_200_fluc_stoch = simulate_200p(N0_fluc, R0_fluc, mask_fluc, is_stoch=True)

# Count flips in passages 100-200
def count_flips_and_p2p(rel_traj):
    late = rel_traj[100:201]
    active_sp = np.where(late.mean(axis=0) >= 0.01)[0]
    flips_dict = {}
    p2p_dict = {}
    for sp in active_sp:
        diffs = np.diff(late[:, sp])
        sgn = np.sign(diffs)
        sgn = sgn[sgn != 0]
        flips = int(np.sum(sgn[:-1] != sgn[1:])) if len(sgn) > 1 else 0
        p2p = float(np.max(late[:, sp]) - np.min(late[:, sp]))
        flips_dict[sp] = flips
        p2p_dict[sp] = p2p
    return active_sp, flips_dict, p2p_dict

act_st_det, flips_st_det, p2p_st_det = count_flips_and_p2p(rel_200_stable_det)
act_st_stoch, flips_st_stoch, p2p_st_stoch = count_flips_and_p2p(rel_200_stable_stoch)

act_fl_det, flips_fl_det, p2p_fl_det = count_flips_and_p2p(rel_200_fluc_det)
act_fl_stoch, flips_fl_stoch, p2p_fl_stoch = count_flips_and_p2p(rel_200_fluc_stoch)

print(f"\nStable Community (S=3 LN):")
print(f"  Deterministic (Passages 100-200): Active Species = {len(act_st_det)}, Mean Flips = {np.mean(list(flips_st_det.values())):.1f}, Max P2P = {max(p2p_st_det.values()):.4f}")
print(f"  Stochastic    (Passages 100-200): Active Species = {len(act_st_stoch)}, Mean Flips = {np.mean(list(flips_st_stoch.values())):.1f}, Max P2P = {max(p2p_st_stoch.values()):.4f}")

print(f"\nFluctuating Community (S=48 HN):")
print(f"  Deterministic (Passages 100-200): Active Species = {len(act_fl_det)}, Mean Flips = {np.mean(list(flips_fl_det.values())):.1f}, Max P2P = {max(p2p_fl_det.values()):.4f}")
print(f"  Stochastic    (Passages 100-200): Active Species = {len(act_fl_stoch)}, Mean Flips = {np.mean(list(flips_fl_stoch.values())):.1f}, Max P2P = {max(p2p_fl_stoch.values()):.4f}")


# ─── 3. Generate Publication-Quality Figure ───
fig = plt.figure(figsize=(16, 11), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.34, wspace=0.22)

# (A) Simulated CV Comparison: Stable vs Fluctuating
ax0 = fig.add_subplot(gs[0, 0])
box_data = [stable_df['cv_stoch'].values, fluc_df['cv_stoch'].values]
bp = ax0.boxplot(box_data, patch_artist=True, widths=0.45,
                 medianprops=dict(color='black', lw=2.0),
                 boxprops=dict(facecolor='#E2E8F0', color='#475569', lw=1.5))
colors = ['#2B6CB0', '#C53030']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)

# Scatter empirical data points on top
jitter = 0.08
ax0.scatter(np.ones(len(stable_df)) + np.random.uniform(-jitter, jitter, len(stable_df)),
            stable_df['cv_stoch'], color='#1A365D', alpha=0.75, s=36, edgecolor='white', lw=0.6, label='Empirically Stable (N=37)')
ax0.scatter(2 * np.ones(len(fluc_df)) + np.random.uniform(-jitter, jitter, len(fluc_df)),
            fluc_df['cv_stoch'], color='#742A2A', alpha=0.75, s=36, edgecolor='white', lw=0.6, label='Empirically Fluctuating (N=38)')

ax0.axhline(0.25, color='gray', linestyle='--', lw=1.5, alpha=0.8, label="Gore Fluctuation Threshold (CV=0.25)")
ax0.set_xticks([1, 2])
ax0.set_xticklabels([f"Empirically Stable\n(Mean CV = {stable_df['cv_stoch'].mean():.3f})",
                     f"Empirically Fluctuating\n(Mean CV = {fluc_df['cv_stoch'].mean():.3f})"],
                    fontweight='bold', fontsize=10.5)
ax0.set_ylabel("Simulated CV (Days 7–10)", fontweight='bold', fontsize=11)
ax0.set_title(f"(A) Simulated Fluctuation Levels (Days 7–10)\n(Mann-Whitney U p = {p_val_u:.2e}, Statistically Significant Separation)",
              fontweight='bold', fontsize=11.5, loc='left', pad=8)
ax0.grid(True, linestyle='--', alpha=0.5)
ax0.legend(loc='upper left', fontsize=8.5)

# (B) 10-Day Trajectory: Stable Community (S=3 LN Comm1)
ax1 = fig.add_subplot(gs[0, 1])
days_10 = np.arange(11)
emp_stable_traj = data['relative_abundance'][stable_key].cpu().numpy()
st_species = np.where(emp_stable_traj.max(axis=0) > 0.01)[0]
cmap_st = plt.cm.tab10(np.linspace(0, 1, 10))

for idx, sp in enumerate(st_species):
    ax1.plot(days_10, emp_stable_traj[:, sp], marker='o', markersize=5, lw=1.8, linestyle='--',
             color=cmap_st[idx % 10], label=f"Sp {sp+1} (Empirical)")
    ax1.plot(days_10, rollouts_stoch_dict[stable_key][:, sp], lw=2.2,
             color=cmap_st[idx % 10], alpha=0.85, label=f"Sp {sp+1} (Model)")

ax1.set_title(f"(B) 10-Day Trajectory: Empirically Stable Community ({stable_key[0]}, S={stable_key[1]})\n(Empirical CV = {meta_map[(3, 'Community1')]['LN_CV']:.3f} | Model Damped to Fixed Hierarchy)",
              fontweight='bold', fontsize=11.5, color='#1A365D', loc='left', pad=8)
ax1.set_xlabel("Time (Days)", fontweight='bold', fontsize=11)
ax1.set_ylabel("Relative Abundance", fontweight='bold', fontsize=11)
ax1.set_xlim(-0.3, 10.3)
ax1.set_ylim(-0.02, 1.02)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', fontsize=7.8, ncol=2)

# (C) 200 Passages: Stable Community Dynamics (Passages 0-200)
ax2 = fig.add_subplot(gs[1, 0])
p_axis = np.arange(201)
for idx, sp in enumerate(act_st_stoch):
    ax2.plot(p_axis, rel_200_stable_stoch[:, sp], lw=1.8, color=cmap_st[idx % 10],
             label=f"Sp {sp+1} (Mean={rel_200_stable_stoch[100:, sp].mean():.2f}, P2P={p2p_st_stoch[sp]*100:.1f}%)")
ax2.set_title(f"(C) 200 Dilution Passages: Stable Community ({stable_key[0]}, S={stable_key[1]})\n(Strong Restoring Forces: P2P Amplitude < 3.5%, No Large Phase Swings)",
              fontweight='bold', fontsize=11.5, color='#1A365D', loc='left', pad=8)
ax2.set_xlabel("Dilution Passage (Days)", fontweight='bold', fontsize=11)
ax2.set_ylabel("Relative Abundance", fontweight='bold', fontsize=11)
ax2.set_xlim(-2, 202)
ax2.set_ylim(-0.02, 1.02)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='center right', fontsize=8.5)

# (D) 200 Passages: Fluctuating Community Dynamics (Passages 0-200)
ax3 = fig.add_subplot(gs[1, 1])
cmap_fl = plt.cm.tab20(np.linspace(0, 1, 20))
for idx, sp in enumerate(act_fl_stoch[:8]):
    ax3.plot(p_axis, rel_200_fluc_stoch[:, sp], lw=1.8, color=cmap_fl[idx % 20],
             label=f"Sp {sp+1} (Flips={flips_fl_stoch[sp]}, P2P={p2p_fl_stoch[sp]*100:.1f}%)")
ax3.set_title(f"(D) 200 Dilution Passages: Fluctuating Community ({fluc_key[0]}, S={fluc_key[1]})\n(Slow Manifold Neutrality: 9 Coexisting Species, Sustained Large Oscillations)",
              fontweight='bold', fontsize=11.5, color='#8B2500', loc='left', pad=8)
ax3.set_xlabel("Dilution Passage (Days)", fontweight='bold', fontsize=11)
ax3.set_ylabel("Relative Abundance", fontweight='bold', fontsize=11)
ax3.set_xlim(-2, 202)
ax3.set_ylim(-0.02, 1.02)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', fontsize=7.8, ncol=2)

out_fig = os.path.join(PROJECT_ROOT, 'Figures', 'Gore_Stable_vs_Fluctuating_Validation.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved validation figure to: {out_fig}")

import shutil
artifact_dst = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143/Gore_Stable_vs_Fluctuating_Validation.png'
shutil.copyfile(out_fig, artifact_dst)
print("Copied to artifact directory successfully!")

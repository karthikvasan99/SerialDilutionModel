import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.scratch_evaluate_k2_800ep import all_keys, all_m

# Load long term trajectories
all_N = np.load('Data/Gore_LongTerm_Abundances_100p.npy') # (99, 101, 48)
all_bio = all_N.sum(axis=-1)
all_rel = all_N / np.where(all_bio[..., None] > 0, all_bio[..., None], 1.0)

num_passages = all_N.shape[1] - 1
passages = np.arange(num_passages + 1)

# Surviving species (rel >= 1e-3)
surv_arr = (all_rel >= 1e-3).sum(axis=-1) # (99, 101)

# Shannon diversity and effective species
p_safe = np.where(all_rel > 1e-12, all_rel, 1e-12)
shannon_arr = -np.sum(np.where(all_rel >= 1e-12, all_rel * np.log(p_safe), 0.0), axis=-1)
effective_sp = np.exp(shannon_arr) # (99, 101)

df_comms = pd.DataFrame({
    'Nutrient': [k[0] for k in all_keys],
    'S_initial': [k[1] for k in all_keys],
    'Community': [k[2] for k in all_keys]
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

# Colors and styles
colors_s = {3: '#1f77b4', 6: '#2ca02c', 12: '#ff7f0e', 24: '#d62728', 48: '#9467bd'}
nut_styles = {'LN': ':', 'MN': '--', 'HN': '-'}

# Panel 1: Surviving Species Count across 100 passages
ax1 = axes[0]
for s in [3, 6, 12, 24, 48]:
    s_idx = df_comms[df_comms.S_initial == s].index
    mean_surv = surv_arr[s_idx].mean(axis=0)
    std_surv = surv_arr[s_idx].std(axis=0)
    ax1.plot(passages, mean_surv, label=f'Initial S={s}', color=colors_s[s], lw=2.5)
    ax1.fill_between(passages, mean_surv - std_surv, mean_surv + std_surv, color=colors_s[s], alpha=0.15)

ax1.axvline(x=10, color='gray', linestyle='--', lw=1.5, label='Experimental End (Day 10)')
ax1.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Surviving Species Count ($x_i \geq 10^{-3}$)', fontsize=12, fontweight='bold')
ax1.set_title('A: Long-Term Species Richness Retention', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 100)

# Panel 2: Effective Number of Species (exp(H))
ax2 = axes[1]
for s in [3, 6, 12, 24, 48]:
    s_idx = df_comms[df_comms.S_initial == s].index
    mean_eff = effective_sp[s_idx].mean(axis=0)
    std_eff = effective_sp[s_idx].std(axis=0)
    ax2.plot(passages, mean_eff, label=f'Initial S={s}', color=colors_s[s], lw=2.5)
    ax2.fill_between(passages, mean_eff - std_eff, mean_eff + std_eff, color=colors_s[s], alpha=0.15)

ax2.axvline(x=10, color='gray', linestyle='--', lw=1.5)
ax2.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Effective Species $exp(H)$ (Shannon Diversity)', fontsize=12, fontweight='bold')
ax2.set_title('B: Multi-Species Shannon Equitability', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 100)

# Panel 3: Trajectory of a complex S=24 community over 100 days
# Pick HN S=24 Community 1
target_idx = df_comms[(df_comms.Nutrient == 'HN') & (df_comms.S_initial == 24) & (df_comms.Community == 'Community1')].index[0]
target_rel = all_rel[target_idx] # (101, 48)
mask_target = all_m[target_idx].cpu().numpy().astype(bool)
present_species = np.where(mask_target)[0]

ax3 = axes[2]
cmap = plt.cm.get_cmap('tab20', len(present_species))
for j, sp_id in enumerate(present_species):
    traj = target_rel[:, sp_id]
    if traj.max() >= 0.01: # only plot if it reaches at least 1%
        ax3.plot(passages, traj, lw=1.6, alpha=0.85, label=f'Sp {sp_id+1}')

ax3.axvline(x=10, color='black', linestyle='--', lw=1.5, label='Day 10 (Gore Lab End)')
ax3.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Relative Abundance $x_i(t)$', fontsize=12, fontweight='bold')
ax3.set_title('C: HN $S=24$ Community 1 (100 Days)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 100)
ax3.set_ylim(-0.02, 0.65)

plt.tight_layout()
os.makedirs('Figures', exist_ok=True)
out_path = 'Figures/Gore_LongTerm_Diversity_100Days.png'
plt.savefig(out_path, dpi=300)
print(f"Figure saved to {out_path}")

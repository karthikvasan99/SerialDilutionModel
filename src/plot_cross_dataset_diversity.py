import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load saved arrays
data_cross = np.load('Data/Cross_Dataset_LongTerm_Diversity.npz')
surv_g = data_cross['surv_g'] # (12, 101)
eff_g = data_cross['eff_g']
surv_p = data_cross['surv_p'] # (45, 101)
eff_p = data_cross['eff_p']
surv_o = data_cross['surv_o'] # (175, 101)
eff_o = data_cross['eff_o']

# Load Gore long term
all_N_gore = np.load('Data/Gore_LongTerm_Abundances_100p.npy') # (99, 101, 48)
all_bio_gore = all_N_gore.sum(axis=-1)
all_rel_gore = all_N_gore / np.where(all_bio_gore[..., None] > 0, all_bio_gore[..., None], 1.0)
surv_gore = (all_rel_gore >= 1e-3).sum(axis=-1) # (99, 101)
p_gore_safe = np.where(all_rel_gore > 1e-12, all_rel_gore, 1e-12)
eff_gore = np.exp(-np.sum(np.where(all_rel_gore >= 1e-12, all_rel_gore * np.log(p_gore_safe), 0.0), axis=-1))

passages = np.arange(101)

fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

colors = {
    'Gore': '#d62728',     # Red
    'Ophelia': '#2ca02c',  # Green
    'German': '#1f77b4',   # Blue
    'Pollard': '#ff7f0e'   # Orange
}

# --- Panel 1: Surviving Species Count Over 100 Passages ---
ax1 = axes[0, 0]
for name, arr, col in [
    ('Gore (S=3..48, K=2)', surv_gore, colors['Gore']),
    ('Pollard (N=50, K=5)', surv_p, colors['Pollard']),
    ('German (N=91, K=12)', surv_g, colors['German']),
    ('Ophelia (N=12, K=2)', surv_o, colors['Ophelia'])
]:
    mean_val = arr.mean(axis=0)
    std_val = arr.std(axis=0)
    ax1.plot(passages, mean_val, label=f"{name} (Mean ± SD)", color=col, lw=2.5)
    ax1.fill_between(passages, np.maximum(0, mean_val - std_val), mean_val + std_val, color=col, alpha=0.15)

ax1.axvline(x=10, color='gray', linestyle='--', lw=1.5, label='Experimental Range (6-10 days)')
ax1.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Surviving Species Count ($x_i \geq 10^{-3}$)', fontsize=12, fontweight='bold')
ax1.set_title('A: Long-Term Species Richness Retention (100 Days)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 100)

# --- Panel 2: Effective Number of Species exp(H) Over 100 Passages ---
ax2 = axes[0, 1]
for name, arr, col in [
    ('Gore (S=3..48, K=2)', eff_gore, colors['Gore']),
    ('Pollard (N=50, K=5)', eff_p, colors['Pollard']),
    ('German (N=91, K=12)', eff_g, colors['German']),
    ('Ophelia (N=12, K=2)', eff_o, colors['Ophelia'])
]:
    mean_val = arr.mean(axis=0)
    std_val = arr.std(axis=0)
    ax2.plot(passages, mean_val, label=name, color=col, lw=2.5)
    ax2.fill_between(passages, np.maximum(0, mean_val - std_val), mean_val + std_val, color=col, alpha=0.15)

ax2.axvline(x=10, color='gray', linestyle='--', lw=1.5)
ax2.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Effective Species $\exp(H)$ (Shannon Diversity)', fontsize=12, fontweight='bold')
ax2.set_title('B: Multi-Species Shannon Equitability Across Datasets', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 100)

# --- Panel 3: Ophelia (Venturelli) Trajectories over 100 Passages ---
ax3 = axes[1, 0]
ophelia_arr = data_cross['ophelia_arr'] # (175, 101, 12)
sample_idx_o = 0 # Representative synthetic community
for sp in range(12):
    traj = ophelia_arr[sample_idx_o, :, sp]
    if traj.max() >= 0.01:
        ax3.plot(passages, traj, lw=2.0, alpha=0.85, label=f"Species {sp+1}")

ax3.axvline(x=6, color='black', linestyle='--', lw=1.5, label='Exp End (Day 6)')
ax3.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Relative Abundance $x_i(t)$', fontsize=12, fontweight='bold')
ax3.set_title('C: Ophelia (Venturelli) Community Trajectory (100 Days)', fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', frameon=True, fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 100)

# --- Panel 4: German Rumen Trajectories over 100 Passages ---
ax4 = axes[1, 1]
german_arr = data_cross['german_arr'] # (12, 101, 91)
sample_idx_g = 0 # BRRS H2:CO2 5 bar
for sp in range(german_arr.shape[-1]):
    traj = german_arr[sample_idx_g, :, sp]
    if traj.max() >= 0.02:
        ax4.plot(passages, traj, lw=1.8, alpha=0.8, label=f"Taxon {sp+1}")

ax4.axvline(x=6, color='black', linestyle='--', lw=1.5, label='Exp End (Day 6)')
ax4.set_xlabel('Dilution Passage (Days)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Relative Abundance $x_i(t)$', fontsize=12, fontweight='bold')
ax4.set_title('D: German Rumen Enrichment Trajectory (100 Days)', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 100)

plt.tight_layout()
os.makedirs('Figures', exist_ok=True)
out_fig = 'Figures/Cross_Dataset_LongTerm_Diversity_Comparison.png'
plt.savefig(out_fig, dpi=300)
print(f"Comparison figure saved to {out_fig}")

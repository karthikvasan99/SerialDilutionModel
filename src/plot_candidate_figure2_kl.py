import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.mixture import GaussianMixture

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def assign_phylum(strain_name):
    s = strain_name.lower()
    if any(k in s for k in ['bacteroides', 'parabacteroides', 'prevotella', 'aliistipes', 'odoribacter', 'barnesiella']):
        return 'Bacteroidota'
    elif any(k in s for k in ['clostridium', 'ruminococcus', 'faecalibacterium', 'coprococcus', 'eubacterium', 
                             'roseburia', 'anaerostipes', 'dorea', 'blautia', 'lachnospira', 'streptococcus', 
                             'enterococcus', 'holdemanella']):
        return 'Bacillota (Firmicutes)'
    elif any(k in s for k in ['veillonella', 'dialister', 'megasphaera', 'phascolarctobacterium']):
        return 'Negativicutes'
    elif any(k in s for k in ['bifidobacterium', 'collinsella', 'eggerthella', 'actinomyces', 'adlercreutzia']):
        return 'Actinomycetota'
    elif any(k in s for k in ['escherichia', 'klebsiella', 'desulfovibrio', 'sutterella', 'bilophila', 'parasutterella']):
        return 'Pseudomonadota'
    else:
        return 'Other'

print("="*75)
print("GENERATING FULL FIGURE 2 WITH KL DIVERGENCE PANEL B")
print("="*75)

# 1. LOAD DATA & MODEL PARAMETERS
multi_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_DualTheta_MultiRun_10Seeds.pt')
multi_data = torch.load(multi_path, map_location='cpu', weights_only=False)
runs_data = multi_data['runs_data']
strains = multi_data['strains']
clean_strains = [s.replace('-', ' ') for s in strains]
phyla = [assign_phylum(s) for s in clean_strains]

ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Winning_CRM.pth')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt['model_state_dict']

import torch.nn.functional as F
Theta_c = F.softplus(state['theta_c_raw']).numpy()
Theta_s = F.softplus(state['theta_s_raw']).numpy()
W_raw = F.softplus(state['w_raw']).numpy()
np.fill_diagonal(W_raw, 0.0)
row_sums = W_raw.sum(axis=1, keepdims=True)
W = W_raw * np.minimum(0.50 / np.maximum(row_sums, 1e-6), 1.0)

# Hierarchical clustering for visualization
Z = linkage(Theta_c, method='ward')
order = leaves_list(Z)
Theta_c_ord = Theta_c[order, :]
Theta_s_ord = Theta_s[order, :]

# Load KL values
kl_data = torch.load(os.path.join(PROJECT_ROOT, 'Data', 'Pollard_KL_Benchmark_Values.pt'), weights_only=False)
kl_crm = kl_data['kl_crm']
kl_loocv = [0.1835, 0.1052, 0.0924]
loocv_std = [0.012, 0.009, 0.008]
kl_w0 = kl_data['kl_w0']
kl_pers = kl_data['kl_pers']
kl_th = kl_data['kl_th_mean']
kl_th_std = kl_data['kl_th_std']
kl_adh = kl_data['kl_adh_mean']
kl_adh_std = kl_data['kl_adh_std']

# Bimodality metrics
ratios_mucin = np.array([r['ratio_mucin'] for r in runs_data])
ratios_plain = np.array([r['ratio_plain'] for r in runs_data])
mean_mucin = np.mean(ratios_mucin, axis=0)
mean_plain = np.mean(ratios_plain, axis=0)
log_mucin = np.log10(mean_mucin)
log_plain = np.log10(mean_plain)

gmm_mucin_2 = GaussianMixture(n_components=2, random_state=42).fit(log_mucin.reshape(-1, 1))
gmm_plain_2 = GaussianMixture(n_components=2, random_state=42).fit(log_plain.reshape(-1, 1))

idx_m = np.argsort(gmm_mucin_2.means_.flatten())
w_m = gmm_mucin_2.weights_[idx_m]
m_m = gmm_mucin_2.means_.flatten()[idx_m]
s_m = np.sqrt(gmm_mucin_2.covariances_.flatten()[idx_m])

idx_p = np.argsort(gmm_plain_2.means_.flatten())
w_p = gmm_plain_2.weights_[idx_p]
m_p = gmm_plain_2.means_.flatten()[idx_p]
s_p = np.sqrt(gmm_plain_2.covariances_.flatten()[idx_p])

# Tradeoff metrics
theta_ratios = np.array([r['theta_ratio'] for r in runs_data])
log_ratios_mucin = np.log10(ratios_mucin)
mean_log_ratio = np.mean(log_ratios_mucin, axis=0)
sem_log_ratio = np.std(log_ratios_mucin, axis=0, ddof=1) / np.sqrt(len(runs_data))
mean_theta_ratio = np.mean(theta_ratios, axis=0)
sem_theta_ratio = np.std(theta_ratios, axis=0, ddof=1) / np.sqrt(len(runs_data))
slope, intercept, r_val, p_val, std_err = linregress(mean_log_ratio, mean_theta_ratio)

# -------------------------------------------------------------
# FIGURE CANVAS & GRIDSPEC LAYOUT
# -------------------------------------------------------------
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig = plt.figure(figsize=(16.8, 13.0), dpi=300)

gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.05], hspace=0.35, wspace=0.25,
                       left=0.06, right=0.96, top=0.94, bottom=0.07)

# -------------------------------------------------------------
# PANEL A: PARAMETER ARCHITECTURE
# -------------------------------------------------------------
gs_a = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 0], width_ratios=[1.0, 1.0, 0.9], wspace=0.32)
ax_a1 = fig.add_subplot(gs_a[0, 0])
ax_a2 = fig.add_subplot(gs_a[0, 1])
ax_a3 = fig.add_subplot(gs_a[0, 2])

ax_a1.text(-0.30, 1.12, 'a', transform=ax_a1.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

vmax_th = max(Theta_c.max(), Theta_s.max())
im1 = ax_a1.imshow(Theta_c_ord, aspect='auto', cmap='YlGnBu', vmin=0, vmax=vmax_th)
ax_a1.set_title(r'Biofilm Traits $\mathbf{\Theta}^{(c)}$', fontsize=12, fontweight='bold', pad=8)
ax_a1.set_xlabel('Resource Dimension', fontsize=10.5, fontweight='bold')
ax_a1.set_ylabel('Strains (Clustered)', fontsize=11, fontweight='bold')
ax_a1.set_xticks([0, 1, 2, 3, 4])
ax_a1.set_xticklabels(['R1', 'R2', 'R3', 'R4', 'R5'], fontsize=9.5)

im2 = ax_a2.imshow(Theta_s_ord, aspect='auto', cmap='YlGnBu', vmin=0, vmax=vmax_th)
ax_a2.set_title(r'Planktonic Traits $\mathbf{\Theta}^{(s)}$', fontsize=12, fontweight='bold', pad=8)
ax_a2.set_xlabel('Resource Dimension', fontsize=10.5, fontweight='bold')
ax_a2.set_yticks([])
ax_a2.set_xticks([0, 1, 2, 3, 4])
ax_a2.set_xticklabels(['R1', 'R2', 'R3', 'R4', 'R5'], fontsize=9.5)

cb_a = fig.colorbar(im2, ax=[ax_a1, ax_a2], orientation='horizontal', fraction=0.045, pad=0.18, shrink=0.85)
cb_a.set_label('Consumption Affinity', fontsize=9.5, fontweight='bold')

im3 = ax_a3.imshow(W, aspect='auto', cmap='Purples', vmin=0, vmax=W.max())
ax_a3.set_title(r'Byproducts $\mathbf{W}$ ($K \times K$)', fontsize=12, fontweight='bold', pad=8)
ax_a3.set_xlabel('Product', fontsize=10.5, fontweight='bold')
ax_a3.set_ylabel('Substrate', fontsize=10.5, fontweight='bold')
ax_a3.set_xticks([0, 1, 2, 3, 4])
ax_a3.set_yticks([0, 1, 2, 3, 4])
ax_a3.set_xticklabels(['R1', 'R2', 'R3', 'R4', 'R5'], fontsize=9.5)
ax_a3.set_yticklabels(['R1', 'R2', 'R3', 'R4', 'R5'], fontsize=9.5)
cb_w = fig.colorbar(im3, ax=ax_a3, orientation='horizontal', fraction=0.045, pad=0.18, shrink=0.85)
cb_w.set_label('Transfer Fraction', fontsize=9.5, fontweight='bold')

# -------------------------------------------------------------
# PANEL B: KL DIVERGENCE BENCHMARK
# -------------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 1])
ax_b.text(-0.12, 1.12, 'b', transform=ax_b.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

cond_names = ['No Carrier\n(Liquid)', 'Plain Agar\n(Inert Bead)', 'Mucin Agar\n(Mucosal Bead)']
x = np.arange(len(cond_names))
w = 0.125
offsets = np.linspace(-2.5*w, 2.5*w, 6)

b1 = ax_b.bar(x + offsets[0], kl_crm, width=w, label='Full Fit (Consensus CRM)', color='#1B6CA8', edgecolor='#0F2942', lw=1.1, zorder=3)
b2 = ax_b.bar(x + offsets[1], kl_loocv, width=w, yerr=loocv_std, capsize=3, label='Held-Out Vessel (LOOCV)', color='#2ECC71', edgecolor='#196F3D', lw=1.1, zorder=3)
b3 = ax_b.bar(x + offsets[2], kl_w0, width=w, label=r'$\mathbf{W}=\mathbf{0}$ (No Cross-feeding)', color='#E08B53', edgecolor='#803010', lw=1.1, zorder=3)
b4 = ax_b.bar(x + offsets[3], kl_pers, width=w, label='Persistence Null', color='#F5A623', edgecolor='#B37400', lw=1.1, zorder=3)
b5 = ax_b.bar(x + offsets[4], kl_th, width=w, yerr=kl_th_std, capsize=3, label=r'Shuffled $\mathbf{\Theta}$', color='#D9534F', edgecolor='#701010', lw=1.1, zorder=3)
b6 = ax_b.bar(x[1:] + offsets[5], kl_adh[1:], width=w, yerr=kl_adh_std[1:], capsize=3, label='Shuffled Adhesion', color='#8E44AD', edgecolor='#4A154B', lw=1.1, zorder=3)

# Add value text
for bar in b1:
    h = bar.get_height()
    ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.04, f'{h:.2f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#1B6CA8')
for bar in b2:
    h = bar.get_height()
    ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.04, f'{h:.2f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#196F3D')
for bar in b3:
    h = bar.get_height()
    ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.04, f'{h:.2f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#803010')
for bar in b4:
    h = bar.get_height()
    ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.04, f'{h:.2f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#B37400')
for i, bar in enumerate(b5):
    h = bar.get_height()
    err = kl_th_std[i]
    ax_b.text(bar.get_x() + bar.get_width()/2., h + err + 0.05, f'{h:.2f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#701010')
for i, bar in enumerate(b6):
    h = bar.get_height()
    err = kl_adh_std[i+1]
    ax_b.text(bar.get_x() + bar.get_width()/2., h + err + 0.04, f'{h:.2f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#4A154B')

ax_b.set_xticks(x)
ax_b.set_xticklabels(cond_names, fontsize=11.5, fontweight='bold', color='#0F2942')
ax_b.set_ylabel(r'Generalized KL Divergence (nats, lower is better $\downarrow$)', fontsize=11.5, fontweight='bold', labelpad=4)
ax_b.set_ylim(0, 3.8)
ax_b.tick_params(axis='y', labelsize=10.5)
ax_b.legend(loc='upper left', ncol=3, fontsize=8.8, frameon=True, framealpha=0.95, edgecolor='#CCCCCC')
ax_b.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)

# -------------------------------------------------------------
# PANEL C: EMERGENT BIMODALITY
# -------------------------------------------------------------
gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.18)
ax_c1 = fig.add_subplot(gs_c[0, 0])
ax_c2 = fig.add_subplot(gs_c[0, 1])
ax_c1.text(-0.25, 1.12, 'c', transform=ax_c1.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

x_eval = np.linspace(-1.3, 1.3, 500)
bins = np.linspace(-1.1, 1.1, 14)

ax_c1.hist(log_plain, bins=bins, density=True, color='#95A5A6', edgecolor='#2C3E50', alpha=0.55, lw=1.1, label='Strains ($N=50$)')
pdf_p1 = w_p[0] * (1.0 / (s_p[0] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_eval - m_p[0]) / s_p[0])**2)
pdf_p2 = w_p[1] * (1.0 / (s_p[1] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_eval - m_p[1]) / s_p[1])**2)
ax_c1.plot(x_eval, pdf_p1 + pdf_p2, color='#2C3E50', lw=2.2, label='Total Density')
ax_c1.plot(x_eval, pdf_p1, color='#2980B9', lw=1.6, linestyle='--', label=f'Planktonic ({w_p[0]*100:.0f}%)')
ax_c1.plot(x_eval, pdf_p2, color='#27AE60', lw=1.6, linestyle=':', label=f'Biofilm ({w_p[1]*100:.0f}%)')
ax_c1.axvline(0, color='red', linestyle='--', alpha=0.7, lw=1.2)
ax_c1.set_title('Plain Agar (Inert Hydrogel)', fontsize=12, fontweight='bold', pad=8)
ax_c1.set_xlabel(r'$\log_{10}(N_c / N_s)$ Colonization Ratio', fontsize=10.5, fontweight='bold')
ax_c1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
ax_c1.legend(loc='upper right', fontsize=8.5, frameon=True)
ax_c1.set_xlim(-1.25, 1.25)
ax_c1.grid(True, linestyle=':', alpha=0.6)

ax_c2.hist(log_mucin, bins=bins, density=True, color='#E67E22', edgecolor='#7E3A04', alpha=0.55, lw=1.1, label='Strains ($N=50$)')
pdf_m1 = w_m[0] * (1.0 / (s_m[0] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_eval - m_m[0]) / s_m[0])**2)
pdf_m2 = w_m[1] * (1.0 / (s_m[1] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_eval - m_m[1]) / s_m[1])**2)
ax_c2.plot(x_eval, pdf_m1 + pdf_m2, color='#7E3A04', lw=2.2, label='Total Density')
ax_c2.plot(x_eval, pdf_m1, color='#2980B9', lw=1.6, linestyle='--', label=f'Planktonic ({w_m[0]*100:.0f}%)')
ax_c2.plot(x_eval, pdf_m2, color='#27AE60', lw=1.6, linestyle=':', label=f'Biofilm ({w_m[1]*100:.0f}%)')
ax_c2.axvline(0, color='red', linestyle='--', alpha=0.7, lw=1.2)
ax_c2.set_title('Mucin Agar (Mucosal Hydrogel)', fontsize=12, fontweight='bold', pad=8)
ax_c2.set_xlabel(r'$\log_{10}(N_c / N_s)$ Colonization Ratio', fontsize=10.5, fontweight='bold')
ax_c2.set_yticks([])
ax_c2.legend(loc='upper left', fontsize=8.5, frameon=True)
ax_c2.set_xlim(-1.25, 1.25)
ax_c2.grid(True, linestyle=':', alpha=0.6)

# -------------------------------------------------------------
# PANEL D: TRADEOFF ("Stick or Specialize")
# -------------------------------------------------------------
ax_d = fig.add_subplot(gs[1, 1])
ax_d.text(-0.12, 1.12, 'd', transform=ax_d.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

phylum_colors = {
    'Bacteroidota': '#E67E22',
    'Bacillota (Firmicutes)': '#27AE60',
    'Negativicutes': '#8E44AD',
    'Actinomycetota': '#2980B9',
    'Pseudomonadota': '#D35400',
    'Other': '#7F8C8D'
}

for p in np.unique(phyla):
    mask = [phy == p for phy in phyla]
    ax_d.errorbar(mean_log_ratio[mask], mean_theta_ratio[mask],
                  xerr=sem_log_ratio[mask], yerr=sem_theta_ratio[mask],
                  fmt='o', color=phylum_colors[p], ecolor='#BDC3C7', elinewidth=1.0, capsize=2,
                  markersize=7.0, alpha=0.88, label=p, zorder=4)

x_fit = np.linspace(-1.05, 1.05, 100)
y_fit = slope * x_fit + intercept
ax_d.plot(x_fit, y_fit, color='#C0392B', lw=2.2, linestyle='-', zorder=5,
          label=f'Ensemble Fit (r = {r_val:.3f}, p = {p_val:.2e})')

ax_d.axvline(0, color='#7F8C8D', linestyle=':', lw=1.2, alpha=0.7)
ax_d.axhline(1.0, color='#7F8C8D', linestyle=':', lw=1.2, alpha=0.7)

ax_d.set_title('"Stick or Specialize": Physical Adhesion vs Metabolic Specialization', fontsize=12, fontweight='bold', pad=8)
ax_d.set_xlabel(r'Colonization Preference $\log_{10}(N_c / N_s)$', fontsize=11, fontweight='bold')
ax_d.set_ylabel(r'Metabolic Divergence $||\mathbf{\Theta}^{(c)} - \mathbf{\Theta}^{(s)}||_2 / ||\mathbf{\Theta}^{(s)}||_2$', fontsize=11, fontweight='bold')
ax_d.legend(loc='upper right', fontsize=8.8, frameon=True, framealpha=0.95, edgecolor='#CCCCCC')
ax_d.grid(True, linestyle='--', alpha=0.5)

full_save_path = os.path.join(PROJECT_ROOT, 'Paper', 'figures', 'Figure2_Pollard_Candidate_KL_Divergence.png')
plt.savefig(full_save_path, dpi=300)
plt.close()
print(f"Saved complete Figure 2 with KL Panel B to: {full_save_path}")

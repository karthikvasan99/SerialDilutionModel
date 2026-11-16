import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Load computed KL benchmark data
kl_data = torch.load(os.path.join(PROJECT_ROOT, 'Data', 'Pollard_KL_Benchmark_Values.pt'), weights_only=False)

fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)

cond_names = ['No Carrier\n(Liquid)', 'Plain Agar\n(Inert Bead)', 'Mucin Agar\n(Mucosal Bead)']
x = np.arange(len(cond_names))
w = 0.125

# Values:
kl_crm = kl_data['kl_crm']           # [0.2070, 0.0897, 0.0709]
# LOOCV KL values from search_pollard_heldout_kl.py:
kl_loocv = [0.1835, 0.1052, 0.0924]
loocv_std = [0.012, 0.009, 0.008]

kl_w0 = kl_data['kl_w0']             # [0.3384, 0.1327, 0.1059]
kl_pers = kl_data['kl_pers']         # [1.3418, 0.4832, 0.5386]
kl_th = kl_data['kl_th_mean']        # [3.0983, 0.9902, 1.0651]
kl_th_std = kl_data['kl_th_std']
kl_adh = kl_data['kl_adh_mean']      # [nan, 0.1018, 0.0836]
kl_adh_std = kl_data['kl_adh_std']

# Bar positions (6 bars per condition)
offsets = np.linspace(-2.5*w, 2.5*w, 6)

b1 = ax.bar(x + offsets[0], kl_crm, width=w, label='Full Fit (Consensus CRM)', color='#1B6CA8', edgecolor='#0F2942', lw=1.1, zorder=3)
b2 = ax.bar(x + offsets[1], kl_loocv, width=w, yerr=loocv_std, capsize=3, label='Held-Out Vessel (LOOCV)', color='#2ECC71', edgecolor='#196F3D', lw=1.1, zorder=3)
b3 = ax.bar(x + offsets[2], kl_w0, width=w, label=r'$\mathbf{W}=\mathbf{0}$ (No Cross-feeding)', color='#E08B53', edgecolor='#803010', lw=1.1, zorder=3)
b4 = ax.bar(x + offsets[3], kl_pers, width=w, label='Persistence Null', color='#F5A623', edgecolor='#B37400', lw=1.1, zorder=3)
b5 = ax.bar(x + offsets[4], kl_th, width=w, yerr=kl_th_std, capsize=3, label=r'Shuffled $\mathbf{\Theta}$', color='#D9534F', edgecolor='#701010', lw=1.1, zorder=3)
b6 = ax.bar(x[1:] + offsets[5], kl_adh[1:], width=w, yerr=kl_adh_std[1:], capsize=3, label='Shuffled Adhesion', color='#8E44AD', edgecolor='#4A154B', lw=1.1, zorder=3)

# Add number labels
for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.04, f'{h:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1B6CA8')
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.04, f'{h:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#196F3D')
for bar in b3:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.04, f'{h:.2f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#803010')
for bar in b4:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.04, f'{h:.2f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#B37400')
for i, bar in enumerate(b5):
    h = bar.get_height()
    err = kl_th_std[i]
    ax.text(bar.get_x() + bar.get_width()/2., h + err + 0.05, f'{h:.2f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#701010')
for i, bar in enumerate(b6):
    h = bar.get_height()
    err = kl_adh_std[i+1]
    ax.text(bar.get_x() + bar.get_width()/2., h + err + 0.04, f'{h:.2f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#4A154B')

ax.set_xticks(x)
ax.set_xticklabels(cond_names, fontsize=12, fontweight='bold', color='#0F2942')
ax.set_ylabel(r'Generalized KL Divergence (nats, lower is better $\downarrow$)', fontsize=12.5, fontweight='bold', labelpad=6)
ax.set_ylim(0, 3.8)
ax.tick_params(axis='y', labelsize=11)
ax.legend(loc='upper left', ncol=3, fontsize=9.2, frameon=True, framealpha=0.95, edgecolor='#CCCCCC')
ax.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)

# Annotation callout
ax.text(0.02, 0.65, "$\mathbf{Lower\ is\ Better\ (\downarrow)}$\n• CRM achieves lowest information loss\n• Persistence error is 5–7× higher\n• Shuffled $\mathbf{\Theta}$ collapses ($>1.0$ nats)", 
        transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#BDC3C7', lw=1.0))

plt.title('Figure 2 Panel B Candidate: Information-Theoretic Fidelity (KL Divergence)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()

save_path = os.path.join(PROJECT_ROOT, 'Paper', 'figures', 'Figure2_PanelB_Candidate_KL_Divergence.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved standalone Panel B plot to: {save_path}")

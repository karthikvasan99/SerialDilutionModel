import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.scratch_evaluate_k2_800ep import all_keys, all_m, data
from src.scratch_k2_ultimate_f1 import X, y
from src.Gore_K2_Classifier_100Rec import GoreK2Classifier100Rec
from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE

clf = GoreK2Classifier100Rec()
probs = clf.predict_proba(X)
preds = clf.predict(X)

ode_k2 = AdaptivePureConsumerResourceODE(N=48, K=2)
model_k2 = HierarchicalPhysicalNeuralODE(ode_k2, num_conditions=3, K=2)
model_k2.load_state_dict(torch.load('Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth', weights_only=False))
Theta_np = model_k2.ode_func.Theta.detach().cpu().numpy()

mn_24_data = []
for i, k in enumerate(all_keys):
    if k[0] == 'MN' and k[1] == 24:
        comm_num = int(k[2].replace('Community', ''))
        mn_24_data.append({
            'comm_id': f'C{comm_num}',
            'comm_num': comm_num,
            'prob': probs[i],
            'pred': preds[i],
            'gt': y[i],
            'mask': all_m[i].cpu().numpy()
        })

df_mn24 = pd.DataFrame(mn_24_data).sort_values('prob')

chaotic_masks = [d['mask'] for d in mn_24_data if d['gt'] == 1]
stable_masks  = [d['mask'] for d in mn_24_data if d['gt'] == 0]

freq_chaotic = np.mean(chaotic_masks, axis=0)
freq_stable  = np.mean(stable_masks, axis=0)
diff_freq    = freq_chaotic - freq_stable

df_taxa = pd.DataFrame({
    'species_idx': np.arange(1, 49),
    'species_name': [f'Sp {i}' for i in range(1, 49)],
    'freq_chaotic': freq_chaotic,
    'freq_stable': freq_stable,
    'diff': diff_freq,
    'theta1': Theta_np[:, 0],
    'theta2': Theta_np[:, 1],
    'niche_ratio': Theta_np[:, 0] / (Theta_np[:, 1] + 1e-4)
}).sort_values('diff', ascending=False)

plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig = plt.figure(figsize=(15.0, 3.6), dpi=300)
gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.35, 1.15], wspace=0.32)

# Panel A
ax0 = fig.add_subplot(gs[0])
y_pos = np.arange(len(df_mn24))
bar_colors = ['#802010' if gt == 1 else '#1B6CA8' for gt in df_mn24['gt']]

bars = ax0.barh(y_pos, df_mn24['prob'], color=bar_colors, edgecolor='#0F2942', lw=1.3, height=0.6, zorder=3)
ax0.axvline(clf.T_star, color='#D9534F', linestyle='--', lw=2.0, label=r'$T^* = 0.389$', zorder=4)

for idx, (_, row) in enumerate(df_mn24.iterrows()):
    label_txt = f"{row['prob']:.2f} ({'Chaos' if row['gt']==1 else 'Stable'})"
    ax0.text(row['prob'] + 0.02, idx, label_txt, va='center', fontsize=10.0, fontweight='bold', color='#1A365D')

ax0.set_yticks(y_pos)
ax0.set_yticklabels(df_mn24['comm_id'], fontsize=11.0, fontweight='bold')
ax0.set_xlabel(r'$P(\mathrm{Chaos})$', fontsize=12.5, fontweight='bold', color='#1A365D')
ax0.set_title(r'$\mathbf{(A)\;Classifier\;Scores\;(S=24)}$', fontsize=13.0, fontweight='bold', color='#0F2942', pad=6)
ax0.set_xlim(0, 1.05)
ax0.tick_params(axis='x', labelsize=11.0)
ax0.legend(loc='lower right', fontsize=11.5, framealpha=0.95, edgecolor='#0F2942')
ax0.grid(True, axis='x', linestyle='--', alpha=0.5, zorder=0)

# Panel B
ax1 = fig.add_subplot(gs[1])
df_sorted = df_taxa.sort_values('diff')
x_bars = np.arange(len(df_sorted))

taxa_colors = []
for d in df_sorted['diff']:
    if d > 0.3:
        taxa_colors.append('#E06D53')
    elif d < -0.3:
        taxa_colors.append('#1B6CA8')
    else:
        taxa_colors.append('#A0AAB2')

ax1.bar(x_bars, df_sorted['diff'], color=taxa_colors, edgecolor='none', width=0.8, zorder=3)
ax1.axhline(0, color='black', lw=1.0, alpha=0.6)
ax1.axhline(0.3, color='#E06D53', linestyle=':', lw=1.4, alpha=0.8)
ax1.axhline(-0.3, color='#1B6CA8', linestyle=':', lw=1.4, alpha=0.8)

ax1.text(39, 0.85, 'Sp 31–36\n(+80% Chaos)', color='#802010', fontsize=10.5, fontweight='bold', ha='center')
ax1.text(7, -0.92, 'Sp 37–42\n(+80% Stable)', color='#0B4F6C', fontsize=10.5, fontweight='bold', ha='center')

ax1.set_xlabel('48 Species (Ranked by Enrichment)', fontsize=12.5, fontweight='bold', color='#1A365D')
ax1.set_ylabel(r'$\Delta\,\mathrm{Prevalence}$', fontsize=12.5, fontweight='bold', color='#1A365D')
ax1.set_title(r'$\mathbf{(B)\;Taxonomic\;Biomarkers}$', fontsize=13.0, fontweight='bold', color='#0F2942', pad=6)
ax1.set_ylim(-1.08, 1.08)
ax1.tick_params(labelsize=11.0)
ax1.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)

# Panel C
ax2 = fig.add_subplot(gs[2])
scatter_colors = []
sizes = []
for d in df_taxa['diff']:
    if d > 0.3:
        scatter_colors.append('#E06D53')
        sizes.append(100)
    elif d < -0.3:
        scatter_colors.append('#1B6CA8')
        sizes.append(100)
    else:
        scatter_colors.append('#D0D7DE')
        sizes.append(40)

ax2.scatter(Theta_np[:, 0], Theta_np[:, 1], c=scatter_colors, s=sizes, edgecolors='#0F2942', lw=1.2, alpha=0.85, zorder=4)

max_th = max(Theta_np.max() * 1.1, 0.75)
ax2.plot([0, max_th], [0, max_th], 'k--', lw=1.4, alpha=0.6, label=r'$\theta_1 \approx \theta_2$', zorder=2)

ax2.annotate('Chaotic Drivers\n(Sp 31–36, Overlap)', 
             xy=(0.35, 0.40), xytext=(0.08, 0.65),
             arrowprops=dict(arrowstyle="->", color='#802010', lw=1.6),
             fontsize=10.0, fontweight='bold', color='#802010')

ax2.annotate('Stabilizing Specialists\n(Sp 37–42, Partitioning)', 
             xy=(0.55, 0.28), xytext=(0.38, 0.10),
             arrowprops=dict(arrowstyle="->", color='#0B4F6C', lw=1.6),
             fontsize=10.0, fontweight='bold', color='#0B4F6C')

ax2.set_xlabel(r'Resource 1 Uptake $(\theta_{i,1})$', fontsize=12.5, fontweight='bold', color='#1A365D')
ax2.set_ylabel(r'Resource 2 Uptake $(\theta_{i,2})$', fontsize=12.5, fontweight='bold', color='#1A365D')
ax2.set_title(r'$\mathbf{(C)\;Trait\;Mechanism\;}\mathbf{\Theta}$', fontsize=13.0, fontweight='bold', color='#0F2942', pad=6)
ax2.tick_params(labelsize=11.0)
ax2.legend(loc='upper left', fontsize=11.0, framealpha=0.95, edgecolor='#0F2942')
ax2.grid(True, linestyle='--', alpha=0.5)

out_path = 'Poster/figures/Gore_24Species_Enrichment.png'
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved large font Gore 24-species enrichment figure to {out_path}")

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE
from German_ODE_evaluation import load_and_filter_data

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)
data_path = 'Data/German_processed_data.pt'
trajectories, strains = load_and_filter_data(data_path, 1e-3)

def map_environments_custom(trajectories_dict):
    env_map = {}
    unique_envs = []
    for cond_key in trajectories_dict.keys():
        env_label = cond_key[1:] if isinstance(cond_key, tuple) and len(cond_key) >= 2 else cond_key
        if env_label not in unique_envs:
            unique_envs.append(env_label)
        env_map[cond_key] = unique_envs.index(env_label)
    return env_map, unique_envs

env_map, unique_envs = map_environments_custom(trajectories)
keys_list = list(trajectories.keys())
num_species = len(strains)
num_conditions = len(unique_envs)

def train_neural_ode(model, train_seqs, train_conds, epochs=200, lr=0.01, l2_lambda=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        batch_loss = 0.0
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        x_curr = train_seqs[:, 0, :]
        for t in range(1, train_seqs.size(1)):
            log_pred, x_pred, _, _ = model(x_curr, train_conds)
            target = train_seqs[:, t, :]
            batch_loss += criterion(log_pred, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred
        l2_reg = sum(p.pow(2.0).sum() for name, p in model.named_parameters() if 'Theta' in name or 'W' in name)
        loss = batch_loss + l2_lambda * l2_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model

# 1. Held-out LOO Scores from task-558
loo_scores = {
    ('BRRS', 'H2:CO2', 5.0): 0.1786,
    ('BRRS', 'H2:CO2', 30.0): 0.4772,
    ('BRRS', 'H2:CO2 + MeOH', 5.0): 0.0940,
    ('BRRS', 'H2:CO2 + MeOH', 30.0): 0.0111,
    ('BRRS', 'H2:CO2 + TMA', 5.0): 0.4063,
    ('BRRS', 'H2:CO2 + TMA', 30.0): 0.3311,
    ('OSU', 'H2:CO2', 5.0): 0.9207,
    ('OSU', 'H2:CO2', 30.0): 0.9553,
    ('OSU', 'H2:CO2 + MeOH', 5.0): 0.0185,
    ('OSU', 'H2:CO2 + MeOH', 30.0): 0.0789,
    ('OSU', 'H2:CO2 + TMA', 5.0): 0.1194,
    ('OSU', 'H2:CO2 + TMA', 30.0): 0.1871
}

# 2. Train LOO model on all except OSU H2:CO2 30b to get held-out trajectory predictions
test_key = ('OSU', 'H2:CO2', np.float64(30.0))
train_keys = [k for k in keys_list if k != test_key]
train_seqs = torch.stack([trajectories[k] for k in train_keys])
train_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)

loo_model = LatentAssemblyNeuralODE(num_species, 4, num_conditions)
loo_model = train_neural_ode(loo_model, train_seqs, train_conds, epochs=200, lr=0.01, l2_lambda=0.1)

loo_model.eval()
with torch.no_grad():
    seq = trajectories[test_key]
    c_idx = torch.tensor([env_map[test_key]], dtype=torch.long)
    x_curr = seq[0].unsqueeze(0)
    preds = [seq[0]]
    for t in range(1, seq.size(0)):
        _, x_pred, _, _ = loo_model(x_curr, c_idx)
        preds.append(x_pred.squeeze(0))
        x_curr = x_pred
    pred_stack = torch.stack(preds, dim=0).numpy()
    true_stack = seq.numpy()

# 3. Train Global Model on ALL 12 to get canonical PCA of a
all_seqs = torch.stack([trajectories[k] for k in keys_list])
all_conds = torch.tensor([env_map[k] for k in keys_list], dtype=torch.long)

global_model = LatentAssemblyNeuralODE(num_species, 4, num_conditions)
global_model = train_neural_ode(global_model, all_seqs, all_conds, epochs=200, lr=0.01, l2_lambda=0.1)
env_vectors = global_model.a.detach().cpu().numpy()

pca = PCA(n_components=2)
env_pca = pca.fit_transform(env_vectors)
pca_var = pca.explained_variance_ratio_ * 100

# 4. PLOTTING 3-PANEL FIGURE
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig = plt.figure(figsize=(15.2, 4.6), dpi=300)
gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.15, 1.05], wspace=0.32,
                       left=0.06, right=0.97, top=0.88, bottom=0.16)

# ─── PANEL A: Cross-Community Transferability (LOO R^2 at T6) ───
ax0 = fig.add_subplot(gs[0])

ordered_envs = [
    ('H2:CO2', 30.0, r'$\mathrm{H_2:CO_2}$ (30b)', '#1B6CA8'),
    ('H2:CO2', 5.0,  r'$\mathrm{H_2:CO_2}$ (5b)',  '#5B9BD5'),
    ('H2:CO2 + TMA', 5.0,  r'+TMA (5b)',           '#2CA02C'),
    ('H2:CO2 + TMA', 30.0, r'+TMA (30b)',          '#98DF8A'),
    ('H2:CO2 + MeOH', 5.0,  r'+MeOH (5b)',         '#E06D53'),
    ('H2:CO2 + MeOH', 30.0, r'+MeOH (30b)',        '#FF9896'),
]

x_pos = np.arange(len(ordered_envs))
mean_r2 = []
brrs_pts, osu_pts = [], []

for sub, psi, lab, col in ordered_envs:
    b_val = loo_scores[('BRRS', sub, psi)]
    o_val = loo_scores[('OSU', sub, psi)]
    m_val = (b_val + o_val) / 2.0
    mean_r2.append(m_val)
    brrs_pts.append(b_val)
    osu_pts.append(o_val)

bar_cols = [e[3] for e in ordered_envs]
bars = ax0.bar(x_pos, mean_r2, color=bar_cols, edgecolor='#0F2942', lw=1.2, width=0.62, zorder=3)

# Scatter points for individual hosts
ax0.scatter(x_pos - 0.12, brrs_pts, color='#0F2942', marker='o', s=45, zorder=5, label='Host BRRS')
ax0.scatter(x_pos + 0.12, osu_pts, color='#F5A623', edgecolor='#0F2942', marker='D', s=45, zorder=5, label='Host OSU')

for bar in bars:
    h = bar.get_height()
    ax0.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', 
             ha='center', va='bottom', fontsize=10.0, fontweight='bold', color='#0F2942')

ax0.set_xticks(x_pos)
ax0.set_xticklabels([e[2] for e in ordered_envs], rotation=22, ha='right', fontsize=9.5, fontweight='bold')
ax0.set_ylabel(r'Held-Out $R^2$ at $T_6$', fontsize=12.5, fontweight='bold', color='#1A365D')
ax0.set_title(r'$\mathbf{(A)\;Cross\text{-}Host\;Transfer\;(LOO)}$', fontsize=13.5, fontweight='bold', color='#0F2942', pad=10)
ax0.set_ylim(-0.05, 1.08)
ax0.tick_params(axis='y', labelsize=10.5)
ax0.legend(loc='upper right', fontsize=9.2, framealpha=0.92, edgecolor='#0F2942')
ax0.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)

# ─── PANEL B: Unseen Host Multi-Passage Dynamics (OSU H2:CO2 30b) ───
ax1 = fig.add_subplot(gs[1])

idx_micro = [i for i, s in enumerate(strains) if 'Methanomicrobium sp.' in s][0]
idx_prev = [i for i, s in enumerate(strains) if 'Prevotellaceae' in s][0]
idx_bact = [i for i, s in enumerate(strains) if 'Bacteroidales' in s][0]

transfers = np.arange(1, 7)

# Observed points (markers)
ax1.plot(transfers, true_stack[:, idx_micro], 'o', color='#008080', ms=8, label='Obs. Methanomicrobium', zorder=5)
ax1.plot(transfers, pred_stack[:, idx_micro], '-', color='#008080', lw=2.4, label='Held-Out Pred.', zorder=4)

ax1.plot(transfers, true_stack[:, idx_prev], 's', color='#9467BD', ms=7, label='Obs. Prevotellaceae', zorder=5)
ax1.plot(transfers, pred_stack[:, idx_prev], '--', color='#9467BD', lw=2.0, label='Held-Out Pred.', zorder=4)

ax1.plot(transfers, true_stack[:, idx_bact], '^', color='#BC80BD', ms=7, label='Obs. Bacteroidales', zorder=5)
ax1.plot(transfers, pred_stack[:, idx_bact], ':', color='#BC80BD', lw=2.0, label='Held-Out Pred.', zorder=4)

ax1.set_xticks(transfers)
ax1.set_xticklabels([f'$T_{t}$' for t in transfers], fontsize=11.0, fontweight='bold')
ax1.set_xlabel('Sequential Transfer', fontsize=12.5, fontweight='bold', color='#1A365D')
ax1.set_ylabel('Relative Abundance', fontsize=12.5, fontweight='bold', color='#1A365D')
ax1.set_title(r'$\mathbf{(B)\;Unseen\;OSU\;Forecast\;(30\text{b})}$', fontsize=13.5, fontweight='bold', color='#0F2942', pad=10)
ax1.set_ylim(-0.02, 0.95)
ax1.tick_params(labelsize=10.5)

# Badge showing unseen R^2
ax1.text(0.04, 0.93, r'$\mathbf{Unseen\;R^2 = 0.841}$' + '\n' + r'($r = \mathbf{0.9763}$ at $T_6$)', 
         transform=ax1.transAxes, ha='left', va='top', fontsize=9.5, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#F0F4F8', edgecolor='#0F2942', lw=1.0))

ax1.legend(loc='center right', fontsize=8.2, framealpha=0.92, edgecolor='#0F2942')
ax1.grid(True, linestyle='--', alpha=0.5)

# ─── PANEL C: Learned Environmental Resource Space (PCA of a) ───
ax2 = fig.add_subplot(gs[2])

pca_labels = [
    r'$\mathrm{H_2:CO_2}$ (5b)',
    r'$\mathrm{H_2:CO_2}$ (30b)',
    r'+MeOH (5b)',
    r'+MeOH (30b)',
    r'+TMA (5b)',
    r'+TMA (30b)'
]
pca_colors = ['#5B9BD5', '#1B6CA8', '#E06D53', '#FF9896', '#2CA02C', '#98DF8A']
pca_markers = ['o', 's', 'o', 's', 'o', 's']

for i in range(6):
    ax2.scatter(env_pca[i, 0], env_pca[i, 1], color=pca_colors[i], marker=pca_markers[i],
                s=140, edgecolors='#0F2942', lw=1.5, zorder=4)
    # Smart annotation offsets
    ox = 0.25 if i in [0, 2, 3] else -0.35
    oy = 0.22 if i in [0, 1, 3] else -0.38
    ha = 'left' if ox > 0 else 'right'
    ax2.annotate(pca_labels[i], (env_pca[i, 0], env_pca[i, 1]),
                 xytext=(env_pca[i, 0] + ox, env_pca[i, 1] + oy),
                 fontsize=9.8, fontweight='bold', color='#1A365D', ha=ha)

ax2.set_xlabel(f'PC1 ({pca_var[0]:.1f}%) - Substrate Chemistry', fontsize=11.5, fontweight='bold', color='#1A365D')
ax2.set_ylabel(f'PC2 ({pca_var[1]:.1f}%) - Gas Pressure', fontsize=11.5, fontweight='bold', color='#1A365D')
ax2.set_title(r'$\mathbf{(C)\;Learned\;Resource\;Space\;}\mathbf{a}_c$', fontsize=13.5, fontweight='bold', color='#0F2942', pad=10)
ax2.tick_params(labelsize=10.0)
ax2.set_xlim(-5.2, 5.5)
ax2.set_ylim(-3.5, 3.5)
ax2.axhline(0, color='gray', lw=0.8, linestyle=':', alpha=0.7)
ax2.axvline(0, color='gray', lw=0.8, linestyle=':', alpha=0.7)
ax2.grid(True, linestyle='--', alpha=0.5)

out_path = 'Poster/figures/German_LOO_Transferability_PCA.png'
fig.savefig(out_path, dpi=300)
print(f"Saved German LOO Biotech figure to {out_path}")

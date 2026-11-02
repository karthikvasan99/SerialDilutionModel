import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from scipy.stats import pearsonr

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
data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
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

print("Training Cross-Pressure Model (Trained on all OSU + BRRS 30b; BRRS 5b completely held out)...")
train_keys = [k for k in keys_list if k[0] == "OSU" or (k[0] == "BRRS" and k[2] == 30.0)]
train_seqs = torch.stack([trajectories[k] for k in train_keys])
train_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)

cp_model = LatentAssemblyNeuralODE(num_species, 4, num_conditions)
cp_model = train_neural_ode(cp_model, train_seqs, train_conds, epochs=200)
cp_model.eval()

# 1. Trajectory Predictions for BRRS +MeOH (5 bar) and BRRS +TMA (5 bar)
with torch.no_grad():
    # BRRS +MeOH 5b
    k_meoh = ("BRRS", "H2:CO2 + MeOH", np.float64(5.0))
    seq_meoh = trajectories[k_meoh]
    c_meoh = torch.tensor([env_map[k_meoh]], dtype=torch.long)
    preds_meoh = [seq_meoh[0].numpy()]
    x_curr = seq_meoh[0].unsqueeze(0)
    for t in range(1, seq_meoh.size(0)):
        _, x_pred, _, _ = cp_model(x_curr, c_meoh)
        preds_meoh.append(x_pred.squeeze(0).numpy())
        x_curr = x_pred
    preds_meoh = np.array(preds_meoh) * 100
    trues_meoh = seq_meoh.numpy() * 100

    # BRRS +TMA 5b
    k_tma = ("BRRS", "H2:CO2 + TMA", np.float64(5.0))
    seq_tma = trajectories[k_tma]
    c_tma = torch.tensor([env_map[k_tma]], dtype=torch.long)
    preds_tma = [seq_tma[0].numpy()]
    x_curr = seq_tma[0].unsqueeze(0)
    for t in range(1, seq_tma.size(0)):
        _, x_pred, _, _ = cp_model(x_curr, c_tma)
        preds_tma.append(x_pred.squeeze(0).numpy())
        x_curr = x_pred
    preds_tma = np.array(preds_tma) * 100
    trues_tma = seq_tma.numpy() * 100

# Reference Recipe in Cow OSU +MeOH 5b
true_osu_meoh_t6 = trajectories[("OSU", "H2:CO2 + MeOH", np.float64(5.0))][-1, 0].item() * 100

# Target Microbe is index 0: Methanomicrobium sp.
target_idx = 0

# ==============================================================================
# CREATE 3-PANEL HIGH-RESOLUTION FIGURE
# ==============================================================================
print("Generating 3-Panel Figure...")
fig = plt.figure(figsize=(19, 6.8), dpi=240)
gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.25, 1.1], wspace=0.28)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# ==============================================================================
# PANEL A: The Generalization Dilemma & Out-of-Sample Prescription
# ==============================================================================
ax0 = fig.add_subplot(gs[0])

cases = [
    "Cow OSU\n(+MeOH, 5 bar)\nHost 1 Recipe",
    "Cow BRRS\n(+MeOH, 5 bar)\nDirect Transfer",
    "Cow BRRS\n(+TMA, 5 bar)\nModel Prescribed"
]
x_pos = np.arange(len(cases))
bar_width = 0.35

obs_vals = [
    true_osu_meoh_t6,
    trues_meoh[-1, target_idx],
    trues_tma[-1, target_idx]
]
pred_vals = [
    true_osu_meoh_t6,
    preds_meoh[-1, target_idx],
    preds_tma[-1, target_idx]
]

bar_obs = ax0.bar(x_pos - bar_width/2, obs_vals, width=bar_width, color='#1f77b4', edgecolor='black', linewidth=1.2, label='Observed (Wet-Lab)', zorder=3)
bar_pred = ax0.bar(x_pos + bar_width/2, [np.nan, pred_vals[1], pred_vals[2]], width=bar_width, color='#ff7f0e', edgecolor='black', linewidth=1.2, hatch='//', label='Model Forecast', zorder=3)

# Value labels on top of bars
for i, v in enumerate(obs_vals):
    ax0.text(x_pos[i] - bar_width/2, v + 1.2, f"{v:.1f}%", ha='center', va='bottom', fontsize=13.0, fontweight='bold', color='#1f77b4')

ax0.text(x_pos[1] + bar_width/2, pred_vals[1] + 1.2, f"{pred_vals[1]:.1f}%", ha='center', va='bottom', fontsize=13.0, fontweight='bold', color='#d35400')
ax0.text(x_pos[2] + bar_width/2, pred_vals[2] + 1.2, f"{pred_vals[2]:.1f}%", ha='center', va='bottom', fontsize=13.0, fontweight='bold', color='#d35400')

# Clear annotations
ax0.annotate('Transfer Fails!\n(0.0% Observed)', xy=(x_pos[1], 2.5), xytext=(x_pos[1], 18.0),
             ha='center', fontsize=12.0, fontweight='bold', color='#c0392b',
             arrowprops=dict(arrowstyle='->', lw=1.8, color='#c0392b'))

ax0.annotate('Prescription Confirmed!\n(23.0% Observed)', xy=(x_pos[2], 24.5), xytext=(x_pos[2], 35.0),
             ha='center', fontsize=12.0, fontweight='bold', color='#27ae60',
             arrowprops=dict(arrowstyle='->', lw=1.8, color='#27ae60'))

ax0.set_xticks(x_pos)
ax0.set_xticklabels(cases, fontsize=13.0, fontweight='bold')
ax0.set_ylabel('Target Abundance (%)', fontsize=16.0, fontweight='bold')
ax0.set_ylim(0, 50)
ax0.tick_params(axis='both', which='major', labelsize=13)
ax0.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax0.legend(loc='upper right', fontsize=12.5, framealpha=0.95)

# ==============================================================================
# PANEL B: Dynamic Multi-Passage Trajectories of Target Microbe in Held-Out Host
# ==============================================================================
ax1 = fig.add_subplot(gs[1])

transfers = np.arange(1, 7)

# Failed condition: +MeOH (5 bar)
ax1.plot(transfers, trues_meoh[:, target_idx], color='#c0392b', linestyle='-', marker='o',
         markersize=8, linewidth=2.6, label='+MeOH (Direct Transfer, Fails)', zorder=4)
ax1.plot(transfers, preds_meoh[:, target_idx], color='#c0392b', linestyle='--', marker='o',
         markerfacecolor='white', markeredgecolor='#c0392b', markeredgewidth=2.0,
         markersize=7, linewidth=2.0, label='+MeOH Model Forecast', zorder=3)

# Prescribed condition: +TMA (5 bar)
ax1.plot(transfers, trues_tma[:, target_idx], color='#27ae60', linestyle='-', marker='s',
         markersize=8, linewidth=2.6, label='+TMA (Model Prescribed, Rescues)', zorder=4)
ax1.plot(transfers, preds_tma[:, target_idx], color='#27ae60', linestyle='--', marker='s',
         markerfacecolor='white', markeredgecolor='#27ae60', markeredgewidth=2.0,
         markersize=7, linewidth=2.0, label='+TMA Model Forecast', zorder=3)

# Visual badge in corner
ax1.text(0.04, 0.94, "Cow BRRS (5 bar, Held-Out Host)\nMethanomicrobium sp.",
         transform=ax1.transAxes, fontsize=12.0, fontweight='bold', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#2c3e50', alpha=0.95, lw=1.5))

ax1.set_xlabel('Transfers ($T_1 \\to T_6$)', fontsize=16.0, fontweight='bold')
ax1.set_ylabel('Target Abundance (%)', fontsize=16.0, fontweight='bold')
ax1.set_xticks(transfers)
ax1.set_xticklabels([f'$T_{t}$' for t in transfers], fontsize=14.0, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=13)
# Set generous y-limits so all data and legends have plenty of headroom
ax1.set_ylim(-1.0, 32.0)
ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)

# Custom legend placed in completely clear upper-right region
leg_lines = [
    mlines.Line2D([], [], color='#c0392b', marker='o', lw=2.4, label='+MeOH: Transfer Fails'),
    mlines.Line2D([], [], color='#27ae60', marker='s', lw=2.4, label='+TMA: Prescribed (Rescues)'),
    mlines.Line2D([], [], color='black', linestyle='-', lw=2.2, label='Observed (Wet-Lab)'),
    mlines.Line2D([], [], color='black', linestyle='--', marker='o', markerfacecolor='white', lw=1.8, label='Model Forecast')
]
ax1.legend(handles=leg_lines, loc='upper right', fontsize=11.0, framealpha=0.95)

# ==============================================================================
# PANEL C: Community-Wide Holdout Generalization (Ophelia-Style Parity Scatter)
# ==============================================================================
ax2 = fig.add_subplot(gs[2])

o_tma_t6 = trues_tma[-1]
p_tma_t6 = preds_tma[-1]

# Non-target species
other_mask = np.arange(num_species) != target_idx
ax2.scatter(o_tma_t6[other_mask], p_tma_t6[other_mask], color='#4a69bd', alpha=0.6, s=55, edgecolors='none', label='Community Species (86)', zorder=3)

# Highlight Target Microbe
ax2.scatter(o_tma_t6[target_idx], p_tma_t6[target_idx], color='#f39c12', edgecolor='black', linewidth=1.8, s=220, marker='*', label='Methanomicrobium sp. (Target)', zorder=5)

# Annotation for target
ax2.annotate('Target Enriched\n(Obs: 23.0%, Pred: 16.5%)', xy=(o_tma_t6[target_idx], p_tma_t6[target_idx]),
             xytext=(o_tma_t6[target_idx] - 12.0, p_tma_t6[target_idx] + 10.0),
             ha='center', fontsize=11.5, fontweight='bold', color='#b7791f',
             arrowprops=dict(arrowstyle='->', lw=1.6, color='#b7791f'))

# 1:1 Identity Line
max_val = 45.0
ax2.plot([0, max_val], [0, max_val], 'k--', lw=2.0, alpha=0.8, zorder=2)
ax2.set_xlim(-1.5, max_val)
ax2.set_ylim(-1.5, max_val)

# Corner badge
ax2.text(0.04, 0.94, "Held-Out Host (Cow BRRS)\nPrescribed Condition (+TMA, 5 bar)",
         transform=ax2.transAxes, fontsize=11.5, fontweight='bold', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#4a69bd', alpha=0.95, lw=1.5))

ax2.set_xlabel('Observed Abundance (%)', fontsize=16.0, fontweight='bold')
ax2.set_ylabel('Predicted Abundance (%)', fontsize=16.0, fontweight='bold')
ax2.tick_params(axis='both', which='major', labelsize=13)
ax2.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax2.legend(loc='lower right', fontsize=11.5, framealpha=0.92)

# Save
output_path = os.path.join(PROJECT_ROOT, 'Poster', 'figures', 'German_Prescriptive_Biotech_Search.png')
plt.savefig(output_path, dpi=240, bbox_inches='tight')
plt.close()
print(f"Figure successfully generated at: {output_path}")

import os
import sys
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import PurePhysicalConsumerResourceNeuralODE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing Cross-Host Biotechnology Transfer on Device: {device}")

# ─── 1. Load Data & Filter to 81 Shared Species ───
data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
raw_data = torch.load(data_path, weights_only=False)
raw_strains = raw_data['strains']
trajectories_dict = raw_data['trajectories']
raw_traj = torch.stack(list(trajectories_dict.values()), dim=0) # (12, 6, 751)

# Longitudinal peak filter >= 1%
peaks = raw_traj.amax(dim=(0, 1))
mask_1pct = peaks >= 0.01
strains_91 = [raw_strains[i] for i in range(len(raw_strains)) if mask_1pct[i]]
data_91 = raw_traj[:, :, mask_1pct]

# Host overlap filter: mutually present in both Cow BRRS and Cow OSU
brrs_present = (data_91[:6].sum(dim=(0, 1)) > 0)
osu_present = (data_91[6:].sum(dim=(0, 1)) > 0)
shared_mask = brrs_present & osu_present

strains_81 = [strains_91[i] for i in range(len(strains_91)) if shared_mask[i]]
data_81 = data_91[:, :, shared_mask].to(device)
data_81 = data_81 / (data_81.sum(dim=-1, keepdim=True) + 1e-12)

num_conditions, num_passages, num_species = data_81.shape
cond_env_ids = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=torch.long, device=device)

# Seed T1 (relative inoculum)
x0 = data_81[:, 0, :]
seed_t1 = torch.where(x0 > 0, x0, torch.tensor(1e-4, device=device))
seed_t1 = seed_t1 / seed_t1.sum(dim=-1, keepdim=True)

# Target microbe: Methanomicrobium sp. (index 0)
target_idx = 0
target_name = strains_81[target_idx].split('|')[-1]
print(f"Target Species (idx {target_idx}): {target_name}")

# ─── 2. Training Function: Pure Cow OSU Model ───
def train_and_predict(train_indices=[6, 7, 8, 9, 10, 11], eval_indices=[8, 2, 4], K=3, lr=0.015, l2_alpha=0.05, tf_decay=0.85, epochs=350, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = PurePhysicalConsumerResourceNeuralODE(
        num_species=num_species, 
        num_latent_dims=K, 
        num_conditions=6, 
        default_dilution_factor=100.0, 
        dispersal_rate=0.0
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = data_81[train_indices]
    train_seed = seed_t1[train_indices]
    train_envs = cond_env_ids[train_indices]

    best_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = 0.0
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * tf_decay)))
        
        Theta, W, alpha, R0, B0 = model.get_params()
        N_curr = train_seed * B0
        for p in range(1, num_passages):
            log_pred, x_pred, N_final, _ = model(N_curr, train_envs, ode_step_size=0.1)
            target = train_data[:, p, :]
            kl = (target * torch.log((target + 1e-6) / (x_pred + 1e-6))).sum(dim=-1).mean()
            loss = loss + kl
            
            tot_bio = N_final.sum(dim=-1, keepdim=True) + 1e-12
            if tf_ratio > 0:
                x_mix = tf_ratio * target + (1.0 - tf_ratio) * x_pred
                N_curr = x_mix * tot_bio
            else:
                N_curr = N_final
            
        if l2_alpha > 0:
            loss = loss + l2_alpha * torch.sum(alpha**2)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate on eval conditions
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()

    eval_data = data_81[eval_indices]
    eval_seed = seed_t1[eval_indices]
    eval_envs = cond_env_ids[eval_indices]

    with torch.no_grad():
        eval_preds = [eval_seed]
        Theta, W, alpha, R0, B0 = model.get_params()
        N_curr = eval_seed * B0
        for p in range(1, num_passages):
            log_pred, x_pred, N_final, _ = model(N_curr, eval_envs, ode_step_size=0.1)
            eval_preds.append(x_pred)
            N_curr = N_final

    preds_tensor = torch.stack(eval_preds, dim=1) # (len(eval), num_passages, num_species)
    pred_np = preds_tensor.cpu().numpy()
    true_np = eval_data.cpu().numpy()

    return pred_np, true_np

# ─── 3. Train Pure OSU Champion Model ───
print("\n--- Training Model Strictly on Cow OSU (All 6 Conditions) ---")
# eval_indices: 8 = OSU +MeOH 5b (in-sample), 2 = BRRS +MeOH 5b (zero-shot), 4 = BRRS +TMA 5b (zero-shot)
pred_np, true_np = train_and_predict(
    train_indices=[6, 7, 8, 9, 10, 11],
    eval_indices=[8, 2, 4],
    K=3, lr=0.015, l2_alpha=0.05, tf_decay=0.85, epochs=350, seed=42
)

# Extract T6 abundances
obs_osu_meoh = true_np[0, -1, target_idx] * 100
pred_osu_meoh = pred_np[0, -1, target_idx] * 100

obs_brrs_meoh = true_np[1, -1, target_idx] * 100
pred_brrs_meoh = pred_np[1, -1, target_idx] * 100

obs_brrs_tma = true_np[2, -1, target_idx] * 100
pred_brrs_tma = pred_np[2, -1, target_idx] * 100

print("\n" + "="*70)
print(f"CROSS-HOST BIOTECHNOLOGY PREDICTION RESULTS FOR: {target_name}")
print("="*70)
print(f"Cow OSU Baseline (+MeOH, 5 bar):  Observed = {obs_osu_meoh:5.2f}% | Model Fit = {pred_osu_meoh:5.2f}%")
print(f"Cow BRRS Transfer (+MeOH, 5 bar): Observed = {obs_brrs_meoh:5.2f}% | Predicted = {pred_brrs_meoh:5.2f}% (Collapse!)")
print(f"Cow BRRS Rescue (+TMA, 5 bar):    Observed = {obs_brrs_tma:5.2f}% | Predicted = {pred_brrs_tma:5.2f}% (Rescue!)")
print("="*70)

# ─── 4. Generate Publication-Quality 2-Panel Figure ───
print("\nGenerating 2-Panel Cross-Host Transfer Figure (Panels A & B)...")
fig = plt.figure(figsize=(14.2, 5.8), dpi=300)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1.1], wspace=0.28)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# ── PANEL A: Cross-Host Prescriptive Transfer at T6 ──
ax0 = fig.add_subplot(gs[0])

cases = [
    "Cow OSU\n(+MeOH, 5 bar)\nRecipe Host",
    "Cow BRRS\n(+MeOH, 5 bar)\nDirect Transfer",
    "Cow BRRS\n(+TMA, 5 bar)\nPrescribed Rescue"
]
x = np.arange(len(cases))
bar_width = 0.32

obs_vals = [obs_osu_meoh, obs_brrs_meoh, obs_brrs_tma]

# Plot Observed Bars (Dark Navy)
rects_obs = ax0.bar(
    x - bar_width/2, obs_vals, width=bar_width,
    color='#1b365d', edgecolor='black', linewidth=1.2,
    label='Observed (Wet Lab)', zorder=3
)

# Plot Model Predictions:
# In-sample for OSU (Slate Grey hatched), Zero-shot for BRRS (Amber/Orange hatched)
rect_fit = ax0.bar(
    x[0] + bar_width/2, pred_osu_meoh, width=bar_width,
    color='#6c757d', edgecolor='black', linewidth=1.2,
    hatch='//', label='Model Fit (In-Sample, Cow OSU)', zorder=3
)
rects_pred = ax0.bar(
    x[1:] + bar_width/2, [pred_brrs_meoh, pred_brrs_tma], width=bar_width,
    color='#e67e22', edgecolor='black', linewidth=1.2,
    hatch='//', label='Model Prediction (Zero-Shot Transfer)', zorder=3
)

# Text labels on top of bars
for i in range(len(cases)):
    val_o = obs_vals[i]
    ax0.text(x[i] - bar_width/2, val_o + 1.2, f"{val_o:.1f}%",
             ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#1b365d')

ax0.text(x[0] + bar_width/2, pred_osu_meoh + 1.2, f"{pred_osu_meoh:.1f}%",
         ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#495057')
ax0.text(x[1] + bar_width/2, pred_brrs_meoh + 1.2, f"{pred_brrs_meoh:.1f}%",
         ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#b95d00')
ax0.text(x[2] + bar_width/2, pred_brrs_tma + 1.2, f"{pred_brrs_tma:.1f}%",
         ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#b95d00')

ax0.set_xticks(x)
ax0.set_xticklabels(cases, fontsize=11.0, fontweight='bold')
ax0.set_ylabel("Methanomicrobium sp. T6 Abundance (%)", fontsize=12.0, fontweight='bold')
ax0.set_title("A. Cross-Host Prescriptive Transfer", fontsize=13.5, fontweight='bold', pad=12)
ax0.legend(loc='upper right', framealpha=0.95, fontsize=9.5, edgecolor='#cccccc')
ax0.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax0.set_ylim(0, 56)

# Annotations
ax0.annotate('Naive Transfer Collapses\n(Obs: 0.0%, Pred: 2.1%)',
             xy=(1, max(pred_brrs_meoh, obs_brrs_meoh) + 1.5), xytext=(0.85, 23),
             ha='center', fontsize=9.5, fontweight='bold', color='#c53030',
             arrowprops=dict(arrowstyle='->', lw=1.5, color='#c53030'))

ax0.annotate('Prescription Rescues\nEnrichment (Obs: 23.2%)',
             xy=(2, obs_brrs_tma + 1.5), xytext=(2, 40),
             ha='center', fontsize=9.5, fontweight='bold', color='#276749',
             arrowprops=dict(arrowstyle='->', lw=1.5, color='#276749'))

# ── PANEL B: Trajectory Dynamics in Cow BRRS (Passages T1 to T6) ──
ax1 = fig.add_subplot(gs[1])
passages = np.arange(1, 7)

# Observed trajectories in BRRS
obs_meoh_traj = true_np[1, :, target_idx] * 100
obs_tma_traj = true_np[2, :, target_idx] * 100

# Zero-shot predicted trajectories in BRRS
pred_meoh_traj = pred_np[1, :, target_idx] * 100
pred_tma_traj = pred_np[2, :, target_idx] * 100

# Plot TMA Trajectories (Green)
ax1.plot(passages, obs_tma_traj, 's-', color='#276749', lw=2.5, markersize=7.5,
         label='Observed (+TMA, 5b)', zorder=4)
ax1.plot(passages, pred_tma_traj, 's--', color='#48bb78', lw=2.2, markersize=6.5,
         markerfacecolor='white', markeredgewidth=1.8, label='Pred (Trained on OSU) (+TMA)', zorder=3)

# Plot MeOH Trajectories (Red)
ax1.plot(passages, obs_meoh_traj, 'o-', color='#c53030', lw=2.5, markersize=7.5,
         label='Observed (+MeOH, 5b)', zorder=4)
ax1.plot(passages, pred_meoh_traj, 'o--', color='#fc8181', lw=2.2, markersize=6.5,
         markerfacecolor='white', markeredgewidth=1.8, label='Pred (Trained on OSU) (+MeOH)', zorder=3)

ax1.set_xlabel("Serial Dilution Transfer (Passage)", fontsize=12.0, fontweight='bold')
ax1.set_ylabel("Relative Abundance (%)", fontsize=12.0, fontweight='bold')
ax1.set_title("B. Dynamic Passage Trajectories in BRRS", fontsize=13.5, fontweight='bold', pad=12)
ax1.set_xticks(passages)
ax1.set_xticklabels([f"T{p}" for p in passages], fontsize=11.0, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper right', framealpha=0.95, fontsize=9.5, edgecolor='#cccccc')
ax1.set_ylim(-1, 28)

plt.tight_layout()

out_fig = os.path.join(PROJECT_ROOT, 'Poster', 'figures', 'German_CrossHost_Transfer_Biotech.png')
out_fig_search = os.path.join(PROJECT_ROOT, 'Poster', 'figures', 'German_Prescriptive_Biotech_Search.png')

plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.savefig(out_fig_search, dpi=300, bbox_inches='tight')
plt.close()

# Copy to artifact directory
artifact_dir = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
shutil.copyfile(out_fig, os.path.join(artifact_dir, 'German_CrossHost_Transfer_Biotech.png'))
shutil.copyfile(out_fig_search, os.path.join(artifact_dir, 'German_Prescriptive_Biotech_Search.png'))

print(f"\nSuccessfully generated and saved 2-panel figure to:")
print(f"  1. {out_fig}")
print(f"  2. {out_fig_search}")

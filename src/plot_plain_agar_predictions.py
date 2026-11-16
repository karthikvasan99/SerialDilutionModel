import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Set publication style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARTIFACT_DIR = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location=device, weights_only=False)

plain_c_tensor = data['plain_c_tensor'] # (3, 6, 50)
plain_s_tensor = data['plain_s_tensor'] # (3, 6, 50)
strains = data['strains']

# Load model checkpoint
ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'OptimalCRM_K4_Model.pt')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

# Recreate model to get predictions
from evaluate_plain_agar_prediction import OptimalCRM, p0, compute_r2

model = OptimalCRM(N=50, K=4, c_w=0.70).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()

with torch.no_grad():
    pred_c, pred_s = model.forward_carrier(p0.to(device), is_mucin=False, tf_ratio=0.0, gamma_detach=1.0)
    pred_c_gamma, pred_s_gamma = model.forward_carrier(p0.to(device), is_mucin=False, tf_ratio=0.0, gamma_detach=2.0)

# Observed mean across 3 biological vessels
obs_s = plain_s_tensor.mean(dim=0).cpu().numpy() # (6, 50)
obs_c = plain_c_tensor.mean(dim=0).cpu().numpy() # (6, 50)
pred_s_mean = pred_s.mean(dim=0).cpu().numpy()   # (6, 50)
pred_c_mean = pred_c.mean(dim=0).cpu().numpy()   # (6, 50)

# 1. Plot Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

# PANEL A: Scatter Predicted vs Observed (Supernatant & Carrier)
ax = axes[0]
obs_s_flat = obs_s.flatten()
pred_s_flat = pred_s_mean.flatten()
obs_c_flat = obs_c.flatten()
pred_c_flat = pred_c_mean.flatten()

r_s, _ = pearsonr(obs_s_flat, pred_s_flat)
r2_s = compute_r2(obs_s_flat, pred_s_flat)
r_c, _ = pearsonr(obs_c_flat, pred_c_flat)
r2_c = compute_r2(obs_c_flat, pred_c_flat)

ax.scatter(obs_s_flat, pred_s_flat, color='#1f77b4', alpha=0.6, s=30, label=f'Supernatant (r={r_s:.2f}, R²={r2_s:.2f})', edgecolors='none')
ax.scatter(obs_c_flat, pred_c_flat, color='#ff7f0e', alpha=0.6, s=30, label=f'Carrier (r={r_c:.2f}, R²={r2_c:.2f})', edgecolors='none')
ax.plot([0, 0.4], [0, 0.4], 'k--', lw=1.2, alpha=0.7, label='1:1 Ideal')

ax.set_xlim(-0.01, 0.42)
ax.set_ylim(-0.01, 0.42)
ax.set_xlabel('Observed Relative Abundance', fontsize=11, fontweight='bold')
ax.set_ylabel('Zero-Shot Predicted Abundance', fontsize=11, fontweight='bold')
ax.set_title('(A) Plain Agar Zero-Shot Transfer (All Passages)', fontsize=12, fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
ax.grid(True, linestyle=':', alpha=0.5)

# PANEL B: Per-Passage R^2 Trajectory (Model vs Null Inoculum Baseline)
ax = axes[1]
df_pass = pd.read_csv(os.path.join(PROJECT_ROOT, 'Data', 'PlainAgar_Passage_ZeroShot_Results.csv'))

passages = [f"P{i+1}" for i in range(6)]
p0_np = p0.mean(dim=0).cpu().numpy()
null_r2_s = [compute_r2(obs_s[p], p0_np) for p in range(6)]
null_r2_c = [compute_r2(obs_c[p], p0_np) for p in range(6)]

ax.plot(passages, df_pass['r2_supernatant'], 'o-', color='#1f77b4', lw=2.2, ms=6, label='Model Supernatant (Zero-Shot)')
ax.plot(passages, df_pass['r2_carrier'], 's-', color='#ff7f0e', lw=2.2, ms=6, label='Model Carrier (Zero-Shot)')
ax.plot(passages, null_r2_s, 'o--', color='#7f7f7f', lw=1.5, ms=5, alpha=0.7, label='Inoculum Baseline (Supernatant)')
ax.plot(passages, null_r2_c, 's--', color='#bcbd22', lw=1.5, ms=5, alpha=0.7, label='Inoculum Baseline (Carrier)')
ax.axhline(0, color='black', lw=0.8, linestyle='-', alpha=0.4)

ax.set_xlabel('Serial Dilution Passage', fontsize=11, fontweight='bold')
ax.set_ylabel('Prediction Accuracy (R²)', fontsize=11, fontweight='bold')
ax.set_title('(B) Per-Passage Zero-Shot Accuracy', fontsize=12, fontweight='bold')
ax.set_ylim(-0.6, 0.6)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
ax.grid(True, linestyle=':', alpha=0.5)

# PANEL C: Trajectory of Dominant Strains in Plain Agar Supernatant
ax = axes[2]
# Find top 4 most abundant strains across Plain Agar supernatant
mean_abund = obs_s.mean(axis=0)
top_strains_idx = np.argsort(mean_abund)[::-1][:4]

colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b']
for idx, strain_i in enumerate(top_strains_idx):
    strain_name = strains[strain_i][:20]
    ax.plot(passages, obs_s[:, strain_i], 'o-', color=colors[idx], lw=2.0, ms=5, label=f'{strain_name} (Obs)')
    ax.plot(passages, pred_s_mean[:, strain_i], 'x--', color=colors[idx], lw=1.5, ms=6, alpha=0.8, label=f'{strain_name} (Pred)')

ax.set_xlabel('Serial Dilution Passage', fontsize=11, fontweight='bold')
ax.set_ylabel('Relative Abundance in Supernatant', fontsize=11, fontweight='bold')
ax.set_title('(C) Dominant Strains Dynamics (Plain Supernatant)', fontsize=12, fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=7.5, ncol=2)
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
fig_path = os.path.join(ARTIFACT_DIR, 'Figure2_PlainAgar_OptimalCRM_Predictions.png')
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"Figure saved to {fig_path}", flush=True)

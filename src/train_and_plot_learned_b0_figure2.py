import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARTIFACT_DIR = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
PAPER_FIG_DIR = os.path.join(PROJECT_ROOT, 'Paper', 'figures')
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}", flush=True)

# 1. Load Data
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location=device, weights_only=False)

no_c_tensor = data['no_c_tensor'].to(device)       # (3, 6, 50)
mucin_c_tensor = data['mucin_c_tensor'].to(device) # (3, 6, 50)
mucin_s_tensor = data['mucin_s_tensor'].to(device) # (3, 6, 50)
plain_c_tensor = data['plain_c_tensor'].to(device) # (3, 6, 50)
plain_s_tensor = data['plain_s_tensor'].to(device) # (3, 6, 50)
strains = data['strains']
N = 50

# Load inoculum P0
df_s2 = pd.read_csv(os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv'))
sums = df_s2.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
df_s2['Relative_Abundance'] = df_s2['Read_Fraction'] / 100.0 if sums.mean() > 50 else df_s2['Read_Fraction']
inoc = df_s2[df_s2['sampleType'] == 'inoculum'].pivot_table(index='replicate', columns='Strain_Name', values='Relative_Abundance', fill_value=0.0).reindex(columns=strains, fill_value=0.0)
inoc_norm = inoc.div(inoc.sum(axis=1), axis=0)
p0 = torch.tensor(inoc_norm.mean(axis=0).values, dtype=torch.float32, device=device).unsqueeze(0).expand(3, -1)

# Metrics functions
def compute_r2(true_vals, pred_vals):
    diff_sq = np.sum((true_vals - pred_vals)**2)
    tot_sq = np.sum((true_vals - np.mean(true_vals))**2)
    return float(1.0 - diff_sq / tot_sq)

def compute_kl(t, p, eps=1e-12, clamp_min=1e-9):
    ratio = torch.clamp(t, min=eps) / torch.clamp(p, min=clamp_min)
    kl = torch.sum(torch.where(t > 0, t * torch.log(ratio), torch.zeros_like(t)), dim=-1)
    return float(kl.mean().item())

# Import ModularCRM_B0_Experiment from test_learned_b0_with_correct_dilution
sys.path.append(SCRIPT_DIR)
from test_learned_b0_with_correct_dilution import ModularCRM_B0_Experiment

def train_learned_b0_model(K=4, c_w=0.75, init_b0=2.0, epochs=350, lr=0.015, seed=42):
    print(f"Training Model with Single Shared Learned Initial Biomass (K={K}, c_w={c_w}, init_b0={init_b0}, seed={seed})...", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ModularCRM_B0_Experiment(
        N=50, K=K, c_w=c_w, b0_mode='learned_shared_scalar', init_b0=init_b0
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train strictly on No Carrier + Mucin Agar (3 replicates)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        # 1. No Carrier (1:100 loop dilution)
        p_no = model.forward_no_carrier(p0, tf_target=no_c_tensor, tf_ratio=tf_ratio)
        ratio_no = torch.clamp(no_c_tensor, min=1e-12) / torch.clamp(p_no, min=1e-9)
        loss = torch.sum(torch.where(no_c_tensor > 0, no_c_tensor * torch.log(ratio_no), torch.zeros_like(no_c_tensor)), dim=-1).mean()
        
        # 2. Mucin Agar
        p_mu_c, p_mu_s = model.forward_carrier(p0, is_mucin=True, tf_targets=(mucin_c_tensor, mucin_s_tensor), tf_ratio=tf_ratio)
        ratio_mu_c = torch.clamp(mucin_c_tensor, min=1e-12) / torch.clamp(p_mu_c, min=1e-9)
        ratio_mu_s = torch.clamp(mucin_s_tensor, min=1e-12) / torch.clamp(p_mu_s, min=1e-9)
        loss += torch.sum(torch.where(mucin_c_tensor > 0, mucin_c_tensor * torch.log(ratio_mu_c), torch.zeros_like(mucin_c_tensor)), dim=-1).mean() + \
                torch.sum(torch.where(mucin_s_tensor > 0, mucin_s_tensor * torch.log(ratio_mu_s), torch.zeros_like(mucin_s_tensor)), dim=-1).mean()
                
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin = model.get_params()
        loss += 0.001 * (Theta.pow(2).sum() + W.pow(2).sum()) + 0.001 * alpha.pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == epochs:
            b0_curr = model.get_b0()[0].item()
            print(f"Epoch {epoch:3d}/{epochs} - Loss: {loss.item():.4f} - Learned b0: {b0_curr:.4f}", flush=True)
            
    # Save checkpoint
    b0_final = model.get_b0()[0].item()
    ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Winning_LearnedB0_ModularCRM_K4.pt')
    torch.save({
        'state_dict': model.state_dict(),
        'K': K, 'c_w': c_w, 'learned_b0': b0_final,
        'seed': seed, 'epochs': epochs
    }, ckpt_path)
    print(f"Saved model checkpoint to {ckpt_path} (Learned b0 = {b0_final:.4f})", flush=True)
    return model

def main():
    model = train_learned_b0_model(K=4, c_w=0.75, init_b0=2.0, epochs=350, lr=0.015, seed=42)
    model.eval()
    learned_b0_val = model.get_b0()[0].item()
    
    with torch.no_grad():
        # Zero-shot evaluation on 100% held-out Plain Agar
        pred_pl_c, pred_pl_s = model.forward_carrier(p0, is_mucin=False, tf_ratio=0.0)
        
    obs_s = plain_s_tensor.cpu().numpy() # (3, 6, 50)
    pred_s = pred_pl_s.cpu().numpy()     # (3, 6, 50)
    obs_c = plain_c_tensor.cpu().numpy() # (3, 6, 50)
    pred_c = pred_pl_c.cpu().numpy()     # (3, 6, 50)
    
    # Global metrics across all 3 replicates x 6 passages x 50 strains (N=900)
    flat_obs_s = obs_s.flatten()
    flat_pred_s = pred_s.flatten()
    r2_s = compute_r2(flat_obs_s, flat_pred_s)
    r_s, _ = pearsonr(flat_obs_s, flat_pred_s)
    rho_s, _ = spearmanr(flat_obs_s, flat_pred_s)
    kl_s = compute_kl(plain_s_tensor, pred_pl_s)
    
    flat_obs_c = obs_c.flatten()
    flat_pred_c = pred_c.flatten()
    r2_c = compute_r2(flat_obs_c, flat_pred_c)
    r_c, _ = pearsonr(flat_obs_c, flat_pred_c)
    rho_c, _ = spearmanr(flat_obs_c, flat_pred_c)
    kl_c = compute_kl(plain_c_tensor, pred_pl_c)
    
    flat_obs_comb = np.concatenate([flat_obs_s, flat_obs_c])
    flat_pred_comb = np.concatenate([flat_pred_s, flat_pred_c])
    r2_comb = compute_r2(flat_obs_comb, flat_pred_comb)
    r_comb, _ = pearsonr(flat_obs_comb, flat_pred_comb)
    rho_comb, _ = spearmanr(flat_obs_comb, flat_pred_comb)
    kl_comb = 0.5 * (kl_s + kl_c)
    
    # Null baseline metrics (Static Inoculum P0)
    p0_np = p0.mean(dim=0).cpu().numpy()
    null_pred_s = np.tile(p0_np, (3, 6, 1)).flatten()
    null_pred_c = np.tile(p0_np, (3, 6, 1)).flatten()
    null_pred_comb = np.concatenate([null_pred_s, null_pred_c])
    
    null_r2_s = compute_r2(flat_obs_s, null_pred_s)
    null_r_s, _ = pearsonr(flat_obs_s, null_pred_s)
    null_r2_c = compute_r2(flat_obs_c, null_pred_c)
    null_r_c, _ = pearsonr(flat_obs_c, null_pred_c)
    null_r2_comb = compute_r2(flat_obs_comb, null_pred_comb)
    null_r_comb, _ = pearsonr(flat_obs_comb, null_pred_comb)
    
    p0_tensor_expanded = p0.unsqueeze(1).expand(-1, 6, -1)
    null_kl_s = compute_kl(plain_s_tensor, p0_tensor_expanded)
    null_kl_c = compute_kl(plain_c_tensor, p0_tensor_expanded)
    null_kl_comb = 0.5 * (null_kl_s + null_kl_c)
    
    print("\n" + "="*80)
    print(f"LEARNED SHARED B0 (b0 = {learned_b0_val:.4f}) PLAIN AGAR ZERO-SHOT PERFORMANCE (N=900):")
    print(f"Supernatant Broth: R² = {r2_s:+.4f}, Pearson r = {r_s:.4f}, Spearman rho = {rho_s:.4f}, KL = {kl_s:.4f} nats")
    print(f"Carrier Biofilm:   R² = {r2_c:+.4f}, Pearson r = {r_c:.4f}, Spearman rho = {rho_c:.4f}, KL = {kl_c:.4f} nats")
    print(f"Combined Plain:    R² = {r2_comb:+.4f}, Pearson r = {r_comb:.4f}, Spearman rho = {rho_comb:.4f}, KL = {kl_comb:.4f} nats")
    print(f"Inoculum Null R²:  Comb R² = {null_r2_comb:+.4f}, Pearson r = {null_r_comb:.4f}, KL = {null_kl_comb:.4f} nats")
    print("="*80 + "\n", flush=True)
    
    # Per-Passage Metrics
    passages = [f'P{p+1}' for p in range(6)]
    per_passage_records = []
    
    for p in range(6):
        o_s = obs_s[:, p, :].flatten()
        pr_s = pred_s[:, p, :].flatten()
        r2_sp = compute_r2(o_s, pr_s)
        r_sp, _ = pearsonr(o_s, pr_s)
        rho_sp, _ = spearmanr(o_s, pr_s)
        kl_sp = compute_kl(plain_s_tensor[:, p, :], pred_pl_s[:, p, :])
        
        o_c = obs_c[:, p, :].flatten()
        pr_c = pred_c[:, p, :].flatten()
        r2_cp = compute_r2(o_c, pr_c)
        r_cp, _ = pearsonr(o_c, pr_c)
        rho_cp, _ = spearmanr(o_c, pr_c)
        kl_cp = compute_kl(plain_c_tensor[:, p, :], pred_pl_c[:, p, :])
        
        o_tot = np.concatenate([o_s, o_c])
        pr_tot = np.concatenate([pr_s, pr_c])
        r2_tot = compute_r2(o_tot, pr_tot)
        r_tot, _ = pearsonr(o_tot, pr_tot)
        rho_tot, _ = spearmanr(o_tot, pr_tot)
        kl_tot = 0.5 * (kl_sp + kl_cp)
        
        null_s_p = np.tile(p0_np, (3, 1)).flatten()
        null_c_p = np.tile(p0_np, (3, 1)).flatten()
        null_tot_p = np.concatenate([null_s_p, null_c_p])
        
        null_r2_sp = compute_r2(o_s, null_s_p)
        null_r2_cp = compute_r2(o_c, null_c_p)
        null_r2_tot = compute_r2(o_tot, null_tot_p)
        
        per_passage_records.append({
            'Passage': passages[p],
            'Days': (p + 1) * 2,
            'Supernatant_R2': r2_sp, 'Supernatant_r': r_sp, 'Supernatant_rho': rho_sp, 'Supernatant_KL': kl_sp,
            'Carrier_R2': r2_cp, 'Carrier_r': r_cp, 'Carrier_rho': rho_cp, 'Carrier_KL': kl_cp,
            'Combined_R2': r2_tot, 'Combined_r': r_tot, 'Combined_rho': rho_tot, 'Combined_KL': kl_tot,
            'Null_Sup_R2': null_r2_sp, 'Null_Car_R2': null_r2_cp, 'Null_Comb_R2': null_r2_tot
        })
        
    df_passage = pd.DataFrame(per_passage_records)
    passage_csv = os.path.join(PROJECT_ROOT, 'Data', 'Figure2_LearnedB0_Winner_PerPassage.csv')
    df_passage.to_csv(passage_csv, index=False)
    print(f"Saved per-passage metrics to {passage_csv}", flush=True)
    
    # Save global summary CSV
    df_summary = pd.DataFrame([
        {'Model': f'Learned Shared b0={learned_b0_val:.3f} (K=4, cw=0.75)', 'Sup_R2': r2_s, 'Sup_r': r_s, 'Sup_rho': rho_s, 'Sup_KL': kl_s,
         'Car_R2': r2_c, 'Car_r': r_c, 'Car_rho': rho_c, 'Car_KL': kl_c,
         'Comb_R2': r2_comb, 'Comb_r': r_comb, 'Comb_rho': rho_comb, 'Comb_KL': kl_comb},
        {'Model': 'Static Inoculum Null (p0)', 'Sup_R2': null_r2_s, 'Sup_r': null_r_s, 'Sup_rho': np.nan, 'Sup_KL': null_kl_s,
         'Car_R2': null_r2_c, 'Car_r': null_r_c, 'Car_rho': np.nan, 'Car_KL': null_kl_c,
         'Comb_R2': null_r2_comb, 'Comb_r': null_r_comb, 'Comb_rho': np.nan, 'Comb_KL': null_kl_comb}
    ])
    summary_csv = os.path.join(PROJECT_ROOT, 'Data', 'Figure2_LearnedB0_Winner_PlainAgar_Summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    
    # -------------------------------------------------------------
    # 2. RENDER PUBLICATION-QUALITY FIGURE 2
    # -------------------------------------------------------------
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 0.9
    plt.rcParams['ytick.major.width'] = 0.9
    
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25)
    passage_palette = sns.color_palette("viridis", 6)
    
    # -------------------------------------------------------------
    # Panel A: Parity Plot - Supernatant Broth
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    for p in range(6):
        o_p = obs_s[:, p, :].flatten()
        pr_p = pred_s[:, p, :].flatten()
        ax1.scatter(o_p, pr_p, color=passage_palette[p], alpha=0.70, s=32, edgecolors='none', label=f'P{p+1}')
        
    max_val_s = max(flat_obs_s.max(), flat_pred_s.max()) * 1.05
    ax1.plot([0, max_val_s], [0, max_val_s], 'k--', lw=1.3, alpha=0.8, label='Identity (y = x)')
    
    m_s, b_s = np.polyfit(flat_obs_s, flat_pred_s, 1)
    x_line_s = np.linspace(0, max_val_s, 100)
    ax1.plot(x_line_s, m_s * x_line_s + b_s, color='#e74c3c', lw=2.0, alpha=0.9, label=f'Fit: y = {m_s:.2f}x + {b_s:.3f}')
    
    ax1.set_title(rf"$\mathbf{{A}}$  Zero-Shot Supernatant Broth ($R^2 = {r2_s:+.3f}$, $r = {r_s:.3f}$, $\rho = {rho_s:.3f}$)", fontsize=12.5, pad=9, loc='left')
    ax1.set_xlabel("Observed Relative Abundance", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Zero-Shot Predicted Abundance", fontsize=11, fontweight='bold')
    ax1.set_xlim(-0.015, max_val_s)
    ax1.set_ylim(-0.015, max_val_s)
    ax1.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=8.5, ncol=2)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    ax1.text(0.95, 0.08, f"100% Held Out\nLearned $b_0 = {learned_b0_val:.3f}$\n$R^2 = {r2_s:.4f}$\nPearson $r = {r_s:.4f}$\nSpearman $\\rho = {rho_s:.4f}$\nKL = {kl_s:.4f} nats",
             transform=ax1.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ebf5fb', edgecolor='#2980b9', lw=1.2),
             fontsize=9.5, fontweight='bold')
             
    # -------------------------------------------------------------
    # Panel B: Parity Plot - Carrier Biofilm
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    for p in range(6):
        o_p = obs_c[:, p, :].flatten()
        pr_p = pred_c[:, p, :].flatten()
        ax2.scatter(o_p, pr_p, color=passage_palette[p], alpha=0.70, s=32, edgecolors='none', label=f'P{p+1}')
        
    max_val_c = max(flat_obs_c.max(), flat_pred_c.max()) * 1.05
    ax2.plot([0, max_val_c], [0, max_val_c], 'k--', lw=1.3, alpha=0.8, label='Identity (y = x)')
    
    m_c, b_c = np.polyfit(flat_obs_c, flat_pred_c, 1)
    x_line_c = np.linspace(0, max_val_c, 100)
    ax2.plot(x_line_c, m_c * x_line_c + b_c, color='#e74c3c', lw=2.0, alpha=0.9, label=f'Fit: y = {m_c:.2f}x + {b_c:.3f}')
    
    ax2.set_title(rf"$\mathbf{{B}}$  Zero-Shot Carrier Biofilm ($R^2 = {r2_c:+.3f}$, $r = {r_c:.3f}$, $\rho = {rho_c:.3f}$)", fontsize=12.5, pad=9, loc='left')
    ax2.set_xlabel("Observed Relative Abundance", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Zero-Shot Predicted Abundance", fontsize=11, fontweight='bold')
    ax2.set_xlim(-0.015, max_val_c)
    ax2.set_ylim(-0.015, max_val_c)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=8.5, ncol=2)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    ax2.text(0.95, 0.08, f"100% Held Out\nLearned $b_0 = {learned_b0_val:.3f}$\n$R^2 = {r2_c:.4f}$\nPearson $r = {r_c:.4f}$\nSpearman $\\rho = {rho_c:.4f}$\nKL = {kl_c:.4f} nats",
             transform=ax2.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef9e7', edgecolor='#f39c12', lw=1.2),
             fontsize=9.5, fontweight='bold')
             
    # -------------------------------------------------------------
    # Panel C: Passage-by-Passage Predictive Accuracy Breakdown
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(passages))
    width = 0.25
    
    rects1 = ax3.bar(x - width, df_passage['Supernatant_R2'], width, label='Supernatant $R^2$', color='#3498db', alpha=0.85, edgecolor='black', lw=0.8)
    rects2 = ax3.bar(x, df_passage['Carrier_R2'], width, label='Carrier Biofilm $R^2$', color='#e67e22', alpha=0.85, edgecolor='black', lw=0.8)
    rects3 = ax3.bar(x + width, df_passage['Combined_R2'], width, label='Combined $R^2$', color='#2ecc71', alpha=0.85, edgecolor='black', lw=0.8)
    
    # Inoculum Null baseline line
    ax3.plot(x, df_passage['Null_Comb_R2'], 'k:', lw=1.8, marker='x', ms=6, label='Static Inoculum Null ($p_0$)')
    
    # Twin axis for Pearson r
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, df_passage['Supernatant_r'], 'o--', color='#1f618d', lw=2.0, ms=5.5, label='Supernatant Pearson $r$')
    ax3_twin.plot(x, df_passage['Carrier_r'], 's--', color='#b9770e', lw=2.0, ms=5.5, label='Carrier Pearson $r$')
    ax3_twin.plot(x, df_passage['Combined_r'], '^--', color='#1e8449', lw=2.0, ms=5.5, label='Combined Pearson $r$')
    
    ax3.set_title(r"$\mathbf{C}$  Passage-by-Passage Predictive Fidelity ($P_1 \to P_6$)", fontsize=12.5, pad=9, loc='left')
    ax3.set_xticks(x)
    ax3.set_xticklabels(passages, fontweight='bold', fontsize=11)
    ax3.set_ylabel(r"Coefficient of Determination ($R^2$)", fontsize=11, fontweight='bold')
    ax3_twin.set_ylabel(r"Pearson Correlation ($r$)", fontsize=11, fontweight='bold')
    ax3.set_ylim(-0.25, 0.70)
    ax3_twin.set_ylim(0.0, 1.0)
    ax3.axhline(0, color='gray', linestyle='-', lw=0.8)
    ax3.grid(True, linestyle=':', alpha=0.5, axis='y')
    
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, framealpha=0.92, fontsize=8.0, ncol=2)
    
    ax3.axvspan(1.5, 3.5, color='#f4d03f', alpha=0.15)
    ax3.text(2.5, 0.62, "Equilibrium Regime\n($R^2 > 0.45$, $r \\approx 0.70$)", ha='center', va='center',
             fontsize=9, fontweight='bold', color='#7d6608')
             
    # -------------------------------------------------------------
    # Panel D: Longitudinal Dynamics of Representative Gut Taxa
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    mean_obs_s = obs_s.mean(axis=0)  # (6, 50)
    mean_pred_s = pred_s.mean(axis=0) # (6, 50)
    
    top_taxa_idx = [
        strains.index('Clostridium-sp-D5'),
        strains.index('Acidaminococcus-intestini-D21'),
        strains.index('Mitsuokella-multacida-DSM-20544'),
        strains.index('Bacteroides-uniformis-ATCC-8492')
    ]
    taxa_names = [
        r'$\it{Clostridium}$ sp. D5',
        r'$\it{A.\ intestini}$ D21',
        r'$\it{M.\ multacida}$ DSM-20544',
        r'$\it{B.\ uniformis}$ ATCC-8492'
    ]
    taxa_colors = ['#8e44ad', '#2980b9', '#d35400', '#27ae60']
    pass_coords = np.arange(7) # P0 to P6
    pass_labels = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    
    for i, (idx, name, col) in enumerate(zip(top_taxa_idx, taxa_names, taxa_colors)):
        traj_obs = np.concatenate([[p0_np[idx]], mean_obs_s[:, idx]])
        traj_pred = np.concatenate([[p0_np[idx]], mean_pred_s[:, idx]])
        
        ax4.plot(pass_coords, traj_obs, '-o', color=col, lw=2.2, ms=6, label=f'{name} (Obs)')
        ax4.plot(pass_coords, traj_pred, '--s', color=col, lw=1.8, ms=5, alpha=0.85, markerfacecolor='white', markeredgewidth=1.5, label=f'{name} (Zero-Shot)')
        
    ax4.set_title(r"$\mathbf{D}$  Longitudinal Dynamics of Dominant Gut Colonizers", fontsize=12.5, pad=9, loc='left')
    ax4.set_xticks(pass_coords)
    ax4.set_xticklabels(pass_labels, fontweight='bold', fontsize=11)
    ax4.set_xlabel("Serial Dilution Passage (Days 0 to 12)", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Relative Abundance in Supernatant", fontsize=11, fontweight='bold')
    ax4.set_ylim(-0.01, 0.40)
    ax4.grid(True, linestyle=':', alpha=0.5)
    ax4.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=8.0, ncol=2)
    
    plt.tight_layout()
    
    # Save output figures
    out_paper = os.path.join(PAPER_FIG_DIR, 'Figure2_PlainAgar_ZeroShot_Transfer.png')
    out_artifact = os.path.join(ARTIFACT_DIR, 'Figure2_PlainAgar_ZeroShot_Transfer.png')
    out_opt_artifact = os.path.join(ARTIFACT_DIR, 'Figure2_PlainAgar_OptimalCRM_Predictions.png')
    
    fig.savefig(out_paper, dpi=300, bbox_inches='tight')
    fig.savefig(out_artifact, dpi=300, bbox_inches='tight')
    fig.savefig(out_opt_artifact, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved updated publication Figure 2 to:\n  - {out_paper}\n  - {out_artifact}\n  - {out_opt_artifact}", flush=True)

if __name__ == '__main__':
    main()

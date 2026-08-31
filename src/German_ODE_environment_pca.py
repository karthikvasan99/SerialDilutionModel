"""
Train Continuous-Time Latent Assembly Neural ODE on the Entire German Microbiome Dataset
and Analyze Environment Vectors ('a') via Principal Component Analysis (PCA).

Hyperparameters (from grid search):
  - Latent resource pools (K): 4
  - Proximal L2 regularization (l2): 0.05
  - Training epochs: 200
  - Learning rate: 0.01
  - Dilution factor: 100.0
  - Soft Scheduled Sampling: linear decay from 1.0 to 0.0 over first 80% of epochs

Outputs:
  - 2D PCA projection of environment vectors ('a' and softplus(a)) with labels
  - Latent resource allocation heatmap across conditions
  - PCA loadings and scree variance plots
  - Saved model weights and PCA coordinates CSV
"""

import os
import sys
import copy
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE

# Matplotlib global styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# ─── Default Reproducibility Seed ───
BASE_SEED = 42


def set_seed(seed=BASE_SEED):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_filter_data(data_path, abundance_threshold=1e-3):
    """
    Loads German dataset and filters ubiquitous species with mean T1 abundance >= threshold.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file '{data_path}' not found.")

    print(f"Loading German microbiome dataset from: {data_path}", flush=True)
    data = torch.load(data_path, weights_only=False)
    strains = data['strains']
    trajectories = data['trajectories']
    num_before = len(strains)

    # Filter bacteria with mean relative abundance >= threshold in T1 AND present in all 12 T1 samples
    t1_passages = torch.stack([seq[0] for seq in trajectories.values()], dim=0)
    mean_t1_abundance = t1_passages.mean(dim=0)
    all_t1_present = (t1_passages > 0).all(dim=0)
    keep_mask = (mean_t1_abundance >= abundance_threshold) & all_t1_present
    num_after = int(keep_mask.sum().item())

    print(f"\nSpecies T1 Abundance Filtering (Threshold >= {abundance_threshold} & Present in All 12 T1 Samples):", flush=True)
    print(f" - Total bacteria BEFORE filter: {num_before}", flush=True)
    print(f" - Total bacteria AFTER filter:  {num_after}", flush=True)
    print(f" - Filtered out: {num_before - num_after} low-abundance / non-ubiquitous species", flush=True)

    filtered_strains = [s for idx, s in enumerate(strains) if keep_mask[idx]]
    filtered_trajectories = {}
    for k in trajectories.keys():
        filt_seq = trajectories[k][:, keep_mask]
        filtered_trajectories[k] = filt_seq / filt_seq.sum(dim=-1, keepdim=True)

    return filtered_trajectories, filtered_strains


def map_environments(trajectories_dict, mode='treatments'):
    """
    Maps trajectory condition keys to environmental condition indices.

    Modes:
      - 'treatments' (default): Decouples rumen origin (BRRS vs OSU), mapping the 12 trajectories
        down to 6 unique treatment conditions (Substrate x Pressure), matching grid search.
      - 'all_12': Gives every individual trajectory (Origin x Substrate x Pressure) its own condition index.
    """
    env_map = {}
    unique_envs = []

    for cond_key in trajectories_dict.keys():
        if mode == 'treatments':
            if isinstance(cond_key, tuple) and len(cond_key) >= 2:
                env_label = cond_key[1:]
            else:
                env_label = cond_key
        else:  # all_12
            env_label = cond_key

        if env_label not in unique_envs:
            unique_envs.append(env_label)

        env_map[cond_key] = unique_envs.index(env_label)

    print(f"\nEnvironment Mapping ({mode} mode):", flush=True)
    print(f" - Total Trajectories: {len(trajectories_dict)}", flush=True)
    print(f" - Unique Condition Vectors: {len(unique_envs)}", flush=True)
    for idx, env_name in enumerate(unique_envs):
        matching_keys = [k for k, v in env_map.items() if v == idx]
        print(f"    Index {idx}: {env_name} -> Trajectories: {matching_keys}", flush=True)

    return env_map, unique_envs


def format_condition_label(env_tuple):
    """Generates a clean, readable string label from a condition tuple."""
    if isinstance(env_tuple, tuple):
        if len(env_tuple) == 3:
            origin, substrate, pressure = env_tuple
            p_val = f"{float(pressure):.0f} bar" if isinstance(pressure, (int, float, np.number)) else str(pressure)
            return f"{origin} | {substrate} ({p_val})"
        elif len(env_tuple) == 2:
            substrate, pressure = env_tuple
            p_val = f"{float(pressure):.0f} bar" if isinstance(pressure, (int, float, np.number)) else str(pressure)
            return f"{substrate} ({p_val})"
        else:
            return " | ".join(str(x) for x in env_tuple)
    return str(env_tuple)


def get_condition_metadata(env_tuple):
    """Extracts origin, substrate, and pressure attributes from a condition tuple."""
    if isinstance(env_tuple, tuple):
        if len(env_tuple) == 3:
            origin, substrate, pressure = env_tuple
            return str(origin), str(substrate), float(pressure)
        elif len(env_tuple) == 2:
            substrate, pressure = env_tuple
            return "Pooled", str(substrate), float(pressure)
    return "Unknown", str(env_tuple), 0.0


def train_full_neural_ode(model, train_seqs, train_conds, epochs=200, lr=0.01,
                           l2_lambda=0.05, verbose=True):
    """
    Trains LatentAssemblyNeuralODE on the entire dataset using Soft Scheduled Sampling
    and proximal L2 regularization on Theta and W.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    seq_len = train_seqs.size(1)

    loss_history = []
    t0 = time.time()

    if verbose:
        print(f"\n" + "="*70)
        print(f"Training Neural ODE on ALL {train_seqs.size(0)} Trajectories (T1 -> T6)")
        print(f"Hyperparameters: K={model.K}, L2={l2_lambda}, Epochs={epochs}, LR={lr}")
        print("="*70, flush=True)

    for epoch in range(epochs):
        # Soft Scheduled Sampling: linear decay from 1.0 to 0.0 over first 80% of epochs
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.8)))

        model.train()
        optimizer.zero_grad()

        x_curr = train_seqs[:, 0, :]
        batch_loss = 0.0

        for t in range(1, seq_len):
            log_pred, x_pred, _, _ = model(x_curr, train_conds)
            target = train_seqs[:, t, :]
            batch_loss = batch_loss + criterion(log_pred, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred

        loss_epoch = batch_loss / (seq_len - 1)
        loss_epoch.backward()
        optimizer.step()

        # Proximal L2 Regularization on ode_func.Theta and ode_func.W
        with torch.no_grad():
            if l2_lambda > 0.0:
                l2_factor = 1.0 / (1.0 + 2.0 * lr * l2_lambda)
                model.ode_func.Theta.data *= l2_factor
                model.ode_func.W.data *= l2_factor

        loss_val = loss_epoch.item()
        loss_history.append(loss_val)

        if verbose and ((epoch + 1) % 25 == 0 or epoch == 0 or epoch == epochs - 1):
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:4d}/{epochs:4d} | KL Div Loss: {loss_val:.6f} | "
                  f"TF Ratio: {tf_ratio:.3f} | Elapsed: {elapsed:.2f}s", flush=True)

    total_time = time.time() - t0
    if verbose:
        print(f"\nFull Dataset Training Completed in {total_time:.2f}s | Final KL Loss: {loss_history[-1]:.6f}")

    return model, loss_history


def compute_pca(data_matrix):
    """
    Computes PCA via SVD on a centered data matrix (M x K).

    Returns:
      coords: (M x min(M, K)) projection coordinates
      var_exp_ratio: (min(M, K),) explained variance ratio per PC
      components: (min(M, K) x K) principal component loading vectors (V^T)
      singular_vals: singular values (S)
    """
    M, K = data_matrix.shape
    mean_vec = np.mean(data_matrix, axis=0)
    centered = data_matrix - mean_vec

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    variance = (S ** 2) / max(1, M - 1)
    total_var = np.sum(variance)
    var_exp_ratio = (variance / total_var) if total_var > 0 else np.zeros_like(variance)
    coords = centered @ Vt.T

    return coords, var_exp_ratio, Vt, mean_vec


def evaluate_rollouts(model, trajectories, env_map):
    """
    Evaluates multi-passage autoregressive rollouts across all trajectories.
    """
    model.eval()
    seq_len = 6
    all_true = []
    all_pred = []
    passage_r2 = {}

    with torch.no_grad():
        for t_step in range(1, seq_len):
            true_step = []
            pred_step = []

            for key, seq in trajectories.items():
                c_idx = env_map[key]
                c_tensor = torch.tensor([c_idx], dtype=torch.long)
                
                curr = seq[0].unsqueeze(0)
                for step in range(1, t_step + 1):
                    _, nxt, _, _ = model(curr, c_tensor)
                    curr = nxt

                true_step.append(seq[t_step].numpy().flatten())
                pred_step.append(nxt.squeeze(0).numpy().flatten())

            true_step_cat = np.concatenate(true_step)
            pred_step_cat = np.concatenate(pred_step)

            all_true.append(true_step_cat)
            all_pred.append(pred_step_cat)

            mask = true_step_cat > 0
            if mask.sum() >= 2:
                r, _ = pearsonr(np.log10(true_step_cat[mask]), np.log10(np.clip(pred_step_cat[mask], 1e-6, None)))
                passage_r2[t_step + 1] = float(r**2) if not np.isnan(r) else 0.0
            else:
                passage_r2[t_step + 1] = 0.0

    all_true_cat = np.concatenate(all_true)
    all_pred_cat = np.concatenate(all_pred)
    mask_all = all_true_cat > 0
    overall_r, _ = pearsonr(np.log10(all_true_cat[mask_all]), np.log10(np.clip(all_pred_cat[mask_all], 1e-6, None)))
    overall_r2 = float(overall_r**2) if not np.isnan(overall_r) else 0.0

    return passage_r2, overall_r2, all_true_cat, all_pred_cat


def plot_environment_pca_analysis(
    model, unique_envs, env_map, trajectories, loss_history,
    coords_a, var_exp_a, Vt_a,
    coords_r0, var_exp_r0, Vt_r0,
    save_paths, condition_mode='treatments'
):
    """
    Generates a 4-panel publication-ready visualization of:
      (A) 2D PCA Projection of Environment Vectors 'a' with condition labels and biplot loadings
      (B) Latent Resource Allocation Heatmap (R_0 = softplus(a)) across conditions
      (C) Scree Plot / Explained Variance & Feature Loadings on PC1 & PC2
      (D) Training Convergence & Autoregressive Multi-Passage Fit
    """
    fig = plt.figure(figsize=(20, 15), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

    # Color & Marker Mappings
    substrate_colors = {
        'H2:CO2': '#1f77b4',        # Blue
        'H2:CO2 + MeOH': '#2ca02c', # Green
        'H2:CO2 + TMA': '#d95f02',  # Orange / Vermillion
    }
    fallback_color = '#6c757d'

    pressure_markers = {
        5.0: ('o', 140),   # Circle
        30.0: ('^', 170),  # Triangle
    }

    # ─────────────────────────────────────────────────────────────
    # PANEL A: 2D PCA Projection of Initial Resource Vectors R_0
    # ─────────────────────────────────────────────────────────────
    ax_pca = fig.add_subplot(gs[0, 0])

    pc1_pct = var_exp_r0[0] * 100
    pc2_pct = var_exp_r0[1] * 100
    total_pct = pc1_pct + pc2_pct

    # Determine plot bounds for scaling
    x_min, x_max = coords_r0[:, 0].min(), coords_r0[:, 0].max()
    y_min, y_max = coords_r0[:, 1].min(), coords_r0[:, 1].max()
    x_span = max(0.4, x_max - x_min)
    y_span = max(0.4, y_max - y_min)

    # Expand limits to give ample breathing room for labels and biplot
    if condition_mode == 'treatments':
        ax_pca.set_xlim(-1.20, 1.20)
        ax_pca.set_ylim(-0.80, 1.05)
    else:
        ax_pca.set_xlim(x_min - 0.45 * x_span, x_max + 0.45 * x_span)
        ax_pca.set_ylim(y_min - 0.45 * y_span, y_max + 0.45 * y_span)

    # Plot condition points
    for idx, env in enumerate(unique_envs):
        origin, substrate, pressure = get_condition_metadata(env)
        label_text = format_condition_label(env)
        color = substrate_colors.get(substrate, fallback_color)
        marker, size = pressure_markers.get(pressure, ('s', 140))

        # Distinct style for origin if in 12-condition mode
        if condition_mode == 'all_12':
            face_col = color if origin == 'BRRS' else 'white'
            edge_col = color if origin == 'OSU' else 'black'
            lw = 2.2
        else:
            face_col = color
            edge_col = 'black'
            lw = 1.5

        x_pt = coords_r0[idx, 0]
        y_pt = coords_r0[idx, 1]

        ax_pca.scatter(
            x_pt, y_pt,
            facecolors=face_col,
            edgecolors=edge_col,
            marker=marker,
            s=size,
            linewidths=lw,
            alpha=0.95,
            zorder=5
        )

        # Smart directional offset for text callouts
        angle = np.arctan2(y_pt, x_pt)
        dist = 0.28 * max(x_span, y_span)
        dx = dist * np.cos(angle)
        dy = dist * np.sin(angle)

        # Precise manual nudges for 6-condition mode
        if condition_mode == 'treatments':
            if idx == 0:  # H2:CO2 5 bar (+0.28, +0.35)
                dx, dy = 0.16, 0.10
            elif idx == 1:  # H2:CO2 30 bar (+0.71, -0.34)
                dx, dy = 0.10, -0.12
            elif idx == 2:  # MeOH 5 bar (-0.55, -0.31)
                dx, dy = -0.16, -0.12
            elif idx == 3:  # MeOH 30 bar (-0.50, -0.11)
                dx, dy = -0.22, 0.10
            elif idx == 4:  # TMA 5 bar (+0.16, -0.17)
                dx, dy = -0.18, -0.14
            elif idx == 5:  # TMA 30 bar (-0.10, +0.58)
                dx, dy = 0.16, 0.08

        ha = 'left' if dx >= 0 else 'right'
        va = 'bottom' if dy >= 0 else 'top'

        ax_pca.annotate(
            label_text,
            xy=(x_pt, y_pt),
            xytext=(x_pt + dx, y_pt + dy),
            fontsize=10.0,
            fontweight='bold',
            color='#1a1a1a',
            ha=ha,
            va=va,
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor=color, alpha=0.92, linewidth=1.2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.06', color='#666666', lw=0.9),
            zorder=6
        )

    # Overlaid Biplot Loadings (arrows for latent resource pools K)
    K = Vt_r0.shape[1]
    max_coord = max(np.max(np.abs(coords_r0[:, 0])), np.max(np.abs(coords_r0[:, 1])))
    arrow_scale = max_coord * 0.75

    for k in range(K):
        u = Vt_r0[0, k] * arrow_scale
        v = Vt_r0[1, k] * arrow_scale
        ax_pca.annotate(
            '', xy=(u, v), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.6, ls='--')
        )
        ax_pca.text(
            u * 1.14, v * 1.14, f"$R_{k+1}$",
            color='#444444', fontsize=11, fontweight='bold',
            ha='center', va='center', zorder=7
        )

    # Reference grid & center lines
    ax_pca.axhline(0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
    ax_pca.axvline(0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
    ax_pca.grid(True, linestyle='--', alpha=0.3)

    ax_pca.set_xlabel(f"Principal Component 1 ({pc1_pct:.1f}% Variance)", fontsize=12.5, fontweight='bold')
    ax_pca.set_ylabel(f"Principal Component 2 ({pc2_pct:.1f}% Variance)", fontsize=12.5, fontweight='bold')
    ax_pca.set_title(rf"A. 2D PCA of Initial Resource Pools $R_0 = \mathrm{{softplus}}(a)$ (2D Var = {total_pct:.1f}%)",
                     fontsize=13.5, fontweight='bold', pad=12)

    # Custom Legend
    if condition_mode == 'all_12':
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=substrate_colors['H2:CO2'],
                       markersize=9, label=r'$\mathrm{H_2:CO_2}$ (Basal)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=substrate_colors['H2:CO2 + MeOH'],
                       markersize=9, label=r'$\mathrm{H_2:CO_2 + MeOH}$'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=substrate_colors['H2:CO2 + TMA'],
                       markersize=9, label=r'$\mathrm{H_2:CO_2 + TMA}$'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#444444',
                       markersize=8, label='5 bar Pressure'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#444444',
                       markersize=9, label='30 bar Pressure'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#444444',
                       markersize=8, label='BRRS Origin (Solid)'),
            plt.Line2D([0], [0], marker='s', color='#444444', markerfacecolor='white',
                       markersize=8, markeredgewidth=2, label='OSU Origin (Hollow)'),
        ]
    else:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=substrate_colors['H2:CO2'],
                       markersize=9, label=r'$\mathrm{H_2:CO_2}$ (Basal)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=substrate_colors['H2:CO2 + MeOH'],
                       markersize=9, label=r'$\mathrm{H_2:CO_2 + MeOH}$'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=substrate_colors['H2:CO2 + TMA'],
                       markersize=9, label=r'$\mathrm{H_2:CO_2 + TMA}$'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#444444',
                       markersize=8, label='5 bar Pressure'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#444444',
                       markersize=9, label='30 bar Pressure'),
        ]
    ax_pca.legend(handles=legend_elements, loc='upper left', fontsize=9.5, frameon=True,
                  framealpha=0.92, edgecolor='#cccccc')

    # ─────────────────────────────────────────────────────────────
    # PANEL B: Latent Resource Allocation Heatmap R_0 = softplus(a)
    # ─────────────────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0, 1])

    a_tensor = model.a.detach().cpu()
    r0_matrix = F.softplus(a_tensor).numpy()
    labels = [format_condition_label(env) for env in unique_envs]
    resource_cols = [f"Pool {k+1} ($R_{k+1}$)" for k in range(K)]

    im = ax_heat.imshow(r0_matrix, cmap='YlGnBu', aspect='auto', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label(r'Initial Resource Concentration $R_0 = \mathrm{softplus}(a)$', fontsize=11, fontweight='bold')

    for i in range(r0_matrix.shape[0]):
        for j in range(r0_matrix.shape[1]):
            val = r0_matrix[i, j]
            txt_color = 'white' if val > (r0_matrix.max() * 0.65) else 'black'
            ax_heat.text(j, i, f"{val:.3f}", ha='center', va='center',
                         color=txt_color, fontsize=10.5, fontweight='bold')

    ax_heat.set_xticks(range(K))
    ax_heat.set_xticklabels(resource_cols, fontsize=11, fontweight='bold')
    ax_heat.set_yticks(range(len(labels)))
    ax_heat.set_yticklabels(labels, fontsize=10.0, fontweight='medium')
    ax_heat.set_title(r"B. Learned Environmental Resource Pools ($R_0$ per Condition)",
                      fontsize=13.5, fontweight='bold', pad=12)

    # ─────────────────────────────────────────────────────────────
    # PANEL C: PCA Explained Variance & Feature Loadings of R_0
    # ─────────────────────────────────────────────────────────────
    ax_loadings = fig.add_subplot(gs[1, 0])

    bar_width = 0.35
    indices = np.arange(K)

    loadings_pc1 = Vt_r0[0, :]
    loadings_pc2 = Vt_r0[1, :]

    bars1 = ax_loadings.bar(indices - bar_width/2, loadings_pc1, bar_width,
                            label=f'PC1 ({pc1_pct:.1f}%)', color='#1f77b4', alpha=0.88, edgecolor='black', lw=1.0)
    bars2 = ax_loadings.bar(indices + bar_width/2, loadings_pc2, bar_width,
                            label=f'PC2 ({pc2_pct:.1f}%)', color='#e76f51', alpha=0.88, edgecolor='black', lw=1.0)

    # Add numeric annotations
    for bar in bars1:
        h = bar.get_height()
        ax_loadings.annotate(f'{h:+.2f}',
                             xy=(bar.get_x() + bar.get_width()/2, h),
                             xytext=(0, 3 if h >= 0 else -11),
                             textcoords="offset points",
                             ha='center', va='bottom' if h >= 0 else 'top',
                             fontsize=9.5, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax_loadings.annotate(f'{h:+.2f}',
                             xy=(bar.get_x() + bar.get_width()/2, h),
                             xytext=(0, 3 if h >= 0 else -11),
                             textcoords="offset points",
                             ha='center', va='bottom' if h >= 0 else 'top',
                             fontsize=9.5, fontweight='bold')

    ax_loadings.axhline(0, color='black', linewidth=0.9)
    ax_loadings.set_xticks(indices)
    ax_loadings.set_xticklabels([f"Resource $R_{k+1}$" for k in range(K)], fontsize=11, fontweight='bold')
    ax_loadings.set_ylabel("Eigenvector Loading (Component Weight)", fontsize=11.5, fontweight='bold')
    ax_loadings.set_title(r"C. PCA Feature Loadings of Resource Pools $R_1 \dots R_4$ on PC1 & PC2",
                          fontsize=13.5, fontweight='bold', pad=12)
    ax_loadings.grid(True, linestyle='--', alpha=0.35, axis='y')
    ax_loadings.set_ylim(-0.85, 1.55)
    ax_loadings.legend(loc='lower left', fontsize=10.5, frameon=True)

    # Scree Inset (Top Right, above bars)
    ax_scree = ax_loadings.inset_axes([0.66, 0.56, 0.31, 0.40])
    pc_num = range(1, len(var_exp_r0) + 1)
    ax_scree.bar(pc_num, var_exp_r0 * 100, color='#457b9d', edgecolor='black', alpha=0.85)
    ax_scree.plot(pc_num, np.cumsum(var_exp_r0) * 100, color='#e63946', marker='o', lw=1.8, ms=4)
    ax_scree.set_xticks(pc_num)
    ax_scree.set_xlabel("PC", fontsize=8.5, fontweight='bold')
    ax_scree.set_ylabel("Var %", fontsize=8.5, fontweight='bold')
    ax_scree.set_title("Scree / Cum. %", fontsize=9, fontweight='bold')
    ax_scree.tick_params(labelsize=8)
    ax_scree.grid(True, linestyle=':', alpha=0.4)

    # ─────────────────────────────────────────────────────────────
    # PANEL D: Training Convergence & Autoregressive Dynamics Fit
    # ─────────────────────────────────────────────────────────────
    ax_train = fig.add_subplot(gs[1, 1])

    epochs_arr = range(1, len(loss_history) + 1)
    ax_train.plot(epochs_arr, loss_history, color='#1d3557', lw=2.4, label='Training KL Divergence Loss')

    passage_r2, overall_r2, _, _ = evaluate_rollouts(model, trajectories, env_map)

    r2_summary_text = (
        f"Neural ODE Full Dataset Fit:\n"
        f" • Final KL Divergence: {loss_history[-1]:.5f}\n"
        f" • Overall Multi-Passage $R^2$: {overall_r2:.4f}\n"
        f" • Passages $T_2 \\to T_6$ Mean $R^2$: {np.mean(list(passage_r2.values())):.4f}\n"
        f"   - $T_2$ $R^2$: {passage_r2.get(2, 0):.3f}   $T_3$ $R^2$: {passage_r2.get(3, 0):.3f}\n"
        f"   - $T_4$ $R^2$: {passage_r2.get(4, 0):.3f}   $T_5$ $R^2$: {passage_r2.get(5, 0):.3f}\n"
        f"   - $T_6$ $R^2$: {passage_r2.get(6, 0):.3f}"
    )

    ax_train.text(
        0.48, 0.45, r2_summary_text,
        transform=ax_train.transAxes,
        fontsize=10.5,
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#1d3557', alpha=0.92, lw=1.2)
    )

    ax_train.set_xlabel("Training Epoch", fontsize=11.5, fontweight='bold')
    ax_train.set_ylabel("KL Divergence Loss", fontsize=11.5, fontweight='bold')
    ax_train.set_title("D. Training Convergence & Autoregressive Multi-Passage Fit",
                       fontsize=13.5, fontweight='bold', pad=12)
    ax_train.grid(True, linestyle='--', alpha=0.35)
    ax_train.legend(loc='upper right', fontsize=10.5, frameon=True)

    # ─────────────────────────────────────────────────────────────
    # Save Figures
    # ─────────────────────────────────────────────────────────────
    for path in save_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA diagnostic figure to: '{path}'", flush=True)

    plt.close()


def save_results_csv(unique_envs, model, coords_r0, var_exp_r0, Vt_r0, csv_path):
    """
    Saves initial resource vectors R_0 = softplus(a), PCA coordinates, and component loadings to CSV.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    a_matrix = model.a.detach().cpu().numpy()
    r0_matrix = F.softplus(model.a.detach().cpu()).numpy()
    K = model.K

    rows = []
    for idx, env in enumerate(unique_envs):
        origin, substrate, pressure = get_condition_metadata(env)
        row = {
            'condition_idx': idx,
            'condition_label': format_condition_label(env),
            'origin': origin,
            'substrate': substrate,
            'pressure_bar': pressure,
            'PC1_coord_R0': coords_r0[idx, 0],
            'PC2_coord_R0': coords_r0[idx, 1],
        }
        for k in range(K):
            row[f'R0_resource_pool_{k+1}'] = r0_matrix[idx, k]
            row[f'a_param_dim_{k+1}'] = a_matrix[idx, k]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved environment resource vectors and PCA coordinates to CSV: '{csv_path}'", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train Neural ODE on entire German dataset and perform PCA on initial resource vectors R_0 = softplus(a)."
    )
    parser.add_argument('--data-path', type=str, default=os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt'),
                        help='Path to German_processed_data.pt')
    parser.add_argument('--abundance-threshold', type=float, default=1e-3,
                        help='T1 abundance threshold for ubiquitous species filter (default: 1e-3)')
    parser.add_argument('--condition-mode', type=str, choices=['treatments', 'all_12'], default='treatments',
                        help="Condition mode: 'treatments' (6 unique media conditions, default) or 'all_12' (12 individual conditions)")
    parser.add_argument('--latent-dims', '-k', type=int, default=4,
                        help='Number of latent resource dimensions K (default: 4 from grid search)')
    parser.add_argument('--l2', type=float, default=0.05,
                        help='Proximal L2 penalty on Theta and W (default: 0.05 from grid search)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200 from grid search)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for Adam optimizer (default: 0.01)')
    parser.add_argument('--dilution-factor', type=float, default=100.0,
                        help='Serial dilution factor (default: 100.0)')
    parser.add_argument('--seed', type=int, default=BASE_SEED,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--model-save-path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'Results', 'German_ODE_full_model.pth'),
                        help='Path to save full trained model state dict')
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Load and filter data
    trajectories, strains = load_and_filter_data(args.data_path, args.abundance_threshold)
    env_map, unique_envs = map_environments(trajectories, mode=args.condition_mode)

    num_species = len(strains)
    num_conditions = len(unique_envs)

    all_keys = list(trajectories.keys())
    train_seqs = torch.stack([trajectories[k] for k in all_keys])
    train_conds = torch.tensor([env_map[k] for k in all_keys], dtype=torch.long)

    # 2. Instantiate and train Neural ODE Model
    model = LatentAssemblyNeuralODE(
        num_species=num_species,
        num_latent_dims=args.latent_dims,
        num_conditions=num_conditions,
        default_dilution_factor=args.dilution_factor,
    )

    trained_model, loss_history = train_full_neural_ode(
        model, train_seqs, train_conds,
        epochs=args.epochs, lr=args.lr, l2_lambda=args.l2, verbose=True
    )

    # 3. Save Model Checkpoint
    if args.model_save_path:
        os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'strains': strains,
            'unique_envs': unique_envs,
            'env_map': env_map,
            'latent_dims': args.latent_dims,
            'l2': args.l2,
            'epochs': args.epochs,
            'lr': args.lr,
            'condition_mode': args.condition_mode,
            'loss_history': loss_history,
        }, args.model_save_path)
        print(f"\nSaved trained Neural ODE model checkpoint to: '{args.model_save_path}'", flush=True)

    # 4. Extract physical initial resource vectors R_0 = softplus(a) and compute PCA
    r0_matrix = F.softplus(trained_model.a.detach().cpu()).numpy()
    coords_r0, var_exp_r0, Vt_r0, mean_r0 = compute_pca(r0_matrix)

    print("\n" + "="*70)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA) ON INITIAL RESOURCE VECTORS R_0 = softplus(a)")
    print("="*70)
    print(f"Total Conditions: {num_conditions} | Latent Dimensions (K): {args.latent_dims}")
    print(f"Explained Variance Ratio per PC:")
    for pc_idx, var_pct in enumerate(var_exp_r0 * 100):
        cum_pct = np.sum(var_exp_r0[:pc_idx+1]) * 100
        print(f"  • PC{pc_idx+1}: {var_pct:5.2f}% of variance  (Cumulative: {cum_pct:5.2f}%)")
    print(f"Total 2D Projection Variance (PC1 + PC2): {(var_exp_r0[0] + var_exp_r0[1])*100:.2f}%")

    print("\n2D PCA Projection Coordinates for Conditions:")
    for idx, env in enumerate(unique_envs):
        lbl = format_condition_label(env)
        print(f"  [{idx:2d}] {lbl:<32} : PC1 = {coords_r0[idx, 0]:+7.4f}, PC2 = {coords_r0[idx, 1]:+7.4f}")

    print("\nLatent Resource Loadings on PC1 & PC2:")
    for k in range(args.latent_dims):
        print(f"  • Resource Pool R_{k+1}: PC1 loading = {Vt_r0[0, k]:+7.4f}, PC2 loading = {Vt_r0[1, k]:+7.4f}")

    # 5. Generate Figures and Save CSV
    images_plot_path = os.path.join(PROJECT_ROOT, 'Images', f'German_ODE_environment_vectors_pca_{args.condition_mode}.png')
    figures_plot_path = os.path.join(PROJECT_ROOT, 'Figures', f'German_ODE_environment_vectors_pca_{args.condition_mode}.png')
    csv_path = os.path.join(PROJECT_ROOT, 'Results', f'German_ODE_environment_vectors_pca_{args.condition_mode}.csv')

    # Also save standard image path for default
    save_paths = [images_plot_path, figures_plot_path]
    if args.condition_mode == 'treatments':
        default_img = os.path.join(PROJECT_ROOT, 'Images', 'German_ODE_environment_vectors_pca.png')
        default_fig = os.path.join(PROJECT_ROOT, 'Figures', 'German_ODE_environment_vectors_pca.png')
        save_paths.extend([default_img, default_fig])

    plot_environment_pca_analysis(
        trained_model, unique_envs, env_map, trajectories, loss_history,
        coords_a=None, var_exp_a=None, Vt_a=None,
        coords_r0=coords_r0, var_exp_r0=var_exp_r0, Vt_r0=Vt_r0,
        save_paths=save_paths,
        condition_mode=args.condition_mode
    )

    save_results_csv(unique_envs, trained_model, coords_r0, var_exp_r0, Vt_r0, csv_path)
    if args.condition_mode == 'treatments':
        default_csv = os.path.join(PROJECT_ROOT, 'Results', 'German_ODE_environment_vectors_pca.csv')
        save_results_csv(unique_envs, trained_model, coords_r0, var_exp_r0, Vt_r0, default_csv)

    print("\nAll tasks completed successfully!")


if __name__ == '__main__':
    main()

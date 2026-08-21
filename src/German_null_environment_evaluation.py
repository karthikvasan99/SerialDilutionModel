"""
Null Environment Evaluation for Latent Assembly Model on German Microbiome Dataset.

Evaluates whether the Latent Assembly Model has learned true environmental specificity
by comparing autoregressive predictions (T1 -> T6) under the TRUE environmental vector (a)
against NULL environmental vectors (all other environmental vectors).
"""

import os
import sys
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Add src to path if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyModel

# Reproducibility Seed
BASE_SEED = 42


def set_seed(seed=BASE_SEED):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_and_filter_data(data_path, threshold=1e-3):
    """
    Loads German dataset and filters for ubiquitous species with mean T1 abundance >= threshold.
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
    keep_mask = (mean_t1_abundance >= threshold) & all_t1_present
    num_after = int(keep_mask.sum().item())

    print(f"\nSpecies T1 Abundance Filtering (Threshold >= {threshold} & Present in All 12 T1 Samples):", flush=True)
    print(f" - Total bacteria BEFORE filter: {num_before}", flush=True)
    print(f" - Total bacteria AFTER filter:  {num_after}", flush=True)
    print(f" - Filtered out: {num_before - num_after} low-abundance / non-ubiquitous noise species", flush=True)

    filtered_strains = [s for idx, s in enumerate(strains) if keep_mask[idx]]
    filtered_trajectories = {}
    for k in trajectories.keys():
        filt_seq = trajectories[k][:, keep_mask]
        filtered_trajectories[k] = filt_seq / filt_seq.sum(dim=-1, keepdim=True)

    return filtered_trajectories, filtered_strains


def map_environments(trajectories_dict):
    """
    Decouples rumen origin from the environmental condition.
    Maps 12 trajectory keys to exactly 6 unique 'a' vector indices based purely
    on Substrate and Hydrogen Pressure.
    """
    env_map = {}
    unique_envs = []

    for cond_key in trajectories_dict.keys():
        if isinstance(cond_key, tuple) and len(cond_key) >= 2:
            env_label = cond_key[1:]
        else:
            env_label = cond_key

        if env_label not in unique_envs:
            unique_envs.append(env_label)

        env_map[cond_key] = unique_envs.index(env_label)

    print(f"\nEnvironment Mapping Complete:", flush=True)
    print(f" - Total Trajectories: {len(trajectories_dict)}", flush=True)
    print(f" - Unique 'a' Vectors (Environments): {len(unique_envs)}", flush=True)
    for idx, env_name in enumerate(unique_envs):
        matching_keys = [k[0] if isinstance(k, tuple) else k for k, v in env_map.items() if v == idx]
        print(f"    Index {idx}: {env_name} -> Tested in origins: {matching_keys}", flush=True)

    return env_map, unique_envs


def format_env_name(env_tuple):
    """Formats an environment tuple into a readable string label."""
    if isinstance(env_tuple, tuple):
        substrate = env_tuple[0]
        pressure = env_tuple[1]
        return f"{substrate}\n({pressure} bar)"
    return str(env_tuple)


def autoregressive_rollout_T6(model, test_seq, c_idx):
    """
    Performs pure autoregressive rollout from T1 to T6 given condition c_idx.
    Returns predicted T6 relative abundance vector (1D numpy array of shape [N]).
    """
    model.eval()
    curr = test_seq[0].unsqueeze(0)  # Shape (1, N)
    c_tensor = torch.tensor([c_idx], dtype=torch.long)

    with torch.no_grad():
        for t in range(1, len(test_seq)):
            _, nxt = model(curr, c_tensor)
            curr = nxt

    return curr.squeeze(0).cpu().numpy()


def compute_metrics(true_T6, pred_T6):
    """
    Computes Pearson correlation (r) and R^2 on log10 scale:
    1. r_nonzero, r2_nonzero: Evaluated ONLY on non-zero ground truth species (mask: true > 0).
    2. r_all_clip, r2_all_clip: Evaluated on ALL filtered species with 1e-6 clipping.
    """
    if isinstance(true_T6, torch.Tensor):
        true_T6 = true_T6.cpu().numpy()
    if isinstance(pred_T6, torch.Tensor):
        pred_T6 = pred_T6.cpu().numpy()

    # Metric 1: Non-zero ground truth species only
    mask = (true_T6 > 0)
    if np.sum(mask) >= 2:
        true_log_nz = np.log10(true_T6[mask])
        pred_log_nz = np.log10(pred_T6[mask])
        r_nz, _ = pearsonr(true_log_nz, pred_log_nz)
        r2_nonzero = float(r_nz ** 2) if not np.isnan(r_nz) else 0.0
        r_nz = float(r_nz) if not np.isnan(r_nz) else 0.0
    else:
        r_nz, r2_nonzero = 0.0, 0.0

    # Metric 2: All filtered species clipped at 1e-6
    true_log_all = np.log10(np.clip(true_T6, 1e-6, None))
    pred_log_all = np.log10(np.clip(pred_T6, 1e-6, None))
    r_all, _ = pearsonr(true_log_all, pred_log_all)
    r2_all_clip = float(r_all ** 2) if not np.isnan(r_all) else 0.0
    r_all = float(r_all) if not np.isnan(r_all) else 0.0

    return r_nz, r2_nonzero, r_all, r2_all_clip


def train_loo_model(model, train_seqs, train_conds, val_seqs, val_conds, epochs=1000, lr=0.01, l1_lambda=0.0, l2_lambda=0.0, verbose=True):
    """
    Trains model on full trajectories (T1 -> T6) using Soft Scheduled Sampling matching German_evaluation.py.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    seq_len = train_seqs.size(1)

    train_loss_hist = []
    val_loss_hist = []

    if verbose:
        print(f"\nTraining on {train_seqs.size(0)} full trajectories (T1->T6), Validating on {val_seqs.size(0)} trajectory...")

    for epoch in range(epochs):
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.8)))

        model.train()
        optimizer.zero_grad()

        x_curr = train_seqs[:, 0, :]
        batch_loss = 0.0
        for t in range(1, seq_len):
            log_pred, x_pred = model(x_curr, train_conds)
            target = train_seqs[:, t, :]
            batch_loss = batch_loss + criterion(log_pred, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred

        (batch_loss / (seq_len - 1)).backward()
        optimizer.step()

        # Proximal Gradient Descent for L1 and L2 Penalties
        with torch.no_grad():
            if l2_lambda > 0.0:
                l2_factor = 1.0 / (1.0 + 2.0 * lr * l2_lambda)
                model.Theta.data *= l2_factor
                model.W.data *= l2_factor

            if l1_lambda > 0.0:
                for param in model.parameters():
                    alpha_l1 = lr * l1_lambda
                    param.data = torch.sign(param.data) * torch.clamp(torch.abs(param.data) - alpha_l1, min=0.0)

        # Validation Loss Tracking
        model.eval()
        with torch.no_grad():
            x_curr = val_seqs[:, 0, :]
            val_loss = 0.0
            for t in range(1, seq_len):
                log_pred, x_pred = model(x_curr, val_conds)
                target = val_seqs[:, t, :]
                val_loss = val_loss + criterion(log_pred, target)
                x_curr = x_pred

        avg_loss = (batch_loss / (seq_len - 1)).item()
        avg_val_loss = (val_loss / (seq_len - 1)).item()

        train_loss_hist.append(avg_loss)
        val_loss_hist.append(avg_val_loss)

        if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:4d}/{epochs} | Train KL: {avg_loss:.6f} | Val KL: {avg_val_loss:.6f}", flush=True)

    return model, train_loss_hist, val_loss_hist


def evaluate_and_plot_null_environments(trained_model, test_key, test_seq, test_c_idx, unique_envs, num_species, output_path):
    """
    Evaluates final T6 autoregressive predictions using the TRUE environment vector vs
    NULL environment vectors (predictions using all other environment vectors).
    Generates a grouped bar chart comparing both evaluation metrics:
    1. All Filtered Species with 1e-6 clip
    2. Non-zero ground-truth species only
    """
    trained_model.eval()
    true_T6 = test_seq[-1].cpu().numpy() if isinstance(test_seq[-1], torch.Tensor) else test_seq[-1]

    r2_nonzero_list = []
    r2_all_clip_list = []
    r_nz_list = []
    r_all_list = []
    env_labels = []
    is_true_env = []

    print("\n==================================================", flush=True)
    print(f"NULL MODEL ENVIRONMENT COMPARISON FOR HELD-OUT TEST DATA", flush=True)
    print(f"Test Trajectory: {test_key}", flush=True)
    print(f"True Environment: Index {test_c_idx} -> {unique_envs[test_c_idx]}", flush=True)
    print("==================================================", flush=True)

    for e_idx, env_info in enumerate(unique_envs):
        pred_T6 = autoregressive_rollout_T6(trained_model, test_seq, e_idx)

        r_nz, r2_nz, r_all, r2_all = compute_metrics(true_T6, pred_T6)

        is_correct = (e_idx == test_c_idx)
        label = format_env_name(env_info)
        if is_correct:
            label += "\n(TRUE)"
        else:
            label += "\n(NULL)"

        r2_nonzero_list.append(r2_nz)
        r2_all_clip_list.append(r2_all)
        r_nz_list.append(r_nz)
        r_all_list.append(r_all)
        env_labels.append(label)
        is_true_env.append(is_correct)

        status = "TRUE ENVIRONMENT" if is_correct else "NULL ENVIRONMENT"
        print(f"Env [{e_idx}] {unique_envs[e_idx]} ({status:16s}): Non-Zero R^2 = {r2_nz:.4f} (r={r_nz:+.4f}) | All Filtered R^2 = {r2_all:.4f} (r={r_all:+.4f})", flush=True)

    # Plotting Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(13, 6.5))

    x = np.arange(len(unique_envs))
    width = 0.35

    # Bars for All Filtered Species (1e-6 Clip)
    colors_all = ['#1f77b4' if is_true else '#aec7e8' for is_true in is_true_env]
    edgecolors_all = ['black' if is_true else '#4682b4' for is_true in is_true_env]
    linewidths_all = [2.0 if is_true else 1.0 for is_true in is_true_env]
    bars_all = ax.bar(
        x - width / 2, r2_all_clip_list, width,
        color=colors_all, edgecolor=edgecolors_all, linewidth=linewidths_all,
        label=f'All Filtered Species ({num_species} taxa, 1e-6 Clip)'
    )

    # Bars for Non-Zero Species Only
    colors_nz = ['#d62728' if is_true else '#ff9896' for is_true in is_true_env]
    edgecolors_nz = ['black' if is_true else '#b22222' for is_true in is_true_env]
    linewidths_nz = [2.0 if is_true else 1.0 for is_true in is_true_env]
    bars_nz = ax.bar(
        x + width / 2, r2_nonzero_list, width,
        color=colors_nz, edgecolor=edgecolors_nz, linewidth=linewidths_nz,
        label='Non-Zero True Species Only'
    )

    # Annotate bar values (R^2 and r)
    for idx, bar in enumerate(bars_all):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, yval + 0.012,
            f"{yval:.3f}",
            ha='center', va='bottom', fontsize=9.5,
            fontweight='bold' if is_true_env[idx] else 'normal'
        )

    for idx, bar in enumerate(bars_nz):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, yval + 0.012,
            f"{yval:.3f}",
            ha='center', va='bottom', fontsize=9.5,
            fontweight='bold' if is_true_env[idx] else 'normal'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(env_labels, fontsize=10.5)
    ax.set_ylabel("Pearson $R^2$ (Log Scale, Final Passage T6)", fontsize=13, fontweight='bold')

    clean_test_name = test_key if isinstance(test_key, str) else " ".join(str(x) for x in test_key)
    ax.set_title(
        f"Latent Assembly Model: Environment Specificity vs Null Models\nTest Trajectory: {clean_test_name}",
        fontsize=14, fontweight='bold', pad=15
    )

    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    max_y = max(max(r2_all_clip_list), max(r2_nonzero_list), 0.1)
    ax.set_ylim(0, max(0.65, max_y * 1.25))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Null Environment Evaluation Bar Chart to: '{output_path}'", flush=True)
    plt.close()

    return {
        'r2_all_clip': r2_all_clip_list,
        'r2_nonzero': r2_nonzero_list,
        'r_all': r_all_list,
        'r_nz': r_nz_list,
        'unique_envs': unique_envs,
        'test_c_idx': test_c_idx,
    }


def main():
    parser = argparse.ArgumentParser(description="Null Environment Evaluation for German Microbiome Dataset")
    parser.add_argument('--data-path', type=str, default=os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt'),
                        help='Path to German_processed_data.pt')
    parser.add_argument('--output-plot', type=str, default=os.path.join(PROJECT_ROOT, 'Images', 'German_null_environment_evaluation.png'),
                        help='Path to save output figure')
    parser.add_argument('--held-out-idx', type=int, default=0, help='Index of held-out trajectory (0 to 11)')
    parser.add_argument('--val-idx', type=int, default=1, help='Index of validation trajectory (0 to 11)')
    parser.add_argument('--latent-dims', type=int, default=4, help='Number of latent metabolic dimensions (K)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 regularization penalty')
    parser.add_argument('--seed', type=int, default=BASE_SEED, help='Random seed for reproducibility')
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Load and filter dataset
    trajectories, strains = load_and_filter_data(args.data_path)
    num_species = len(strains)

    # 2. Map 12 trajectories to 6 unique environmental conditions
    env_map, unique_envs = map_environments(trajectories)
    num_conditions = len(unique_envs)

    # 3. Select Held-Out Test, Validation, and Training Sets
    all_keys = list(trajectories.keys())
    held_out_key = all_keys[args.held_out_idx]
    val_key = all_keys[args.val_idx]
    train_keys = [k for k in all_keys if k not in (held_out_key, val_key)]

    test_seq = trajectories[held_out_key]
    test_c_idx = env_map[held_out_key]

    val_seqs = torch.stack([trajectories[val_key]])
    val_conds = torch.tensor([env_map[val_key]], dtype=torch.long)

    train_seqs = torch.stack([trajectories[k] for k in train_keys])
    train_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)

    print(f"\nLeave-One-Out Experimental Split:", flush=True)
    print(f" - Training Set: {len(train_keys)} trajectories (full T1->T6)", flush=True)
    print(f" - Validation Set: [{val_key}] (full T1->T6)", flush=True)
    print(f" - Held-Out Test Set: [{held_out_key}] (full T1->T6)", flush=True)
    print(f" - Model Hyperparameters: Latent Dims={args.latent_dims}, Epochs={args.epochs}, LR={args.lr}, L2={args.l2}", flush=True)

    # 4. Initialize and Train Model
    model = LatentAssemblyModel(
        num_species=num_species,
        num_latent_dims=args.latent_dims,
        num_conditions=num_conditions
    )

    trained_model, train_loss_hist, val_loss_hist = train_loo_model(
        model, train_seqs, train_conds, val_seqs, val_conds,
        epochs=args.epochs, lr=args.lr, l1_lambda=0.0, l2_lambda=args.l2, verbose=True
    )

    # 5. Evaluate and Plot Null Environments
    evaluate_and_plot_null_environments(
        trained_model=trained_model,
        test_key=held_out_key,
        test_seq=test_seq,
        test_c_idx=test_c_idx,
        unique_envs=unique_envs,
        num_species=num_species,
        output_path=args.output_plot
    )


if __name__ == "__main__":
    main()

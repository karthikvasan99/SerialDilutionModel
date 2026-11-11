"""
Parallel Hyperparameter Grid Search for the Continuous-Time Neural ODE Model.

Searches over:
  - Number of latent resource pools (K)
  - Proximal L2 regularization penalty
  - Number of training epochs

Evaluates Leave-One-Out autoregressive multi-passage rollout on BRRS H2:CO2 (5.0 bar)
using BRRS H2:CO2 (30.0 bar) as validation.
"""

import os
import sys
import copy
import time
import argparse
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
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

# ─── Reproducibility Base Seed ───
BASE_SEED = 42
torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)

# ─── Hyperparameter Grid ───
LATENT_DIMS_GRID = [2, 3, 4, 5]
L2_GRID = [0.0, 0.01, 0.05, 0.10, 0.25]
EPOCHS_GRID = [200, 400, 600]

LR = 0.01
DILUTION_FACTOR = 100.0
ABUNDANCE_THRESHOLD = 1e-3
DEFAULT_NUM_WORKERS = 8


def load_and_filter_data(data_path, abundance_threshold=ABUNDANCE_THRESHOLD):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file '{data_path}' not found.")

    data = torch.load(data_path, weights_only=False)
    strains = data['strains']
    trajectories = data['trajectories']

    t1_passages = torch.stack([seq[0] for seq in trajectories.values()], dim=0)
    mean_t1_abundance = t1_passages.mean(dim=0)
    all_t1_present = (t1_passages > 0).all(dim=0)
    keep_mask = (mean_t1_abundance >= abundance_threshold) & all_t1_present

    for k in trajectories.keys():
        s = trajectories[k][:, keep_mask]
        trajectories[k] = s / s.sum(dim=-1, keepdim=True)

    strains = [s for i, s in enumerate(strains) if keep_mask[i]]
    return trajectories, strains


def map_environments(trajectories_dict):
    env_map = {}
    unique_envs = []
    for cond_key in trajectories_dict.keys():
        env_label = cond_key[1:] if isinstance(cond_key, tuple) and len(cond_key) >= 2 else cond_key
        if env_label not in unique_envs:
            unique_envs.append(env_label)
        env_map[cond_key] = unique_envs.index(env_label)
    return env_map, unique_envs


def evaluate_autoregressive_ode(model, test_seq, c_idx):
    """Run continuous autoregressive rollout from T1 to T6 and compute per-passage R^2."""
    model.eval()
    c_tensor = torch.tensor([c_idx], dtype=torch.long)
    passage_r2 = {}
    passage_r2_all = {}

    with torch.no_grad():
        curr = test_seq[0].unsqueeze(0)
        for t in range(1, len(test_seq)):
            _, nxt, _, _ = model(curr, c_tensor)
            true_t = test_seq[t].numpy()
            pred_t = nxt.squeeze(0).numpy()

            mask = true_t > 0
            if mask.sum() >= 2:
                r, _ = pearsonr(
                    np.log10(true_t[mask]),
                    np.log10(np.clip(pred_t[mask], 1e-6, None))
                )
                passage_r2[t + 1] = float(r**2) if not np.isnan(r) else 0.0
            else:
                passage_r2[t + 1] = 0.0

            r_all, _ = pearsonr(
                np.log10(np.clip(true_t, 1e-6, None)),
                np.log10(np.clip(pred_t, 1e-6, None))
            )
            passage_r2_all[t + 1] = float(r_all**2) if not np.isnan(r_all) else 0.0

            curr = nxt

    return passage_r2, passage_r2_all


def train_and_eval_ode_combo(params):
    """
    Worker task: trains one Neural ODE model for (K, l2, epochs) using Soft Scheduled Sampling
    and proximal regularization, restoring the best validation state.
    """
    (
        combo_idx,
        K,
        l2,
        epochs,
        num_species,
        num_conditions,
        train_seqs,
        train_conds,
        val_seqs,
        val_conds,
        test_seq,
        test_c_idx,
        lr,
        dilution_factor,
        seed,
    ) = params

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    seq_len = train_seqs.size(1)

    model = LatentAssemblyNeuralODE(
        num_species=num_species,
        num_latent_dims=K,
        num_conditions=num_conditions,
        default_dilution_factor=dilution_factor,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')

    best_val_loss = float('inf')
    best_model_state = None

    t0 = time.time()

    for epoch in range(epochs):
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.8)))

        # Training Step
        model.train()
        optimizer.zero_grad()
        x_curr = train_seqs[:, 0, :]
        batch_loss = 0.0

        for t in range(1, seq_len):
            log_pred, x_pred, _, _ = model(x_curr, train_conds)
            target = train_seqs[:, t, :]
            batch_loss = batch_loss + criterion(log_pred, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred

        (batch_loss / (seq_len - 1)).backward()
        optimizer.step()

        # Proximal L2 updates
        with torch.no_grad():
            if l2 > 0.0:
                l2_factor = 1.0 / (1.0 + 2.0 * lr * l2)
                model.ode_func.Theta.data *= l2_factor
                model.ode_func.W.data *= l2_factor

        # Validation Step
        model.eval()
        with torch.no_grad():
            x_curr_val = val_seqs[:, 0, :]
            val_loss = 0.0
            for t in range(1, seq_len):
                log_pred, x_pred, _, _ = model(x_curr_val, val_conds)
                target = val_seqs[:, t, :]
                val_loss += criterion(log_pred, target).item()
                x_curr_val = x_pred

        avg_val = val_loss / (seq_len - 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = copy.deepcopy(model.state_dict())

    # Load best checkpoint for testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    avg_train = (batch_loss / (seq_len - 1)).item()
    passage_r2, passage_r2_all = evaluate_autoregressive_ode(model, test_seq, test_c_idx)

    result = {
        'combo_idx': combo_idx,
        'K': K,
        'l2': l2,
        'epochs': epochs,
        'train_kl': avg_train,
        'val_kl': best_val_loss,
    }
    for p, r2 in passage_r2.items():
        result[f'R2_T{p}'] = r2
    result['R2_mean'] = float(np.mean(list(passage_r2.values())))
    result['R2_final'] = passage_r2.get(len(test_seq), 0.0)
    result['R2_final_all'] = passage_r2_all.get(len(test_seq), 0.0)

    dt = time.time() - t0
    return combo_idx, K, l2, epochs, best_val_loss, result['R2_final'], dt, result


def plot_ode_grid_search_results(df, images_dir):
    os.makedirs(images_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ─── Panel A: Best T6 R2 vs Latent Dims (K) ───
    best_per_k = df.loc[df.groupby('K')['R2_final'].idxmax()]
    axes[0, 0].bar(range(len(best_per_k)), best_per_k['R2_final'].values, color='steelblue', edgecolor='black')
    axes[0, 0].set_xticks(range(len(best_per_k)))
    axes[0, 0].set_xticklabels(best_per_k['K'].values)
    axes[0, 0].set_xlabel('Latent Resources (K)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Best $R^2$ at T6', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('A. Best T6 $R^2$ by Latent Resources (K)', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
    for i, (_, row) in enumerate(best_per_k.iterrows()):
        axes[0, 0].text(i, row['R2_final'] + 0.005, f"{row['R2_final']:.3f}", ha='center', fontsize=9)

    # ─── Panel B: Best T6 R2 vs L2 Penalty ───
    best_per_l2 = df.loc[df.groupby('l2')['R2_final'].idxmax()]
    axes[0, 1].plot(best_per_l2['l2'], best_per_l2['R2_final'], 'o-', color='darkorange', lw=2, markersize=6)
    axes[0, 1].set_xlabel('L2 Penalty', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Best $R^2$ at T6', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('B. Best T6 $R^2$ by L2 Penalty', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # ─── Panel C: Best T6 R2 vs Epochs ───
    best_per_ep = df.loc[df.groupby('epochs')['R2_final'].idxmax()]
    axes[0, 2].plot(best_per_ep['epochs'], best_per_ep['R2_final'], 's-', color='crimson', lw=2, markersize=6)
    axes[0, 2].set_xlabel('Epochs', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('Best $R^2$ at T6', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('C. Best T6 $R^2$ by Epochs', fontsize=13, fontweight='bold')
    axes[0, 2].grid(True, linestyle='--', alpha=0.5)

    # ─── Panel D: Heatmap — Best R2_final for each (K, L2) ───
    pivot = df.loc[df.groupby(['K', 'l2'])['R2_final'].idxmax()].pivot(index='K', columns='l2', values='R2_final')
    im = axes[1, 0].imshow(pivot.values, cmap='YlOrRd', aspect='auto', origin='lower')
    axes[1, 0].set_xticks(range(len(pivot.columns)))
    axes[1, 0].set_xticklabels([f"{v:.3f}" for v in pivot.columns], rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_yticks(range(len(pivot.index)))
    axes[1, 0].set_yticklabels(pivot.index.values)
    axes[1, 0].set_xlabel('L2 Penalty', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Latent Resources (K)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('D. T6 $R^2$ Heatmap (K × L2)', fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # ─── Panel E: Heatmap — Best R2_mean for each (K, L2) ───
    pivot_mean = df.loc[df.groupby(['K', 'l2'])['R2_mean'].idxmax()].pivot(index='K', columns='l2', values='R2_mean')
    im2 = axes[1, 1].imshow(pivot_mean.values, cmap='YlGnBu', aspect='auto', origin='lower')
    axes[1, 1].set_xticks(range(len(pivot_mean.columns)))
    axes[1, 1].set_xticklabels([f"{v:.3f}" for v in pivot_mean.columns], rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_yticks(range(len(pivot_mean.index)))
    axes[1, 1].set_yticklabels(pivot_mean.index.values)
    axes[1, 1].set_xlabel('L2 Penalty', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Latent Resources (K)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('E. Mean $R^2$ Heatmap (K × L2)', fontsize=13, fontweight='bold')
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # ─── Panel F: Per-Passage R^2 for Top 5 Configs ───
    top5 = df.sort_values('R2_final', ascending=False).head(5)
    passage_cols = [c for c in df.columns if c.startswith('R2_T')]
    passages = [int(c.replace('R2_T', '')) for c in passage_cols]
    colors_top5 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for i, (_, row) in enumerate(top5.iterrows()):
        label = f"K={int(row['K'])}, L2={row['l2']:.3f}, ep={int(row['epochs'])}"
        vals = [row[c] for c in passage_cols]
        axes[1, 2].plot(passages, vals, 'o-', color=colors_top5[i], lw=2, markersize=6, label=label)
    axes[1, 2].set_xlabel('Passage', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Pearson $R^2$ (Non-Zero, Log)', fontsize=12, fontweight='bold')
    axes[1, 2].set_title('F. Per-Passage $R^2$ (Top 5 by T6)', fontsize=13, fontweight='bold')
    axes[1, 2].legend(fontsize=9, loc='best')
    axes[1, 2].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Continuous Neural ODE — Extensive Hyperparameter Grid Search', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(images_dir, 'German_ODE_grid_search_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved Neural ODE grid search visualization to '{plot_path}'")
    plt.close()


def main(num_workers=DEFAULT_NUM_WORKERS):
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
    results_dir = os.path.join(PROJECT_ROOT, 'Results')
    images_dir = os.path.join(PROJECT_ROOT, 'Images')

    trajectories, strains = load_and_filter_data(data_path)
    env_map, unique_envs = map_environments(trajectories)
    num_species = len(strains)
    num_conditions = len(unique_envs)

    all_keys = list(trajectories.keys())
    held_out_key = all_keys[0]   # BRRS H2:CO2 5.0
    val_key = all_keys[1]        # BRRS H2:CO2 30.0
    train_keys = [k for k in all_keys if k not in (held_out_key, val_key)]

    train_seqs = torch.stack([trajectories[k] for k in train_keys])
    train_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)
    val_seqs = torch.stack([trajectories[val_key]])
    val_conds = torch.tensor([env_map[val_key]], dtype=torch.long)
    test_seq = trajectories[held_out_key]
    test_c_idx = env_map[held_out_key]

    print(f"Neural ODE Grid Search Configuration:")
    print(f"  Parallel workers: {num_workers}")
    print(f"  Species (N): {num_species}")
    print(f"  Conditions: {num_conditions}")
    print(f"  Held-out test: {held_out_key}")
    print(f"  Validation:    {val_key}")
    print(f"  Training trajectories: {len(train_keys)} (full T1->T6 sequences)")
    print(f"\n  Latent dims grid ({len(LATENT_DIMS_GRID)}): {LATENT_DIMS_GRID}")
    print(f"  L2 penalty grid ({len(L2_GRID)}): {L2_GRID}")
    print(f"  Epochs grid ({len(EPOCHS_GRID)}): {EPOCHS_GRID}")
    total_combos = len(LATENT_DIMS_GRID) * len(L2_GRID) * len(EPOCHS_GRID)
    print(f"\n  Total combinations: {total_combos}\n")

    tasks = []
    combo_idx = 0
    for K in LATENT_DIMS_GRID:
        for l2 in L2_GRID:
            for ep in EPOCHS_GRID:
                combo_idx += 1
                seed = BASE_SEED + combo_idx
                tasks.append((
                    combo_idx,
                    K,
                    l2,
                    ep,
                    num_species,
                    num_conditions,
                    train_seqs,
                    train_conds,
                    val_seqs,
                    val_conds,
                    test_seq,
                    test_c_idx,
                    LR,
                    DILUTION_FACTOR,
                    seed,
                ))

    results = []
    completed_count = 0
    t_start_global = time.time()

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        for combo_idx, K, l2, ep, best_val_loss, r2_final, dt, result in pool.imap_unordered(train_and_eval_ode_combo, tasks):
            completed_count += 1
            elapsed = time.time() - t_start_global
            print(f"  [{completed_count:>2}/{total_combos}] (Job #{combo_idx:>2}) K={K:>2}, L2={l2:.3f}, ep={ep:3d} | "
                  f"val_kl={best_val_loss:.5f} | R2_T6={r2_final:.4f} | "
                  f"{dt:.1f}s (elapsed {elapsed:.0f}s)")
            results.append(result)

    # ─── Save Results ───
    df = pd.DataFrame(results)
    df = df.sort_values(['K', 'l2', 'epochs']).reset_index(drop=True)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'German_ODE_grid_search_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} results to '{csv_path}'")

    # ─── Print Top Results ───
    print("\n" + "="*80)
    print("TOP 20 NEURAL ODE CONFIGURATIONS BY FINAL PASSAGE R^2 (T6)")
    print("="*80)
    df_sorted = df.sort_values('R2_final', ascending=False).head(20)
    print(df_sorted[['K', 'l2', 'epochs', 'val_kl', 'R2_T2', 'R2_T3', 'R2_T4', 'R2_T5', 'R2_T6', 'R2_mean']].to_string(index=False))

    print("\n" + "="*80)
    print("TOP 20 NEURAL ODE CONFIGURATIONS BY MEAN R^2 ACROSS ALL PASSAGES")
    print("="*80)
    df_sorted_mean = df.sort_values('R2_mean', ascending=False).head(20)
    print(df_sorted_mean[['K', 'l2', 'epochs', 'val_kl', 'R2_T2', 'R2_T3', 'R2_T4', 'R2_T5', 'R2_T6', 'R2_mean']].to_string(index=False))

    print("\n" + "="*80)
    print("TOP 20 NEURAL ODE CONFIGURATIONS BY BEST VALIDATION LOSS")
    print("="*80)
    df_sorted_val = df.sort_values('val_kl', ascending=True).head(20)
    print(df_sorted_val[['K', 'l2', 'epochs', 'val_kl', 'R2_T2', 'R2_T3', 'R2_T4', 'R2_T5', 'R2_T6', 'R2_mean']].to_string(index=False))

    # ─── Visualization ───
    plot_ode_grid_search_results(df, images_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel Grid Search for Neural ODE Latent Assembly Model")
    parser.add_argument('--num-workers', '--cores', dest='num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help=f'Number of parallel worker processes to use (default: {DEFAULT_NUM_WORKERS})')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(num_workers=args.num_workers)

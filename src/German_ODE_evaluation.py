import os
import sys
import copy
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torchdiffeq import odeint

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE

# ─── Reproducibility Base Seed ───
BASE_SEED = 42


def set_seed(seed=BASE_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_filter_data(data_path, abundance_threshold=1e-3):
    """Loads dataset and filters ubiquitous species with T1 mean >= threshold."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file '{data_path}' not found.")

    data = torch.load(data_path, weights_only=False)
    strains = data['strains']
    trajectories = data['trajectories']
    num_before = len(strains)

    t1_passages = torch.stack([seq[0] for seq in trajectories.values()], dim=0)
    mean_t1_abundance = t1_passages.mean(dim=0)
    all_t1_present = (t1_passages > 0).all(dim=0)
    keep_mask = (mean_t1_abundance >= abundance_threshold) & all_t1_present
    num_after = int(keep_mask.sum().item())

    print(f"\nSpecies T1 Abundance Filtering (Threshold >= {abundance_threshold} & Present in All 12 T1 Samples):")
    print(f" - Total bacteria BEFORE filter: {num_before}")
    print(f" - Total bacteria AFTER filter:  {num_after}")
    print(f" - Filtered out: {num_before - num_after} low-abundance / noise species")

    strains = [s for idx, s in enumerate(strains) if keep_mask[idx]]
    for k in trajectories.keys():
        filt_seq = trajectories[k][:, keep_mask]
        trajectories[k] = filt_seq / filt_seq.sum(dim=-1, keepdim=True)

    return trajectories, strains


def map_environments(trajectories_dict):
    """Maps 12 trajectory condition tuples down to unique environmental media indices."""
    env_map = {}
    unique_envs = []
    for cond_key in trajectories_dict.keys():
        env_label = cond_key[1:] if isinstance(cond_key, tuple) and len(cond_key) >= 2 else cond_key
        if env_label not in unique_envs:
            unique_envs.append(env_label)
        env_map[cond_key] = unique_envs.index(env_label)
    return env_map, unique_envs


def train_neural_ode(model, train_seqs, train_conds, val_seqs, val_conds,
                     epochs=500, lr=0.01, l2_lambda=0.05, verbose=True):
    """
    Trains LatentAssemblyNeuralODE using Soft Scheduled Sampling and proximal L2 regularization.
    Restores the best model state based on validation loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')

    seq_len = train_seqs.size(1)
    train_loss_hist = []
    val_loss_hist = []

    best_val_loss = float('inf')
    best_model_state = None

    if verbose:
        print(f"\nTraining Neural ODE on {train_seqs.size(0)} trajectories (T1->T6), Validating on {val_seqs.size(0)} trajectory...")

    t0 = time.time()
    for epoch in range(epochs):
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.8)))

        # --- Training ---
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

        # Proximal L2 Regularization on ode_func.Theta and ode_func.W
        with torch.no_grad():
            if l2_lambda > 0.0:
                l2_factor = 1.0 / (1.0 + 2.0 * lr * l2_lambda)
                model.ode_func.Theta.data *= l2_factor
                model.ode_func.W.data *= l2_factor

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            x_curr_val = val_seqs[:, 0, :]
            val_loss = 0.0
            for t in range(1, seq_len):
                log_pred, x_pred, _, _ = model(x_curr_val, val_conds)
                target = val_seqs[:, t, :]
                val_loss += criterion(log_pred, target).item()
                x_curr_val = x_pred

        avg_train_loss = (batch_loss / (seq_len - 1)).item()
        avg_val_loss = val_loss / (seq_len - 1)

        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0 or epoch == epochs - 1):
            dt = time.time() - t0
            print(f"Epoch {epoch+1:4d}/{epochs} | Train KL: {avg_train_loss:.6f} | Val KL: {avg_val_loss:.6f} (best: {best_val_loss:.6f}) | {dt:.1f}s")

    # Load best checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best Neural ODE checkpoint with Validation KL = {best_val_loss:.6f}")

    return model, train_loss_hist, val_loss_hist


def evaluate_and_plot_ode(model, test_key, test_seq, test_c_idx,
                          train_loss_hist, val_loss_hist,
                          train_keys, trajectories, env_map,
                          images_dir):
    """
    Evaluates Neural ODE autoregressive predictions and plots 8-panel diagnostic figure.
    """
    model.eval()
    seq_len = test_seq.size(0)

    # 1. Evaluate Train Autoregressive Rollouts
    train_true_all = []
    train_pred_all = []
    with torch.no_grad():
        for k in train_keys:
            seq = trajectories[k]
            c_idx = env_map[k]
            curr = seq[0].unsqueeze(0)
            c_tensor = torch.tensor([c_idx], dtype=torch.long)
            for t in range(1, seq_len):
                _, nxt, _, _ = model(curr, c_tensor)
                train_true_all.append(seq[t].numpy().flatten())
                train_pred_all.append(nxt.squeeze(0).numpy().flatten())
                curr = nxt

    train_true_flat = np.concatenate(train_true_all) if train_true_all else np.array([])
    train_pred_flat = np.concatenate(train_pred_all) if train_pred_all else np.array([])

    # 2. Evaluate Held-Out Test Trajectory Rollout
    test_true_dict = {}
    test_pred_dict = {}
    test_true_all = []
    test_pred_all = []

    with torch.no_grad():
        curr = test_seq[0].unsqueeze(0)
        c_tensor = torch.tensor([test_c_idx], dtype=torch.long)
        for t in range(1, seq_len):
            _, nxt, _, _ = model(curr, c_tensor)
            true_t = test_seq[t].numpy().flatten()
            pred_t = nxt.squeeze(0).numpy().flatten()

            test_true_dict[t] = true_t
            test_pred_dict[t] = pred_t
            test_true_all.append(true_t)
            test_pred_all.append(pred_t)
            curr = nxt

    test_true_flat = np.concatenate(test_true_all) if test_true_all else np.array([])
    test_pred_flat = np.concatenate(test_pred_all) if test_pred_all else np.array([])

    # 3. 8-Panel Figure Construction
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # Panel 0: Loss Curve
    axes[0].plot(train_loss_hist, color='steelblue', lw=2, label='Train Loss (KL)')
    axes[0].plot(val_loss_hist, color='darkorange', lw=2, label='Val Loss (KL)')
    axes[0].set_title("Neural ODE Training & Validation Loss", fontsize=15, fontweight='bold')
    axes[0].set_xlabel("Epochs", fontsize=13)
    axes[0].set_ylabel("KL Divergence", fontsize=13)
    axes[0].set_yscale('log')
    axes[0].legend(loc='upper right', fontsize=12)
    axes[0].grid(True, ls='--', alpha=0.5)

    def plot_scatter(ax, true_vals, pred_vals, title, color):
        mask_nz = (true_vals > 0)
        if np.sum(mask_nz) >= 2:
            true_log_nz = np.log10(true_vals[mask_nz])
            pred_log_nz = np.log10(np.clip(pred_vals[mask_nz], 1e-6, None))
            r_nz, _ = pearsonr(true_log_nz, pred_log_nz)
            r2_nz = float(r_nz ** 2) if not np.isnan(r_nz) else 0.0
        else:
            r2_nz = 0.0

        true_log_all = np.log10(np.clip(true_vals, 1e-6, None))
        pred_log_all = np.log10(np.clip(pred_vals, 1e-6, None))
        r_all, _ = pearsonr(true_log_all, pred_log_all)
        r2_all = float(r_all ** 2) if not np.isnan(r_all) else 0.0

        clip_true = np.clip(true_vals, 1e-6, None)
        clip_pred = np.clip(pred_vals, 1e-6, None)

        is_zero = (true_vals == 0) | (pred_vals <= 1e-6)
        is_active = ~is_zero

        if np.any(is_active):
            ax.scatter(
                clip_true[is_active], clip_pred[is_active],
                alpha=0.65, edgecolors='k', c=color, s=55,
                label=f'Active Non-Zero ($R^2_{{nz}}={r2_nz:.2f}$)'
            )

        if np.any(is_zero):
            ax.scatter(
                clip_true[is_zero], clip_pred[is_zero],
                alpha=0.85, edgecolors='k', c='#9467bd', marker='D', s=50,
                label=fr'Extinct / Floor ($\leq 10^{{-6}}$, $R^2_{{all}}={r2_all:.2f}$)'
            )
        else:
            ax.scatter([], [], c='#9467bd', marker='D', label=f'All Taxa ($R^2_{{all}}={r2_all:.2f}$)')

        ax.axvline(1e-6, color='gray', linestyle=':', alpha=0.5, lw=1)
        ax.axhline(1e-6, color='gray', linestyle=':', alpha=0.5, lw=1)

        plot_min = 4e-7
        plot_max = 1.5
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label='y=x')

        ax.set_xlim([plot_min, plot_max])
        ax.set_ylim([plot_min, plot_max])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10.5, framealpha=0.9)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.2)

    # Panel 1: Train Scatter
    plot_scatter(axes[1], train_true_flat, train_pred_flat, "Train Autoregressive (All)", 'steelblue')

    # Panel 2: Test Scatter
    plot_scatter(axes[2], test_true_flat, test_pred_flat, "Test Autoregressive (All)", 'darkorange')

    # Panels 3-7: Test Scatter by Passage (T2 to T6)
    for idx in range(1, 6):
        ax_idx = 2 + idx
        if idx in test_true_dict:
            plot_scatter(axes[ax_idx], test_true_dict[idx], test_pred_dict[idx], f"Test Passage T{idx+1}", 'crimson')
        else:
            axes[ax_idx].axis('off')

    fig.supxlabel('True Relative Abundance', fontsize=18, y=0.02)
    fig.supylabel('Predicted Relative Abundance (Neural ODE)', fontsize=18, x=0.01)
    plt.tight_layout(rect=[0.03, 0.04, 1, 1])

    os.makedirs(images_dir, exist_ok=True)
    if isinstance(test_key, tuple):
        clean_name = "_".join(str(item).replace(":", "_").replace(" ", "") for item in test_key)
    else:
        clean_name = str(test_key).replace(" ", "_").replace("|", "").replace(":", "")

    plot_filename = os.path.join(images_dir, f'German_ODE_loo_eval_{clean_name}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved Neural ODE evaluation plot to '{plot_filename}'")
    plt.close()


def plot_continuous_dynamics(model, test_key, test_seq, test_c_idx, strains, images_dir):
    """
    Simulates continuous within-passage biomass growth N_i(t) and latent resource depletion R_k(t)
    across all 5 serial dilution passage cycles.
    """
    model.eval()
    seq_len = test_seq.size(0)
    c_tensor = torch.tensor([test_c_idx], dtype=torch.long)

    t_eval = torch.linspace(0.0, 1.0, 60)
    passage_N_curves = []
    passage_R_curves = []
    passage_rel_curves = []

    with torch.no_grad():
        curr_N = test_seq[0].unsqueeze(0)  # Inoculum (T1)

        for p in range(1, seq_len):
            N_0 = curr_N / model.default_dilution_factor
            a_batch = model.a[c_tensor]
            R_0 = F.softplus(a_batch)
            S_0 = torch.cat([torch.clamp(N_0, min=0.0), R_0], dim=-1)

            S_t = odeint(model.ode_func, S_0, t_eval, method='dopri5')  # (60, 1, N + K)
            N_t = S_t[:, 0, :model.N].numpy()  # (60, N)
            R_t = S_t[:, 0, model.N:].numpy()  # (60, K)

            rel_t = N_t / (np.sum(N_t, axis=-1, keepdims=True) + 1e-12)

            passage_N_curves.append(N_t)
            passage_R_curves.append(R_t)
            passage_rel_curves.append(rel_t)

            curr_N = torch.clamp(S_t[-1, 0:1, :model.N], min=0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11), sharex=True)

    # Time axis: 5 passages of duration 1.0 each (total time 0 to 5.0)
    time_pts = []
    all_rel = []
    all_R = []
    for p_idx in range(len(passage_rel_curves)):
        offset = float(p_idx)
        time_pts.append(t_eval.numpy() + offset)
        all_rel.append(passage_rel_curves[p_idx])
        all_R.append(passage_R_curves[p_idx])

    time_pts_concat = np.concatenate(time_pts)
    rel_concat = np.concatenate(all_rel, axis=0)
    R_concat = np.concatenate(all_R, axis=0)

    # Panel 1: Top 8 most abundant species relative abundances over continuous time
    mean_abundance = rel_concat.mean(axis=0)
    top_indices = np.argsort(mean_abundance)[::-1][:8]

    cmap = plt.colormaps['tab10']
    for rank, idx in enumerate(top_indices):
        raw_name = strains[idx].split('|')[-1]
        label = f"{raw_name[:25]}"
        ax1.plot(time_pts_concat, rel_concat[:, idx], lw=2.2, label=label, color=cmap(rank))

    # Add vertical dashed lines indicating serial dilution passage resets
    for p in range(1, seq_len - 1):
        ax1.axvline(p, color='gray', linestyle='--', alpha=0.7)
        ax2.axvline(p, color='gray', linestyle='--', alpha=0.7)
        ax1.text(p + 0.05, 0.90, f"Dilution (T{p+1})", color='dimgray', fontsize=10, fontweight='bold', transform=ax1.get_xaxis_transform())

    ax1.set_ylabel("Relative Abundance $x_i(t)$", fontsize=13, fontweight='bold')
    ax1.set_title(f"A. Continuous Microbial Community Dynamics Across Serial Dilutions ({test_key})", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.22, 1.0), fontsize=9.5, frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Panel 2: Latent Resource Depletion Dynamics
    for k in range(model.K):
        ax2.plot(time_pts_concat, R_concat[:, k], lw=2.2, label=f"Latent Resource $R_{k+1}(t)$")

    ax2.set_xlabel("Continuous Serial Dilution Time (Passage Cycle Index)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Latent Resource Conc. $R_k(t)$", fontsize=13, fontweight='bold')
    ax2.set_title("B. Continuous Latent Resource Depletion & Cross-Feeding Kinetics", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.22, 1.0), fontsize=10, frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    if isinstance(test_key, tuple):
        clean_name = "_".join(str(item).replace(":", "_").replace(" ", "") for item in test_key)
    else:
        clean_name = str(test_key).replace(" ", "_").replace("|", "").replace(":", "")

    dynamics_path = os.path.join(images_dir, f'German_ODE_continuous_dynamics_{clean_name}.png')
    plt.savefig(dynamics_path, dpi=300, bbox_inches='tight')
    print(f"Saved continuous within-passage kinetics plot to '{dynamics_path}'")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Neural ODE Leave-One-Out Evaluation on German Dataset")
    parser.add_argument('--held-out-idx', type=int, default=0, help='Index of held-out trajectory (0 to 11)')
    parser.add_argument('--val-idx', type=int, default=1, help='Index of validation trajectory (0 to 11)')
    parser.add_argument('--latent-dims', type=int, default=4, help='Number of latent resource dimensions (K, default: 4)')
    parser.add_argument('--l2', type=float, default=0.05, help='Proximal L2 penalty on Theta and W (default: 0.05)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--dilution-factor', type=float, default=100.0, help='Passage dilution factor (default: 100.0)')
    parser.add_argument('--seed', type=int, default=BASE_SEED, help='Random seed for reproducibility')
    args = parser.parse_args()

    set_seed(args.seed)
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
    images_dir = os.path.join(PROJECT_ROOT, 'Images')

    trajectories, strains = load_and_filter_data(data_path)
    env_map, unique_envs = map_environments(trajectories)
    num_species = len(strains)
    num_conditions = len(unique_envs)

    all_keys = list(trajectories.keys())
    held_out_key = all_keys[args.held_out_idx]
    val_key = all_keys[args.val_idx]
    train_keys = [k for k in all_keys if k not in (held_out_key, val_key)]

    test_seq = trajectories[held_out_key]
    test_c_idx = env_map[held_out_key]

    print(f"\nNeural ODE Setup:")
    print(f" - Model Type: LatentAssemblyNeuralODE (coupled continuous consumer-resource ODEs)")
    print(f" - Species (N): {num_species}")
    print(f" - Latent Resources (K): {args.latent_dims}")
    print(f" - Environmental Media Conditions: {num_conditions}")
    print(f" - Dilution Factor: {args.dilution_factor}")
    print(f" - Held-Out Test Trajectory: {held_out_key}")
    print(f" - Validation Trajectory:    {val_key}")
    print(f" - Training Trajectories:    {len(train_keys)}")

    train_seqs = torch.stack([trajectories[k] for k in train_keys])
    train_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)
    val_seqs = torch.stack([trajectories[val_key]])
    val_conds = torch.tensor([env_map[val_key]], dtype=torch.long)

    # Initialize Neural ODE Model
    model = LatentAssemblyNeuralODE(
        num_species=num_species,
        num_latent_dims=args.latent_dims,
        num_conditions=num_conditions,
        default_dilution_factor=args.dilution_factor,
    )

    # Train Neural ODE with validation checkpointing
    trained_model, train_loss_hist, val_loss_hist = train_neural_ode(
        model, train_seqs, train_conds, val_seqs, val_conds,
        epochs=args.epochs, lr=args.lr, l2_lambda=args.l2, verbose=True
    )

    # Evaluate 8-Panel Diagnostic Figure
    evaluate_and_plot_ode(
        trained_model, held_out_key, test_seq, test_c_idx,
        train_loss_hist, val_loss_hist,
        train_keys, trajectories, env_map,
        images_dir
    )

    # Generate Continuous Within-Passage Trajectory Simulation
    plot_continuous_dynamics(
        trained_model, held_out_key, test_seq, test_c_idx,
        strains, images_dir
    )


if __name__ == '__main__':
    main()


import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import defaultdict

# Import the Linear Markov / Softmax Ablation Model
from SerialDilutionModels import LinearSoftmaxModel


def map_environments(trajectories_dict):
    """
    Decouples rumen origin from the environmental condition.
    Maps 12 trajectory keys to exactly 6 unique condition indices based purely
    on Substrate and Hydrogen Pressure.
    """
    env_map = {}
    unique_envs = []
    
    for cond_key in trajectories_dict.keys():
        # Keys are tuples from pandas groupby: (Origin, Substrate, Pressure)
        # We slice the tuple to isolate the treatment (Substrate, Pressure)
        if isinstance(cond_key, tuple) and len(cond_key) >= 2:
            env_label = cond_key[1:]
        else:
            env_label = cond_key
            
        if env_label not in unique_envs:
            unique_envs.append(env_label)
            
        env_map[cond_key] = unique_envs.index(env_label)
        
    print(f"Environment Mapping Complete:")
    print(f" - Total Trajectories: {len(trajectories_dict)}")
    print(f" - Unique Condition Indices (Environments): {len(unique_envs)}")
    for idx, env_name in enumerate(unique_envs):
        matching_keys = [k[0] if isinstance(k, tuple) else k for k, v in env_map.items() if v == idx]
        print(f"    Index {idx}: {env_name} -> Tested in origins: {matching_keys}")
        
    return env_map, unique_envs


def train_loo_model(model, train_seqs, train_conds, val_seqs, val_conds, epochs=1500, lr=0.02, l1_lambda=0.0, l2_lambda=0.25, verbose=True):
    """
    Trains the Linear Markov / Softmax ablation model on full trajectories (T1 -> T6) using Soft Scheduled Sampling.
    Tracks both training and validation loss with proximal gradient regularization.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    seq_len = train_seqs.size(1)
    
    train_loss_hist = []
    val_loss_hist = []
    
    if verbose:
        print(f"\nTraining Linear Markov Model on {train_seqs.size(0)} full trajectories (T1->T6), Validating on {val_seqs.size(0)} trajectory...")
        
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
                if hasattr(model, 'A'):
                    model.A.data *= l2_factor
                if hasattr(model, 'b'):
                    model.b.data *= l2_factor

            if l1_lambda > 0.0:
                for param in model.parameters():
                    alpha_l1 = lr * l1_lambda
                    param.data = torch.sign(param.data) * torch.clamp(torch.abs(param.data) - alpha_l1, min=0.0)
                    
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
        
        if verbose and (epoch % 300 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d}/{epochs} | Train KL: {avg_loss:.6f} | Val KL: {avg_val_loss:.6f}")
            
    return model, train_loss_hist, val_loss_hist


def evaluate_held_out_autoregressive(model, test_key, test_seq, c_idx, unique_envs, train_loss_hist, val_loss_hist, train_keys, trajectories, env_map):
    """
    Performs a purely autoregressive rollout on the held-out test condition and the training conditions.
    Plots a 2x4 grid with Loss, Train Scatter, Test Overall, and Test by Passage.
    """
    print(f"\n==================================================")
    print(f"Evaluating Held-Out Condition: '{test_key}' [Linear Markov Baseline]")
    print(f"Assigned Condition Index: {c_idx} ({unique_envs[c_idx]})")
    
    model.eval()
    
    # 1. Autoregressive Rollout on Training Set
    train_true_all, train_pred_all = [], []
    with torch.no_grad():
        for k in train_keys:
            seq = trajectories[k]
            c = torch.tensor([env_map[k]], dtype=torch.long)
            curr = seq[0].unsqueeze(0)
            for t in range(1, len(seq)):
                _, nxt = model(curr, c)
                train_true_all.append(seq[t].numpy().flatten())
                train_pred_all.append(nxt.squeeze(0).numpy().flatten())
                curr = nxt
                
    train_true_flat = np.concatenate(train_true_all) if train_true_all else np.array([])
    train_pred_flat = np.concatenate(train_pred_all) if train_pred_all else np.array([])
    
    # 2. Autoregressive Rollout on Held-Out Test Set
    test_true_dict = {}
    test_pred_dict = {}
    test_true_all, test_pred_all = [], []
    
    curr = test_seq[0].unsqueeze(0)
    c = torch.tensor([c_idx], dtype=torch.long)
    with torch.no_grad():
        for t in range(1, len(test_seq)):
            _, nxt = model(curr, c)
            true_t = test_seq[t].numpy().flatten()
            pred_t = nxt.squeeze(0).numpy().flatten()
            
            test_true_dict[t] = true_t
            test_pred_dict[t] = pred_t
            
            test_true_all.append(true_t)
            test_pred_all.append(pred_t)
            curr = nxt
            
    test_true_flat = np.concatenate(test_true_all) if test_true_all else np.array([])
    test_pred_flat = np.concatenate(test_pred_all) if test_pred_all else np.array([])
    
    # 3. Plotting (2x4 Grid)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    
    # Panel 0: Loss Curve
    axes[0].plot(train_loss_hist, color='steelblue', lw=2, label='Train Loss (KL)')
    axes[0].plot(val_loss_hist, color='darkorange', lw=2, label='Val Loss (KL)')
    axes[0].set_title("Training & Validation Loss (Linear Markov)", fontsize=16, fontweight='bold')
    axes[0].set_xlabel("Epochs", fontsize=14)
    axes[0].set_ylabel("KL Divergence", fontsize=14)
    axes[0].set_yscale('log')
    axes[0].legend(loc='upper right', fontsize=12)
    axes[0].grid(True, ls='--', alpha=0.5)
    
    # Helper function for scatter panels (evaluates only species with non-zero true abundance)
    def plot_scatter(ax, true_vals, pred_vals, title, color):
        # 1. Filter strictly for bacteria that have non-zero abundance in experimental true data
        mask = (true_vals > 0)
        if np.sum(mask) < 2:
            ax.axis('off')
            return
            
        true_active = true_vals[mask]
        pred_active = pred_vals[mask]
        
        # 2. Calculate Log R^2 directly on true non-zero values without clamping
        true_log = np.log10(true_active)
        pred_log = np.log10(pred_active)
        r, _ = pearsonr(true_log, pred_log)
        r2 = r**2 if not np.isnan(r) else 0.0
        
        # 3. Scatter only the true non-zero bacteria
        ax.scatter(true_active, pred_active, alpha=0.5, edgecolors='k', c=color, s=50, label=f'$R^2$={r2:.2f}')
        
        # Dynamic axis limits based on active data
        min_v = true_active.min() * 0.5
        max_v = min(true_active.max() * 1.5, 1.5)
        
        min_y = pred_active.min() * 0.5
        max_y = min(pred_active.max() * 1.5, 1.5)
        
        plot_min = min(min_v, min_y)
        plot_max = max(max_v, max_y)
        
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label='y=x')
        ax.set_xlim([plot_min, plot_max])
        ax.set_ylim([plot_min, plot_max])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
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
            
    fig.supxlabel('True Relative Abundance', fontsize=20, y=0.02)
    fig.supylabel('Predicted Relative Abundance', fontsize=20, x=0.01)
    
    plt.tight_layout(rect=[0.03, 0.04, 1, 1])
    
    # Save Image
    output_dir = "../Images"
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(test_key, tuple):
        clean_name = "_".join(str(item).replace(":", "_").replace(" ", "") for item in test_key)
    else:
        clean_name = str(test_key).replace(" ", "_").replace("|", "").replace(":", "")
        
    plot_filename = os.path.join(output_dir, f'loo_eval_linear_markov_{clean_name}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved held-out linear Markov evaluation plot to '{plot_filename}'")


if __name__ == "__main__":
    DATA_PATH = '../Data/German_processed_data.pt'
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data file '{DATA_PATH}' not found. Please run the parser script first.")
        
    print(f"Loading data from {DATA_PATH}...")
    data = torch.load(DATA_PATH, weights_only=False)
    strains = data['strains']
    trajectories = data['trajectories']
    num_before = len(strains)
    
    # Filter bacteria with mean relative abundance >= 1e-3 in T1 across all samples AND non-zero in all 12 T1 samples
    t1_passages = torch.stack([seq[0] for seq in trajectories.values()], dim=0)
    mean_t1_abundance = t1_passages.mean(dim=0)
    all_t1_present = (t1_passages > 0).all(dim=0)
    threshold = 1e-3
    keep_mask = (mean_t1_abundance >= threshold) & all_t1_present
    num_after = int(keep_mask.sum().item())
    
    print(f"\nSpecies T1 Abundance Filtering (Threshold >= {threshold} & Present in All 12 T1 Samples):")
    print(f" - Total bacteria BEFORE filter: {num_before}")
    print(f" - Total bacteria AFTER filter:  {num_after}")
    print(f" - Filtered out: {num_before - num_after} low-abundance / non-ubiquitous noise species")
    
    strains = [s for idx, s in enumerate(strains) if keep_mask[idx]]
    for k in trajectories.keys():
        filt_seq = trajectories[k][:, keep_mask]
        trajectories[k] = filt_seq / filt_seq.sum(dim=-1, keepdim=True)
        
    num_species = len(strains)
    
    # 1. Map 12 trajectories down to 6 unique environmental conditions
    env_map, unique_envs = map_environments(trajectories)
    num_conditions = len(unique_envs)
    
    # 2. Select Held-Out Condition (LOO Split)
    all_keys = list(trajectories.keys())
    held_out_idx = 0 
    val_idx = 1
    
    held_out_key = all_keys[held_out_idx]
    val_key = all_keys[val_idx]
    
    train_keys = [k for k in all_keys if k not in (held_out_key, val_key)]
    
    test_seq = trajectories[held_out_key]
    test_c_idx = env_map[held_out_key]
    
    print(f"\nLeave-One-Out Split Configured:")
    print(f" - Training Set: {len(train_keys)} trajectories (full T1->T6)")
    print(f" - Validation Set: [{val_key}] (full T1->T6)")
    print(f" - Held-Out Test Set: [{held_out_key}] (full T1->T6)")
    
    # Full trajectory tensors (T1 -> T6)
    train_seqs = torch.stack([trajectories[k] for k in train_keys]) # Shape: (10, 6, num_species)
    train_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)
    val_seqs = torch.stack([trajectories[val_key]]) # Shape: (1, 6, num_species)
    val_conds = torch.tensor([env_map[val_key]], dtype=torch.long)
    
    # 3. Initialize Linear Markov / Softmax Ablation Model
    # enforce_inoculum_mask=True enforces x_{T+1} = (x_T * exp(logits)) / Z
    model = LinearSoftmaxModel(num_species=num_species, num_conditions=num_conditions, enforce_inoculum_mask=True)
    
    # 4. Train on Full Trajectories (10 Train trajectories, 1 Val trajectory)
    trained_model, train_loss_hist, val_loss_hist = train_loo_model(
        model, train_seqs, train_conds, val_seqs, val_conds,
        epochs=1500, lr=0.02, l1_lambda=0.0, l2_lambda=0.25, verbose=True
    )
    
    # 5. Evaluate Purely Autoregressively on Held-Out Trajectory
    evaluate_held_out_autoregressive(
        trained_model, held_out_key, test_seq, test_c_idx, unique_envs, 
        train_loss_hist, val_loss_hist, train_keys, trajectories, env_map
    )

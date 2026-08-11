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

# Import your model
from SerialDilutionModels import LatentAssemblyModel

def map_environments(trajectories_dict):
    """
    Decouples rumen origin from the environmental condition.
    Maps 12 trajectory keys to exactly 6 unique 'a' vector indices based purely
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
            env_label = cond_key # Fallback if formatting differs
            
        if env_label not in unique_envs:
            unique_envs.append(env_label)
            
        env_map[cond_key] = unique_envs.index(env_label)
        
    print(f"Environment Mapping Complete:")
    print(f" - Total Trajectories: {len(trajectories_dict)}")
    print(f" - Unique 'a' Vectors (Environments): {len(unique_envs)}")
    for idx, env_name in enumerate(unique_envs):
        matching_keys = [k[0] if isinstance(k, tuple) else k for k, v in env_map.items() if v == idx]
        print(f"    Index {idx}: {env_name} -> Tested in origins: {matching_keys}")
        
    return env_map, unique_envs


def train_loo_model(model, train_windows, val_windows, epochs=1500, lr=0.02, l1_lambda=0.0, l2_lambda=0.4, verbose=True):
    """
    Trains the model on causal windows using Soft Scheduled Sampling.
    Tracks both training and validation loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    train_grouped = defaultdict(list)
    for seq, c_idx in train_windows:
        train_grouped[len(seq)].append((seq, c_idx))
        
    train_batches = []
    for length, items in train_grouped.items():
        seqs = torch.stack([item[0] for item in items])
        conds = torch.tensor([item[1] for item in items], dtype=torch.long)
        train_batches.append((seqs, conds))
        
    val_grouped = defaultdict(list)
    for seq, c_idx in val_windows:
        val_grouped[len(seq)].append((seq, c_idx))
        
    val_batches = []
    for length, items in val_grouped.items():
        seqs = torch.stack([item[0] for item in items])
        conds = torch.tensor([item[1] for item in items], dtype=torch.long)
        val_batches.append((seqs, conds))
        
    train_loss_hist = []
    val_loss_hist = []
    
    if verbose:
        print(f"\nTraining on {len(train_windows)} causal windows, Validating on {len(val_windows)} windows...")
        
    for epoch in range(epochs):
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.8)))

        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        total_steps = 0
        
        for seqs, conds in train_batches:
            seq_len = seqs.size(1)
            x_curr = seqs[:, 0, :] 
            
            batch_loss = 0.0
            for t in range(1, seq_len):
                log_pred, x_pred = model(x_curr, conds)
                target = seqs[:, t, :]
                
                batch_loss = batch_loss + criterion(log_pred, target)
                # Soft mixing between ground truth and model prediction
                x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred 
                total_steps += 1
                
            (batch_loss / (seq_len - 1)).backward()
            epoch_loss += batch_loss.item()
            
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
                    
        model.eval()
        val_epoch_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for seqs, conds in val_batches:
                seq_len = seqs.size(1)
                x_curr = seqs[:, 0, :]
                
                batch_loss = 0.0
                for t in range(1, seq_len):
                    log_pred, x_pred = model(x_curr, conds)
                    target = seqs[:, t, :]
                    batch_loss = batch_loss + criterion(log_pred, target)
                    x_curr = x_pred # Purely autoregressive feedback
                    val_steps += 1
                val_epoch_loss += batch_loss.item()
        
        avg_loss = epoch_loss / total_steps if total_steps > 0 else 0
        avg_val_loss = val_epoch_loss / val_steps if val_steps > 0 else 0
        
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
    print(f"Evaluating Held-Out Condition: '{test_key}'")
    print(f"Assigned 'a' Vector Index: {c_idx} ({unique_envs[c_idx]})")
    
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
    axes[0].set_title("Training & Validation Loss", fontsize=16, fontweight='bold')
    axes[0].set_xlabel("Epochs", fontsize=14)
    axes[0].set_ylabel("KL Divergence", fontsize=14)
    axes[0].set_yscale('log')
    axes[0].legend(loc='upper right', fontsize=12)
    axes[0].grid(True, ls='--', alpha=0.5)
    
    # Helper function for scatter panels
    def plot_scatter(ax, true_vals, pred_vals, title, color):
        if len(true_vals) == 0:
            ax.axis('off')
            return
        
        # Calculate Log R^2
        true_log = np.log10(np.clip(true_vals, 1e-6, None))
        pred_log = np.log10(np.clip(pred_vals, 1e-6, None))
        r, _ = pearsonr(true_log, pred_log)
        r2 = r**2
        
        ax.scatter(true_vals, pred_vals, alpha=0.5, edgecolors='k', c=color, s=50, label=f'$R^2$={r2:.2f}')
        
        # Dynamic axis limits based on data
        plottable = true_vals > 0
        min_v = true_vals[plottable].min() * 0.5 if np.any(plottable) else 1e-4
        max_v = min(true_vals.max() * 1.5, 1.5)
        
        valid_y = pred_vals > 0
        min_y = pred_vals[valid_y].min() * 0.5 if np.any(valid_y) else 1e-4
        max_y = min(pred_vals.max() * 1.5, 1.5)
        
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
    # Sequence typically has length 6 (T1...T6), so t goes from 1 to 5
    for idx in range(1, 6):
        ax_idx = 2 + idx
        if idx in test_true_dict:
            plot_scatter(axes[ax_idx], test_true_dict[idx], test_pred_dict[idx], f"Test Passage T{idx+1}", 'crimson')
        else:
            axes[ax_idx].axis('off') # Hide panel if passage doesn't exist
            
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
        
    plot_filename = os.path.join(output_dir, f'loo_eval_{clean_name}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved held-out evaluation plot to '{plot_filename}'")

if __name__ == "__main__":
    DATA_PATH = '../Data/German_processed_data.pt'
    LATENT_DIMS = 2
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data file '{DATA_PATH}' not found. Please run the parser script first.")
        
    print(f"Loading data from {DATA_PATH}...")
    # Add weights_only=False to allow loading the list of strings (strains) and tuple keys
    data = torch.load(DATA_PATH, weights_only=False)
    strains = data['strains']
    trajectories = data['trajectories']
    num_species = len(strains)
    
    # 1. Map 12 trajectories down to 6 unique environmental 'a' vectors
    env_map, unique_envs = map_environments(trajectories)
    num_conditions = len(unique_envs)
    
    # 2. Select Held-Out Condition
    # You can change `held_out_idx` (0 to 11) to test different held-out trajectories
    all_keys = list(trajectories.keys())
    held_out_idx = 0 
    val_idx = 1 # Select the next trajectory to act as the validation set
    
    held_out_key = all_keys[held_out_idx]
    val_key = all_keys[val_idx]
    
    train_keys = [k for k in all_keys if k not in (held_out_key, val_key)]
    
    test_seq = trajectories[held_out_key]
    test_c_idx = env_map[held_out_key]
    
    print(f"\nLeave-One-Out Split Configured:")
    print(f" - Training Set: {len(train_keys)} trajectories")
    print(f" - Validation Set: [{val_key}]")
    print(f" - Held-Out Test Set: [{held_out_key}]")
    
    def get_causal_windows(keys):
        windows = []
        for k in keys:
            seq = trajectories[k]
            c_idx = env_map[k]
            n = len(seq)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    windows.append((seq[i:j+1], c_idx))
        return windows
        
    train_windows = get_causal_windows(train_keys)
    val_windows = get_causal_windows([val_key])
    
    # 3. Initialize Model with 6 conditions
    model = LatentAssemblyModel(num_species=num_species, num_latent_dims=LATENT_DIMS, num_conditions=num_conditions)
    
    # 4. Train on Causal Windows (10 Train trajectories, 1 Val trajectory)
    trained_model, train_loss_hist, val_loss_hist = train_loo_model(
        model, train_windows, val_windows,
        epochs=1500, lr=0.02, l1_lambda=0.0, l2_lambda=0.25, verbose=True
    )
    
    # 5. Evaluate Purely Autoregressively on Held-Out Trajectory
    evaluate_held_out_autoregressive(
        trained_model, held_out_key, test_seq, test_c_idx, unique_envs, 
        train_loss_hist, val_loss_hist, train_keys, trajectories, env_map
    )
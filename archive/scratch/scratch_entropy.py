import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import defaultdict
from scipy.io import loadmat
from SerialDilutionModels import LatentAssemblyModel, LatentAssemblyNeuralODE

def load_venturelli_arrays(train_array, val_array, test_array, max_grouping_length=None):
    """
    Parses 3D RNN Tensors of shape (num_samples, time_steps, num_species).
    Converts absolute abundances to relative abundances and extracts full multi-passage trajectories.
    """
    print(f"Parsing Venturelli datasets...")
    
    num_species = train_array.shape[2]
    
    def process_array(data_array, is_train=True):
        trajectories = []
        for sample_idx in range(data_array.shape[0]):
            sample_seq = data_array[sample_idx]
            
            valid_timepoints = []
            valid_labels = [] 
            
            for t in range(sample_seq.shape[0]):
                if not np.isnan(sample_seq[t]).any():
                    clean_seq = np.clip(sample_seq[t], 0, None)
                    total_od = np.sum(clean_seq)
                    
                    if total_od > 1e-8:
                        # Normalize to relative abundance (across species)
                        rel_abundance = clean_seq / total_od
                        valid_timepoints.append(rel_abundance)
                        valid_labels.append(f"t={t*12}h")
            
            if len(valid_timepoints) < 2:
                continue # Need at least t0 and t1 to train/eval
                
            if is_train:
                # Keep full multi-passage trajectory for training/validation
                trajectories.append((torch.tensor(np.array(valid_timepoints), dtype=torch.float32), 0))
            else:
                # Test set keeps the full trajectory to test multi-step unrolling
                trajectories.append((torch.tensor(np.array(valid_timepoints), dtype=torch.float32), 0, f"Sample_{sample_idx}", valid_labels))
                
        return trajectories

    train_trajectories = process_array(train_array, is_train=True)
    val_trajectories = process_array(val_array, is_train=True)
    test_trajectories = process_array(test_array, is_train=False)
    
    strains = [f"Species_{i+1}" for i in range(num_species)]
    valid_conds = ["standard_media"] 
    
    print(f"Extracted {len(train_trajectories)} Train, {len(val_trajectories)} Val, and {len(test_trajectories)} Test full trajectories.")
    return train_trajectories, val_trajectories, test_trajectories, num_species, strains, valid_conds

def train_model(model, train_trajectories, eval_trajectories, epochs=1500, lr=0.02, l1_lambda=0.0, l2_lambda=0.4, verbose=True):
    """
    Trains the Neural ODE model autoregressively across multi-passage trajectories.
    Evaluates against the provided Validation Set (eval_trajectories).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    train_grouped = defaultdict(list)
    for seq, c in train_trajectories:
        train_grouped[len(seq)].append((seq, c))
        
    train_batches = []
    for length, items in train_grouped.items():
        seqs = torch.stack([item[0] for item in items])
        conds = torch.tensor([item[1] for item in items], dtype=torch.long)
        train_batches.append((seqs, conds))
        
    eval_grouped = defaultdict(list)
    for seq, c in eval_trajectories:
        eval_grouped[len(seq)].append((seq, c))
        
    eval_batches = []
    for length, items in eval_grouped.items():
        seqs = torch.stack([item[0] for item in items])
        conds = torch.tensor([item[1] for item in items], dtype=torch.long)
        eval_batches.append((seqs, conds))
        
    train_loss_hist = []
    val_loss_hist = []
    
    if verbose:
        print("\nStarting Multi-Passage Autoregressive ODE Training...")
        
    for epoch in range(epochs):
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.8)))

        model.train()
        optimizer.zero_grad()
        epoch_train_loss = 0.0
        total_train_steps = 0
        
        for seqs, conds in train_batches:
            seq_len = seqs.size(1)
            x_curr = seqs[:, 0, :] 
            
            batch_loss = 0.0
            for t in range(1, seq_len):
                # Neural ODE returns 4 values: log_pred, x_pred, N_final, R_final
                log_pred, x_pred, _, _ = model(x_curr, conds)
                target = seqs[:, t, :]
                
                batch_loss = batch_loss + criterion(log_pred, target)
                # Soft Mixing (Teacher Forcing)
                x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred 
                total_train_steps += 1
                
            (batch_loss / (seq_len - 1)).backward()
            epoch_train_loss += batch_loss.item()
            
        optimizer.step()
        
        with torch.no_grad():
            if l2_lambda > 0.0:
                l2_factor = 1.0 / (1.0 + 2.0 * lr * l2_lambda)
                # For the ODE model, Theta and W are stored in ode_func
                model.ode_func.Theta.data *= l2_factor
                model.ode_func.W.data *= l2_factor

            if l1_lambda > 0.0:
                for param in model.parameters():
                    alpha_l1 = lr * l1_lambda
                    param.data = torch.sign(param.data) * torch.clamp(torch.abs(param.data) - alpha_l1, min=0.0)
        
        # Evaluate validation loss
        model.eval()
        epoch_eval_loss = 0.0
        total_eval_steps = 0
        with torch.no_grad():
            for seqs, conds in eval_batches:
                seq_len = seqs.size(1)
                x_curr = seqs[:, 0, :]
                for t in range(1, seq_len):
                    log_pred, x_pred, _, _ = model(x_curr, conds)
                    target = seqs[:, t, :]
                    epoch_eval_loss += criterion(log_pred, target).item()
                    x_curr = x_pred
                    total_eval_steps += 1
                    
        avg_train_loss = epoch_train_loss / total_train_steps if total_train_steps > 0 else 0
        avg_eval_loss = epoch_eval_loss / total_eval_steps if total_eval_steps > 0 else 0
        
        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_eval_loss)
        
        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d}/{epochs} | Train KL (Avg/Step): {avg_train_loss:.6f} | Val KL (Avg/Step): {avg_eval_loss:.6f}")
            
    return model, train_loss_hist, val_loss_hist

def evaluate_predictions(model, train_trajectories, test_trajectories, strains, valid_conds, train_loss_hist, val_loss_hist, plot_eval=False, save_data=False, name=None):
    """
    Evaluates the model on RAW relative abundances using a 2x3 layout (6 panels).
    Filters metrics to only include bacteria that were present at t=0.
    """
    model.eval()
    
    # Include mask lists to track which bacteria were present at t=0
    train_true_all, train_pred_all, train_mask_all = [], [], []
    test_true_all, test_pred_all, test_p_list, test_mask_all = [], [], [], []
    
    successful_test_reps = 0
    total_test_reps = len(test_trajectories)
    
    with torch.no_grad():
        for seq, c_idx in train_trajectories:
            x_curr = seq[0].unsqueeze(0)
            c_curr = torch.tensor([c_idx], dtype=torch.long)
            # Create a boolean mask of species present at t=0
            t0_mask = (seq[0].numpy() > 0)
            
            for t in range(1, len(seq)):
                # Neural ODE unpacking
                _, x_pred, _, _ = model(x_curr, c_curr)
                train_pred_all.append(x_pred.squeeze(0).numpy())
                train_true_all.append(seq[t].numpy())
                train_mask_all.append(t0_mask)
                x_curr = x_pred
                
        for seq, c_idx, rep, time_labels in test_trajectories:
            x_curr = seq[0].unsqueeze(0) 
            c_curr = torch.tensor([c_idx], dtype=torch.long)
            # Create a boolean mask of species present at t=0
            t0_mask = (seq[0].numpy() > 0)
            
            rep_has_error = False
            temp_true, temp_pred, temp_p, temp_mask = [], [], [], []
            
            for t in range(1, len(seq)):
                # Neural ODE unpacking
                _, x_pred, _, _ = model(x_curr, c_curr)
                
                if torch.isnan(x_pred).any() or torch.isinf(x_pred).any():
                    rep_has_error = True
                    break
                    
                temp_true.append(seq[t].numpy())
                temp_pred.append(x_pred.squeeze(0).numpy())
                temp_p.append(time_labels[t])
                temp_mask.append(t0_mask)
                x_curr = x_pred
                
            if not rep_has_error:
                successful_test_reps += 1
                test_true_all.extend(temp_true)
                test_pred_all.extend(temp_pred)
                test_p_list.extend(temp_p)
                test_mask_all.extend(temp_mask)
                
    # ======================== METRICS CALCULATION ========================
    train_true_mat = np.array(train_true_all)
    train_pred_mat = np.array(train_pred_all)
    train_mask_mat = np.array(train_mask_all)
    
    # Apply the mask to flatten and extract only active bacteria (present at t=0)
    train_true = train_true_mat[train_mask_mat]
    train_pred = train_pred_mat[train_mask_mat]
    
    # Calculate Squared Pearson Correlation (r^2) on Linear-Scale Data
    r_train, _ = pearsonr(train_true, train_pred)
    r2_train = r_train**2
    
    test_true_mat = np.array(test_true_all)
    test_pred_mat = np.array(test_pred_all)
    test_mask_mat = np.array(test_mask_all)
    p_test_arr = np.array(test_p_list)
    
    test_metrics = {}
    timepoints_to_plot = ['t=12h', 't=24h', 't=36h', 't=48h', 't=60h']
    
    for p in timepoints_to_plot:
        time_mask = (p_test_arr == p)
        if np.any(time_mask):
            true_p_mat = test_true_mat[time_mask]
            pred_p_mat = test_pred_mat[time_mask]
            mask_p_mat = test_mask_mat[time_mask]
            
            # Apply the mask to flatten and extract only active bacteria (present at t=0)
            true_p = true_p_mat[mask_p_mat]
            pred_p = pred_p_mat[mask_p_mat]
            
            if len(true_p) > 1: # ensure at least 2 points to compute pearsonr
                r_p, _ = pearsonr(true_p, pred_p)
                r2_p = r_p**2
            else:
                r2_p = np.nan
            
            test_metrics[p] = (true_p, pred_p, r2_p)
        else:
            test_metrics[p] = None
            
    print(f"\n==================================================")
    print(f"Successfully predicted {successful_test_reps}/{total_test_reps} test sets without NaNs.")
    print(f"Train Scatter | Abundance R^2={r2_train:.3f}")
    for p in timepoints_to_plot:
        if test_metrics[p] is not None:
            _, _, r2_p = test_metrics[p]
            print(f"Test t=0 -> {p} | Model R^2={r2_p:.3f}")
    print("==================================================")
    

    if save_data:
        # ======================== SAVE DATA ========================
        save_dict = {
            'train_true': train_true,
            'train_pred': train_pred,
            'r2_train': r2_train,
            'timepoints_to_plot': timepoints_to_plot,
            # Note: test_true and test_pred are preserved as full matrices here for downstream 
            # uses, but metrics and plotting use the filtered 1D arrays inside test_metrics.
            'test_true': test_true_mat,
            'test_pred': test_pred_mat
        }
        for p in timepoints_to_plot:
            if test_metrics[p] is not None:
                save_dict[f'true_{p}'] = test_metrics[p][0]
                save_dict[f'pred_{p}'] = test_metrics[p][1]
                save_dict[f'r2_{p}'] = test_metrics[p][2]

        if name is None:
            np.savez('../Results/venturelli_ode_plot_data.npz', **save_dict)
        else:
            np.savez(f'../Results/{name}_ode_plot_data.npz', **save_dict)
    
    if plot_eval:
        # ======================== PLOTTING ========================
        # 2x3 grid to perfectly fit 1 Train + 5 Test scatters (no loss curve)
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        axes = axes.flatten()
        
        # --- PANEL 0: Train Scatter ---
        axes[0].scatter(train_true, train_pred, alpha=0.3, edgecolors='k', c='steelblue', label=f'Predicted ($R^2$={r2_train:.2f})')
        
        # Calculate limits independently for x and y
        plottable_mask = train_true > 0
        min_x = train_true[plottable_mask].min() if np.any(plottable_mask) else 1e-4
        max_x = train_true.max()
        axes[0].plot([min_x, max_x], [min_x, max_x], 'r--', lw=3, label='y=x')
        
        valid_y_mask = train_true > 0
        min_y = train_pred[valid_y_mask].min() if np.any(valid_y_mask) else 1e-4
        max_y = train_pred.max()
        
        plot_min_x = min_x * 0.5
        plot_max_x = min(max_x * 1.5, 1.5)
        plot_min_y = min_y * 0.5
        plot_max_y = min(max_y * 1.5, 1.5)
        
        # Decoupled axes limits for Train Set ONLY
        axes[0].set_xlim([plot_min_x, plot_max_x])
        axes[0].set_ylim([plot_min_y, plot_max_y])

        axes[0].set_title('Train Set (All Timepoints)', fontsize=26)
        axes[0].tick_params(axis='both', which='major', labelsize=22)
        axes[0].legend(loc='upper left', fontsize=20)
        axes[0].grid(True, which="both", ls="--", alpha=0.2)
        
        # --- PANELS 1-5: Test Scatters ---
        for idx, p in enumerate(timepoints_to_plot):
            ax = axes[idx + 1]
            if test_metrics[p] is not None:
                true_p, pred_p, r2_p = test_metrics[p]
                
                ax.scatter(true_p, pred_p, alpha=0.6, edgecolors='k', c='darkorange', label=f'Predicted ($R^2$={r2_p:.2f})')
                
                # Calculate limits independently for x and y
                plottable_mask_p = true_p > 0
                min_x_test = true_p[plottable_mask_p].min() if np.any(plottable_mask_p) else 1e-4
                max_x_test = true_p.max()
                ax.plot([min_x_test, max_x_test], [min_x_test, max_x_test], 'r--', lw=3, label='y=x')
                
                valid_y_mask_test = true_p > 0
                min_y_test = pred_p[valid_y_mask_test].min() if np.any(valid_y_mask_test) else 1e-4
                max_y_test = pred_p.max()
                
                plot_min_x_test = min_x_test * 0.5
                plot_max_x_test = min(max_x_test * 1.5, 1.5)
                plot_min_y_test = min_y_test * 0.5
                plot_max_y_test = min(max_y_test * 1.5, 1.5)
                
                # Coupled axes limits for Test Sets to maintain square aspect
                ax.set_xlim([min(plot_min_x_test, plot_min_y_test), max(plot_max_x_test, plot_max_y_test)])
                ax.set_ylim([min(plot_min_x_test, plot_min_y_test), max(plot_max_x_test, plot_max_y_test)])

                
                ax.set_title(f"Test Evaluation ({p})", fontsize=26)
                ax.tick_params(axis='both', which='major', labelsize=22)
                ax.legend(loc='upper left', fontsize=20)
                ax.grid(True, which="both", ls="--", alpha=0.2)
                
        # Shared labels for the entire figure
        fig.supxlabel('Actual Relative Abundance', fontsize=30, y=0.02)
        fig.supylabel('Predicted Relative Abundance', fontsize=30, x=0.01)
                
        plt.tight_layout(rect=[0.03, 0.04, 1, 1])
        
        # Ensure Images directory exists and save
        output_dir = "Images"
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, 'venturelli_ode_evaluation.png')
        
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved ODE evaluation plot to '{plot_filename}'")

if __name__ == "__main__":
    MAX_TRAJECTORY_LENGTH = 7  
    
    # -------------------------------------------------------------
    # LOAD EXPERIMENTAL TENSORS
    # -------------------------------------------------------------
    if os.path.exists('../Data/train_data.mat') and os.path.exists('../Data/test_data.mat'):
        train_data_red = loadmat('../Data/train_data.mat')
        X_train_full = train_data_red['train_data']

        test_data_red  = loadmat('../Data/test_data.mat')
        X_test  = test_data_red['test_data']
    else:
        # STRICT REQUIREMENT: Fails if the actual experimental data is missing
        raise FileNotFoundError("Data files 'train_data.mat' and 'test_data.mat' not found. Please place them in the working directory.")
        
    # --- 10% Validation Split ---
    np.random.seed(42) # Ensure reproducible splits
    num_train_samples = X_train_full.shape[0]
    indices = np.random.permutation(num_train_samples)
    val_size = max(1, int(num_train_samples * 0.10))
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_train_split = X_train_full[train_indices]
    X_val = X_train_full[val_indices]
    
    print(f"\nSplit complete: {X_train_split.shape[0]} Train arrays, {X_val.shape[0]} Val arrays.")
    
    # Process into causal trajectories
    train_trajectories, val_trajectories, test_trajectories, NUM_SPECIES, strains, valid_conds = load_venturelli_arrays(
        train_array=X_train_split,
        val_array=X_val,
        test_array=X_test, 
        max_grouping_length=MAX_TRAJECTORY_LENGTH
    )
    num_conds = len(valid_conds)
    
    # --- Grid Search ---
    latent_dims_grid = [2]
    # Note: Neural ODEs take significantly longer to integrate and train.
    # You might want to lower epochs if training runs too long.
    epochs_grid = [800] 
    l2_lambda_grid = [0]
    
    best_val_loss = float('inf')
    best_params = {}
    best_model = None
    best_train_hist = None
    best_val_hist = None
    
    total_iterations = len(latent_dims_grid) * len(epochs_grid) * len(l2_lambda_grid)
    current_iter = 0
    
    print(f"\nStarting Grid Search ({total_iterations} combinations)...")
    
    for k_val in latent_dims_grid:
        for epochs_val in epochs_grid:
            for l2_val in l2_lambda_grid:
                current_iter += 1
                
                # --- Update Progress Bar ---
                progress = current_iter / total_iterations
                bar_length = 40
                block = int(round(bar_length * progress))
                text = f"\rProgress: [{'=' * block + ' ' * (bar_length - block)}] {current_iter}/{total_iterations} "
                sys.stdout.write(text)
                sys.stdout.flush()
                
                # Instantiate the Neural ODE Model instead of the discrete map
                model = LatentAssemblyNeuralODE(num_species=NUM_SPECIES, num_latent_dims=k_val, num_conditions=num_conds)
                
                trained_model, train_loss_hist, val_loss_hist = train_model(
                    model, train_trajectories, val_trajectories,
                    epochs=epochs_val, lr=0.02, l1_lambda=0.0, l2_lambda=l2_val, verbose=False
                )
                
                final_val_loss = val_loss_hist[-1]
                
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_params = {'latent_dims': k_val, 'epochs': epochs_val, 'l2_lambda': l2_val}
                    best_model = copy.deepcopy(trained_model)
                    best_train_hist = train_loss_hist
                    best_val_hist = val_loss_hist
                    
    print() # New line after progress bar finishes
                
    print("\n==================================================")
    print("Grid Search Complete!")
    print(f"Best Parameters: {best_params}")
    print(f"Best Val KL Loss: {best_val_loss:.6f}")
    print("==================================================")
    
    print("\nRunning final evaluation and plotting on the best model...")
    evaluate_predictions(best_model, train_trajectories, test_trajectories, strains, valid_conds, best_train_hist, best_val_hist, save_data=True, plot_eval=True)
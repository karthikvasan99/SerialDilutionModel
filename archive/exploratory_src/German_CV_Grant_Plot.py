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

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE
from German_ODE_evaluation import load_and_filter_data, map_environments
from torchdiffeq import odeint

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
def train_neural_ode(model, train_seqs, train_conds, val_seqs, val_conds, epochs=200, lr=0.01, l2_lambda=0.05, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    train_loss_hist = []
    val_loss_hist = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        batch_loss = 0.0
        # Softmax Teacher Forcing Decay (1.0 -> 0.0)
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        # Batched Train Loop
        x_curr = train_seqs[:, 0, :]
        for t in range(1, train_seqs.size(1)):
            log_pred, x_pred, _, _ = model(x_curr, train_conds)
            target = train_seqs[:, t, :]
            batch_loss += criterion(log_pred, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred
                
        # L2 Regularization
        l2_reg = sum(p.pow(2.0).sum() for name, p in model.named_parameters() if 'Theta' in name or 'W' in name)
        loss = batch_loss + l2_lambda * l2_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        # Batched Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            x_curr_val = val_seqs[:, 0, :]
            for t in range(1, val_seqs.size(1)):
                log_pred, x_pred, _, _ = model(x_curr_val, val_conds)
                target = val_seqs[:, t, :]
                val_loss += criterion(log_pred, target).item()
                x_curr_val = x_pred
                    
        train_loss_hist.append(batch_loss.item())
        val_loss_hist.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d}/{epochs} | Train KL: {batch_loss.item():.6f} | Val KL: {val_loss:.6f}")
            
    model.load_state_dict(best_model_state)
    return model, train_loss_hist, val_loss_hist, best_val_loss

def main():
    set_seed(42)
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
    
    # 1. Load Data
    trajectories, strains = load_and_filter_data(data_path, 1e-3)
    env_map, unique_envs = map_environments(trajectories)
    num_species = len(strains)
    num_conditions = len(unique_envs)
    
    # Define Target Test Set explicitly
    test_key = ('BRRS', 'H2:CO2 + MeOH', np.float64(30.0))
    if test_key not in trajectories:
        print(f"Error: Could not find {test_key} in trajectories!")
        return
        
    test_seq = trajectories[test_key].unsqueeze(0)
    test_cond = torch.tensor([env_map[test_key]], dtype=torch.long)
    
    # Pool for CV training
    train_keys = [k for k in trajectories.keys() if k != test_key]
    train_pool_seqs = torch.stack([trajectories[k] for k in train_keys])
    train_pool_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)
    
    print(f"Total Trajectories: {len(trajectories)}")
    print(f"Test Trajectory: {test_key}")
    print(f"Training Pool Size: {len(train_keys)}")
    
    # 2. Hyperparameter Grid
    latent_dims_grid = [4]
    l2_grid = [0.05, 0.1]
    epochs_grid = [200]
    
    best_avg_val_loss = float('inf')
    best_params = {}
    
    print("\nStarting 5-Fold Cross Validation...")
    
    # K-Fold CV without sklearn
    num_train_seqs = len(train_keys)
    indices = np.random.permutation(num_train_seqs)
    folds = [indices[i::5] for i in range(5)]
    
    total_combs = len(latent_dims_grid) * len(l2_grid) * len(epochs_grid)
    comb_idx = 1
    
    for k in latent_dims_grid:
        for epochs in epochs_grid:
            for l2 in l2_grid:
                fold_val_losses = []
                print(f"\n[{comb_idx}/{total_combs}] Grid: K={k}, Epochs={epochs}, L2={l2}")
                comb_idx += 1
                
                for fold in range(5):
                    val_idx = folds[fold]
                    train_idx = np.setdiff1d(np.arange(num_train_seqs), val_idx)
                    
                    fold_train_seqs = train_pool_seqs[train_idx]
                    fold_train_conds = train_pool_conds[train_idx]
                    fold_val_seqs = train_pool_seqs[val_idx]
                    fold_val_conds = train_pool_conds[val_idx]
                    
                    model = LatentAssemblyNeuralODE(num_species, k, num_conditions)
                    
                    trained_model, _, _, fold_best_val = train_neural_ode(
                        model, fold_train_seqs, fold_train_conds, fold_val_seqs, fold_val_conds,
                        epochs=epochs, lr=0.01, l2_lambda=l2, verbose=False
                    )
                    fold_val_losses.append(fold_best_val)
                    
                avg_val_loss = np.mean(fold_val_losses)
                print(f"  -> Avg Val Loss: {avg_val_loss:.6f}")
                
                if avg_val_loss < best_avg_val_loss:
                    best_avg_val_loss = avg_val_loss
                    best_params = {'K': k, 'epochs': epochs, 'l2': l2}
                    
    print("\n========================================")
    print(f"Best Hyperparameters Found: {best_params} (Avg Val KL = {best_avg_val_loss:.6f})")
    print("========================================")
    
    # 3. Retrain on ALL 11 trajectories (no validation split here, train until end)
    print("\nRetraining final model on the entire training pool...")
    final_model = LatentAssemblyNeuralODE(num_species, best_params['K'], num_conditions)
    # We pass train_pool as BOTH train and val just to reuse the function, 
    # but we will just take the final model state after all epochs.
    final_model, train_hist, _, _ = train_neural_ode(
        final_model, train_pool_seqs, train_pool_conds, train_pool_seqs, train_pool_conds,
        epochs=best_params['epochs'], lr=0.01, l2_lambda=best_params['l2'], verbose=True
    )
    
    # 4. Save Final Model
    save_path = os.path.join(PROJECT_ROOT, 'Results', 'best_german_cv_model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(final_model.state_dict(), save_path)
    print(f"\nSaved best model to {save_path}")
    
    # 5. Evaluate on Test Set and Plot (Linear Scale)
    print("\nEvaluating on Test Set...")
    final_model.eval()
    with torch.no_grad():
        seq = trajectories[test_key]
        c_idx = torch.tensor([env_map[test_key]], dtype=torch.long)
        
        test_true_all = []
        test_pred_all = []
        
        x_curr = seq[0].unsqueeze(0)
        for t in range(1, seq.size(0)):
            _, x_pred, _, _ = final_model(x_curr, c_idx)
            test_true_all.append(seq[t].numpy())
            test_pred_all.append(x_pred.squeeze(0).numpy())
            x_curr = x_pred
            
    test_true = np.concatenate(test_true_all)
    test_pred = np.concatenate(test_pred_all)
    
    # R2 Calculation (Linear)
    # Mask out exactly zeros in true if needed, but linear can handle them. 
    # The user asked "so that we don't have to deal with handling zeros" implying linear Pearson on everything.
    r_val, _ = pearsonr(test_true, test_pred)
    r2_val = float(r_val**2) if not np.isnan(r_val) else 0.0
    
    print(f"Test Linear R^2 = {r2_val:.4f}")
    
    # Linear Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(test_true, test_pred, alpha=0.7, edgecolors='k', s=60, c='darkblue', label=f'Model Predictions ($R^2 = {r2_val:.3f}$)')
    
    # y=x line
    max_val = max(np.max(test_true), np.max(test_pred))
    plt.plot([0, max_val * 1.1], [0, max_val * 1.1], 'r--', lw=2, label='y=x (Perfect Prediction)')
    
    plt.title(f"Grant-Ready Evaluation (Linear Scale)\nTest Set: {test_key}", fontsize=15, fontweight='bold')
    plt.xlabel("True Relative Abundance", fontsize=13)
    plt.ylabel("Predicted Relative Abundance", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    plot_save_path = os.path.join(PROJECT_ROOT, 'Images', 'German_CV_Linear_Evaluation.png')
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved grant-ready linear plot to {plot_save_path}")

if __name__ == '__main__':
    main()

import os
import sys
import copy
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE
from German_ODE_evaluation import load_and_filter_data

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def map_environments_custom(trajectories_dict):
    env_map = {}
    unique_envs = []
    for cond_key in trajectories_dict.keys():
        env_label = cond_key[1:] if isinstance(cond_key, tuple) and len(cond_key) >= 2 else cond_key
        if env_label not in unique_envs:
            unique_envs.append(env_label)
        env_map[cond_key] = unique_envs.index(env_label)
    return env_map, unique_envs

def train_neural_ode(model, train_seqs, train_conds, val_seqs, val_conds, epochs=200, lr=0.01, l2_lambda=0.1, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    
    train_loss_hist = []
    val_loss_hist = []
    
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
    env_map, unique_envs = map_environments_custom(trajectories)
    num_species = len(strains)
    num_conditions = len(unique_envs)
    
    keys_list = list(trajectories.keys())
    
    # Prepare data tensors
    all_seqs = torch.stack([trajectories[k] for k in keys_list])
    all_conds = torch.tensor([env_map[k] for k in keys_list], dtype=torch.long)
    
    print(f"Total Trajectories: {len(keys_list)}")
    print(f"Unique Environments: {num_conditions}")
    
    # 2. Hyperparameter Grid
    latent_dims_grid = [2, 4, 6]
    l2_grid = [0.01, 0.05, 0.1]
    epochs_grid = [200]
    
    best_avg_val_loss = float('inf')
    best_params = {}
    
    print("\nStarting 4-Fold Cross Validation...")
    
    # K-Fold CV
    num_seqs = len(keys_list)
    indices = np.random.permutation(num_seqs)
    folds = [indices[i::4] for i in range(4)] # 4-fold since 12 trajectories
    
    total_combs = len(latent_dims_grid) * len(l2_grid) * len(epochs_grid)
    comb_idx = 1
    
    for k in latent_dims_grid:
        for epochs in epochs_grid:
            for l2 in l2_grid:
                fold_val_losses = []
                print(f"\n[{comb_idx}/{total_combs}] Grid: K={k}, Epochs={epochs}, L2={l2}")
                comb_idx += 1
                
                for fold in range(4):
                    val_idx = folds[fold]
                    train_idx = np.setdiff1d(np.arange(num_seqs), val_idx)
                    
                    fold_train_seqs = all_seqs[train_idx]
                    fold_train_conds = all_conds[train_idx]
                    fold_val_seqs = all_seqs[val_idx]
                    fold_val_conds = all_conds[val_idx]
                    
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
    
    # 3. Train ONE Final Model on ALL data using best parameters
    print("\nTraining final model on all data for PCA using best parameters...")
    final_model = LatentAssemblyNeuralODE(num_species, best_params['K'], num_conditions)
    # Using all_seqs for both train and val just to reuse the function, taking best state
    final_model, _, _, _ = train_neural_ode(
        final_model, all_seqs, all_conds, all_seqs, all_conds, 
        epochs=best_params['epochs'], lr=0.01, l2_lambda=best_params['l2'], verbose=True
    )
    
    # Extract R_0 (embedded environment conditions)
    env_vectors = final_model.a.detach().cpu().numpy()
    
    # Perform PCA
    pca = PCA(n_components=2)
    env_pca = pca.fit_transform(env_vectors)
    var_explained = pca.explained_variance_ratio_ * 100
    
    # 4. PLOTTING
    fig, ax1 = plt.subplots(figsize=(8, 7))
    
    # Define aesthetic mapping matching earlier plots
    substrate_colors = {'H2:CO2': '#e74c3c', 'H2:CO2 + MeOH': '#3498db', 'H2:CO2 + TMA': '#2ecc71'}
    pressure_markers = {5.0: 'o', 30.0: 's'}
    
    for i, env in enumerate(unique_envs):
        substrate, pressure = env
        color = substrate_colors.get(substrate, 'k')
        marker = pressure_markers.get(float(pressure), 'X')
        
        ax1.scatter(env_pca[i, 0], env_pca[i, 1], s=250, c=color, marker=marker, 
                    edgecolors='black', linewidth=1.5, zorder=3)
                    
        # Add labels offset
        ax1.text(env_pca[i, 0], env_pca[i, 1] + 0.15, f"{substrate}\n{pressure} bar", 
                 fontsize=10, ha='center', va='bottom')

    # Custom legend for PCA
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([], [], color='#e74c3c', marker='o', linestyle='None', markersize=10, label='H2:CO2 (5 bar)'),
        mlines.Line2D([], [], color='#e74c3c', marker='s', linestyle='None', markersize=10, label='H2:CO2 (30 bar)'),
        mlines.Line2D([], [], color='#3498db', marker='o', linestyle='None', markersize=10, label='H2:CO2+MeOH (5 bar)'),
        mlines.Line2D([], [], color='#3498db', marker='s', linestyle='None', markersize=10, label='H2:CO2+MeOH (30 bar)'),
        mlines.Line2D([], [], color='#2ecc71', marker='o', linestyle='None', markersize=10, label='H2:CO2+TMA (5 bar)'),
        mlines.Line2D([], [], color='#2ecc71', marker='s', linestyle='None', markersize=10, label='H2:CO2+TMA (30 bar)'),
    ]
    ax1.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.9)
    ax1.set_xlabel(f'PC1 ({var_explained[0]:.1f}% Variance)', fontsize=14)
    ax1.set_ylabel(f'PC2 ({var_explained[1]:.1f}% Variance)', fontsize=14)
    ax1.set_title(f'Learned Environmental Resources ($R_0$)', fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, 'Images', 'German_Grant_PCA_CV.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved Grant Plot to {save_path}")

if __name__ == '__main__':
    main()

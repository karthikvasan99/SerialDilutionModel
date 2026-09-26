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

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
def map_environments_custom(trajectories_dict):
    """Maps 12 trajectory condition tuples down to 6 unique environmental media indices."""
    env_map = {}
    unique_envs = []
    for cond_key in trajectories_dict.keys():
        env_label = cond_key[1:] if isinstance(cond_key, tuple) and len(cond_key) >= 2 else cond_key
        if env_label not in unique_envs:
            unique_envs.append(env_label)
        env_map[cond_key] = unique_envs.index(env_label)
    return env_map, unique_envs

def train_neural_ode(model, train_seqs, train_conds, epochs=200, lr=0.01, l2_lambda=0.1, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
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
        
    return model

def main():
    set_seed(42)
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
    
    # 1. Load Data
    trajectories, strains = load_and_filter_data(data_path, 1e-3)
    env_map, unique_envs = map_environments_custom(trajectories)
    num_species = len(strains)
    num_conditions = len(unique_envs)
    
    keys_list = list(trajectories.keys())
    
    # Best params found previously
    K = 4
    epochs = 200
    l2 = 0.1
    
    print(f"Total Trajectories: {len(keys_list)}")
    print(f"Unique Environments: {num_conditions}")
    print("\nStarting Leave-One-Out Evaluation...")
    
    loo_r2_scores = {}
    
    # 2. Leave-One-Out Loop over all 12 trajectories
    for test_idx, test_key in enumerate(keys_list):
        print(f"[{test_idx+1}/12] Held out: {test_key}")
        
        train_keys = [k for k in keys_list if k != test_key]
        train_seqs = torch.stack([trajectories[k] for k in train_keys])
        train_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)
        
        # Train model on the 11 trajectories
        model = LatentAssemblyNeuralODE(num_species, K, num_conditions)
        model = train_neural_ode(model, train_seqs, train_conds, epochs=epochs, lr=0.01, l2_lambda=l2, verbose=False)
        
        # Evaluate on the held-out trajectory
        model.eval()
        with torch.no_grad():
            seq = trajectories[test_key]
            c_idx = torch.tensor([env_map[test_key]], dtype=torch.long)
            
            x_curr = seq[0].unsqueeze(0)
            for t in range(1, seq.size(0)):
                _, x_pred, _, _ = model(x_curr, c_idx)
                x_curr = x_pred
                
            # We only care about T_6 R2 (the final timepoint)
            final_true = seq[-1].numpy()
            final_pred = x_pred.squeeze(0).numpy()
            
            r_val, _ = pearsonr(final_true, final_pred)
            r2_val = float(r_val**2) if not np.isnan(r_val) else 0.0
            
            loo_r2_scores[test_key] = r2_val
            print(f"   -> T_6 LOO R^2: {r2_val:.4f}")

    # 3. Train ONE Final Model on ALL data to extract R_0
    print("\nTraining final model on all data for PCA...")
    all_seqs = torch.stack([trajectories[k] for k in keys_list])
    all_conds = torch.tensor([env_map[k] for k in keys_list], dtype=torch.long)
    
    final_model = LatentAssemblyNeuralODE(num_species, K, num_conditions)
    final_model = train_neural_ode(final_model, all_seqs, all_conds, epochs=epochs, lr=0.01, l2_lambda=l2, verbose=False)
    
    # Extract R_0 (embedded environment conditions)
    # The latent environments are stored in the embedding weights
    env_vectors = final_model.a.detach().cpu().numpy()
    
    # Perform PCA
    pca = PCA(n_components=2)
    env_pca = pca.fit_transform(env_vectors)
    var_explained = pca.explained_variance_ratio_ * 100
    
    # 4. PLOTTING (1x2 Grid)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # PANEL A: PCA
    ax1 = axes[0]
    
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
    ax1.set_title(f'A) Learned Environmental Resources ($R_0$)', fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # PANEL B: LOO R2 Bar Chart
    ax2 = axes[1]
    
    # Group R2 by unique environment (averaging BRRS and OSU)
    env_r2 = {}
    for env in unique_envs:
        r2_vals = [loo_r2_scores[k] for k in keys_list if k[1:] == env]
        env_r2[env] = np.mean(r2_vals)
        
    x_labels = [f"{env[0]}\n{env[1]} bar" for env in unique_envs]
    y_vals = [env_r2[env] for env in unique_envs]
    
    # Colors matching PCA
    bar_colors = [substrate_colors.get(env[0], 'gray') for env in unique_envs]
    
    bars = ax2.bar(x_labels, y_vals, color=bar_colors, edgecolor='black', zorder=3)
    ax2.set_ylim([0, 1.0])
    ax2.set_ylabel('Leave-One-Out $R^2$ at $T_6$', fontsize=14)
    ax2.set_title('B) Cross-Community Transferability', fontsize=16, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    # Add exact values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, 'Images', 'German_Grant_PCA_LOO.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved Grant Plot to {save_path}")

if __name__ == '__main__':
    main()

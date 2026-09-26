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
    
    # Best Params
    K = 4
    epochs = 200
    l2 = 0.05
    
    print(f"Total Trajectories: {len(keys_list)}")
    print(f"Unique Environments: {num_conditions}")
    print(f"\nStarting Leave-One-Out Evaluation with K={K}, Epochs={epochs}, L2={l2}...")
    
    loo_r2_scores = {}
    
    # 2. Leave-One-Out Loop over all 12 trajectories
    for test_idx, test_key in enumerate(keys_list):
        print(f"[{test_idx+1}/12] Held out: {test_key}")
        
        train_keys = [k for k in keys_list if k != test_key]
        train_seqs = torch.stack([trajectories[k] for k in train_keys])
        train_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)
        
        model = LatentAssemblyNeuralODE(num_species, K, num_conditions)
        model = train_neural_ode(model, train_seqs, train_conds, epochs=epochs, lr=0.01, l2_lambda=l2, verbose=False)
        
        model.eval()
        with torch.no_grad():
            seq = trajectories[test_key]
            c_idx = torch.tensor([env_map[test_key]], dtype=torch.long)
            
            x_curr = seq[0].unsqueeze(0)
            for t in range(1, seq.size(0)):
                _, x_pred, _, _ = model(x_curr, c_idx)
                x_curr = x_pred
                
            final_true = seq[-1].numpy()
            final_pred = x_pred.squeeze(0).numpy()
            
            r_val, _ = pearsonr(final_true, final_pred)
            r2_val = float(r_val**2) if not np.isnan(r_val) else 0.0
            
            loo_r2_scores[test_key] = r2_val
            print(f"   -> T_6 LOO R^2: {r2_val:.4f}")

    # 3. PLOTTING (Just the Bar Chart)
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Define aesthetic mapping matching earlier plots
    substrate_colors = {'H2:CO2': '#e74c3c', 'H2:CO2 + MeOH': '#3498db', 'H2:CO2 + TMA': '#2ecc71'}
    
    # Group R2 by unique environment (averaging BRRS and OSU)
    env_r2 = {}
    for env in unique_envs:
        r2_vals = [loo_r2_scores[k] for k in keys_list if k[1:] == env or (k[1] == env[0] and float(k[2]) == float(env[1]))]
        env_r2[env] = np.mean(r2_vals)
        
    x_labels = [f"{env[0]}\n{env[1]} bar" for env in unique_envs]
    y_vals = [env_r2[env] for env in unique_envs]
    
    bar_colors = [substrate_colors.get(env[0], 'gray') for env in unique_envs]
    
    bars = ax.bar(x_labels, y_vals, color=bar_colors, edgecolor='black', linewidth=1.5, zorder=3)
    ax.set_ylim([0, 1.0])
    ax.set_ylabel('Leave-One-Out $R^2$ at $T_6$', fontsize=14, fontweight='bold')
    ax.set_title('Cross-Community Transferability (Zero-Shot)', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    # Add exact values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, 'Images'), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, 'Images', 'Grant_Cross_Community_Transferability.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved Grant Plot to {save_path}")

if __name__ == '__main__':
    main()

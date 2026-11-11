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

def train_neural_ode(model, train_seqs, train_conds, epochs=200, lr=0.01, l2_lambda=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        batch_loss = 0.0
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        x_curr = train_seqs[:, 0, :]
        for t in range(1, train_seqs.size(1)):
            log_pred, x_pred, _, _ = model(x_curr, train_conds)
            target = train_seqs[:, t, :]
            batch_loss += criterion(log_pred, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred
                
        l2_reg = sum(p.pow(2.0).sum() for name, p in model.named_parameters() if 'Theta' in name or 'W' in name)
        loss = batch_loss + l2_lambda * l2_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
    return model

def main():
    set_seed(42)
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
    
    trajectories, strains = load_and_filter_data(data_path, 1e-3)
    env_map, unique_envs = map_environments_custom(trajectories)
    num_species = len(strains)
    num_conditions = len(unique_envs)
    
    keys_list = list(trajectories.keys())
    
    # We only care about TMA at 5 bar
    target_env = ('H2:CO2 + TMA', np.float64(5.0))
    # Or as stored in the dictionary (often float instead of np.float64, so let's match by string representation or tuple match)
    
    test_keys = [k for k in keys_list if k[1:] == target_env or (k[1] == 'H2:CO2 + TMA' and float(k[2]) == 5.0)]
    
    if len(test_keys) == 0:
        print("Could not find the TMA 5 bar trajectories!")
        return
        
    print(f"Targeting specific trajectories for fine-tuning: {test_keys}")
    
    k_grid = [2, 3, 4, 5]
    epochs_grid = [100, 200, 300]
    l2_grid = [0.01, 0.05, 0.1]
    
    best_avg_r2 = -float('inf')
    best_params = {}
    
    total_combs = len(k_grid) * len(epochs_grid) * len(l2_grid)
    comb_idx = 1
    
    for k in k_grid:
        for epochs in epochs_grid:
            for l2 in l2_grid:
                print(f"[{comb_idx}/{total_combs}] Testing K={k}, Epochs={epochs}, L2={l2}")
                comb_idx += 1
                
                r2_vals = []
                # Leave-One-Out for the two TMA 5 bar trajectories
                for test_key in test_keys:
                    train_keys = [k_i for k_i in keys_list if k_i != test_key]
                    train_seqs = torch.stack([trajectories[k_i] for k_i in train_keys])
                    train_conds = torch.tensor([env_map[k_i] for k_i in train_keys], dtype=torch.long)
                    
                    model = LatentAssemblyNeuralODE(num_species, k, num_conditions)
                    model = train_neural_ode(model, train_seqs, train_conds, epochs=epochs, lr=0.01, l2_lambda=l2)
                    
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
                        r2 = float(r_val**2) if not np.isnan(r_val) else 0.0
                        r2_vals.append(r2)
                        
                avg_r2 = np.mean(r2_vals)
                print(f"  -> Avg TMA (5 bar) LOO R^2: {avg_r2:.4f}")
                
                if avg_r2 > best_avg_r2:
                    best_avg_r2 = avg_r2
                    best_params = {'K': k, 'epochs': epochs, 'l2': l2}
                    print(f"  *** New Best! {best_params} R^2 = {best_avg_r2:.4f} ***")

    print("\n========================================")
    print(f"Best Hyperparameters for TMA 5.0 bar: {best_params} (Avg R^2 = {best_avg_r2:.4f})")
    print("========================================")

if __name__ == '__main__':
    main()

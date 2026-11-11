import os
import sys
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
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
    
    # Target trajectories: Both TMA 5.0 and TMA 30.0
    tma_keys = [k for k in keys_list if 'TMA' in k[1]]
    print(f"Targeting specific trajectories for fine-tuning: {tma_keys}")
    
    # Grid
    k_grid = [2, 3, 4]
    lr_grid = [0.01, 0.005, 0.001]
    l2_grid = [0.01, 0.05, 0.1]
    epochs = 200
    
    results = []
    
    total_combs = len(k_grid) * len(lr_grid) * len(l2_grid)
    comb_idx = 1
    
    for k in k_grid:
        for lr in lr_grid:
            for l2 in l2_grid:
                print(f"[{comb_idx}/{total_combs}] Testing K={k}, LR={lr}, L2={l2}")
                comb_idx += 1
                
                r2_scores = {}
                
                for test_key in tma_keys:
                    train_keys = [k_i for k_i in keys_list if k_i != test_key]
                    train_seqs = torch.stack([trajectories[k_i] for k_i in train_keys])
                    train_conds = torch.tensor([env_map[k_i] for k_i in train_keys], dtype=torch.long)
                    
                    model = LatentAssemblyNeuralODE(num_species, k, num_conditions)
                    model = train_neural_ode(model, train_seqs, train_conds, epochs=epochs, lr=lr, l2_lambda=l2)
                    
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
                        r2_scores[str(test_key)] = r2
                
                res_entry = {
                    'K': k,
                    'LR': lr,
                    'L2': l2,
                    'Epochs': epochs,
                    'BRRS_TMA_5': r2_scores.get(str(('BRRS', 'H2:CO2 + TMA', np.float64(5.0))), 0.0),
                    'OSU_TMA_5': r2_scores.get(str(('OSU', 'H2:CO2 + TMA', np.float64(5.0))), 0.0),
                    'BRRS_TMA_30': r2_scores.get(str(('BRRS', 'H2:CO2 + TMA', np.float64(30.0))), 0.0),
                    'OSU_TMA_30': r2_scores.get(str(('OSU', 'H2:CO2 + TMA', np.float64(30.0))), 0.0),
                }
                res_entry['Avg_TMA_5'] = (res_entry['BRRS_TMA_5'] + res_entry['OSU_TMA_5']) / 2.0
                res_entry['Avg_TMA_30'] = (res_entry['BRRS_TMA_30'] + res_entry['OSU_TMA_30']) / 2.0
                
                print(f"   -> Avg TMA 5.0 bar: {res_entry['Avg_TMA_5']:.4f}")
                print(f"   -> Avg TMA 30.0 bar: {res_entry['Avg_TMA_30']:.4f}")
                
                results.append(res_entry)
                
    os.makedirs(os.path.join(PROJECT_ROOT, 'Results'), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(PROJECT_ROOT, 'Results', 'TMA_Finetune_Results.csv'), index=False)
    
    best_5 = df.loc[df['Avg_TMA_5'].idxmax()]
    best_30 = df.loc[df['Avg_TMA_30'].idxmax()]
    
    print("\n========================================")
    print(f"Best Params for TMA 5.0 bar: K={best_5['K']}, LR={best_5['LR']}, L2={best_5['L2']} (R2={best_5['Avg_TMA_5']:.4f})")
    print(f"Best Params for TMA 30.0 bar: K={best_30['K']}, LR={best_30['LR']}, L2={best_30['L2']} (R2={best_30['Avg_TMA_30']:.4f})")
    print("========================================")

if __name__ == '__main__':
    main()

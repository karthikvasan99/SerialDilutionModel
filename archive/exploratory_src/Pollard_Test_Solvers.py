import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['sampleType'] != 'inoculum']
    
    sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
    if sums.mean() > 50:
        df['Relative_Abundance'] = df['Read_Fraction'] / 100.0
    else:
        df['Relative_Abundance'] = df['Read_Fraction']
        
    sample_cols = ['sampleType', 'passage', 'replicate']
    num_samples = len(df.drop_duplicates(subset=sample_cols))
    mean_abundances = df.groupby('Strain_Name')['Relative_Abundance'].sum() / num_samples
    
    strains_keep = mean_abundances[mean_abundances >= 1e-3].index
    df_filtered = df[df['Strain_Name'].isin(strains_keep)].copy()
    
    pivot_df = df_filtered.pivot_table(index=['sampleType', 'replicate', 'passage'], 
                                       columns='Strain_Name', 
                                       values='Relative_Abundance', 
                                       fill_value=0.0)
    
    pivot_df = pivot_df + 1e-6
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)
    
    unique_envs = list(pivot_df.index.get_level_values('sampleType').unique())
    unique_reps = list(pivot_df.index.get_level_values('replicate').unique())
    
    num_species = pivot_df.shape[1]
    
    trajectories = []
    conditions = []
    
    for env in unique_envs:
        env_idx = unique_envs.index(env)
        for rep in unique_reps:
            if (env, rep) in pivot_df.index:
                seq = pivot_df.loc[(env, rep)].values
                if seq.shape[0] == 6:
                    trajectories.append(torch.tensor(seq, dtype=torch.float32))
                    conditions.append(env_idx)
                    
    trajectories = torch.stack(trajectories).to(device)
    conditions = torch.tensor(conditions, dtype=torch.long).to(device)
    
    return trajectories, conditions, unique_envs, num_species

def train_model(model, train_seqs, train_conds, val_seqs, val_conds, epochs, ode_method, ode_step_size, lr=0.01, l2_lambda=0.01):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        batch_loss = 0.0
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        x_curr = train_seqs[:, 0, :]
        for t in range(1, train_seqs.size(1)):
            log_pred, x_pred, _, _ = model(x_curr, train_conds, ode_method=ode_method, ode_step_size=ode_step_size)
            target = train_seqs[:, t, :]
            batch_loss += criterion(log_pred, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred
                
        l2_reg = sum(p.pow(2.0).sum() for name, p in model.named_parameters() if 'Theta' in name or 'W' in name)
        loss = batch_loss + l2_lambda * l2_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            x_curr_val = val_seqs[:, 0, :]
            for t in range(1, val_seqs.size(1)):
                log_pred, x_pred, _, _ = model(x_curr_val, val_conds, ode_method=ode_method, ode_step_size=ode_step_size)
                target = val_seqs[:, t, :]
                val_loss += criterion(log_pred, target).item()
                x_curr_val = x_pred
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
    elapsed = time.time() - start_time
    return best_val_loss, elapsed

def main():
    set_seed(42)
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv')
    
    trajectories, conditions, unique_envs, num_species = prepare_data(data_path)
    num_conditions = len(unique_envs)
    
    num_seqs = trajectories.shape[0]
    indices = np.random.permutation(num_seqs)
    
    # Just 1 fold for speed testing
    val_idx = indices[0::5]
    train_idx = np.setdiff1d(np.arange(num_seqs), val_idx)
    
    train_seqs = trajectories[train_idx]
    train_conds = conditions[train_idx]
    val_seqs = trajectories[val_idx]
    val_conds = conditions[val_idx]
    
    print(f"Testing Solvers on GPU for 100 Epochs (K=5, L2=0.01)...", flush=True)
    print(f"{'Solver':<20} | {'Step Size':<10} | {'Time (s)':<10} | {'Val KL':<10}", flush=True)
    print("-" * 60, flush=True)
    
    configs = [
        ('rk4', 0.05),     # original
        ('rk4', 0.1),      # fewer steps
        ('rk4', 0.2),      # even fewer steps
        ('euler', 0.05),   # fast but maybe unstable
        ('euler', 0.1),
        ('dopri5', None)   # adaptive
    ]
    
    for method, step_size in configs:
        set_seed(42)
        model = LatentAssemblyNeuralODE(num_species, 5, num_conditions)
        val_loss, elapsed = train_model(model, train_seqs, train_conds, val_seqs, val_conds, 
                                        epochs=100, ode_method=method, ode_step_size=step_size)
        step_str = str(step_size) if step_size else 'adaptive'
        print(f"{method:<20} | {step_str:<10} | {elapsed:<10.2f} | {val_loss:<10.4f}", flush=True)

if __name__ == '__main__':
    main()

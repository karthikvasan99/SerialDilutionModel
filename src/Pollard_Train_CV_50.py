import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyModel

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
    
    # Filter for the 50 highly abundant strains
    strains_keep = mean_abundances[mean_abundances >= 1e-3].index
    df_filtered = df[df['Strain_Name'].isin(strains_keep)].copy()
    
    pivot_df = df_filtered.pivot_table(index=['sampleType', 'replicate', 'passage'], 
                                       columns='Strain_Name', 
                                       values='Relative_Abundance', 
                                       fill_value=0.0)
    
    # Apply pseudocount of 1e-6 and re-normalize
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
            # Get passage 1 to 6
            if (env, rep) in pivot_df.index:
                seq = pivot_df.loc[(env, rep)].values
                if seq.shape[0] == 6:
                    trajectories.append(torch.tensor(seq, dtype=torch.float32))
                    conditions.append(env_idx)
                    
    trajectories = torch.stack(trajectories)
    conditions = torch.tensor(conditions, dtype=torch.long)
    
    return trajectories, conditions, unique_envs, num_species

def train_model(model, train_seqs, train_conds, val_seqs, val_conds, epochs=300, lr=0.01, l2_lambda=0.05):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        batch_loss = 0.0
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        x_curr = train_seqs[:, 0, :]
        for t in range(1, train_seqs.size(1)):
            log_pred, x_pred = model(x_curr, train_conds)
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
                log_pred, x_pred = model(x_curr_val, val_conds)
                target = val_seqs[:, t, :]
                val_loss += criterion(log_pred, target).item()
                x_curr_val = x_pred
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_state)
    return model, best_val_loss

def main():
    set_seed(42)
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv')
    
    trajectories, conditions, unique_envs, num_species = prepare_data(data_path)
    num_conditions = len(unique_envs)
    
    print(f"Data Loaded. Trajectories: {trajectories.shape[0]}, Length: {trajectories.shape[1]}, Species: {num_species}")
    
    num_seqs = trajectories.shape[0]
    indices = np.random.permutation(num_seqs)
    folds = [indices[i::5] for i in range(5)]
    
    K_grid = [2, 3, 4, 5]
    l2_grid = [0.01, 0.05, 0.1]
    epochs_grid = [300, 500]
    
    best_avg_val_loss = float('inf')
    best_params = {}
    
    print("\nStarting Hyperparameter Grid Search with 5-Fold CV...")
    total_combs = len(K_grid) * len(l2_grid) * len(epochs_grid)
    comb_idx = 1
    
    for k in K_grid:
        for epochs in epochs_grid:
            for l2 in l2_grid:
                fold_val_losses = []
                print(f"[{comb_idx}/{total_combs}] K={k}, Epochs={epochs}, L2={l2}", end=" ")
                comb_idx += 1
                
                for fold in range(5):
                    val_idx = folds[fold]
                    train_idx = np.setdiff1d(np.arange(num_seqs), val_idx)
                    
                    train_seqs = trajectories[train_idx]
                    train_conds = conditions[train_idx]
                    val_seqs = trajectories[val_idx]
                    val_conds = conditions[val_idx]
                    
                    model = LatentAssemblyModel(num_species, k, num_conditions)
                    trained_model, fold_best_val = train_model(
                        model, train_seqs, train_conds, val_seqs, val_conds,
                        epochs=epochs, lr=0.01, l2_lambda=l2
                    )
                    fold_val_losses.append(fold_best_val)
                    
                avg_val_loss = np.mean(fold_val_losses)
                print(f"-> Avg Val KL: {avg_val_loss:.6f}")
                
                if avg_val_loss < best_avg_val_loss:
                    best_avg_val_loss = avg_val_loss
                    best_params = {'K': k, 'epochs': epochs, 'l2': l2}
                    
    print("\n========================================")
    print(f"Best Hyperparameters: {best_params} (Avg Val KL = {best_avg_val_loss:.6f})")
    print("========================================")
    
    print("Training final model on all data...")
    final_model = LatentAssemblyModel(num_species, best_params['K'], num_conditions)
    final_model, _ = train_model(
        final_model, trajectories, conditions, trajectories, conditions,
        epochs=best_params['epochs'], lr=0.01, l2_lambda=best_params['l2']
    )
    
    save_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_50Strains_model.pth')
    torch.save(final_model.state_dict(), save_path)
    print(f"Saved best model to {save_path}")

if __name__ == '__main__':
    main()

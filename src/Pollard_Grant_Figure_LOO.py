import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

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
                    
    trajectories = torch.stack(trajectories)
    conditions = torch.tensor(conditions, dtype=torch.long)
    
    return trajectories, conditions, unique_envs, num_species

def train_model(model, train_seqs, train_conds, epochs=600, lr=0.01, l2_lambda=0.005):
    model = model.to(device)
    train_seqs = train_seqs.to(device)
    train_conds = train_conds.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        batch_loss = 0.0
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        x_curr = train_seqs[:, 0, :]
        for t in range(1, train_seqs.size(1)):
            log_pred, x_pred, _, _ = model(x_curr, train_conds, ode_method='euler', ode_step_size=0.1)
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
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv')
    
    trajectories, conditions, unique_envs, num_species = prepare_data(data_path)
    num_conditions = len(unique_envs)
    
    print("Computing Cross-Validation R^2 (Transferability) for Panel B...")
    
    num_seqs = trajectories.shape[0]
    indices = np.random.permutation(num_seqs)
    folds = [indices[i::5] for i in range(5)]
    
    K = 5
    l2 = 0.005
    epochs = 600
    
    r2_scores = np.zeros(num_seqs)
    
    for fold in range(5):
        print(f"Running Fold {fold+1}/5...")
        val_idx = folds[fold]
        train_idx = np.setdiff1d(np.arange(num_seqs), val_idx)
        
        train_seqs = trajectories[train_idx]
        train_conds = conditions[train_idx]
        val_seqs = trajectories[val_idx]
        val_conds = conditions[val_idx]
        
        model = LatentAssemblyNeuralODE(num_species, K, num_conditions)
        model = train_model(model, train_seqs, train_conds, epochs=epochs, lr=0.01, l2_lambda=l2)
        
        model.eval()
        with torch.no_grad():
            x_curr = val_seqs[:, 0, :].to(device)
            c_idx = val_conds.to(device)
            
            for t in range(1, val_seqs.size(1)):
                _, x_pred, _, _ = model(x_curr, c_idx, ode_method='euler', ode_step_size=0.1)
                x_curr = x_pred
                
            final_true = val_seqs[:, -1, :].numpy()
            final_pred = x_pred.cpu().numpy()
            
            for i in range(len(val_idx)):
                true_vec = final_true[i]
                pred_vec = final_pred[i]
                r_val, _ = pearsonr(true_vec, pred_vec)
                r2_val = float(r_val**2) if not np.isnan(r_val) else 0.0
                r2_scores[val_idx[i]] = r2_val

    # Group R2 by environment
    env_r2 = {}
    for i, env in enumerate(unique_envs):
        # find all trajectories with condition == i
        idx = (conditions == i).nonzero(as_tuple=True)[0].numpy()
        env_r2[env] = np.mean(r2_scores[idx])

    print("Generating Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- PANEL A: PCA of Learned Environmental Resources ---
    # Load the already trained best model
    final_model = LatentAssemblyNeuralODE(num_species, K, num_conditions)
    final_model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'Data', 'Pollard_50Strains_ODE_model_Extended_v2.pth'), map_location='cpu', weights_only=True))
    
    env_vectors = final_model.a.detach().numpy()
    pca = PCA(n_components=2)
    env_pca = pca.fit_transform(env_vectors)
    var_explained = pca.explained_variance_ratio_ * 100

    ax1 = axes[0]
    
    # Clean up labels for 2 lines
    labels_raw = unique_envs
    labels_clean = [l.replace(' supernatant', '\nsupernatant').replace(' carriers', '\ncarriers') for l in labels_raw]
    
    colors = ['#e74c3c', '#e74c3c', '#95a5a6', '#3498db', '#3498db']
    markers = ['s', 'o', 'X', 's', 'o']
    
    for i in range(5):
        ax1.scatter(env_pca[i, 0], env_pca[i, 1], s=400, c=colors[i], marker=markers[i], edgecolors='black', linewidth=1.5, zorder=3, label=labels_raw[i])
        ax1.text(env_pca[i, 0], env_pca[i, 1] + 0.15, labels_clean[i], fontsize=14, ha='center', va='bottom')

    # Increase plot limits so labels don't get cut off
    x_min, x_max = ax1.get_xlim()
    y_min, y_max = ax1.get_ylim()
    ax1.set_xlim([x_min - abs(x_min)*0.2, x_max + abs(x_max)*0.2])
    ax1.set_ylim([y_min - abs(y_min)*0.2, y_max + abs(y_max)*0.3])

    ax1.set_xlabel(f'PC1 ({var_explained[0]:.1f}% Variance)', fontsize=16)
    ax1.set_ylabel(f'PC2 ({var_explained[1]:.1f}% Variance)', fontsize=16)
    ax1.set_title('A) Learned Environmental Resources', fontsize=18, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Increase legend spacing and markers, move to lower left to avoid floating in middle
    ax1.legend(fontsize=14, loc='lower left', labelspacing=1.5, borderpad=1.0)

    # --- PANEL B: Transferability R2 ---
    ax2 = axes[1]
    x_labels = labels_clean
    y_vals = [env_r2[env] for env in unique_envs]
    
    bars = ax2.bar(x_labels, y_vals, color=colors, edgecolor='black', zorder=3)
    ax2.set_ylim([0, 1.0])
    ax2.set_ylabel('Cross-Validation $R^2$ at $T_6$', fontsize=16)
    ax2.set_title('B) Cross-Community Transferability', fontsize=18, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Rotate x labels for better fit
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, 'Images', 'Pollard_Grant_Combined_Figure_Transferability.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == '__main__':
    main()

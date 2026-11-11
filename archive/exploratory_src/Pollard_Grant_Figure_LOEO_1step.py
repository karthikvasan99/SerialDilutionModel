import os
import sys
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

def prepare_data_averaged_p0(csv_path):
    df = pd.read_csv(csv_path)
    
    sums = df[df['sampleType'] != 'inoculum'].groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
    if sums.mean() > 50:
        df['Relative_Abundance'] = df['Read_Fraction'] / 100.0
    else:
        df['Relative_Abundance'] = df['Read_Fraction']
        
    df_non_inoc = df[df['sampleType'] != 'inoculum'].copy()
    
    sample_cols = ['sampleType', 'passage', 'replicate']
    num_samples = len(df_non_inoc.drop_duplicates(subset=sample_cols))
    mean_abundances = df_non_inoc.groupby('Strain_Name')['Relative_Abundance'].sum() / num_samples
    strains_keep = mean_abundances[mean_abundances >= 1e-3].index
    
    df_filtered = df[df['Strain_Name'].isin(strains_keep)].copy()
    
    df_p0 = df_filtered[df_filtered['sampleType'] == 'inoculum']
    p0_avg = df_p0.groupby('Strain_Name')['Relative_Abundance'].mean()
    
    df_envs = df_filtered[df_filtered['sampleType'] != 'inoculum']
    env_avg = df_envs.groupby(['sampleType', 'passage', 'Strain_Name'])['Relative_Abundance'].mean().reset_index()
    
    pivot_envs = env_avg.pivot_table(index=['sampleType', 'passage'], columns='Strain_Name', values='Relative_Abundance', fill_value=0.0)
    
    unique_envs = list(pivot_envs.index.get_level_values('sampleType').unique())
    passages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    num_species = len(strains_keep)
    
    p0_vec = np.zeros(num_species)
    for i, strain in enumerate(pivot_envs.columns):
        if strain in p0_avg.index:
            p0_vec[i] = p0_avg[strain]
            
    trajectories = []
    conditions = []
    
    for env_idx, env in enumerate(unique_envs):
        seq = [p0_vec]
        for p in passages:
            if (env, p) in pivot_envs.index:
                seq.append(pivot_envs.loc[(env, p)].values)
            else:
                seq.append(np.zeros(num_species))
        
        seq_arr = np.array(seq) + 1e-6
        seq_arr = seq_arr / seq_arr.sum(axis=1, keepdims=True)
        trajectories.append(torch.tensor(seq_arr, dtype=torch.float32))
        conditions.append(env_idx)
        
    trajectories = torch.stack(trajectories).to(device)
    conditions = torch.tensor(conditions, dtype=torch.long).to(device)
    
    return trajectories, conditions, unique_envs, num_species

def train_base_model(model, train_seqs, train_conds, epochs=600, lr=0.01, l2_lambda=0.005):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        batch_loss = 0.0
        # For 1-step ahead training (Teacher Forcing = 1.0)
        # We predict each passage from the true previous passage
        for t in range(1, train_seqs.size(1)):
            x_curr_true = train_seqs[:, t-1, :]
            log_pred, _, _, _ = model(x_curr_true, train_conds, ode_method='euler', ode_step_size=0.1)
            target = train_seqs[:, t, :]
            batch_loss += criterion(log_pred, target)
                
        l2_reg = sum(p.pow(2.0).sum() for name, p in model.named_parameters() if 'Theta' in name or 'W' in name)
        loss = batch_loss + l2_lambda * l2_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
    return model

def finetune_a_only(base_model, test_seq, test_cond, num_species, K, epochs=200, lr=0.01):
    class FinetuneModel(nn.Module):
        def __init__(self, base, N, K):
            super().__init__()
            self.base = base
            self.a_new = nn.Parameter(torch.randn(1, K, device=device) * 0.1)
            
        def forward(self, N_prev, T_end=1.0):
            orig_a = self.base.a
            self.base.a = self.a_new
            cond_idx = torch.zeros(N_prev.size(0), dtype=torch.long, device=N_prev.device)
            res = self.base(N_prev, cond_idx, T_end=T_end, ode_method='euler', ode_step_size=0.1)
            self.base.a = orig_a
            return res

    for param in base_model.parameters():
        param.requires_grad = False
        
    ft_model = FinetuneModel(base_model, num_species, K).to(device)
    optimizer = optim.Adam([ft_model.a_new], lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    train_inputs = test_seq[0:2].unsqueeze(1) # P0, P1
    train_targets = test_seq[1:3].unsqueeze(1) # P1, P2
    
    for epoch in range(epochs):
        ft_model.train()
        optimizer.zero_grad()
        
        loss = 0.0
        for t in range(2):
            x_curr = train_inputs[t]
            log_pred, _, _, _ = ft_model(x_curr)
            loss += criterion(log_pred, train_targets[t])
            
        loss.backward()
        optimizer.step()
        
    for param in base_model.parameters():
        param.requires_grad = True
        
    return ft_model

def main():
    set_seed(42)
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv')
    
    trajectories, conditions, unique_envs, num_species = prepare_data_averaged_p0(data_path)
    num_conditions = len(unique_envs)
    
    print(f"Data Loaded. Averaged trajectories: {trajectories.shape[0]}, Passages: {trajectories.shape[1]}")
    
    K = 5
    l2 = 0.005
    epochs = 600
    
    r2_scores = []
    
    for test_idx in range(num_conditions):
        print(f"\n--- Leave-One-Environment-Out: {unique_envs[test_idx]} ---")
        train_idx = [i for i in range(num_conditions) if i != test_idx]
        
        train_seqs = trajectories[train_idx]
        train_conds = torch.arange(4, dtype=torch.long, device=device)
        
        test_seq = trajectories[test_idx]
        
        print("Training base model on 4 environments (1-step ahead)...")
        base_model = LatentAssemblyNeuralODE(num_species, K, 4).to(device)
        base_model = train_base_model(base_model, train_seqs, train_conds, epochs=epochs, lr=0.01, l2_lambda=l2)
        
        print("Fine-tuning 'a' on first two passages (P0->P1, P1->P2) of held-out environment...")
        ft_model = finetune_a_only(base_model, test_seq, test_idx, num_species, K, epochs=200, lr=0.01)
        
        ft_model.eval()
        with torch.no_grad():
            true_all = []
            pred_all = []
            
            # Predict P3, P4, P5, P6 purely from true data of previous passage (P2, P3, P4, P5)
            # test_seq indices: 0=P0, 1=P1, 2=P2, 3=P3, 4=P4, 5=P5, 6=P6
            for t in range(2, 6): # t=2(P2)->predict P3, t=3(P3)->predict P4, etc.
                x_curr_true = test_seq[t].unsqueeze(0)
                _, x_pred, _, _ = ft_model(x_curr_true)
                
                target = test_seq[t+1].cpu().numpy()
                pred = x_pred.squeeze(0).cpu().numpy()
                
                true_all.extend(target)
                pred_all.extend(pred)
                
            r_val, _ = pearsonr(true_all, pred_all)
            r2_val = float(r_val**2) if not np.isnan(r_val) else 0.0
            r2_scores.append(r2_val)
            print(f"-> 1-Step Ahead R^2 (across P3-P6): {r2_val:.4f}")

    print("\nTraining final model on all 5 environments for Panel A...")
    final_conds = torch.arange(5, dtype=torch.long, device=device)
    final_model = LatentAssemblyNeuralODE(num_species, K, 5).to(device)
    final_model = train_base_model(final_model, trajectories, final_conds, epochs=epochs, lr=0.01, l2_lambda=l2)
    
    print("Generating Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- PANEL A: PCA of Learned Environmental Resources ---
    env_vectors = final_model.a.detach().numpy()
    pca = PCA(n_components=2)
    env_pca = pca.fit_transform(env_vectors)
    var_explained = pca.explained_variance_ratio_ * 100

    ax1 = axes[0]
    
    labels_raw = unique_envs
    labels_clean = [l.replace(' supernatant', '\nsupernatant').replace(' carriers', '\ncarriers') for l in labels_raw]
    
    colors = ['#e74c3c', '#e74c3c', '#95a5a6', '#3498db', '#3498db']
    markers = ['s', 'o', 'X', 's', 'o']
    
    for i in range(5):
        ax1.scatter(env_pca[i, 0], env_pca[i, 1], s=400, c=colors[i], marker=markers[i], edgecolors='black', linewidth=1.5, zorder=3, label=labels_raw[i])
        ax1.text(env_pca[i, 0], env_pca[i, 1] + 0.15, labels_clean[i], fontsize=14, ha='center', va='bottom')

    x_min, x_max = ax1.get_xlim()
    y_min, y_max = ax1.get_ylim()
    ax1.set_xlim([x_min - abs(x_min)*0.2, x_max + abs(x_max)*0.2])
    ax1.set_ylim([y_min - abs(y_min)*0.2, y_max + abs(y_max)*0.3])

    ax1.set_xlabel(f'PC1 ({var_explained[0]:.1f}% Variance)', fontsize=16)
    ax1.set_ylabel(f'PC2 ({var_explained[1]:.1f}% Variance)', fontsize=16)
    ax1.set_title('A) Learned Environmental Resources', fontsize=18, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    ax1.legend(fontsize=14, loc='lower left', labelspacing=1.5, borderpad=1.0)

    # --- PANEL B: Transferability R2 ---
    ax2 = axes[1]
    x_labels = labels_clean
    y_vals = r2_scores
    
    bars = ax2.bar(x_labels, y_vals, color=colors, edgecolor='black', zorder=3)
    ax2.set_ylim([0, 1.0])
    ax2.set_ylabel('1-Step Ahead $R^2$ (Passages 3-6)', fontsize=16)
    ax2.set_title('B) Cross-Environment Transferability', fontsize=18, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, 'Images', 'Pollard_Grant_Combined_Figure_LOEO_1step.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == '__main__':
    main()

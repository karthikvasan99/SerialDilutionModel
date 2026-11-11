import os
import sys
import torch
import torch.nn as nn
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
    for env in unique_envs:
        seq = [p0_vec]
        for p in passages:
            if (env, p) in pivot_envs.index:
                seq.append(pivot_envs.loc[(env, p)].values)
            else:
                seq.append(np.zeros(num_species))
        seq_arr = np.array(seq) + 1e-6
        seq_arr = seq_arr / seq_arr.sum(axis=1, keepdims=True)
        trajectories.append(torch.tensor(seq_arr, dtype=torch.float32))
        
    trajectories = torch.stack(trajectories).to(device)
    return trajectories, unique_envs, num_species

def get_p0_to_p6_r2(model, a_vec, test_seq):
    model.eval()
    orig_a = model.a
    model.a = nn.Parameter(a_vec.detach())
    
    with torch.no_grad():
        x_curr = test_seq[0].unsqueeze(0) # P0
        cond_idx = torch.zeros(1, dtype=torch.long, device=device)
        
        # Unroll P0 -> P6 (6 passages)
        for t in range(1, test_seq.size(0)):
            _, x_pred, _, _ = model(x_curr, cond_idx, ode_method='euler', ode_step_size=0.1)
            x_curr = x_pred
            
        final_true = test_seq[-1].cpu().numpy() # P6 true
        final_pred = x_pred.squeeze(0).cpu().numpy() # P6 pred
            
    model.a = orig_a
    r_val, _ = pearsonr(final_true, final_pred)
    return float(r_val**2) if not np.isnan(r_val) else 0.0

def main():
    torch.manual_seed(42)
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv')
    trajectories, unique_envs, num_species = prepare_data_averaged_p0(data_path)
    
    K = 5
    model = LatentAssemblyNeuralODE(num_species, K, 5).to(device)
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'Data', 'Pollard_50Strains_ODE_model_P0.pth'), map_location=device, weights_only=True))
    
    ode_r2 = []
    null_r2 = []
    null_std = []
    
    print("Evaluating P0 -> P6 Predictions...")
    for env_idx in range(len(unique_envs)):
        test_seq = trajectories[env_idx]
        
        # 1. True environment vector (slice the exact embedding for this env)
        a_correct = model.a[env_idx:env_idx+1] # shape (1, K)
        
        r2_correct = get_p0_to_p6_r2(model, a_correct, test_seq)
        ode_r2.append(r2_correct)
        
        # 2. Shuffled environment vector (50 permutations for smooth std)
        shuffled_scores = []
        for _ in range(50):
            idx_perm = torch.randperm(K)
            a_shuffled = a_correct[:, idx_perm]
            r2_shuf = get_p0_to_p6_r2(model, a_shuffled, test_seq)
            shuffled_scores.append(r2_shuf)
            
        avg_null_r2 = np.mean(shuffled_scores)
        std_null_r2 = np.std(shuffled_scores)
        null_r2.append(avg_null_r2)
        null_std.append(std_null_r2)
        print(f"{unique_envs[env_idx]}: Correct={r2_correct:.4f}, Shuffled={avg_null_r2:.4f} ± {std_null_r2:.4f}")

    # Generate Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    env_vectors = model.a.detach().cpu().numpy()
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
        # Place mucin-agar carriers (index 0) below its point, all others above
        if labels_raw[i] == 'mucin-agar carriers':
            ax1.text(env_pca[i, 0], env_pca[i, 1] - 0.25, labels_clean[i], fontsize=14, ha='center', va='top')
        else:
            ax1.text(env_pca[i, 0], env_pca[i, 1] + 0.25, labels_clean[i], fontsize=14, ha='center', va='bottom')

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

    # --- PANEL B: Grouped Bars ---
    ax2 = axes[1]
    x = np.arange(len(labels_clean))
    width = 0.35
    
    bars_ode = ax2.bar(x - width/2, ode_r2, width, label='True Environment Vector', color=colors, edgecolor='black', zorder=3)
    bars_null = ax2.bar(x + width/2, null_r2, width, yerr=null_std, capsize=5, label='Shuffled Vector', color='lightgray', edgecolor='black', hatch='//', zorder=3)
    
    ax2.set_ylim([0, 1.25])
    ax2.set_ylabel('$R^2$ of Predicted Final Composition ($P_6$)', fontsize=16)
    ax2.set_title('B) Importance of Environmental Specificity', fontsize=18, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_clean, fontsize=14)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")
    ax2.legend(fontsize=14, loc='upper right')

    for bar in bars_ode:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
    for bar in bars_null:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, 'Images', 'Pollard_Grant_Combined_Figure_Final.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == '__main__':
    main()

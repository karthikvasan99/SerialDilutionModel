import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE

def main():
    K = 5
    num_species = 50
    unique_envs = ['mucin-agar carriers', 'mucin-agar supernatant', 'no carriers', 'plain-agar carriers', 'plain-agar supernatant']
    
    # Fast load model
    final_model = LatentAssemblyNeuralODE(num_species, K, 5)
    
    # We will just generate random env vectors if the model is not found, but it should be found
    # wait, I didn't save final_model in the script! I commented it out/deleted it! 
    # But wait, in the previous task `Pollard_Grant_Figure_LOEO.py`, I saved `Pollard_50Strains_ODE_model_P0.pth`.
    try:
        final_model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'Data', 'Pollard_50Strains_ODE_model_P0.pth'), map_location='cpu', weights_only=True))
    except:
        final_model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'Data', 'Pollard_50Strains_ODE_model_Extended_v2.pth'), map_location='cpu', weights_only=True))
        
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    env_vectors = final_model.a.detach().cpu().numpy()
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
    ode_r2 = [0.9142, 0.8979, 0.6962, 0.9472, 0.8857]
    null_r2 = [0.9382, 0.9205, 0.7196, 0.9536, 0.8883]
    ax2 = axes[1]
    
    x = np.arange(len(labels_clean))
    width = 0.35
    
    # We will use the colors for ODE, and a muted version for Null
    bars_ode = ax2.bar(x - width/2, ode_r2, width, label='Neural ODE', color=colors, edgecolor='black', zorder=3)
    bars_null = ax2.bar(x + width/2, null_r2, width, label='Null Model ($x_{t+1}=x_t$)', color='lightgray', edgecolor='black', hatch='//', zorder=3)
    
    ax2.set_ylim([0, 1.05])
    ax2.set_ylabel('1-Step Ahead $R^2$ (Passages 3-6)', fontsize=16)
    ax2.set_title('B) Cross-Environment Transferability', fontsize=18, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_clean, fontsize=14)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")
    
    ax2.legend(fontsize=14, loc='lower left')

    for bar in bars_ode:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
    for bar in bars_null:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, 'Images', 'Pollard_Grant_Combined_Figure_LOEO_1step.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == '__main__':
    main()

import os
import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

sys.path.append('/home/karthik/SerialDilutionModel/src/')
from SerialDilutionModels import LatentAssemblyModel

def sample_simplex_torch(n, num_samples=1, device='cpu', dtype=torch.float32):
    exp_dist = torch.distributions.Exponential(rate=torch.tensor([1.0], device=device, dtype=dtype))
    raw_samples = exp_dist.sample((num_samples, n)).squeeze(-1)
    return raw_samples / raw_samples.sum(dim=-1, keepdim=True)

def main():
    # Load model
    print("Loading Pollard model...")
    model = LatentAssemblyModel(125, 3, 3)
    model.load_state_dict(torch.load('/home/karthik/SerialDilutionModel/Data/Pollard_model.pth', map_location='cpu', weights_only=True))
    model.eval()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- PANEL A: PCA of Learned Environmental Resources ---
    print("Generating Panel A...")
    # Extract 'a' from model
    env_vectors = model.a.detach().numpy()
    
    # Do PCA
    pca = PCA(n_components=2)
    env_pca = pca.fit_transform(env_vectors)
    var_explained = pca.explained_variance_ratio_ * 100

    ax1 = axes[0]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    labels = ['Condition 0', 'Condition 1', 'Condition 2']
    
    for i in range(3):
        ax1.scatter(env_pca[i, 0], env_pca[i, 1], s=300, c=colors[i], edgecolors='black', linewidth=1.5, zorder=3, label=labels[i])
        ax1.text(env_pca[i, 0], env_pca[i, 1] + 0.05, labels[i], fontsize=12, ha='center', va='bottom')

    ax1.set_xlabel(f'PC1 ({var_explained[0]:.1f}% Variance)', fontsize=14)
    ax1.set_ylabel(f'PC2 ({var_explained[1]:.1f}% Variance)', fontsize=14)
    ax1.set_title('A) Learned Environmental Resources (Pollard)', fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax1.legend(fontsize=12)

    # --- PANEL B: Steady-State Entropies ---
    print("Generating Panel B...")
    samples = sample_simplex_torch(125, 100)
    entropy_list = []
    
    with torch.no_grad():
        for sample in samples:
            entropy_sub_list = []
            for cond in range(3):
                # find_steady_state returns: x_steady, converged, passages
                res = model.find_steady_state(sample.unsqueeze(0), torch.tensor([cond]))
                x_steady = res[0].squeeze(0).numpy()
                entropy = sp.stats.entropy(x_steady + 1e-12)
                entropy_sub_list.append(entropy)
            entropy_list.append(entropy_sub_list)
            
    entropy_list = np.array(entropy_list)
    
    ax2 = axes[1]
    for i in range(3):
        ax2.hist(entropy_list[:, i], bins=15, alpha=0.6, color=colors[i], label=labels[i], edgecolor='black')
        
    ax2.set_xlabel('Steady-State Shannon Entropy', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.set_title('B) Predicted Community Diversity by Environment', fontsize=16, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    os.makedirs('/home/karthik/SerialDilutionModel/Images', exist_ok=True)
    save_path = '/home/karthik/SerialDilutionModel/Images/Pollard_Grant_Combined_Figure.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == '__main__':
    main()

import os
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

sys.path.append('/home/karthik/SerialDilutionModel/src/')
from SerialDilutionModels import LatentAssemblyModel

def main():
    # Model parameters
    N = 124
    K = 5
    num_conditions = 5
    
    print("Loading optimal 5-condition model...")
    model = LatentAssemblyModel(N, K, num_conditions)
    model.load_state_dict(torch.load('/home/karthik/SerialDilutionModel/Data/Pollard_5Condition_model.pth', map_location='cpu', weights_only=True))
    model.eval()

    # Extract 'a' from model (shape: 5 x 5)
    env_vectors = model.a.detach().numpy()
    
    # Do PCA
    pca = PCA(n_components=2)
    env_pca = pca.fit_transform(env_vectors)
    var_explained = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(10, 8))
    
    labels = ['mucin-agar carriers', 'mucin-agar supernatant', 'no carriers', 'plain-agar carriers', 'plain-agar supernatant']
    # Use distinct markers/colors for clarity
    colors = ['#e74c3c', '#e74c3c', '#95a5a6', '#3498db', '#3498db']
    markers = ['s', 'o', 'X', 's', 'o']  # square for carriers, circle for supernatant, X for no carriers
    
    for i in range(5):
        plt.scatter(env_pca[i, 0], env_pca[i, 1], s=400, c=colors[i], marker=markers[i], edgecolors='black', linewidth=1.5, zorder=3, label=labels[i])
        plt.text(env_pca[i, 0], env_pca[i, 1] + 0.05, labels[i], fontsize=12, ha='center', va='bottom')

    plt.xlabel(f'PC1 ({var_explained[0]:.1f}% Variance)', fontsize=14)
    plt.ylabel(f'PC2 ({var_explained[1]:.1f}% Variance)', fontsize=14)
    plt.title('Learned Environmental Resources (5 Conditions)', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    plt.legend(fontsize=11, loc='best')

    plt.tight_layout()
    save_path = '/home/karthik/SerialDilutionModel/Images/Pollard_5Condition_PCA.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == '__main__':
    main()

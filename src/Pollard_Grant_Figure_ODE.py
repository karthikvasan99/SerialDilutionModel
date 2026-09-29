import os
import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

sys.path.append('/home/karthik/SerialDilutionModel/src/')
from SerialDilutionModels import LatentAssemblyNeuralODE

def sample_simplex_torch(n, num_samples=1, device='cpu', dtype=torch.float32):
    exp_dist = torch.distributions.Exponential(rate=torch.tensor([1.0], device=device, dtype=dtype))
    raw_samples = exp_dist.sample((num_samples, n)).squeeze(-1)
    return raw_samples / raw_samples.sum(dim=-1, keepdim=True)

def find_steady_state_ode(model, initial_x, condition_idx, max_passages=50, tol=1e-5):
    """
    Find steady state by repeatedly simulating serial dilution passages with the ODE model.
    """
    x_T = initial_x
    for passage in range(max_passages):
        _, x_next, _, _ = model(x_T, condition_idx, ode_method='euler', ode_step_size=0.1)
        max_diff = torch.max(torch.abs(x_next - x_T))
        if max_diff < tol:
            return x_next
        x_T = x_next
    return x_T

def main():
    print("Loading optimal 50-strain Neural ODE model...")
    num_species = 50
    K = 5
    num_conditions = 5
    
    model = LatentAssemblyNeuralODE(num_species, K, num_conditions)
    # The model weights are on CPU since we called .cpu() before saving
    model.load_state_dict(torch.load('/home/karthik/SerialDilutionModel/Data/Pollard_50Strains_ODE_model_Extended_v2.pth', map_location='cpu', weights_only=True))
    model.eval()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- PANEL A: PCA of Learned Environmental Resources ---
    print("Generating Panel A (ODE Environments)...")
    env_vectors = model.a.detach().numpy()
    
    pca = PCA(n_components=2)
    env_pca = pca.fit_transform(env_vectors)
    var_explained = pca.explained_variance_ratio_ * 100

    ax1 = axes[0]
    labels = ['mucin-agar carriers', 'mucin-agar supernatant', 'no carriers', 'plain-agar carriers', 'plain-agar supernatant']
    colors = ['#e74c3c', '#e74c3c', '#95a5a6', '#3498db', '#3498db']
    markers = ['s', 'o', 'X', 's', 'o']
    
    for i in range(5):
        ax1.scatter(env_pca[i, 0], env_pca[i, 1], s=400, c=colors[i], marker=markers[i], edgecolors='black', linewidth=1.5, zorder=3, label=labels[i])
        ax1.text(env_pca[i, 0], env_pca[i, 1] + 0.05, labels[i], fontsize=11, ha='center', va='bottom')

    ax1.set_xlabel(f'PC1 ({var_explained[0]:.1f}% Variance)', fontsize=14)
    ax1.set_ylabel(f'PC2 ({var_explained[1]:.1f}% Variance)', fontsize=14)
    ax1.set_title('A) Learned Environmental Resources (Neural ODE)', fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax1.legend(fontsize=11, loc='best')

    # --- PANEL B: Steady-State Entropies ---
    print("Generating Panel B (ODE Steady-States)...")
    # Using 100 random initial compositions per environment
    samples = sample_simplex_torch(num_species, 100)
    entropy_list = []
    
    with torch.no_grad():
        for sample in samples:
            entropy_sub_list = []
            for cond in range(num_conditions):
                cond_tensor = torch.tensor([cond])
                x_steady = find_steady_state_ode(model, sample.unsqueeze(0), cond_tensor)
                x_steady_np = x_steady.squeeze(0).numpy()
                entropy = sp.stats.entropy(x_steady_np + 1e-12)
                entropy_sub_list.append(entropy)
            entropy_list.append(entropy_sub_list)
            
    entropy_list = np.array(entropy_list)
    
    ax2 = axes[1]
    for i in range(num_conditions):
        ax2.hist(entropy_list[:, i], bins=15, alpha=0.5, color=colors[i], label=labels[i], edgecolor='black', histtype='bar')
        
    ax2.set_xlabel('Steady-State Shannon Entropy', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.set_title('B) Predicted Community Diversity (Neural ODE)', fontsize=16, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax2.legend(fontsize=11, loc='best')

    plt.tight_layout()
    os.makedirs('/home/karthik/SerialDilutionModel/Images', exist_ok=True)
    save_path = '/home/karthik/SerialDilutionModel/Images/Pollard_Grant_Combined_Figure_ODE.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == '__main__':
    main()

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.SerialDilutionModels import LatentAssemblyModel

# --- Configuration ---
N = 100
K = 4
num_models = 10
num_ic_per_model = 5
num_steps = 500

# Store entropy trajectories: shape (num_models, num_ic_per_model, num_steps)
entropy_trajectories = np.zeros((num_models, num_ic_per_model, num_steps))

for m in range(num_models):
    # 1. Initialize model
    model = LatentAssemblyModel(num_species=N, num_latent_dims=K, num_conditions=1)
    
    # 2. Sample parameters from Uniform(-1, 1)
    with torch.no_grad():
        model.Theta.uniform_(-1, 1)
        model.W.uniform_(-1, 1)
        model.a.uniform_(-1, 1)
        
    # 3. Generate random initial conditions from Dirichlet(1) -> uniform over simplex
    # PyTorch doesn't have a direct Dirichlet with scalar alpha easily, we can use Gamma(1, 1) which is Exponential(1)
    # and normalize.
    x_0 = torch.rand(num_ic_per_model, N)
    x_0 = -torch.log(x_0) # Exponential(1)
    x_0 = x_0 / x_0.sum(dim=1, keepdim=True)
    
    condition_idx = torch.zeros(num_ic_per_model, dtype=torch.long)
    
    x_T = x_0
    
    # 4. Simulate for num_steps
    for step in range(num_steps):
        # Calculate Shannon entropy
        # Add small epsilon to avoid log(0)
        x_T_np = x_T.detach().numpy()
        entropy = -np.sum(x_T_np * np.log(x_T_np + 1e-12), axis=1)
        entropy_trajectories[m, :, step] = entropy
        
        # Forward pass
        with torch.no_grad():
            _, x_T = model(x_T, condition_idx)

# --- Plotting ---
plt.figure(figsize=(10, 6))
cmap = plt.get_cmap('tab10')

for m in range(num_models):
    color = cmap(m % 10)
    for ic in range(num_ic_per_model):
        plt.plot(entropy_trajectories[m, ic, :], color=color, alpha=0.3, 
                 label=f'Model {m+1}' if ic == 0 else "")

plt.title(f'Shannon Entropy over Dilution Steps\n(Discrete LatentAssemblyModel, K={K}, N={N}, Uniform(-1,1) Params)')
plt.xlabel('Dilution Steps')
plt.ylabel('Shannon Entropy')
# Only show legend for the first trajectory of each model
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('Figures/discrete_model_entropy_diversity.png', dpi=300)
print("Plot saved to Figures/discrete_model_entropy_diversity.png")

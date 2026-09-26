import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path to import models and functions
sys.path.append('src')
from SerialDilutionModels import LatentAssemblyNeuralODE
from Ophelia_ODE_evaluation import parse_venturelli_data, split_data, train_model

def calculate_entropy(x):
    """Calculate Shannon entropy for a batch of relative abundances.
    x shape: (batch_size, N)
    """
    x_safe = torch.clamp(x, min=1e-9)
    entropy = -torch.sum(x_safe * torch.log(x_safe), dim=-1)
    return entropy.numpy()

def main():
    NUM_SPECIES = 12
    print("Loading data...")
    train_data, val_data, test_data, num_species, strains, valid_conds = split_data(*parse_venturelli_data())
    num_conds = len(valid_conds)

    # Best parameters for the ODE model from our previous run
    k_val = 2
    epochs_val = 800
    l2_val = 0.0

    print("Training Neural ODE Model...")
    model = LatentAssemblyNeuralODE(num_species=NUM_SPECIES, num_latent_dims=k_val, num_conditions=num_conds)
    
    # Train the model
    trained_model, _, _ = train_model(
        model, train_data, val_data,
        epochs=epochs_val, lr=0.02, l1_lambda=0.0, l2_lambda=l2_val, verbose=True
    )
    
    print("\nSimulating 100 random initial conditions for 500 serial dilutions...")
    trained_model.eval()
    
    num_simulations = 100
    num_passages = 500
    
    # Generate 100 random initial relative abundances
    x_curr = torch.rand(num_simulations, NUM_SPECIES)
    x_curr = x_curr / x_curr.sum(dim=1, keepdim=True)
    
    # Use standard MS media condition (index 0)
    cond_idx = torch.zeros(num_simulations, dtype=torch.long)
    
    entropy_trajectories = np.zeros((num_simulations, num_passages + 1))
    entropy_trajectories[:, 0] = calculate_entropy(x_curr)
    
    with torch.no_grad():
        for passage in range(1, num_passages + 1):
            if passage % 50 == 0:
                print(f"Passage {passage}/{num_passages}...")
            
            # Predict the next passage
            _, x_next, _, _ = trained_model(x_curr, cond_idx)
            
            entropy_trajectories[:, passage] = calculate_entropy(x_next)
            x_curr = x_next
            
    print("Simulation complete! Plotting results...")
    
    # Plotting
    mean_entropy = np.mean(entropy_trajectories, axis=0)
    std_entropy = np.std(entropy_trajectories, axis=0)
    passages = np.arange(num_passages + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot all 100 trajectories with low alpha
    for i in range(num_simulations):
        plt.plot(passages, entropy_trajectories[i], color='gray', alpha=0.1)
        
    # Plot the mean
    plt.plot(passages, mean_entropy, color='red', linewidth=2, label='Mean Entropy')
    plt.fill_between(passages, mean_entropy - std_entropy, mean_entropy + std_entropy, color='red', alpha=0.2)
    
    plt.title('Microbiome Diversity (Entropy) over 500 Serial Dilutions\nNeural ODE (Monod Kinetics)', fontsize=16)
    plt.xlabel('Serial Dilutions (Passages)', fontsize=14)
    plt.ylabel('Shannon Entropy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Save the plot
    os.makedirs('/home/karthik/.gemini/antigravity-cli/brain/d905d499-a67b-4e9a-a4f0-07d637a9912d', exist_ok=True)
    save_path = '/home/karthik/.gemini/antigravity-cli/brain/d905d499-a67b-4e9a-a4f0-07d637a9912d/entropy_retention_monod.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {save_path}")

if __name__ == '__main__':
    main()

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE
from German_ODE_evaluation import load_and_filter_data, map_environments

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def map_environments_custom(trajectories_dict):
    env_map = {}
    unique_envs = []
    for cond_key in trajectories_dict.keys():
        env_label = cond_key[1:] if isinstance(cond_key, tuple) and len(cond_key) >= 2 else cond_key
        if env_label not in unique_envs:
            unique_envs.append(env_label)
        env_map[cond_key] = unique_envs.index(env_label)
    return env_map, unique_envs

def train_neural_ode(model, train_seqs, train_conds, epochs=200, lr=0.01, l2_lambda=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        batch_loss = 0.0
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        x_curr = train_seqs[:, 0, :]
        for t in range(1, train_seqs.size(1)):
            log_pred, x_pred, _, _ = model(x_curr, train_conds)
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
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
    
    # 1. Load Data
    trajectories, strains = load_and_filter_data(data_path, 1e-3)
    env_map, unique_envs = map_environments_custom(trajectories)
    num_species = len(strains)
    num_conditions = len(unique_envs)
    
    keys_list = list(trajectories.keys())
    
    # Best Params
    K = 4
    epochs = 200
    l2 = 0.05
    
    print(f"Training final model on all data (K={K}, Epochs={epochs}, L2={l2})...")
    
    all_seqs = torch.stack([trajectories[k] for k in keys_list])
    all_conds = torch.tensor([env_map[k] for k in keys_list], dtype=torch.long)
    
    model = LatentAssemblyNeuralODE(num_species, K, num_conditions)
    model = train_neural_ode(model, all_seqs, all_conds, epochs=epochs, lr=0.01, l2_lambda=l2)
    
    # 2. Extract Theta
    Theta = model.ode_func.Theta.detach().cpu().numpy()
    
    # Shorten taxonomy names for plotting (e.g., get last two ranks)
    short_strains = []
    is_methanogen = []
    
    for s in strains:
        parts = s.split('|')
        short_name = parts[-1] if len(parts) > 0 else s
        if 'Methano' in s:
            short_strains.append(f"*{short_name}*")
            is_methanogen.append(True)
        else:
            short_strains.append(short_name)
            is_methanogen.append(False)
            
    # Create DataFrame for Seaborn
    df_theta = pd.DataFrame(Theta, index=short_strains, columns=[f"Latent Resource {i+1}" for i in range(K)])
    
    # Create Row Colors to highlight Methanogens
    methanogen_color = '#e74c3c'  # Red
    other_color = '#bdc3c7'       # Light Grey
    row_colors = [methanogen_color if m else other_color for m in is_methanogen]
    
    # 3. Plotting Clustermap
    sns.set_theme(style="white", font="sans-serif")
    
    g = sns.clustermap(df_theta, 
                       cmap="vlag", 
                       center=0,
                       col_cluster=False, # Don't cluster the 2 resources, just species
                       row_colors=row_colors,
                       figsize=(8, 10),
                       dendrogram_ratio=(0.2, 0.0),
                       cbar_pos=(0.02, 0.8, 0.05, 0.18),
                       linewidths=0.5,
                       tree_kws={'linewidth': 1.5})
                       
    # Formatting the Clustermap
    g.ax_heatmap.set_title(r"Learned Species-Resource Traits ($\Theta$)", fontsize=16, fontweight='bold', pad=20)
    g.ax_heatmap.set_ylabel("Microbial Species (Row Clustered)", fontsize=14, fontweight='bold')
    g.ax_heatmap.set_xlabel("Learned Latent Resources", fontsize=14, fontweight='bold')
    
    # Format y-tick labels (italicize methanogens, bold text)
    for tick_label in g.ax_heatmap.axes.get_yticklabels():
        tick_text = tick_label.get_text()
        if tick_text.startswith('*') and tick_text.endswith('*'):
            tick_label.set_text(tick_text.strip('*'))
            tick_label.set_color('#c0392b')
            tick_label.set_fontweight('bold')
        else:
            tick_label.set_color('#34495e')
            tick_label.set_fontsize(10)
            
    # Add a custom legend for the row colors
    import matplotlib.patches as mpatches
    meth_patch = mpatches.Patch(color=methanogen_color, label='Methanogen')
    other_patch = mpatches.Patch(color=other_color, label='Other Bacteria')
    g.fig.legend(handles=[meth_patch, other_patch], loc='lower right', bbox_to_anchor=(0.95, 0.05), fontsize=12, frameon=True)
    
    os.makedirs(os.path.join(PROJECT_ROOT, 'Images'), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, 'Images', 'Grant_Theta_Clustermap.png')
    g.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved Clustermap to {save_path}")

if __name__ == '__main__':
    main()

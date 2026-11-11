import os
import sys
import copy
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE
from German_ODE_evaluation import load_and_filter_data

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
def map_environments_custom(trajectories_dict):
    """Maps 12 trajectory condition tuples down to 6 unique environmental media indices."""
    env_map = {}
    unique_envs = []
    for cond_key in trajectories_dict.keys():
        env_label = cond_key[1:] if isinstance(cond_key, tuple) and len(cond_key) >= 2 else cond_key
        if env_label not in unique_envs:
            unique_envs.append(env_label)
        env_map[cond_key] = unique_envs.index(env_label)
    return env_map, unique_envs

def train_neural_ode(model, train_seqs, train_conds, val_seqs=None, val_conds=None, epochs=200, lr=0.01, l2_lambda=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        batch_loss = 0.0
        # Softmax Teacher Forcing Decay (1.0 -> 0.0)
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        # Batched Train Loop
        x_curr = train_seqs[:, 0, :]
        for t in range(1, train_seqs.size(1)):
            log_pred, x_pred, _, _ = model(x_curr, train_conds)
            target = train_seqs[:, t, :]
            batch_loss += criterion(log_pred, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred
                
        # L2 Regularization
        l2_reg = sum(p.pow(2.0).sum() for name, p in model.named_parameters() if 'Theta' in name or 'W' in name)
        loss = batch_loss + l2_lambda * l2_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if val_seqs is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                x_curr_val = val_seqs[:, 0, :]
                for t in range(1, val_seqs.size(1)):
                    log_pred, x_pred, _, _ = model(x_curr_val, val_conds)
                    target = val_seqs[:, t, :]
                    val_loss += criterion(log_pred, target).item()
                    x_curr_val = x_pred
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                
    if val_seqs is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        return model, best_val_loss
    
    return model, None

def main():
    set_seed(42)
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
    
    trajectories, strains = load_and_filter_data(data_path, 1e-3)
    env_map, unique_envs = map_environments_custom(trajectories)
    num_species = len(strains)
    num_conditions = len(unique_envs)
    
    keys_list = list(trajectories.keys())
    
    K = 4
    epochs_grid = [200, 500, 800]
    lr_grid = [0.01, 0.02]
    l2_grid = [0.0, 0.05, 0.1, 0.5]
    
    total_combs = len(epochs_grid) * len(lr_grid) * len(l2_grid)
    
    loo_r2_scores = {}
    best_params_per_env = {}
    
    # Precompute host mappings
    hosts = {}
    for k in keys_list:
        host = k[0]
        if host not in hosts:
            hosts[host] = []
        hosts[host].append(k)
        
    print(f"Starting Cross-Host Validation on {len(keys_list)} test cases...")
    print(f"Grid size: {total_combs} combinations per test case.\n")
    
    for test_idx, test_key in enumerate(keys_list):
        test_host = test_key[0]
        train_host = [h for h in hosts.keys() if h != test_host][0]
        
        # Test = 1 trajectory
        # Val = All other trajectories from the same host
        val_keys = [k for k in hosts[test_host] if k != test_key]
        
        # Train = All trajectories from the OTHER host
        train_keys = hosts[train_host]
        
        val_seqs = torch.stack([trajectories[k] for k in val_keys])
        val_conds = torch.tensor([env_map[k] for k in val_keys], dtype=torch.long)
        
        train_seqs = torch.stack([trajectories[k] for k in train_keys])
        train_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)
        
        print(f"[{test_idx+1}/12] Target: {test_key} | Tuning on {len(train_keys)} train, {len(val_keys)} val...")
        
        best_val_loss = float('inf')
        best_params = {}
        
        for ep in epochs_grid:
            for lr in lr_grid:
                for l2 in l2_grid:
                    # Reset seed before each model initialization for fairness
                    set_seed(42 + ep + int(l2*100) + int(lr*1000))
                    model = LatentAssemblyNeuralODE(num_species, K, num_conditions)
                    trained_model, v_loss = train_neural_ode(model, train_seqs, train_conds, val_seqs, val_conds, epochs=ep, lr=lr, l2_lambda=l2)
                    
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        best_params = {'epochs': ep, 'lr': lr, 'l2': l2}
                        
        print(f"   -> Best Params: {best_params} (Val KL = {best_val_loss:.4f})")
        best_params_per_env[test_key] = best_params
        
        # Now retrain final model on Train + Val (all 11 trajectories)
        all_train_keys = train_keys + val_keys
        all_train_seqs = torch.stack([trajectories[k] for k in all_train_keys])
        all_train_conds = torch.tensor([env_map[k] for k in all_train_keys], dtype=torch.long)
        
        print(f"   -> Retraining on 11 trajectories...")
        set_seed(42)
        final_model = LatentAssemblyNeuralODE(num_species, K, num_conditions)
        final_model, _ = train_neural_ode(final_model, all_train_seqs, all_train_conds, 
                                          epochs=best_params['epochs'], 
                                          lr=best_params['lr'], 
                                          l2_lambda=best_params['l2'])
                                          
        # Evaluate on Test
        final_model.eval()
        with torch.no_grad():
            seq = trajectories[test_key]
            c_idx = torch.tensor([env_map[test_key]], dtype=torch.long)
            x_curr = seq[0].unsqueeze(0)
            for t in range(1, seq.size(0)):
                _, x_pred, _, _ = final_model(x_curr, c_idx)
                x_curr = x_pred
            
            final_true = seq[-1].numpy()
            final_pred = x_pred.squeeze(0).numpy()
            r_val, _ = pearsonr(final_true, final_pred)
            r2_val = float(r_val**2) if not np.isnan(r_val) else 0.0
            loo_r2_scores[test_key] = r2_val
            print(f"   => Test T_6 R^2: {r2_val:.4f}\n")

    # Generate the PCA + Bar plot
    print("Training one final global model for PCA...")
    all_seqs = torch.stack([trajectories[k] for k in keys_list])
    all_conds = torch.tensor([env_map[k] for k in keys_list], dtype=torch.long)
    # Just use the average of the best parameters found
    avg_epochs = int(np.mean([p['epochs'] for p in best_params_per_env.values()]))
    avg_lr = np.mean([p['lr'] for p in best_params_per_env.values()])
    avg_l2 = np.mean([p['l2'] for p in best_params_per_env.values()])
    
    set_seed(42)
    global_model = LatentAssemblyNeuralODE(num_species, K, num_conditions)
    global_model, _ = train_neural_ode(global_model, all_seqs, all_conds, epochs=avg_epochs, lr=avg_lr, l2_lambda=avg_l2)
    
    env_vectors = global_model.a.detach().cpu().numpy()
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    env_pca = pca.fit_transform(env_vectors)
    var_explained = pca.explained_variance_ratio_ * 100
    
    # PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1 = axes[0]
    substrate_colors = {'H2:CO2': '#e74c3c', 'H2:CO2 + MeOH': '#3498db', 'H2:CO2 + TMA': '#2ecc71'}
    pressure_markers = {5.0: 'o', 30.0: 's'}
    
    for i, env in enumerate(unique_envs):
        substrate, pressure = env
        color = substrate_colors.get(substrate, 'k')
        marker = pressure_markers.get(float(pressure), 'X')
        ax1.scatter(env_pca[i, 0], env_pca[i, 1], s=250, c=color, marker=marker, edgecolors='black', linewidth=1.5, zorder=3)
        ax1.text(env_pca[i, 0], env_pca[i, 1] + 0.15, f"{substrate}\n{pressure} bar", fontsize=10, ha='center', va='bottom')

    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([], [], color='#e74c3c', marker='o', linestyle='None', markersize=10, label='H2:CO2 (5 bar)'),
        mlines.Line2D([], [], color='#e74c3c', marker='s', linestyle='None', markersize=10, label='H2:CO2 (30 bar)'),
        mlines.Line2D([], [], color='#3498db', marker='o', linestyle='None', markersize=10, label='H2:CO2+MeOH (5 bar)'),
        mlines.Line2D([], [], color='#3498db', marker='s', linestyle='None', markersize=10, label='H2:CO2+MeOH (30 bar)'),
        mlines.Line2D([], [], color='#2ecc71', marker='o', linestyle='None', markersize=10, label='H2:CO2+TMA (5 bar)'),
        mlines.Line2D([], [], color='#2ecc71', marker='s', linestyle='None', markersize=10, label='H2:CO2+TMA (30 bar)'),
    ]
    ax1.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.9)
    ax1.set_xlabel(f'PC1 ({var_explained[0]:.1f}% Variance)', fontsize=14)
    ax1.set_ylabel(f'PC2 ({var_explained[1]:.1f}% Variance)', fontsize=14)
    ax1.set_title(f'A) Learned Environmental Resources ($R_0$)', fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    ax2 = axes[1]
    env_r2 = {}
    for env in unique_envs:
        r2_vals = [loo_r2_scores[k] for k in keys_list if k[1:] == env]
        env_r2[env] = np.mean(r2_vals)
        
    x_labels = [f"{env[0]}\n{env[1]} bar" for env in unique_envs]
    y_vals = [env_r2[env] for env in unique_envs]
    bar_colors = [substrate_colors.get(env[0], 'gray') for env in unique_envs]
    
    bars = ax2.bar(x_labels, y_vals, color=bar_colors, edgecolor='black', zorder=3)
    ax2.set_ylim([0, 1.0])
    ax2.set_ylabel('Leave-One-Out $R^2$ at $T_6$', fontsize=14)
    ax2.set_title('B) Cross-Community Transferability (Tuned)', fontsize=16, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, 'Images', 'German_Grant_PCA_LOO_Tuned.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved tuned Grant Plot to {save_path}")

if __name__ == '__main__':
    main()

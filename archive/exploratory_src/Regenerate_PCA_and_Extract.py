import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from SerialDilutionModels import LatentAssemblyNeuralODE
from German_ODE_evaluation import load_and_filter_data

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
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

set_seed(123)
data_path = 'Data/German_processed_data.pt'
trajectories, strains = load_and_filter_data(data_path, 1e-3)
keys_list = list(trajectories.keys())
env_map, unique_envs = map_environments_custom(trajectories)
num_species = len(strains)
num_conditions = len(unique_envs)

all_seqs = torch.stack([trajectories[k] for k in keys_list])
all_conds = torch.tensor([env_map[k] for k in keys_list], dtype=torch.long)

K, epochs, l2 = 4, 200, 0.05
final_model = LatentAssemblyNeuralODE(num_species, K, num_conditions)
final_model = train_neural_ode(final_model, all_seqs, all_conds, epochs=epochs, l2_lambda=l2)

env_vectors = final_model.a.detach().cpu().numpy()
pca = PCA(n_components=2)
env_pca = pca.fit_transform(env_vectors)
var_explained = pca.explained_variance_ratio_ * 100

fig, ax1 = plt.subplots(figsize=(8, 7))
substrate_colors = {'H2:CO2': '#e74c3c', 'H2:CO2 + MeOH': '#3498db', 'H2:CO2 + TMA': '#2ecc71'}
pressure_markers = {5.0: 'o', 30.0: '^'}

for i, env in enumerate(unique_envs):
    substrate, pressure = env
    color = substrate_colors.get(substrate, 'k')
    marker = pressure_markers.get(float(pressure), 'X')
    ax1.scatter(env_pca[i, 0], env_pca[i, 1], s=250, c=color, marker=marker, edgecolors='black', linewidth=1.5, zorder=3)
    ax1.text(env_pca[i, 0], env_pca[i, 1] + 0.15, f"{substrate}\n{pressure} bar", fontsize=10, ha='center', va='bottom')

import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([0], [0], marker='o', color='w', label='H2:CO2', markerfacecolor='#e74c3c', markersize=10, markeredgecolor='black'),
    mlines.Line2D([0], [0], marker='o', color='w', label='H2:CO2 + MeOH', markerfacecolor='#3498db', markersize=10, markeredgecolor='black'),
    mlines.Line2D([0], [0], marker='o', color='w', label='H2:CO2 + TMA', markerfacecolor='#2ecc71', markersize=10, markeredgecolor='black'),
    mlines.Line2D([0], [0], marker='o', color='w', label='5.0 bar', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
    mlines.Line2D([0], [0], marker='^', color='w', label='30.0 bar', markerfacecolor='gray', markersize=10, markeredgecolor='black')
]
ax1.legend(handles=legend_elements, loc='best', fontsize=12, frameon=True)
ax1.set_xlabel(f"Principal Component 1 ({var_explained[0]:.1f}%)", fontsize=14, fontweight='bold')
ax1.set_ylabel(f"Principal Component 2 ({var_explained[1]:.1f}%)", fontsize=14, fontweight='bold')
ax1.set_title("PCA of Learned Environmental Resources ($R_0$)", fontsize=16, fontweight='bold', pad=15)
ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)

plt.tight_layout()
save_path = 'Images/German_Grant_PCA_CV.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print("\n=== PCA COORDS ===")
print(f"Variance: PC1 {var_explained[0]:.1f}%, PC2 {var_explained[1]:.1f}%")
for i, env in enumerate(unique_envs):
    print(f"{env}: PC1={env_pca[i, 0]:.2f}, PC2={env_pca[i, 1]:.2f}")

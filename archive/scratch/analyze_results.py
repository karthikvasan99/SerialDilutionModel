import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
import torch
import numpy as np
from sklearn.decomposition import PCA
from SerialDilutionModels import LatentAssemblyNeuralODE
from German_ODE_evaluation import load_and_filter_data

def map_environments_custom(trajectories_dict):
    env_map = {}
    unique_envs = []
    for cond_key in trajectories_dict.keys():
        env_label = cond_key[1:] if isinstance(cond_key, tuple) and len(cond_key) >= 2 else cond_key
        if env_label not in unique_envs:
            unique_envs.append(env_label)
        env_map[cond_key] = unique_envs.index(env_label)
    return env_map, unique_envs

data_path = 'Data/German_processed_data.pt'
trajectories, strains = load_and_filter_data(data_path, 1e-3)
keys = list(trajectories.keys())
env_map, unique_envs = map_environments_custom(trajectories)
num_species = len(strains)
num_conds = len(unique_envs)

model = LatentAssemblyNeuralODE(num_species, 4, num_conds)
train_seqs = torch.stack([trajectories[k] for k in keys])
train_conds = torch.tensor([env_map[k] for k in keys], dtype=torch.long)

opt = torch.optim.Adam(model.parameters(), lr=0.01)
crit = torch.nn.KLDivLoss(reduction='batchmean')
for ep in range(150):
    model.train()
    opt.zero_grad()
    loss = 0
    x_c = train_seqs[:, 0, :]
    for t in range(1, train_seqs.size(1)):
        l, x_p, _, _ = model(x_c, train_conds)
        loss += crit(l, train_seqs[:, t, :])
        x_c = x_p
    reg = sum(p.pow(2).sum() for n, p in model.named_parameters() if 'Theta' in n or 'W' in n)
    (loss + 0.05 * reg).backward()
    opt.step()

a_val = model.a.detach().numpy()
pca = PCA(n_components=2)
a_pca = pca.fit_transform(a_val)
print("\n=== PCA RESULTS ===")
print("PCA Explained Variance:", pca.explained_variance_ratio_)
for i, env in enumerate(unique_envs):
    print(f"Env {env}: PCA = {a_pca[i]}")

Theta = model.ode_func.Theta.detach().numpy()
meth_idx = [i for i, s in enumerate(strains) if 'Methano' in s]
print("\n=== THETA RESULTS ===")
print(f"Number of Methanogens: {len(meth_idx)}")
if len(meth_idx) > 1:
    meth_theta = Theta[meth_idx]
    mean_meth = meth_theta.mean(axis=0)
    print("Mean Methanogen Theta Profile (K=4):", mean_meth)
    other_theta = np.delete(Theta, meth_idx, axis=0)
    mean_other = other_theta.mean(axis=0)
    print("Mean Non-Methanogen Theta Profile (K=4):", mean_other)


import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'scratch' else SCRIPT_DIR
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from SerialDilutionModels import LatentAssemblyNeuralODE
from German_ODE_evaluation import load_and_filter_data

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)
data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
trajectories, strains = load_and_filter_data(data_path, 1e-3)

def map_environments_custom(trajectories_dict):
    env_map = {}
    unique_envs = []
    for cond_key in trajectories_dict.keys():
        env_label = cond_key[1:] if isinstance(cond_key, tuple) and len(cond_key) >= 2 else cond_key
        if env_label not in unique_envs:
            unique_envs.append(env_label)
        env_map[cond_key] = unique_envs.index(env_label)
    return env_map, unique_envs

env_map, unique_envs = map_environments_custom(trajectories)
keys_list = list(trajectories.keys())
num_species = len(strains)
num_conditions = len(unique_envs)

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

print("Training Cross-Pressure Model (Trained on all OSU + BRRS 30b; BRRS 5b completely held out)...")
train_keys = [k for k in keys_list if k[0] == "OSU" or (k[0] == "BRRS" and k[2] == 30.0)]
train_seqs = torch.stack([trajectories[k] for k in train_keys])
train_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)

cp_model = LatentAssemblyNeuralODE(num_species, 4, num_conditions)
cp_model = train_neural_ode(cp_model, train_seqs, train_conds, epochs=200)
cp_model.eval()

# Rollouts for BRRS +MeOH (5 bar), BRRS +TMA (5 bar), and OSU +MeOH (5 bar)
with torch.no_grad():
    # BRRS +MeOH 5b (Held out)
    k_meoh = ("BRRS", "H2:CO2 + MeOH", np.float64(5.0))
    seq_meoh = trajectories[k_meoh]
    c_meoh = torch.tensor([env_map[k_meoh]], dtype=torch.long)
    preds_meoh = [seq_meoh[0].numpy()]
    x_curr = seq_meoh[0].unsqueeze(0)
    for t in range(1, seq_meoh.size(0)):
        _, x_pred, _, _ = cp_model(x_curr, c_meoh)
        preds_meoh.append(x_pred.squeeze(0).numpy())
        x_curr = x_pred
    preds_meoh = np.array(preds_meoh) * 100
    trues_meoh = seq_meoh.numpy() * 100

    # BRRS +TMA 5b (Held out)
    k_tma = ("BRRS", "H2:CO2 + TMA", np.float64(5.0))
    seq_tma = trajectories[k_tma]
    c_tma = torch.tensor([env_map[k_tma]], dtype=torch.long)
    preds_tma = [seq_tma[0].numpy()]
    x_curr = seq_tma[0].unsqueeze(0)
    for t in range(1, seq_tma.size(0)):
        _, x_pred, _, _ = cp_model(x_curr, c_tma)
        preds_tma.append(x_pred.squeeze(0).numpy())
        x_curr = x_pred
    preds_tma = np.array(preds_tma) * 100
    trues_tma = seq_tma.numpy() * 100

    # OSU +MeOH 5b (Training fit)
    k_osu_meoh = ("OSU", "H2:CO2 + MeOH", np.float64(5.0))
    seq_osu_meoh = trajectories[k_osu_meoh]
    c_osu_meoh = torch.tensor([env_map[k_osu_meoh]], dtype=torch.long)
    preds_osu_meoh = [seq_osu_meoh[0].numpy()]
    x_curr = seq_osu_meoh[0].unsqueeze(0)
    for t in range(1, seq_osu_meoh.size(0)):
        _, x_pred, _, _ = cp_model(x_curr, c_osu_meoh)
        preds_osu_meoh.append(x_pred.squeeze(0).numpy())
        x_curr = x_pred
    preds_osu_meoh = np.array(preds_osu_meoh) * 100
    trues_osu_meoh = seq_osu_meoh.numpy() * 100

target_idx = 0
true_osu_meoh_t6 = trues_osu_meoh[-1, target_idx]
pred_osu_meoh_t6 = preds_osu_meoh[-1, target_idx]

out_cache = os.path.join(PROJECT_ROOT, 'Data', 'Poster_Cache_German.npz')
np.savez(out_cache,
    trues_meoh = trues_meoh,                 # (6, 87)
    preds_meoh = preds_meoh,                 # (6, 87)
    trues_tma  = trues_tma,                  # (6, 87)
    preds_tma  = preds_tma,                  # (6, 87)
    trues_osu_meoh = trues_osu_meoh,         # (6, 87)
    preds_osu_meoh = preds_osu_meoh,         # (6, 87)
    true_osu_meoh_t6 = np.array(true_osu_meoh_t6),  # scalar
    pred_osu_meoh_t6 = np.array(pred_osu_meoh_t6),  # scalar
    target_idx = np.array(target_idx),              # 0
    strains = np.array(strains),                    # (87,)
)
print(f"Successfully saved {out_cache}!")
for k, v in np.load(out_cache).items():
    print(f"  {k:18s}: {v.shape}")

print(f"Observed target (OSU MeOH, BRRS MeOH, BRRS TMA): [{true_osu_meoh_t6:.1f}%, {trues_meoh[-1, target_idx]:.1f}%, {trues_tma[-1, target_idx]:.1f}%]")
print(f"Model target    (OSU MeOH, BRRS MeOH, BRRS TMA): [{pred_osu_meoh_t6:.1f}%, {preds_meoh[-1, target_idx]:.1f}%, {preds_tma[-1, target_idx]:.1f}%]")

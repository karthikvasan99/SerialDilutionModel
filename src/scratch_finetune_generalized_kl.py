import sys; sys.path.append(".")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Fine-tuning on device: {device}")

data = torch.load('Data/Gore_processed_data.pt', weights_only=False)
phase_c_ckpt = torch.load('Data/Gore_Global_ODE_Model_PhaseC_Best.pth', weights_only=False)

N_species = 48
K = 7
delta_t = 0.10
dilution_factor = 30.0
dispersal_rate = 1e-6
t_span = torch.tensor([0.0, 1.0], device=device)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

# Prepare data
b11_N_true, b11_od, b11_rel, b11_c, b11_m, b11_keys = [], [], [], [], [], []
b4_N_true, b4_od, b4_rel, b4_c, b4_m, b4_keys = [], [], [], [], [], []

for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    cond_idx = cond_to_idx[nutrient]
    mask = data['presence_masks'][(S, comm)]
    od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
    
    if traj_rel.shape[0] == 11:
        N_true = traj_rel * od_mean.unsqueeze(-1)
        b11_N_true.append(N_true)
        b11_od.append(od_mean)
        b11_rel.append(traj_rel)
        b11_c.append(cond_idx)
        b11_m.append(mask)
        b11_keys.append((nutrient, S, comm))
    elif traj_rel.shape[0] == 4:
        od_4 = od_mean[[0, 7, 8, 10]]
        N_true_4 = traj_rel * od_4.unsqueeze(-1)
        b4_N_true.append(N_true_4)
        b4_od.append(od_mean)
        b4_rel.append(traj_rel)
        b4_c.append(cond_idx)
        b4_m.append(mask)
        b4_keys.append((nutrient, S, comm))

b11_N_true = torch.stack(b11_N_true).to(device)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)

b4_N_true = torch.stack(b4_N_true).to(device)
b4_c = torch.tensor(b4_c, device=device)
b4_m = torch.stack(b4_m).to(device)

from src.Gore_Train_GeneralizedKL import PureConsumerResourceODE, PhysicalAssemblyNeuralODE, dixit_generalized_kl

ode_func = PureConsumerResourceODE(N_species, K)
model = PhysicalAssemblyNeuralODE(ode_func, num_conditions=3, K=K).to(device)

# Load Phase C weights into model
model.ode_func.Theta.data = phase_c_ckpt['ode_func.Theta'].to(device)
model.ode_func.W.data = phase_c_ckpt['ode_func.W'].to(device)
model.ode_func.alpha_raw.data = phase_c_ckpt['ode_func.alpha_raw'].to(device)
model.a.data = phase_c_ckpt['a'].to(device)

print("Phase C weights loaded successfully into PhysicalAssemblyNeuralODE.")

# Train with smaller learning rate to fine-tune
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

epochs = 150
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    tf_ratio = max(0.10, 1.0 - (epoch / (epochs * 0.75)))
    
    N_curr = b11_N_true[:, 0, :]
    loss11 = 0.0
    for t in range(1, 11):
        N_pred = model.step_forward(N_curr, b11_c, b11_m)
        target_N = b11_N_true[:, t, :]
        loss11 += dixit_generalized_kl(target_N, N_pred, b11_m)
        N_curr = tf_ratio * target_N + (1.0 - tf_ratio) * N_pred
        
    N_curr_4 = b4_N_true[:, 0, :]
    loss4 = 0.0
    d4_map = {7: 1, 8: 2, 10: 3}
    for day in range(1, 11):
        N_pred_4 = model.step_forward(N_curr_4, b4_c, b4_m)
        if day in d4_map:
            t_idx = d4_map[day]
            target_4 = b4_N_true[:, t_idx, :]
            loss4 += dixit_generalized_kl(target_4, N_pred_4, b4_m)
            N_curr_4 = tf_ratio * target_4 + (1.0 - tf_ratio) * N_pred_4
        else:
            N_curr_4 = N_pred_4
            
    loss = (loss11 / 10.0) + (loss4 / 3.0)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if epoch % 30 == 0:
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} (TF: {tf_ratio:.2f})")

# Evaluate unrolled Linear R^2
model.eval()
with torch.no_grad():
    all_N0 = torch.cat([b11_N_true[:, 0, :], b4_N_true[:, 0, :]], dim=0)
    all_conds = torch.cat([b11_c, b4_c], dim=0)
    all_masks = torch.cat([b11_m, b4_m], dim=0)
    all_keys = b11_keys + b4_keys
    
    N_curr = all_N0.clone()
    pred_trajs_N = [N_curr.cpu().numpy()]
    for day in range(1, 11):
        N_pred = model.step_forward(N_curr, all_conds, all_masks)
        pred_trajs_N.append(N_pred.cpu().numpy())
        N_curr = N_pred
        
    pred_trajs_N = np.stack(pred_trajs_N, axis=1) # (99, 11, 48)
    pred_biomass = pred_trajs_N.sum(axis=-1)
    tot_safe = np.where(pred_biomass[..., None] > 0, pred_biomass[..., None], 1.0)
    pred_rel = pred_trajs_N / tot_safe
    
    r2_rel_list, r2_bio_list = [], []
    for i, (nutrient, S, comm) in enumerate(all_keys):
        mask_np = all_masks[i].cpu().numpy().astype(bool)
        true_od = data['biomass_od'][(nutrient, S, comm)].mean(dim=0).numpy()
        pred_od = pred_biomass[i]
        ss_res_b = np.sum((true_od - pred_od)**2)
        ss_tot_b = np.sum((true_od - np.mean(true_od))**2)
        r2_bio_list.append(1.0 - ss_res_b / (ss_tot_b + 1e-9))
        
        days = data['timepoints'][nutrient][(S, comm)]
        day_nums = np.array([int(d.replace('Day', '')) for d in days])
        true_rel = data['relative_abundance'][(nutrient, S, comm)].numpy()
        sub_pred_rel = pred_rel[i, day_nums]
        y_t = true_rel[:, mask_np].flatten()
        y_p = sub_pred_rel[:, mask_np].flatten()
        ss_res_r = np.sum((y_t - y_p)**2)
        ss_tot_r = np.sum((y_t - np.mean(y_t))**2)
        r2_rel_list.append(1.0 - ss_res_r / (ss_tot_r + 1e-9))
        
    print(f"\nFine-tuned Results:")
    print(f"Mean Relative Abundance Linear R^2: {np.mean(r2_rel_list):.4f}")
    print(f"Median Relative Abundance Linear R^2: {np.median(r2_rel_list):.4f}")
    print(f"Mean Biomass OD Linear R^2: {np.mean(r2_bio_list):.4f}")
    print(f"Median Biomass OD Linear R^2: {np.median(r2_bio_list):.4f}")

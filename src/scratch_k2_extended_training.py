import os
import sys
sys.path.insert(0, '.')
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load dataset
data_path = 'Data/Gore_processed_data.pt'
data = torch.load(data_path, weights_only=False)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

b11_N, b11_od, b11_rel, b11_c, b11_m, b11_keys = [], [], [], [], [], []
b4_N, b4_od, b4_rel, b4_c, b4_m, b4_keys = [], [], [], [], [], []

for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    c_idx = cond_to_idx[nutrient]
    mask = data['presence_masks'][(S, comm)]
    od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
    
    if traj_rel.shape[0] == 11:
        b11_N.append(traj_rel * od_mean.unsqueeze(-1))
        b11_od.append(od_mean)
        b11_rel.append(traj_rel)
        b11_c.append(c_idx)
        b11_m.append(mask)
        b11_keys.append((nutrient, S, comm))
    elif traj_rel.shape[0] == 4:
        od_4 = od_mean[[0, 7, 8, 10]]
        b4_N.append(traj_rel * od_4.unsqueeze(-1))
        b4_od.append(od_mean)
        b4_rel.append(traj_rel)
        b4_c.append(c_idx)
        b4_m.append(mask)
        b4_keys.append((nutrient, S, comm))

b11_N = torch.stack(b11_N).to(device)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)

b4_N = torch.stack(b4_N).to(device)
b4_c = torch.tensor(b4_c, device=device)
b4_m = torch.stack(b4_m).to(device)

all_N0 = torch.cat([b11_N[:, 0, :], b4_N[:, 0, :]], dim=0).to(device)
all_c = torch.cat([b11_c, b4_c], dim=0).to(device)
all_m = torch.cat([b11_m, b4_m], dim=0).to(device)
all_keys = b11_keys + b4_keys

# Ground truth binary chaotic labels per Gore et al.
gt_labels = []
for (nutrient, S, comm) in all_keys:
    mask = data['presence_masks'][(S, comm)].numpy().astype(bool)
    rel = data['relative_abundance'][(nutrient, S, comm)].numpy()
    od = data['biomass_od'][(nutrient, S, comm)].mean(dim=0).numpy()
    if rel.shape[0] == 11:
        N_abs = rel * od[:, None]
        late_N = N_abs[7:11, mask]
    else:
        od_4 = od[[0, 7, 8, 10]]
        N_abs = rel * od_4[:, None]
        late_N = N_abs[1:, mask]
    mu_N = late_N.mean(axis=0)
    sig_N = late_N.std(axis=0)
    cv_N = np.mean(sig_N / (0.01 + mu_N))
    od_7_10 = data['biomass_od'][(nutrient, S, comm)].numpy()[:, 7:11]
    od_cv = np.mean(np.std(od_7_10, axis=1) / np.mean(od_7_10, axis=1))
    gt_labels.append(bool(cv_N >= 0.25 or od_cv >= 0.14))

gt_labels = np.array(gt_labels)

from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE, dixit_generalized_kl

# Model setup for K=2
K = 2
total_epochs = 800
anneal_epochs = 200 # anneal from 1.0 to 0.10 over first 200 epochs, then 600 epochs at 0.10!
lr = 0.05
l2_reg = 0.001
delta_t = 0.10

ode_func = AdaptivePureConsumerResourceODE(N=48, K=K)
model = HierarchicalPhysicalNeuralODE(ode_func, num_conditions=3, K=K).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-3)

def evaluate_checkpoint(current_epoch):
    model.eval()
    with torch.no_grad():
        N_curr = all_N0.clone()
        preds_N = [N_curr.cpu().numpy()]
        for day in range(1, 11):
            N_pred = model.step_forward(N_curr, all_c, all_m, delta_t=delta_t)
            preds_N.append(N_pred.cpu().numpy())
            N_curr = N_pred
            
        preds_N = np.stack(preds_N, axis=1) # (99, 11, 48)
        pred_bio = preds_N.sum(axis=-1)     # (99, 11)
        tot_safe = np.where(pred_bio[..., None] > 0, pred_bio[..., None], 1.0)
        pred_rel = preds_N / tot_safe
        
        r2_rel_list, m_cvs = [], []
        for i, (nutrient, S, comm) in enumerate(all_keys):
            mask_np = all_m[i].cpu().numpy().astype(bool)
            
            days = data['timepoints'][nutrient][(S, comm)]
            day_nums = np.array([int(d.replace('Day', '')) for d in days])
            true_rel = data['relative_abundance'][(nutrient, S, comm)].numpy()
            sub_pred_rel = pred_rel[i, day_nums]
            y_t = true_rel[:, mask_np].flatten()
            y_p = sub_pred_rel[:, mask_np].flatten()
            ss_res_r = np.sum((y_t - y_p)**2)
            ss_tot_r = np.sum((y_t - np.mean(y_t))**2)
            r2_rel_list.append(1.0 - ss_res_r / (ss_tot_r + 1e-9))
            
            late_m = pred_rel[i, 7:11][:, mask_np]
            mu_m = late_m.mean(axis=0)
            sig_m = late_m.std(axis=0)
            surv_m = mu_m >= 1e-3
            m_cv = np.mean(sig_m[surv_m] / (0.01 + mu_m[surv_m])) if surv_m.any() else 0.0
            m_cvs.append(m_cv)
            
        m_cvs = np.array(m_cvs)
        
        all_true_od = np.array([data['biomass_od'][(n, s, c)].mean(dim=0).numpy() for (n, s, c) in all_keys]).flatten()
        all_pred_od = pred_bio.flatten()
        r_bio_global, _ = pearsonr(all_true_od, all_pred_od)
        
        # Stability correlation
        log_m = np.log(m_cvs + 1e-4)
        
        # Ground truth empirical CVs on absolute abundance
        exp_cvs = []
        for (nutrient, S, comm) in all_keys:
            mask = data['presence_masks'][(S, comm)].numpy().astype(bool)
            rel = data['relative_abundance'][(nutrient, S, comm)].numpy()
            od = data['biomass_od'][(nutrient, S, comm)].mean(dim=0).numpy()
            if rel.shape[0] == 11:
                N_abs = rel * od[:, None]
                late_N = N_abs[7:11, mask]
            else:
                od_4 = od[[0, 7, 8, 10]]
                N_abs = rel * od_4[:, None]
                late_N = N_abs[1:, mask]
            mu_N = late_N.mean(axis=0)
            sig_N = late_N.std(axis=0)
            exp_cvs.append(np.mean(sig_N / (0.01 + mu_N)))
        exp_cvs = np.array(exp_cvs)
        log_e = np.log(exp_cvs + 1e-4)
        
        r_log_overall, _ = pearsonr(log_m, log_e)
        auc_chaos = roc_auc_score(gt_labels, m_cvs)
        
        cond_stats = {}
        for nut in ['LN', 'MN', 'HN']:
            idxs = [k for k, key in enumerate(all_keys) if key[0] == nut]
            r_sub, _ = pearsonr(log_m[idxs], log_e[idxs])
            rel_sub = np.mean([r2_rel_list[k] for k in idxs])
            cond_stats[nut] = (rel_sub, r_sub)
            
        return {
            'epoch': current_epoch,
            'mean_rel_r2': np.mean(r2_rel_list),
            'median_rel_r2': np.median(r2_rel_list),
            'global_bio_r': r_bio_global,
            'log_r_overall': r_log_overall,
            'auc_chaos': auc_chaos,
            'ln_rel_r2': cond_stats['LN'][0],
            'mn_rel_r2': cond_stats['MN'][0],
            'hn_rel_r2': cond_stats['HN'][0],
            'ln_log_r': cond_stats['LN'][1],
            'mn_log_r': cond_stats['MN'][1],
            'hn_log_r': cond_stats['HN'][1]
        }

print(f"\n--- Starting K=2 Extended Training ({total_epochs} Epochs: 200 Annealing + 600 at TF=10%) ---")
records = []
start_time = time.time()

for epoch in range(1, total_epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    # Anneal TF from 1.0 to 0.10 over first 200 epochs, then stay at 0.10!
    if epoch <= anneal_epochs:
        tf_ratio = 1.0 - (0.90 * (epoch / anneal_epochs))
    else:
        tf_ratio = 0.10
        
    # Batch 11
    N_curr = b11_N[:, 0, :]
    loss11 = 0.0
    for t in range(1, 11):
        N_pred = model.step_forward(N_curr, b11_c, b11_m, delta_t=delta_t)
        target = b11_N[:, t, :]
        loss11 += dixit_generalized_kl(target, N_pred, b11_m)
        N_curr = tf_ratio * target + (1.0 - tf_ratio) * N_pred
        
    # Batch 4
    N_curr_4 = b4_N[:, 0, :]
    loss4 = 0.0
    d4_map = {7: 1, 8: 2, 10: 3}
    for day in range(1, 11):
        N_pred_4 = model.step_forward(N_curr_4, b4_c, b4_m, delta_t=delta_t)
        if day in d4_map:
            t_idx = d4_map[day]
            target_4 = b4_N[:, t_idx, :]
            loss4 += dixit_generalized_kl(target_4, N_pred_4, b4_m)
            N_curr_4 = tf_ratio * target_4 + (1.0 - tf_ratio) * N_pred_4
        else:
            N_curr_4 = N_pred_4
            
    reg_penalty = l2_reg * (model.ode_func.Theta.pow(2.0).sum() + model.ode_func.W.pow(2.0).sum())
    loss = (loss11 / 10.0) + (loss4 / 3.0) + reg_penalty
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    
    # Evaluate checkpoints at milestones
    if epoch in [100, 200, 300, 400, 500, 600, 700, 800]:
        elapsed = time.time() - start_time
        eval_metrics = evaluate_checkpoint(epoch)
        eval_metrics['train_time'] = round(elapsed, 1)
        eval_metrics['loss'] = round(loss.item(), 4)
        eval_metrics['tf_ratio'] = round(tf_ratio, 2)
        records.append(eval_metrics)
        print(f"Epoch {epoch:3d} (TF: {tf_ratio:.2f}, Loss: {loss.item():.4f}) | Rel R^2: {eval_metrics['mean_rel_r2']:.4f} (Med: {eval_metrics['median_rel_r2']:.4f}) | Bio r: {eval_metrics['global_bio_r']:.4f} | Log r: {eval_metrics['log_r_overall']:.4f} | HN Log r: {eval_metrics['hn_log_r']:.3f} | AUC: {eval_metrics['auc_chaos']:.3f} | Time: {elapsed:.1f}s")

df_progression = pd.DataFrame(records)
df_progression.to_csv('Results/Gore_PhysODE_K2_ExtendedProgression.csv', index=False)
torch.save(model.state_dict(), 'Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth')
print("\nSaved extended training progression to Results/Gore_PhysODE_K2_ExtendedProgression.csv")
print("Saved final model checkpoint to Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth")

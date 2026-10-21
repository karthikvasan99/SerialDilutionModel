import sys, os
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE

def evaluate_model(model_path, K):
    ode_func = AdaptivePureConsumerResourceODE(N=48, K=K)
    model = HierarchicalPhysicalNeuralODE(ode_func, num_conditions=3, K=K).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()
    
    with torch.no_grad():
        N_curr = all_N0.clone()
        preds_N = [N_curr.cpu().numpy()]
        for day in range(1, 11):
            N_pred = model.step_forward(N_curr, all_c, all_m, delta_t=0.10)
            preds_N.append(N_pred.cpu().numpy())
            N_curr = N_pred
            
        preds_N = np.stack(preds_N, axis=1) # (99, 11, 48)
        pred_bio = preds_N.sum(axis=-1)
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
        
        return r2_rel_list, m_cvs, r_bio_global

def compute_detailed_confusion(m_cvs, threshold_percentile=77.78):
    # Overall threshold: 22/99 chaotic in ground truth = 22.2% chaotic, 77.8% stable
    thresh = np.percentile(m_cvs, threshold_percentile)
    preds = m_cvs >= thresh
    
    results = {}
    
    # Overall
    cm = confusion_matrix(gt_labels, preds)
    results['Overall'] = {
        'CM': cm,
        'TN': cm[0, 0], 'FP': cm[0, 1], 'FN': cm[1, 0], 'TP': cm[1, 1],
        'Accuracy': accuracy_score(gt_labels, preds),
        'Precision': precision_score(gt_labels, preds, zero_division=0),
        'Recall': recall_score(gt_labels, preds, zero_division=0),
        'F1': f1_score(gt_labels, preds, zero_division=0),
        'AUC': roc_auc_score(gt_labels, m_cvs)
    }
    
    # Nutrient subsets
    for nut in ['LN', 'MN', 'HN']:
        idxs = [i for i, key in enumerate(all_keys) if key[0] == nut]
        y_true_nut = gt_labels[idxs]
        y_pred_nut = preds[idxs]
        
        # Unique classes
        labels_present = np.unique(y_true_nut)
        if len(labels_present) == 1:
            if labels_present[0] == False:
                cm_nut = np.array([[np.sum(~y_pred_nut), np.sum(y_pred_nut)], [0, 0]])
            else:
                cm_nut = np.array([[0, 0], [np.sum(~y_pred_nut), np.sum(y_pred_nut)]])
        else:
            cm_nut = confusion_matrix(y_true_nut, y_pred_nut)
            
        results[nut] = {
            'CM': cm_nut,
            'TN': cm_nut[0, 0], 'FP': cm_nut[0, 1], 'FN': cm_nut[1, 0], 'TP': cm_nut[1, 1],
            'Accuracy': accuracy_score(y_true_nut, y_pred_nut),
            'Precision': precision_score(y_true_nut, y_pred_nut, zero_division=0),
            'Recall': recall_score(y_true_nut, y_pred_nut, zero_division=0),
            'F1': f1_score(y_true_nut, y_pred_nut, zero_division=0),
        }
        if len(np.unique(y_true_nut)) > 1:
            results[nut]['AUC'] = roc_auc_score(y_true_nut, m_cvs[idxs])
        else:
            results[nut]['AUC'] = float('nan')
            
    return results

if __name__ == '__main__':
    models = {
        'PhysODE_K2_400ep': ('Data/Gore_Global_ODE_Model_PhysODE_BestLowK.pth', 2),
        'PhysODE_K2_800ep': ('Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth', 2),
        'PhysODE_K7_400ep': ('Data/Gore_Global_ODE_Model_PhysODE_K7_400ep.pth', 7),
    }
    
    for name, (path, K) in models.items():
        if not os.path.exists(path):
            print(f"Skipping {name}, checkpoint {path} does not exist yet.")
            continue
        print(f"\n==================== {name} ====================")
        r2_list, m_cvs, r_bio = evaluate_model(path, K)
        print(f"Mean Rel R^2: {np.mean(r2_list):.4f} (Med: {np.median(r2_list):.4f}) | Biomass r: {r_bio:.4f}")
        
        conf = compute_detailed_confusion(m_cvs)
        for cond, metrics in conf.items():
            print(f"--- Condition: {cond} ---")
            print(f"TN: {metrics['TN']:2d} | FP: {metrics['FP']:2d}")
            print(f"FN: {metrics['FN']:2d} | TP: {metrics['TP']:2d}")
            print(f"Accuracy: {metrics['Accuracy']*100:.1f}% | Precision: {metrics['Precision']:.3f} | Recall: {metrics['Recall']:.3f} | F1: {metrics['F1']:.3f}" + (f" | AUC: {metrics['AUC']:.3f}" if not np.isnan(metrics['AUC']) else ""))

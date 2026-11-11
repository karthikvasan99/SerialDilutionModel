import sys, os
sys.path.insert(0, '.')
import torch
import sys, os
sys.path.insert(0, '.')
import torch.nn as nn
import sys, os
sys.path.insert(0, '.')
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

data_path = 'Data/Gore_processed_data.pt'
data = torch.load(data_path, weights_only=False)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

# Prepare dataset
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

dataset = {
    'b11_N': b11_N, 'b11_c': b11_c, 'b11_m': b11_m,
    'b4_N': b4_N, 'b4_c': b4_c, 'b4_m': b4_m,
    'all_N0': all_N0, 'all_c': all_c, 'all_m': all_m,
    'all_keys': all_keys, 'raw_data': data
}

from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE, dixit_generalized_kl, train_and_eval_cfg

# Check if K=7 checkpoint exists, if not train it
ckpt_k7_path = 'Data/Gore_Global_ODE_Model_PhysODE_K7_400ep.pth'
import os
if not os.path.exists(ckpt_k7_path):
    print("Training PhysODE_K7 for 400 epochs...")
    cfg_k7 = {'name': 'PhysODE_K7', 'K': 7, 'l2_reg': 0.001, 'delta_t': 0.10, 'epochs': 400, 'lr': 0.05, 'tf_min': 0.10}
    summary_k7, model_k7 = train_and_eval_cfg(cfg_k7, dataset)
    torch.save(model_k7.state_dict(), ckpt_k7_path)
    print(f"PhysODE_K7 trained and saved to {ckpt_k7_path}")

# Load K=2 model
ode_k2 = AdaptivePureConsumerResourceODE(N=48, K=2)
model_k2 = HierarchicalPhysicalNeuralODE(ode_k2, num_conditions=3, K=2).to(device)
model_k2.load_state_dict(torch.load('Data/Gore_Global_ODE_Model_PhysODE_BestLowK.pth', weights_only=False))
model_k2.eval()

# Load K=7 model
ode_k7 = AdaptivePureConsumerResourceODE(N=48, K=7)
model_k7 = HierarchicalPhysicalNeuralODE(ode_k7, num_conditions=3, K=7).to(device)
model_k7.load_state_dict(torch.load(ckpt_k7_path, weights_only=False))
model_k7.eval()

print("Both PhysODE_K2 and PhysODE_K7 models loaded successfully.")

# Evaluate unrolled predictions for both models
def get_predictions(model, K):
    with torch.no_grad():
        N_curr = all_N0.clone()
        preds_N = [N_curr.cpu().numpy()]
        for day in range(1, 11):
            N_pred = model.step_forward(N_curr, all_c, all_m, delta_t=0.10)
            preds_N.append(N_pred.cpu().numpy())
            N_curr = N_pred
            
        preds_N = np.stack(preds_N, axis=1) # (99, 11, 48)
        pred_bio = preds_N.sum(axis=-1)     # (99, 11)
        tot_safe = np.where(pred_bio[..., None] > 0, pred_bio[..., None], 1.0)
        pred_rel = preds_N / tot_safe
        
        records = []
        for i, (nutrient, S, comm) in enumerate(all_keys):
            mask_np = all_m[i].cpu().numpy().astype(bool)
            
            # Late-stage model CV (Days 7..10)
            late_m = pred_rel[i, 7:11][:, mask_np]
            mu_m = late_m.mean(axis=0)
            sig_m = late_m.std(axis=0)
            surv_m = mu_m >= 1e-3
            m_cv = np.mean(sig_m[surv_m] / (0.01 + mu_m[surv_m])) if surv_m.any() else 0.0
            
            # Model Biomass CV
            bio_7_10 = pred_bio[i, 7:11]
            m_bio_cv = np.std(bio_7_10) / (np.mean(bio_7_10) + 1e-6)
            
            # Empirical CV
            days = data['timepoints'][nutrient][(S, comm)]
            day_nums = np.array([int(d.replace('Day', '')) for d in days])
            true_rel = data['relative_abundance'][(nutrient, S, comm)].numpy()
            late_idx = np.where(day_nums >= 7)[0]
            late_e = true_rel[late_idx][:, mask_np]
            mu_e = late_e.mean(axis=0)
            sig_e = late_e.std(axis=0)
            surv_e = mu_e >= 1e-3
            e_cv = np.mean(sig_e[surv_e] / (0.01 + mu_e[surv_e])) if surv_e.any() else 0.0
            
            od_7_10 = data['biomass_od'][(nutrient, S, comm)].numpy()[:, 7:11]
            od_cv = np.mean(np.std(od_7_10, axis=1) / np.mean(od_7_10, axis=1))
            
            # True classification per Gore et al.
            gt_chaotic = bool(e_cv >= 0.25 or od_cv >= 0.14)
            
            records.append({
                'nutrient': nutrient,
                'S': S,
                'comm': comm,
                'model_cv': m_cv,
                'model_bio_cv': m_bio_cv,
                'exp_cv': e_cv,
                'od_cv': od_cv,
                'gt_chaotic': gt_chaotic
            })
        return pd.DataFrame(records)

df_k2 = get_predictions(model_k2, K=2)
df_k7 = get_predictions(model_k7, K=7)

print(f"\nGround Truth Class Distribution across 99 communities:")
for nut in ['LN', 'MN', 'HN']:
    sub = df_k2[df_k2['nutrient'] == nut]
    print(f"  {nut}: {sub['gt_chaotic'].sum()} Chaotic / {len(sub)} total ({sub['gt_chaotic'].mean()*100:.1f}%)")
print(f"  Total: {df_k2['gt_chaotic'].sum()} Chaotic / {len(df_k2)} total ({df_k2['gt_chaotic'].mean()*100:.1f}%)")

# Determine optimal threshold on ROC curve or use standard threshold (e.g. 0.20 or 0.25)
# In Gore et al., empirical threshold is CV >= 0.25.
# Let's inspect the distribution of model CVs
print("\nModel CV percentiles:")
print("PhysODE_K2:", np.percentile(df_k2['model_cv'], [10, 25, 50, 75, 90]))
print("PhysODE_K7:", np.percentile(df_k7['model_cv'], [10, 25, 50, 75, 90]))

# Let's compute confusion matrices using threshold that maximizes balanced accuracy or matches ground truth rate
for name, df in [("PhysODE_K2 (K=2)", df_k2), ("PhysODE_K7 (K=7)", df_k7)]:
    print(f"\n=======================================================")
    print(f"CONFUSION MATRICES: {name}")
    print(f"=======================================================")
    
    # Let's find best F1/accuracy threshold overall
    from sklearn.metrics import precision_recall_curve, roc_curve
    fpr, tpr, thresholds = roc_curve(df['gt_chaotic'], df['model_cv'])
    # Youden's J statistic = tpr - fpr
    best_idx = np.argmax(tpr - fpr)
    thresh = thresholds[best_idx]
    print(f"Optimal ROC Decision Threshold on Model CV: {thresh:.4f}")
    
    df['pred_chaotic'] = df['model_cv'] >= thresh
    
    overall_cm = confusion_matrix(df['gt_chaotic'], df['pred_chaotic'])
    overall_acc = accuracy_score(df['gt_chaotic'], df['pred_chaotic'])
    overall_auc = roc_auc_score(df['gt_chaotic'], df['model_cv'])
    
    print(f"\nOVERALL (99 Communities) | Accuracy: {overall_acc*100:.1f}% | AUC: {overall_auc:.3f}")
    print(f"                  Predicted Stable  Predicted Chaotic")
    print(f"  True Stable:          {overall_cm[0, 0]:2d}                {overall_cm[0, 1]:2d}")
    print(f"  True Chaotic:         {overall_cm[1, 0]:2d}                {overall_cm[1, 1]:2d}")
    
    for nut in ['LN', 'MN', 'HN']:
        sub = df[df['nutrient'] == nut]
        y_t = sub['gt_chaotic']
        y_p = sub['pred_chaotic']
        cm = confusion_matrix(y_t, y_p, labels=[False, True])
        acc = accuracy_score(y_t, y_p)
        print(f"\n--- {nut} ({len(sub)} Communities) | Accuracy: {acc*100:.1f}% ---")
        print(f"                  Predicted Stable  Predicted Chaotic")
        print(f"  True Stable:          {cm[0, 0]:2d}                {cm[0, 1]:2d}")
        print(f"  True Chaotic:         {cm[1, 0]:2d}                {cm[1, 1]:2d}")

# Also evaluate Phase C Comp-ODE for direct comparison!
print("\n=======================================================")
print("REFERENCE: Comp-ODE (Simplex Model Phase C Best)")
print("=======================================================")
phase_c_df = pd.read_csv('Results/Gore_PhaseC_Optimization_Detailed.csv') if os.path.exists('Results/Gore_PhaseC_Optimization_Detailed.csv') else None
if phase_c_df is not None:
    print("Found Phase C detailed results table!")

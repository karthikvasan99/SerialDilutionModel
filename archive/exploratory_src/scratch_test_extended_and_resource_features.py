import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix
from src.scratch_evaluate_k2_800ep import all_keys, gt_labels, data, all_m, all_N0, all_c
from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load K=2 800ep model
ode_func = AdaptivePureConsumerResourceODE(N=48, K=2)
model = HierarchicalPhysicalNeuralODE(ode_func, num_conditions=3, K=2).to(device)
model.load_state_dict(torch.load('Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth', weights_only=False))
model.eval()

# Let's simulate for 25 passages!
# During simulation, record both species N and latent resources R at each passage
num_passages = 25
delta_t = 0.10

with torch.no_grad():
    N_curr = all_N0.clone()
    
    # Store history
    all_N_history = [N_curr.cpu().numpy()] # list of (99, 48)
    all_R_end_history = []                 # list of (99, K)
    
    for p in range(1, num_passages + 1):
        R0_table = model.get_R0()
        R0 = R0_table[all_c]
        yield_scale = torch.nn.functional.softplus(model.yield_raw)[all_c].unsqueeze(-1)
        
        N0 = N_curr / model.dilution_factor + model.dispersal_rate * all_m
        N0_safe = torch.clamp(N0, min=0.0)
        S0 = torch.cat([N0_safe, R0], dim=-1)
        
        from torchdiffeq import odeint
        t_span = torch.tensor([0.0, 1.0], device=device)
        S_t = odeint(model.ode_func, S0, t_span, method='euler', options={'step_size': delta_t})
        
        S_final = S_t[-1]
        N_final = torch.clamp(S_final[..., :48] * all_m, min=0.0) * yield_scale
        R_final = S_final[..., 48:]
        
        N_curr = N_final
        all_N_history.append(N_final.cpu().numpy())
        all_R_end_history.append(R_final.cpu().numpy())

all_N = np.stack(all_N_history, axis=1) # (99, 26, 48)
all_R = np.stack(all_R_end_history, axis=1) # (99, 25, 2)
all_bio = all_N.sum(axis=-1) # (99, 26)
all_rel = all_N / np.where(all_bio[..., None] > 0, all_bio[..., None], 1.0)

# Build features under 3 configurations:
# Config 1: Baseline (Days 7-10 species only, exactly as before)
# Config 2: Extended simulation (Days 15-25 species dynamics)
# Config 3: Combined Extended + Resource dynamics!

def extract_features(use_extended=False, use_resources=False):
    feat_list = []
    
    # Time window
    if use_extended:
        # Long-term attractor (Passages 15-25)
        t_slice = slice(15, 26)
        r_slice = slice(14, 25)
    else:
        # Standard experimental window (Passages 7-10)
        t_slice = slice(7, 11)
        r_slice = slice(6, 10)
        
    for i, (nutrient, S, comm) in enumerate(all_keys):
        mask_np = all_m[i].cpu().numpy().astype(bool)
        
        # Species dynamics
        sub_rel = all_rel[i, t_slice][:, mask_np]
        mu_rel = sub_rel.mean(axis=0)
        sig_rel = sub_rel.std(axis=0)
        surv = mu_rel >= 1e-3
        sp_cvs = sig_rel[surv] / (0.01 + mu_rel[surv]) if surv.any() else np.array([0.0])
        
        mean_sp_cv = np.mean(sp_cvs)
        max_sp_cv = np.max(sp_cvs)
        median_sp_cv = np.median(sp_cvs)
        
        sub_bio = all_bio[i, t_slice]
        bio_cv = np.std(sub_bio) / (np.mean(sub_bio) + 1e-6)
        bio_final = sub_bio[-1]
        
        surv_count = np.sum(surv)
        extinct_ratio = (S - surv_count) / S
        path_len = np.sum(np.abs(np.diff(sub_rel, axis=0)))
        
        c_ln = 1.0 if nutrient == 'LN' else 0.0
        c_mn = 1.0 if nutrient == 'MN' else 0.0
        c_hn = 1.0 if nutrient == 'HN' else 0.0
        
        f = [mean_sp_cv, max_sp_cv, median_sp_cv, bio_cv, bio_final,
             surv_count, extinct_ratio, path_len, S, c_ln, c_mn, c_hn]
        
        if use_resources:
            # Latent resource features over time window
            # all_R is (99, 25, 2)
            sub_R = all_R[i, r_slice] # (window, 2)
            # 1. Mean residual resource unconsumed
            r_mean = np.mean(sub_R, axis=0)
            # 2. Volatility of residual resources across passages
            r_std = np.std(sub_R, axis=0)
            r_cv = r_std / (r_mean + 1e-6)
            # 3. Minimum resource level reached (resource limitation severity)
            r_min = np.min(sub_R, axis=0)
            
            # Combine resource features
            f.extend([
                r_mean[0], r_mean[1],
                r_cv[0], r_cv[1],
                r_min[0], r_min[1]
            ])
            
        feat_list.append(f)
    return np.array(feat_list)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y = gt_labels.astype(int)

configs = [
    ("1. Baseline (Passages 7-10, Species Only)", False, False),
    ("2. Extended (Passages 15-25, Species Only)", True, False),
    ("3. Baseline + Latent Resources (Passages 7-10)", False, True),
    ("4. Full Model: Extended + Latent Resources (Passages 15-25)", True, True)
]

print("===================================================================")
print("TESTING LONGER SIMULATION & LATENT RESOURCE DYNAMICS")
print("===================================================================")

for cfg_name, ext, res in configs:
    X = extract_features(use_extended=ext, use_resources=res)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=42)
    y_prob_rf = cross_val_predict(rf, X, y, cv=skf, method='predict_proba')[:, 1]
    y_pred_rf = cross_val_predict(rf, X, y, cv=skf)
    acc_rf = accuracy_score(y, y_pred_rf)
    bacc_rf = balanced_accuracy_score(y, y_pred_rf)
    auc_rf = roc_auc_score(y, y_prob_rf)
    f1_rf = f1_score(y, y_pred_rf)
    
    # Logistic Regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    y_prob_lr = cross_val_predict(lr, X, y, cv=skf, method='predict_proba')[:, 1]
    y_pred_lr = cross_val_predict(lr, X, y, cv=skf)
    acc_lr = accuracy_score(y, y_pred_lr)
    bacc_lr = balanced_accuracy_score(y, y_pred_lr)
    auc_lr = roc_auc_score(y, y_prob_lr)
    f1_lr = f1_score(y, y_pred_lr)
    
    print(f"\nConfiguration: {cfg_name} [Features: {X.shape[1]}]")
    print(f"  Random Forest : Acc = {acc_rf*100:.1f}% | Balanced Acc = {bacc_rf*100:.1f}% | AUC = {auc_rf:.4f} | F1 = {f1_rf:.3f}")
    print(f"  Logistic Reg. : Acc = {acc_lr*100:.1f}% | Balanced Acc = {bacc_lr*100:.1f}% | AUC = {auc_lr:.4f} | F1 = {f1_lr:.3f}")

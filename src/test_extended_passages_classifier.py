import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)

from src.scratch_evaluate_k2_800ep import all_keys, gt_labels, data, all_m, all_N0, all_c
from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Pure K=2 model
ode_k2 = AdaptivePureConsumerResourceODE(N=48, K=2)
model_k2 = HierarchicalPhysicalNeuralODE(ode_k2, num_conditions=3, K=2).to(device)
model_k2.load_state_dict(torch.load('Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth', weights_only=False))
model_k2.eval()

# Let's simulate for 30 passages!
num_passages = 30
delta_t = 0.05
tau_steps = int(1.0 / delta_t) + 1 # 21 points per passage

print(f"Simulating pure K=2 model for {num_passages} serial dilution passages...")

with torch.no_grad():
    N_curr = all_N0.clone()
    all_N_passages = [N_curr.cpu().numpy()]
    all_R_paths = []
    all_R0_list = []
    
    for p in range(1, num_passages + 1):
        R0_table = model_k2.get_R0()
        R0 = R0_table[all_c]
        yield_scale = torch.nn.functional.softplus(model_k2.yield_raw)[all_c].unsqueeze(-1)
        
        N0 = N_curr / model_k2.dilution_factor + model_k2.dispersal_rate * all_m
        N0_safe = torch.clamp(N0, min=0.0)
        S0 = torch.cat([N0_safe, R0], dim=-1)
        
        from torchdiffeq import odeint
        t_span = torch.linspace(0.0, 1.0, tau_steps, device=device)
        S_t = odeint(model_k2.ode_func, S0, t_span, method='euler', options={'step_size': delta_t})
        S_t = S_t.permute(1, 0, 2) # (99, tau_steps, 50)
        
        N_path = S_t[..., :48] * all_m.unsqueeze(1)
        R_path = S_t[..., 48:] # (99, tau_steps, 2)
        
        N_final = torch.clamp(N_path[:, -1, :], min=0.0) * yield_scale
        N_curr = N_final
        
        all_N_passages.append(N_final.cpu().numpy())
        all_R_paths.append(R_path.cpu().numpy())
        if p == 1:
            all_R0_list = R0.cpu().numpy()

all_N = np.stack(all_N_passages, axis=1) # (99, 31, 48)
all_R = np.stack(all_R_paths, axis=1)    # (99, 30, tau_steps, 2)
all_bio = all_N.sum(axis=-1)             # (99, 31)
all_rel = all_N / np.where(all_bio[..., None] > 0, all_bio[..., None], 1.0)

# Niche Overlap
Theta_np = model_k2.ode_func.Theta.detach().cpu().numpy()
theta_norms = np.linalg.norm(Theta_np, axis=1, keepdims=True) + 1e-12
Theta_normed = Theta_np / theta_norms

# Function to extract features over a specific passage window [p_start, p_end]
def extract_feature_matrix(p_start=20, p_end=30):
    feature_matrix = []
    
    for i, (nutrient, S, comm) in enumerate(all_keys):
        mask_np = all_m[i].cpu().numpy().astype(bool)
        
        # 1. Species Volatility over late passages
        sub_rel = all_rel[i, p_start:p_end+1][:, mask_np]
        mu_rel = sub_rel.mean(axis=0)
        sig_rel = sub_rel.std(axis=0)
        surv = mu_rel >= 1e-3
        sp_cvs = sig_rel[surv] / (0.01 + mu_rel[surv]) if surv.any() else np.array([0.0])
        mean_sp_cv = np.mean(sp_cvs)
        max_sp_cv = np.max(sp_cvs)
        
        # 2. Biomass Dynamics
        sub_bio = all_bio[i, p_start:p_end+1]
        bio_cv = np.std(sub_bio) / (np.mean(sub_bio) + 1e-6)
        bio_final = sub_bio[-1]
        bio_diff_consec = np.mean(np.abs(np.diff(sub_bio)))
        
        # 3. Path Length
        path_len = np.sum(np.abs(np.diff(sub_rel, axis=0)))
        
        # 4. MacArthur Niche Overlap
        present_indices = np.where(mask_np)[0]
        if len(present_indices) > 1:
            sub_theta = Theta_normed[present_indices]
            overlap_matrix = np.dot(sub_theta, sub_theta.T)
            triu_idx = np.triu_indices(len(present_indices), k=1)
            mean_overlap = np.mean(overlap_matrix[triu_idx])
        else:
            mean_overlap = 0.0
            
        # 5. Poincaré Return Map Shift
        # Absolute biomass entering passages
        inoc = all_N[i, p_start-1:p_end, mask_np] / 30.0
        poincare_diffs = np.linalg.norm(np.diff(inoc, axis=0), axis=1)
        mean_poincare_shift = np.mean(poincare_diffs)
        
        # 6. Intra-Passage Resource Dynamics
        window_R = all_R[i, p_start-1:p_end] # (N_passages, tau_steps, 2)
        
        # Trajectory Total Variation
        tv_paths = []
        for p in range(window_R.shape[0]):
            step_diffs = np.linalg.norm(np.diff(window_R[p], axis=0), axis=1)
            tv_paths.append(np.sum(step_diffs))
        mean_tv_path = np.mean(tv_paths)
        std_tv_path = np.std(tv_paths)
        
        # Path divergence across consecutive cycles
        p_diffs = []
        for p in range(1, window_R.shape[0]):
            d_tau = window_R[p] - window_R[p-1]
            p_diffs.append(np.mean(np.linalg.norm(d_tau, axis=1)))
        mean_path_diff = np.mean(p_diffs)
        
        # Endpoint shifts
        end_pts = window_R[:, -1, :]
        mean_end_diff = np.mean(np.linalg.norm(np.diff(end_pts, axis=0), axis=1))
        end_cv = np.mean(np.std(end_pts, axis=0) / (np.mean(end_pts, axis=0) + 1e-5))
        
        # 7. Media indicators
        c_ln = 1.0 if nutrient == 'LN' else 0.0
        c_hn = 1.0 if nutrient == 'HN' else 0.0
        
        row = [
            mean_tv_path, mean_poincare_shift, mean_overlap,
            bio_cv, max_sp_cv, bio_final, path_len, S,
            c_ln, c_hn, mean_path_diff, mean_end_diff, end_cv, std_tv_path
        ]
        feature_matrix.append(row)
        
    f_names = [
        'mean_tv_path', 'mean_poincare_shift', 'mean_overlap',
        'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S',
        'c_ln', 'c_hn', 'mean_path_diff', 'mean_end_diff', 'end_cv', 'std_tv_path'
    ]
    return np.array(feature_matrix), f_names

y_true = gt_labels.astype(int)
loo = LeaveOneOut()

print("\n" + "="*85)
print("COMPARING PASSAGE WINDOWS FOR FEATURE EXTRACTION")
print("="*85)

windows = [
    ("Passages 7 to 10 (Original Baseline)", 7, 10),
    ("Passages 11 to 20 (Intermediate)", 11, 20),
    ("Passages 21 to 30 (Late Fully Equilibrated)", 21, 30),
    ("Passages 25 to 30 (Ultra-Late Equilibrated)", 25, 30),
]

for w_name, p_s, p_e in windows:
    X_mat, feat_names = extract_feature_matrix(p_s, p_e)
    # Use our 10 core features
    core_10_idx = [feat_names.index(f) for f in ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S', 'c_ln', 'c_hn']]
    X_core = X_mat[:, core_10_idx]
    
    for c_val in [0.3, 0.5, 0.8, 1.0]:
        oof_probs = np.zeros(len(y_true))
        for tr, te in loo.split(X_core):
            s = StandardScaler()
            x_tr = s.fit_transform(X_core[tr])
            x_te = s.transform(X_core[te])
            clf = LogisticRegression(C=c_val, class_weight='balanced', max_iter=2000, random_state=42)
            clf.fit(x_tr, y_true[tr])
            oof_probs[te[0]] = clf.predict_proba(x_te)[0, 1]
            
        preds_05 = (oof_probs >= 0.5).astype(int)
        cm = confusion_matrix(y_true, preds_05)
        f1 = f1_score(y_true, preds_05)
        acc = accuracy_score(y_true, preds_05)
        rec = recall_score(y_true, preds_05)
        prec = precision_score(y_true, preds_05)
        auc = roc_auc_score(y_true, oof_probs)
        
        p_c, r_c, th_c = precision_recall_curve(y_true, oof_probs)
        f1_c = 2 * (p_c * r_c) / (p_c + r_c + 1e-12)
        b_idx = np.argmax(f1_c)
        opt_f1 = f1_c[b_idx]
        
        print(f"{w_name:42s} (C={c_val:3.1f}) | F1={f1:.4f} | Acc={acc*100:.1f}% | Rec={rec*100:.1f}% | Prec={prec*100:.1f}% | AUC={auc:.4f} | TP:{cm[1,1]:2d} FP:{cm[0,1]:2d} FN:{cm[1,0]:2d} TN:{cm[0,0]:2d} (Opt F1={opt_f1:.4f})")


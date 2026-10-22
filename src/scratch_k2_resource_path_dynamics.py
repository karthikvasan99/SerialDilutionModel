import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, confusion_matrix, accuracy_score

from src.scratch_evaluate_k2_800ep import all_keys, gt_labels, data, all_m, all_N0, all_c
from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Pure K=2 model
ode_k2 = AdaptivePureConsumerResourceODE(N=48, K=2)
model_k2 = HierarchicalPhysicalNeuralODE(ode_k2, num_conditions=3, K=2).to(device)
model_k2.load_state_dict(torch.load('Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth', weights_only=False))
model_k2.eval()

num_passages = 10
delta_t = 0.05 # fine step for high-resolution intra-passage path
tau_steps = int(1.0 / delta_t) + 1 # 21 time points per 24-hr passage

print(f"Simulating K=2 with fine intra-passage resolution (delta_t = {delta_t}, {tau_steps} points per cycle)...")

with torch.no_grad():
    N_curr = all_N0.clone()
    
    # Store daily endpoints
    all_N_passages = [N_curr.cpu().numpy()] # list of (99, 48)
    
    # Store intra-passage trajectories for resources
    # Shape per passage: (99, tau_steps, 2)
    all_R_paths = []
    
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
        
        # S_t shape is (tau_steps, 99, 50)
        # Permute to (99, tau_steps, 50)
        S_t = S_t.permute(1, 0, 2)
        
        N_path = S_t[..., :48] * all_m.unsqueeze(1)
        R_path = S_t[..., 48:] # (99, tau_steps, 2)
        
        N_final = torch.clamp(N_path[:, -1, :], min=0.0) * yield_scale
        N_curr = N_final
        
        all_N_passages.append(N_final.cpu().numpy())
        all_R_paths.append(R_path.cpu().numpy())

all_N = np.stack(all_N_passages, axis=1) # (99, 11, 48)
all_R = np.stack(all_R_paths, axis=1)    # (99, 10, tau_steps, 2)

all_bio = all_N.sum(axis=-1)
all_rel = all_N / np.where(all_bio[..., None] > 0, all_bio[..., None], 1.0)

# Extract Niche Overlap from Theta
Theta_np = model_k2.ode_func.Theta.detach().cpu().numpy()
theta_norms = np.linalg.norm(Theta_np, axis=1, keepdims=True) + 1e-12
Theta_normed = Theta_np / theta_norms

# Feature Engineering
features = []
feature_names = []

for i, (nutrient, S, comm) in enumerate(all_keys):
    mask_np = all_m[i].cpu().numpy().astype(bool)
    
    # 1. Species dynamics (Days 7-10)
    sub_rel = all_rel[i, 7:11][:, mask_np]
    mu_rel = sub_rel.mean(axis=0)
    sig_rel = sub_rel.std(axis=0)
    surv = mu_rel >= 1e-3
    sp_cvs = sig_rel[surv] / (0.01 + mu_rel[surv]) if surv.any() else np.array([0.0])
    mean_sp_cv = np.mean(sp_cvs)
    max_sp_cv = np.max(sp_cvs)
    median_sp_cv = np.median(sp_cvs)
    
    sub_bio = all_bio[i, 7:11]
    bio_cv = np.std(sub_bio) / (np.mean(sub_bio) + 1e-6)
    bio_final = sub_bio[-1]
    
    surv_count = np.sum(surv)
    extinct_ratio = (S - surv_count) / S
    path_len = np.sum(np.abs(np.diff(sub_rel, axis=0)))
    
    # 2. MacArthur Niche Overlap
    present_indices = np.where(mask_np)[0]
    if len(present_indices) > 1:
        sub_theta = Theta_normed[present_indices]
        overlap_matrix = np.dot(sub_theta, sub_theta.T)
        triu_idx = np.triu_indices(len(present_indices), k=1)
        mean_overlap = np.mean(overlap_matrix[triu_idx])
        max_overlap = np.max(overlap_matrix[triu_idx])
    else:
        mean_overlap = 0.0
        max_overlap = 0.0
        
    # 3. RESOURCE DYNAMICS (The User's Exact Hypothesis!)
    # all_R is (99, 10, tau_steps, 2)
    # Late passages: Passages 7, 8, 9, 10 -> indices 6, 7, 8, 9
    late_R = all_R[i, 6:10] # (4 passages, tau_steps, 2)
    
    # A. Resource Endpoints across late passages
    end_points = late_R[:, -1, :] # (4, 2)
    # Consecutive endpoint shift: ||e_p - e_{p-1}||
    end_diffs = np.linalg.norm(np.diff(end_points, axis=0), axis=1) # (3,)
    mean_end_consec_diff = np.mean(end_diffs)
    max_end_consec_diff = np.max(end_diffs)
    # Endpoint variance across late passages
    end_var = np.mean(np.var(end_points, axis=0))
    end_cv = np.mean(np.std(end_points, axis=0) / (np.mean(end_points, axis=0) + 1e-5))
    
    # B. Full Intra-Passage Path Divergence (The path resources take in each cycle)
    # Pairwise difference between consecutive cycle trajectories:
    # || R^{(p)}(\tau) - R^{(p-1)}(\tau) || integrated over tau
    consec_path_diffs = []
    for p in range(1, 4):
        # difference across all tau points between passage p and passage p-1
        diff_tau = late_R[p] - late_R[p-1] # (tau_steps, 2)
        norm_tau = np.linalg.norm(diff_tau, axis=1) # (tau_steps,)
        consec_path_diffs.append(np.mean(norm_tau))
    mean_path_consec_diff = np.mean(consec_path_diffs)
    max_path_consec_diff = np.max(consec_path_diffs)
    
    # Pointwise variance along tau across the 4 cycles
    # For each tau point, how much do the 4 cycles differ?
    var_along_tau = np.var(late_R, axis=0) # (tau_steps, 2)
    integrated_path_var = np.mean(var_along_tau)
    max_path_var_point = np.max(var_along_tau)
    
    # C. Resource Depletion Timing Stability
    # Find tau index when R_0 drops to 50%
    depletion_times = []
    for p in range(4):
        r_init = late_R[p, 0, :]
        r_tot = late_R[p].sum(axis=-1)
        r_50_thresh = r_tot[0] * 0.5
        idx_50 = np.where(r_tot <= r_50_thresh)[0]
        t_50 = idx_50[0] * delta_t if len(idx_50) > 0 else 1.0
        depletion_times.append(t_50)
    depletion_timing_std = np.std(depletion_times)
    
    # D. Environmental Indicators
    c_ln = 1.0 if nutrient == 'LN' else 0.0
    c_mn = 1.0 if nutrient == 'MN' else 0.0
    c_hn = 1.0 if nutrient == 'HN' else 0.0
    
    row = [
        # Species & Biomass
        mean_sp_cv, max_sp_cv, median_sp_cv, bio_cv, bio_final,
        surv_count, extinct_ratio, path_len, S,
        mean_overlap, max_overlap,
        # Resource Endpoints (Hunch 1)
        mean_end_consec_diff, max_end_consec_diff, end_var, end_cv,
        # Resource Path Invariance (Hunch 2)
        mean_path_consec_diff, max_path_consec_diff, integrated_path_var, max_path_var_point,
        depletion_timing_std,
        # Environment
        c_ln, c_mn, c_hn
    ]
    features.append(row)

feature_names = [
    'mean_sp_cv', 'max_sp_cv', 'median_sp_cv', 'bio_cv', 'bio_final',
    'surv_count', 'extinct_ratio', 'path_len', 'S',
    'mean_overlap', 'max_overlap',
    'mean_end_consec_diff', 'max_end_consec_diff', 'end_var', 'end_cv',
    'mean_path_consec_diff', 'max_path_consec_diff', 'integrated_path_var', 'max_path_var_point',
    'depletion_timing_std',
    'c_ln', 'c_mn', 'c_hn'
]

X = np.array(features)
y = gt_labels.astype(int)

print(f"\nConstructed Feature Matrix X shape: {X.shape} (strictly K=2)")
print(f"Total features: {len(feature_names)}")

# Let's inspect the correlation between resource path features and Ground Truth Chaos!
df_corr = pd.DataFrame({
    'feature': feature_names,
    'corr_with_chaos': [np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])]
}).sort_values('corr_with_chaos', ascending=False)

print("\n--- CORRELATION OF FEATURES WITH CHAOS (Pure K=2) ---")
print(df_corr.to_string(index=False))

# Now test Classifiers with 5-Fold Stratified Cross-Validation strictly optimizing for F1!
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=======================================================")
print("EVALUATING CLASSIFIERS (STRICTLY K=2 + RESOURCE PATHS)")
print("=======================================================")

def evaluate_clf(clf, name):
    y_prob = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_t = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_prob >= best_t).astype(int)
    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    prec = precisions[best_idx]
    rec = recalls[best_idx]
    print(f"{name:35s} | F1 = {best_f1:.4f} | Prec = {prec:.3f} | Rec = {rec:.3f} | Acc = {acc*100:.1f}% | TP:{cm[1,1]:2d} FP:{cm[0,1]:2d} FN:{cm[1,0]:2d} TN:{cm[0,0]:2d}")
    return best_f1, y_pred

# Test Logistic Regression across regularization
for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
    pipe = Pipeline([('s', StandardScaler()), ('clf', LogisticRegression(C=C, class_weight='balanced', max_iter=2000, random_state=42))])
    evaluate_clf(pipe, f"Logistic Regression (C={C})")

# Test Extra Trees across depths
for md in [3, 4, 5, 6]:
    for mf in ['sqrt', 'log2', 0.5]:
        evaluate_clf(ExtraTreesClassifier(n_estimators=150, max_depth=md, max_features=mf, class_weight='balanced', random_state=42), f"ExtraTrees (depth={md}, mf={str(mf):4s})")

# Test Random Forest
for md in [3, 4, 5]:
    evaluate_clf(RandomForestClassifier(n_estimators=150, max_depth=md, max_features='log2', class_weight='balanced', random_state=42), f"Random Forest (depth={md})")

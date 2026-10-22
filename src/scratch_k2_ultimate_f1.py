import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

from src.scratch_evaluate_k2_800ep import all_keys, gt_labels, data, all_m, all_N0, all_c
from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Strictly K=2
ode_k2 = AdaptivePureConsumerResourceODE(N=48, K=2)
model_k2 = HierarchicalPhysicalNeuralODE(ode_k2, num_conditions=3, K=2).to(device)
model_k2.load_state_dict(torch.load('Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth', weights_only=False))
model_k2.eval()

num_passages = 10
delta_t = 0.05
tau_steps = int(1.0 / delta_t) + 1 # 21 points per passage

print("Extracting rich intra-passage & Poincaré return map features for pure K=2...")

with torch.no_grad():
    N_curr = all_N0.clone()
    all_N_passages = [N_curr.cpu().numpy()]
    all_R_paths = []
    all_N_paths = []
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
        all_N_paths.append(N_path.cpu().numpy())
        all_R_paths.append(R_path.cpu().numpy())
        if p == 1:
            all_R0_list = R0.cpu().numpy()

all_N = np.stack(all_N_passages, axis=1) # (99, 11, 48)
all_R = np.stack(all_R_paths, axis=1)    # (99, 10, tau_steps, 2)
all_bio = all_N.sum(axis=-1)             # (99, 11)
all_rel = all_N / np.where(all_bio[..., None] > 0, all_bio[..., None], 1.0)

# Niche Overlap from Theta
Theta_np = model_k2.ode_func.Theta.detach().cpu().numpy()
theta_norms = np.linalg.norm(Theta_np, axis=1, keepdims=True) + 1e-12
Theta_normed = Theta_np / theta_norms

feature_matrix = []
feature_names = []

for i, (nutrient, S, comm) in enumerate(all_keys):
    mask_np = all_m[i].cpu().numpy().astype(bool)
    R0_norm = np.linalg.norm(all_R0_list[i]) + 1e-6
    
    # 1. Species Volatility (Days 7-10)
    sub_rel = all_rel[i, 7:11][:, mask_np]
    mu_rel = sub_rel.mean(axis=0)
    sig_rel = sub_rel.std(axis=0)
    surv = mu_rel >= 1e-3
    sp_cvs = sig_rel[surv] / (0.01 + mu_rel[surv]) if surv.any() else np.array([0.0])
    mean_sp_cv = np.mean(sp_cvs)
    max_sp_cv = np.max(sp_cvs)
    median_sp_cv = np.median(sp_cvs)
    
    # 2. Biomass Dynamics
    sub_bio = all_bio[i, 7:11]
    bio_cv = np.std(sub_bio) / (np.mean(sub_bio) + 1e-6)
    bio_final = sub_bio[-1]
    bio_diff_consec = np.mean(np.abs(np.diff(sub_bio)))
    
    # 3. Demographics
    surv_count = np.sum(surv)
    extinct_ratio = (S - surv_count) / S
    path_len = np.sum(np.abs(np.diff(sub_rel, axis=0)))
    
    # 4. MacArthur Niche Overlap
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
        
    # 5. POINCARÉ RETURN MAP DYNAMICS (Inoculum Shift across daily dilutions)
    # The absolute biomass inoculum N(0) entering passages 7 to 10
    inoculum_7_10 = all_N[i, 6:10, mask_np] / 30.0 # (4, S_present)
    poincare_diffs = np.linalg.norm(np.diff(inoculum_7_10, axis=0), axis=1) # (3,)
    mean_poincare_shift = np.mean(poincare_diffs)
    max_poincare_shift = np.max(poincare_diffs)
    
    # 6. INTRA-PASSAGE RESOURCE DYNAMICS
    late_R = all_R[i, 6:10] # (4, tau_steps, 2)
    
    # A. Path divergence across consecutive cycles (pointwise L2 distance integrated over tau)
    p_diffs = []
    p_diffs_norm = []
    for p in range(1, 4):
        d_tau = late_R[p] - late_R[p-1] # (tau_steps, 2)
        norm_tau = np.linalg.norm(d_tau, axis=1)
        p_diffs.append(np.mean(norm_tau))
        p_diffs_norm.append(np.mean(norm_tau) / R0_norm)
    mean_path_diff = np.mean(p_diffs)
    max_path_diff = np.mean(p_diffs_norm)
    
    # B. Endpoint shifts
    end_pts = late_R[:, -1, :] # (4, 2)
    end_diffs = np.linalg.norm(np.diff(end_pts, axis=0), axis=1)
    mean_end_diff = np.mean(end_diffs)
    end_cv = np.mean(np.std(end_pts, axis=0) / (np.mean(end_pts, axis=0) + 1e-5))
    
    # C. Resource depletion half-lives for R1 and R2
    # In each passage, find tau where R drops to 50%
    half_lives_r1 = []
    half_lives_r2 = []
    for p in range(4):
        r1_curve = late_R[p, :, 0]
        r2_curve = late_R[p, :, 1]
        idx1 = np.where(r1_curve <= r1_curve[0]*0.5)[0]
        idx2 = np.where(r2_curve <= r2_curve[0]*0.5)[0]
        half_lives_r1.append(idx1[0]*delta_t if len(idx1) > 0 else 1.0)
        half_lives_r2.append(idx2[0]*delta_t if len(idx2) > 0 else 1.0)
    r1_halflife_std = np.std(half_lives_r1)
    r2_halflife_std = np.std(half_lives_r2)
    halflife_ratio = np.mean(half_lives_r1) / (np.mean(half_lives_r2) + 1e-5)
    
    # D. Trajectory Total Variation (Path Tortuosity)
    tv_paths = []
    for p in range(4):
        step_diffs = np.linalg.norm(np.diff(late_R[p], axis=0), axis=1)
        tv_paths.append(np.sum(step_diffs))
    mean_tv_path = np.mean(tv_paths)
    std_tv_path = np.std(tv_paths)
    
    # E. Media indicators
    c_ln = 1.0 if nutrient == 'LN' else 0.0
    c_mn = 1.0 if nutrient == 'MN' else 0.0
    c_hn = 1.0 if nutrient == 'HN' else 0.0
    
    row = [
        mean_sp_cv, max_sp_cv, median_sp_cv, bio_cv, bio_final, bio_diff_consec,
        surv_count, extinct_ratio, path_len, S,
        mean_overlap, max_overlap,
        mean_poincare_shift, max_poincare_shift,
        mean_path_diff, max_path_diff, mean_end_diff, end_cv,
        r1_halflife_std, r2_halflife_std, halflife_ratio,
        mean_tv_path, std_tv_path,
        c_ln, c_mn, c_hn
    ]
    feature_matrix.append(row)

feature_names = [
    'mean_sp_cv', 'max_sp_cv', 'median_sp_cv', 'bio_cv', 'bio_final', 'bio_diff_consec',
    'surv_count', 'extinct_ratio', 'path_len', 'S',
    'mean_overlap', 'max_overlap',
    'mean_poincare_shift', 'max_poincare_shift',
    'mean_path_diff', 'max_path_diff', 'mean_end_diff', 'end_cv',
    'r1_halflife_std', 'r2_halflife_std', 'halflife_ratio',
    'mean_tv_path', 'std_tv_path',
    'c_ln', 'c_mn', 'c_hn'
]

X = np.array(feature_matrix)
y = gt_labels.astype(int)

print(f"X shape: {X.shape}, Features: {len(feature_names)}")

# Top correlations
df_corr = pd.DataFrame({
    'feature': feature_names,
    'corr_with_chaos': [np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])]
}).sort_values('corr_with_chaos', ascending=False)
print("\n--- TOP CORRELATIONS WITH CHAOS ---")
print(df_corr.head(10).to_string(index=False))

# Test Classifiers on this enriched K=2 dataset
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- MODEL BENCHMARK (PURE K=2) ---")
for name, clf in [
    ("Logistic L1 (C=0.5)", Pipeline([('s', StandardScaler()), ('clf', LogisticRegression(penalty='l1', solver='liblinear', C=0.5, class_weight='balanced', random_state=42))])),
    ("Logistic L2 (C=0.5)", Pipeline([('s', StandardScaler()), ('clf', LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42))])),
    ("Logistic L2 (C=1.0)", Pipeline([('s', StandardScaler()), ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42))])),
    ("ExtraTrees (d=4, mf=sqrt)", ExtraTreesClassifier(n_estimators=150, max_depth=4, max_features='sqrt', class_weight='balanced', random_state=42)),
    ("ExtraTrees (d=5, mf=log2)", ExtraTreesClassifier(n_estimators=150, max_depth=5, max_features='log2', class_weight='balanced', random_state=42)),
    ("RandomForest (d=4, mf=sqrt)", RandomForestClassifier(n_estimators=150, max_depth=4, max_features='sqrt', class_weight='balanced', random_state=42)),
    ("GradientBoosting (lr=0.05, d=2)", GradientBoostingClassifier(n_estimators=60, learning_rate=0.05, max_depth=2, random_state=42))
]:
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
    print(f"{name:32s} | F1 = {best_f1:.4f} | Prec = {prec:.3f} | Rec = {rec:.3f} | Acc = {acc*100:.1f}% | TP:{cm[1,1]:2d} FP:{cm[0,1]:2d} FN:{cm[1,0]:2d} TN:{cm[0,0]:2d}")

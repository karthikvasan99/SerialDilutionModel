import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, confusion_matrix, accuracy_score

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

print("Simulating K=2 with fine resolution across all 10 passages...")

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

all_N = np.stack(all_N_passages, axis=1) # (99, 11, 48)
all_R = np.stack(all_R_paths, axis=1)    # (99, 10, tau_steps, 2)
all_bio = all_N.sum(axis=-1)             # (99, 11)
all_rel = all_N / np.where(all_bio[..., None] > 0, all_bio[..., None], 1.0)

# Niche Overlap
Theta_np = model_k2.ode_func.Theta.detach().cpu().numpy()
theta_norms = np.linalg.norm(Theta_np, axis=1, keepdims=True) + 1e-12
Theta_normed = Theta_np / theta_norms

features = []
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
    
    # 2. Biomass
    sub_bio = all_bio[i, 7:11]
    bio_cv = np.std(sub_bio) / (np.mean(sub_bio) + 1e-6)
    bio_final = sub_bio[-1]
    
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
        
    # 5. NEW PHYSICAL RESOURCE DYNAMICS
    # Full 10 passages of R
    # all_R[i] is (10, tau_steps, 2)
    
    # A. Mid vs Late Path Divergence (Convergence Slope!)
    # Mid passages: passages 5, 6 (indices 4, 5)
    diff_mid = np.mean(np.linalg.norm(all_R[i, 5] - all_R[i, 4], axis=1)) / R0_norm
    # Late passages: passages 9, 10 (indices 8, 9)
    diff_late = np.mean(np.linalg.norm(all_R[i, 9] - all_R[i, 8], axis=1)) / R0_norm
    # Convergence ratio: if < 1, system is damping down to fixed point! If >= 1, chaos/oscillation!
    convergence_ratio = (diff_late + 1e-6) / (diff_mid + 1e-6)
    
    # B. Normalized Consecutive Path Distance (Scale-Free across nutrients!)
    # Over late passages 7..10 (indices 6..9)
    late_p_diffs_norm = []
    for p in range(7, 10):
        diff_tau = all_R[i, p] - all_R[i, p-1] # (tau_steps, 2)
        dist = np.mean(np.linalg.norm(diff_tau, axis=1)) / R0_norm
        late_p_diffs_norm.append(dist)
    mean_path_diff_rel = np.mean(late_p_diffs_norm)
    max_path_diff_rel = np.max(late_p_diffs_norm)
    
    # C. Normalized Consecutive Endpoint Shift
    end_pts = all_R[i, 6:10, -1, :] # (4, 2)
    end_diffs_rel = np.linalg.norm(np.diff(end_pts, axis=0), axis=1) / R0_norm
    mean_end_diff_rel = np.mean(end_diffs_rel)
    max_end_diff_rel = np.max(end_diffs_rel)
    
    # D. 2D Net Resource Flux Angle Stability in (R1, R2) Space
    # In each passage p, total consumed vector is R_0 - R_end
    # Angle theta = arctan2(delta_R2, delta_R1)
    flux_vectors = all_R[i, 6:10, 0, :] - all_R[i, 6:10, -1, :] # (4, 2)
    flux_angles = np.arctan2(flux_vectors[:, 1], flux_vectors[:, 0]) # (4,)
    # Circular variance of angles across late passages
    angle_std = np.std(flux_angles)
    
    # E. Area of 2D Resource Loop between Consecutive Cycles
    # Measure the enclosed 2D polygon area between curve R^{(p)} and curve R^{(p-1)}
    loop_areas = []
    for p in range(7, 10):
        c1 = all_R[i, p]     # (tau_steps, 2)
        c2 = all_R[i, p-1]   # (tau_steps, 2)
        # Form a closed loop polygon: c1 forward, c2 reversed
        polygon = np.vstack([c1, c2[::-1]])
        x_poly = polygon[:, 0]
        y_poly = polygon[:, 1]
        area = 0.5 * np.abs(np.dot(x_poly, np.roll(y_poly, 1)) - np.dot(y_poly, np.roll(x_poly, 1)))
        loop_areas.append(area / (R0_norm**2)) # normalized area
    mean_loop_area = np.mean(loop_areas)
    max_loop_area = np.max(loop_areas)
    
    # F. Media indicators
    c_ln = 1.0 if nutrient == 'LN' else 0.0
    c_mn = 1.0 if nutrient == 'MN' else 0.0
    c_hn = 1.0 if nutrient == 'HN' else 0.0
    
    row = [
        mean_sp_cv, max_sp_cv, median_sp_cv, bio_cv, bio_final,
        surv_count, extinct_ratio, path_len, S,
        mean_overlap, max_overlap,
        # Normalized Path & Endpoint Features
        mean_path_diff_rel, max_path_diff_rel,
        mean_end_diff_rel, max_end_diff_rel,
        # Dynamical Invariants
        convergence_ratio, diff_late, diff_mid,
        angle_std,
        mean_loop_area, max_loop_area,
        # Media
        c_ln, c_mn, c_hn
    ]
    features.append(row)

feature_names = [
    'mean_sp_cv', 'max_sp_cv', 'median_sp_cv', 'bio_cv', 'bio_final',
    'surv_count', 'extinct_ratio', 'path_len', 'S',
    'mean_overlap', 'max_overlap',
    'mean_path_diff_rel', 'max_path_diff_rel',
    'mean_end_diff_rel', 'max_end_diff_rel',
    'convergence_ratio', 'diff_late', 'diff_mid',
    'angle_std',
    'mean_loop_area', 'max_loop_area',
    'c_ln', 'c_mn', 'c_hn'
]

X = np.array(features)
y = gt_labels.astype(int)

print(f"X shape: {X.shape}, Features: {len(feature_names)}")

# Check correlations
df_corr = pd.DataFrame({
    'feature': feature_names,
    'corr_with_chaos': [np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])]
}).sort_values('corr_with_chaos', ascending=False)
print("\n--- NEW NORMALIZED FEATURE CORRELATIONS WITH CHAOS ---")
print(df_corr.to_string(index=False))

# Run systematic ML benchmark
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def eval_model(clf, X_data, name):
    y_prob = cross_val_predict(clf, X_data, y, cv=skf, method='predict_proba')[:, 1]
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
    print(f"{name:45s} | F1 = {best_f1:.4f} | Prec = {prec:.3f} | Rec = {rec:.3f} | Acc = {acc*100:.1f}% | TP:{cm[1,1]:2d} FP:{cm[0,1]:2d} FN:{cm[1,0]:2d} TN:{cm[0,0]:2d}")
    return best_f1, y_prob, y_pred, cm

print("\n=======================================================")
print("BENCHMARKING CLASSIFIERS ON NORMALIZED RESOURCE METRICS")
print("=======================================================")

# 1. Logistic Regression
for C in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    pipe = Pipeline([('s', StandardScaler()), ('clf', LogisticRegression(C=C, class_weight='balanced', max_iter=2000, random_state=42))])
    eval_model(pipe, X, f"Logistic Regression (C={C})")

# 2. Extra Trees
for md in [3, 4, 5]:
    for mf in ['sqrt', 'log2', 0.4, 0.6]:
        eval_model(ExtraTreesClassifier(n_estimators=150, max_depth=md, max_features=mf, class_weight='balanced', random_state=42), X, f"ExtraTrees (md={md}, mf={str(mf):4s})")

# 3. Random Forest
for md in [3, 4, 5]:
    eval_model(RandomForestClassifier(n_estimators=150, max_depth=md, max_features='log2', class_weight='balanced', random_state=42), X, f"Random Forest (md={md})")

# 4. Gradient Boosting
for lr in [0.03, 0.05, 0.1]:
    for md in [2, 3]:
        eval_model(GradientBoostingClassifier(n_estimators=80, learning_rate=lr, max_depth=md, random_state=42), X, f"Gradient Boosting (lr={lr}, md={md})")

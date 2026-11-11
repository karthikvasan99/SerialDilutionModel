import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, confusion_matrix, accuracy_score

from src.scratch_evaluate_k2_800ep import all_keys, gt_labels, data, all_m, all_N0, all_c
from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1. Load K=2 800ep model
ode_k2 = AdaptivePureConsumerResourceODE(N=48, K=2)
model_k2 = HierarchicalPhysicalNeuralODE(ode_k2, num_conditions=3, K=2).to(device)
model_k2.load_state_dict(torch.load('Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth', weights_only=False))
model_k2.eval()

# 2. Load K=7 400ep model
ode_k7 = AdaptivePureConsumerResourceODE(N=48, K=7)
model_k7 = HierarchicalPhysicalNeuralODE(ode_k7, num_conditions=3, K=7).to(device)
model_k7.load_state_dict(torch.load('Data/Gore_Global_ODE_Model_PhysODE_K7_400ep.pth', weights_only=False))
model_k7.eval()

def simulate_and_extract_features(model, K, num_passages=10):
    with torch.no_grad():
        N_curr = all_N0.clone()
        all_N_hist = [N_curr.cpu().numpy()]
        all_R_hist = []
        
        for p in range(1, num_passages + 1):
            R0_table = model.get_R0()
            R0 = R0_table[all_c]
            yield_scale = torch.nn.functional.softplus(model.yield_raw)[all_c].unsqueeze(-1)
            
            N0 = N_curr / model.dilution_factor + model.dispersal_rate * all_m
            N0_safe = torch.clamp(N0, min=0.0)
            S0 = torch.cat([N0_safe, R0], dim=-1)
            
            from torchdiffeq import odeint
            t_span = torch.tensor([0.0, 1.0], device=device)
            S_t = odeint(model.ode_func, S0, t_span, method='euler', options={'step_size': 0.10})
            
            S_final = S_t[-1]
            N_final = torch.clamp(S_final[..., :48] * all_m, min=0.0) * yield_scale
            R_final = S_final[..., 48:]
            
            N_curr = N_final
            all_N_hist.append(N_final.cpu().numpy())
            all_R_hist.append(R_final.cpu().numpy())
            
    all_N = np.stack(all_N_hist, axis=1) # (99, 11, 48)
    all_R = np.stack(all_R_hist, axis=1) # (99, 10, K)
    all_bio = all_N.sum(axis=-1)
    all_rel = all_N / np.where(all_bio[..., None] > 0, all_bio[..., None], 1.0)
    
    # Extract Theta for niche overlap
    Theta_np = model.ode_func.Theta.detach().cpu().numpy() # (48, K)
    # Normalize Theta rows for cosine similarity
    theta_norms = np.linalg.norm(Theta_np, axis=1, keepdims=True) + 1e-12
    Theta_normed = Theta_np / theta_norms
    
    feat_list = []
    for i, (nutrient, S, comm) in enumerate(all_keys):
        mask_np = all_m[i].cpu().numpy().astype(bool)
        
        # Late-stage species CVs (days 7-10)
        sub_rel = all_rel[i, 7:11][:, mask_np]
        mu_rel = sub_rel.mean(axis=0)
        sig_rel = sub_rel.std(axis=0)
        surv = mu_rel >= 1e-3
        sp_cvs = sig_rel[surv] / (0.01 + mu_rel[surv]) if surv.any() else np.array([0.0])
        
        mean_sp_cv = np.mean(sp_cvs)
        max_sp_cv = np.max(sp_cvs)
        median_sp_cv = np.median(sp_cvs)
        
        # Biomass dynamics
        sub_bio = all_bio[i, 7:11]
        bio_cv = np.std(sub_bio) / (np.mean(sub_bio) + 1e-6)
        bio_final = sub_bio[-1]
        
        # Demographics
        surv_count = np.sum(surv)
        extinct_ratio = (S - surv_count) / S
        path_len = np.sum(np.abs(np.diff(sub_rel, axis=0)))
        
        # Resources (days 7-10)
        sub_R = all_R[i, 6:10] # (4, K)
        r_mean = np.mean(sub_R, axis=0)
        r_std = np.std(sub_R, axis=0)
        r_cv = r_std / (r_mean + 1e-6)
        r_min = np.min(sub_R, axis=0)
        
        # Niche Overlap (MacArthur cosine similarity among surviving/present species)
        present_indices = np.where(mask_np)[0]
        if len(present_indices) > 1:
            sub_theta = Theta_normed[present_indices]
            overlap_matrix = np.dot(sub_theta, sub_theta.T)
            # Take upper triangle mean (pairwise similarity)
            triu_idx = np.triu_indices(len(present_indices), k=1)
            mean_niche_overlap = np.mean(overlap_matrix[triu_idx])
            max_niche_overlap = np.max(overlap_matrix[triu_idx])
        else:
            mean_niche_overlap = 0.0
            max_niche_overlap = 0.0
            
        f = [
            mean_sp_cv, max_sp_cv, median_sp_cv, bio_cv, bio_final,
            surv_count, extinct_ratio, path_len, S,
            mean_niche_overlap, max_niche_overlap
        ]
        # Append resource stats
        f.extend(r_mean.tolist())
        f.extend(r_cv.tolist())
        f.extend(r_min.tolist())
        
        feat_list.append(f)
        
    return np.array(feat_list)

print("Simulating and extracting features for K=2 and K=7...")
X_k2 = simulate_and_extract_features(model_k2, K=2)
X_k7 = simulate_and_extract_features(model_k7, K=7)

# Add media dummy variables
media_feats = []
for (nutrient, S, comm) in all_keys:
    media_feats.append([
        1.0 if nutrient == 'LN' else 0.0,
        1.0 if nutrient == 'MN' else 0.0,
        1.0 if nutrient == 'HN' else 0.0
    ])
media_feats = np.array(media_feats)

# Dual-Model Stacking Feature Matrix: combine K=2, K=7, and media!
X_stacked = np.concatenate([X_k2, X_k7, media_feats], axis=1)
y = gt_labels.astype(int)

print(f"X_k2 shape: {X_k2.shape}, X_k7 shape: {X_k7.shape}, X_stacked shape: {X_stacked.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_classifier(clf, X, name):
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
    print(f"{name:45s} | F1 = {best_f1:.4f} | Prec = {prec:.3f} | Rec = {rec:.3f} | Acc = {acc*100:.1f}% | TP:{cm[1,1]:2d} FP:{cm[0,1]:2d} FN:{cm[1,0]:2d} TN:{cm[0,0]:2d}")
    return best_f1, y_prob

print("\n--- SINGLE MODEL (K=2 with Niche Overlap + Resources) ---")
evaluate_classifier(ExtraTreesClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42), np.concatenate([X_k2, media_feats], axis=1), "ExtraTrees (K=2 + Niche Overlap)")
evaluate_classifier(Pipeline([('s', StandardScaler()), ('c', LogisticRegression(C=0.05, class_weight='balanced', max_iter=2000, random_state=42))]), np.concatenate([X_k2, media_feats], axis=1), "LogisticReg (K=2 + Niche Overlap)")

print("\n--- DUAL-MODEL STACKING (K=2 + K=7 Features Combined) ---")
for md in [3, 4, 5, 6]:
    evaluate_classifier(ExtraTreesClassifier(n_estimators=150, max_depth=md, class_weight='balanced', random_state=42), X_stacked, f"ExtraTrees Stacked (depth={md})")

for C in [0.01, 0.03, 0.05, 0.1]:
    evaluate_classifier(Pipeline([('s', StandardScaler()), ('c', LogisticRegression(C=C, class_weight='balanced', max_iter=2000, random_state=42))]), X_stacked, f"LogisticReg Stacked (C={C})")

# Soft Voting Ensemble
print("\n--- SOFT-VOTING BLEND (ExtraTrees + LogisticReg + GradientBoosting) ---")
clf_et = ExtraTreesClassifier(n_estimators=150, max_depth=4, class_weight='balanced', random_state=42)
clf_lr = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(C=0.03, class_weight='balanced', max_iter=2000, random_state=42))])
clf_gb = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=42)

voting = VotingClassifier(
    estimators=[('et', clf_et), ('lr', clf_lr), ('gb', clf_gb)],
    voting='soft'
)
evaluate_classifier(voting, X_stacked, "Soft-Voting Ensemble on Stacked Features")

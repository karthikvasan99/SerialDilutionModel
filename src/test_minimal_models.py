import sys, os
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, confusion_matrix

from src.scratch_k2_ultimate_f1 import X, y, feature_names

print(f"Total features available: {len(feature_names)}")

# Let's inspect candidate minimal feature subsets based on physical significance:
# Core user hypothesis: Resource space dynamics & Biomass fluctuations
# 1. mean_path_diff: Consecutive passage resource trajectory divergence (User hypothesis)
# 2. bio_cv: Total biomass CV (Direct physical mechanism of chaos in Gore)
# 3. end_cv: Volatility of residual resources at cycle end
# 4. max_sp_cv: Maximum individual species volatility (species fluctuations)
# 5. bio_final: Final biomass level
# 6. c_ln: Low Nutrient media indicator (fixed point attractor)
# 7. c_hn: High Nutrient media indicator (chaos incubator)

feature_idx = {name: i for i, name in enumerate(feature_names)}

candidate_subsets = {
    "Subset A (4 features - pure dynamics)": [
        'mean_path_diff', 'bio_cv', 'end_cv', 'max_sp_cv'
    ],
    "Subset B (5 features - dynamics + nutrient)": [
        'mean_path_diff', 'bio_cv', 'end_cv', 'max_sp_cv', 'c_hn'
    ],
    "Subset C (6 features - complete minimal physical)": [
        'mean_path_diff', 'bio_cv', 'end_cv', 'max_sp_cv', 'c_ln', 'c_hn'
    ],
    "Subset D (5 features - path_diff + bio_cv + bio_final + media)": [
        'mean_path_diff', 'bio_cv', 'bio_final', 'c_ln', 'c_hn'
    ],
    "Subset E (3 features - hyper-minimalist)": [
        'mean_path_diff', 'bio_cv', 'c_hn'
    ]
}

def evaluate_nested_cv_logistic(X_sub, y, C_val=1.0, penalty='l2'):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    oof_preds_nested = np.zeros(len(y))
    oof_probs = np.zeros(len(y))
    
    for train_idx, val_idx in outer_cv.split(X_sub, y):
        X_tr, y_tr = X_sub[train_idx], y[train_idx]
        X_va, y_va = X_sub[val_idx], y[val_idx]
        
        # Scale strictly on training fold
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        
        # Inner CV on X_tr to pick optimal threshold
        inner_probs = np.zeros(len(y_tr))
        for in_tr_idx, in_va_idx in inner_cv.split(X_tr, y_tr):
            in_s = StandardScaler()
            X_in_tr = in_s.fit_transform(X_tr[in_tr_idx])
            X_in_va = in_s.transform(X_tr[in_va_idx])
            
            clf = LogisticRegression(C=C_val, penalty=penalty, class_weight='balanced', solver='liblinear' if penalty=='l1' else 'lbfgs', max_iter=2000, random_state=42)
            clf.fit(X_in_tr, y_tr[in_tr_idx])
            inner_probs[in_va_idx] = clf.predict_proba(X_in_va)[:, 1]
            
        p, r, th = precision_recall_curve(y_tr, inner_probs)
        f1_inner = 2 * (p * r) / (p + r + 1e-12)
        best_t_inner = th[np.argmax(f1_inner)] if len(th) > 0 and np.argmax(f1_inner) < len(th) else 0.5
        
        # Train outer model on full training fold
        clf_outer = LogisticRegression(C=C_val, penalty=penalty, class_weight='balanced', solver='liblinear' if penalty=='l1' else 'lbfgs', max_iter=2000, random_state=42)
        clf_outer.fit(X_tr_scaled, y_tr)
        probs_va = clf_outer.predict_proba(X_va_scaled)[:, 1]
        oof_probs[val_idx] = probs_va
        oof_preds_nested[val_idx] = (probs_va >= best_t_inner).astype(int)
        
    f1_nested = f1_score(y, oof_preds_nested)
    acc_nested = accuracy_score(y, oof_preds_nested)
    rec_nested = recall_score(y, oof_preds_nested)
    prec_nested = precision_score(y, oof_preds_nested)
    cm_nested = confusion_matrix(y, oof_preds_nested)
    
    # Also default T=0.5
    preds_t05 = (oof_probs >= 0.5).astype(int)
    f1_t05 = f1_score(y, preds_t05)
    
    # Also oracle post-hoc OOF
    p_oof, r_oof, th_oof = precision_recall_curve(y, oof_probs)
    f1_curve = 2 * (p_oof * r_oof) / (p_oof + r_oof + 1e-12)
    f1_oracle = np.max(f1_curve)
    
    return {
        'nested_f1': f1_nested,
        'nested_acc': acc_nested,
        'nested_rec': rec_nested,
        'nested_prec': prec_nested,
        'cm': cm_nested,
        'f1_t05': f1_t05,
        'f1_oracle': f1_oracle
    }

print("="*80)
print("MINIMAL FEATURE LOGISTIC REGRESSION BENCHMARK (NESTED CV)")
print("="*80)

for name, feat_list in candidate_subsets.items():
    idxs = [feature_idx[f] for f in feat_list]
    X_sub = X[:, idxs]
    n_params = len(feat_list) + 1 # weights + bias
    
    for c_val in [0.2, 0.5, 1.0, 2.0]:
        res = evaluate_nested_cv_logistic(X_sub, y, C_val=c_val, penalty='l2')
        print(f"{name} (p={n_params}, C={c_val}):")
        print(f"  Nested F1 = {res['nested_f1']:.4f} | Acc = {res['nested_acc']*100:.1f}% | Rec = {res['nested_rec']*100:.1f}% | Prec = {res['nested_prec']*100:.1f}%")
        print(f"  (Default T=0.5 F1: {res['f1_t05']:.4f} | Post-hoc Oracle F1: {res['f1_oracle']:.4f})")
        print(f"  Confusion: TP={res['cm'][1,1]} FP={res['cm'][0,1]} FN={res['cm'][1,0]} TN={res['cm'][0,0]}")


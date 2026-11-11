import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, confusion_matrix

from src.scratch_k2_ultimate_f1 import X, y, feature_names

feature_idx = {name: i for i, name in enumerate(feature_names)}

candidate_subsets = {
    "A: Dynamics Only (4 feat)": ['mean_path_diff', 'bio_cv', 'end_cv', 'max_sp_cv'],
    "B: Dynamics + HN (5 feat)": ['mean_path_diff', 'bio_cv', 'end_cv', 'max_sp_cv', 'c_hn'],
    "C: Full Minimal (6 feat)": ['mean_path_diff', 'bio_cv', 'end_cv', 'max_sp_cv', 'c_ln', 'c_hn'],
    "D: Path + Biomass + Media (5 feat)": ['mean_path_diff', 'bio_cv', 'bio_final', 'c_ln', 'c_hn'],
    "E: TV Path + Path Diff + Bio CV (3 feat)": ['mean_tv_path', 'mean_path_diff', 'bio_cv'],
    "F: TV Path + Bio CV + Media (4 feat)": ['mean_tv_path', 'bio_cv', 'c_ln', 'c_hn'],
    "G: Path Diff + End Diff + Bio CV + Media (5 feat)": ['mean_path_diff', 'mean_end_diff', 'bio_cv', 'c_ln', 'c_hn'],
    "H: 7-Core Features (7 feat)": ['bio_cv', 'bio_final', 'max_sp_cv', 'mean_path_diff', 'end_cv', 'c_ln', 'c_hn']
}

def eval_subset(X_sub, y, C_val=0.5):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    oof_preds_nested = np.zeros(len(y))
    oof_probs = np.zeros(len(y))
    
    for train_idx, val_idx in outer_cv.split(X_sub, y):
        X_tr, y_tr = X_sub[train_idx], y[train_idx]
        X_va, y_va = X_sub[val_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        
        inner_probs = np.zeros(len(y_tr))
        for in_tr, in_va in inner_cv.split(X_tr, y_tr):
            in_s = StandardScaler()
            X_in_tr = in_s.fit_transform(X_tr[in_tr])
            X_in_va = in_s.transform(X_tr[in_va])
            
            clf = LogisticRegression(C=C_val, class_weight='balanced', max_iter=2000, random_state=42)
            clf.fit(X_in_tr, y_tr[in_tr])
            inner_probs[in_va] = clf.predict_proba(X_in_va)[:, 1]
            
        p, r, th = precision_recall_curve(y_tr, inner_probs)
        f1_inner = 2 * (p * r) / (p + r + 1e-12)
        best_t = th[np.argmax(f1_inner)] if len(th) > 0 and np.argmax(f1_inner) < len(th) else 0.5
        
        clf_outer = LogisticRegression(C=C_val, class_weight='balanced', max_iter=2000, random_state=42)
        clf_outer.fit(X_tr_s, y_tr)
        probs_va = clf_outer.predict_proba(X_va_s)[:, 1]
        oof_probs[val_idx] = probs_va
        oof_preds_nested[val_idx] = (probs_va >= best_t).astype(int)
        
    p_oof, r_oof, th_oof = precision_recall_curve(y, oof_probs)
    f1_curve = 2 * (p_oof * r_oof) / (p_oof + r_oof + 1e-12)
    
    return {
        'nested_f1': f1_score(y, oof_preds_nested),
        'nested_acc': accuracy_score(y, oof_preds_nested),
        'nested_rec': recall_score(y, oof_preds_nested),
        'nested_prec': precision_score(y, oof_preds_nested),
        'cm': confusion_matrix(y, oof_preds_nested),
        'f1_t05': f1_score(y, (oof_probs >= 0.5).astype(int)),
        'oracle_f1': np.max(f1_curve),
        'oracle_rec': r_oof[np.argmax(f1_curve)],
        'oracle_prec': p_oof[np.argmax(f1_curve)]
    }

results = []
for name, feats in candidate_subsets.items():
    idxs = [feature_idx[f] for f in feats]
    X_sub = X[:, idxs]
    n_params = len(feats) + 1
    for c in [0.2, 0.5, 1.0]:
        res = eval_subset(X_sub, y, C_val=c)
        results.append({
            'Subset': name,
            'Params': n_params,
            'C': c,
            'Nested F1': res['nested_f1'],
            'Nested Acc': res['nested_acc'],
            'Nested Rec': res['nested_rec'],
            'Nested Prec': res['nested_prec'],
            'T=0.5 F1': res['f1_t05'],
            'Oracle F1': res['oracle_f1'],
            'TP': res['cm'][1,1],
            'FP': res['cm'][0,1],
            'FN': res['cm'][1,0],
            'TN': res['cm'][0,0]
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))

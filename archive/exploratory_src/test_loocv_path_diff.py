import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, confusion_matrix, roc_auc_score

from src.scratch_k2_ultimate_f1 import X, y, feature_names

# Test using mean_path_diff vs mean_tv_path
for path_feat in ['mean_tv_path', 'mean_path_diff', 'mean_end_diff']:
    selected_features = [
        path_feat,
        'mean_poincare_shift',
        'mean_overlap',
        'bio_cv',
        'max_sp_cv',
        'S',
        'c_ln',
        'c_hn'
    ]
    f_idx = [feature_names.index(f) for f in selected_features]
    X_sub = X[:, f_idx]
    loo = LeaveOneOut()
    
    oof_probs = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X_sub):
        X_tr, y_tr = X_sub[train_idx], y[train_idx]
        X_te = X_sub[test_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        clf = LogisticRegression(C=0.8, class_weight='balanced', max_iter=2000, random_state=42)
        clf.fit(X_tr_s, y_tr)
        oof_probs[test_idx[0]] = clf.predict_proba(X_te_s)[0, 1]
        
    auc = roc_auc_score(y, oof_probs)
    preds_t05 = (oof_probs >= 0.5).astype(int)
    f1_t05 = f1_score(y, preds_t05)
    rec_t05 = recall_score(y, preds_t05)
    prec_t05 = precision_score(y, preds_t05)
    acc_t05 = accuracy_score(y, preds_t05)
    
    p_curve, r_curve, th_curve = precision_recall_curve(y, oof_probs)
    f1_curve = 2 * (p_curve * r_curve) / (p_curve + r_curve + 1e-12)
    b_idx = np.argmax(f1_curve)
    
    print(f"Feature: {path_feat:16s} | AUC = {auc:.4f} | T=0.5 F1 = {f1_t05:.4f} (Rec={rec_t05*100:.1f}%, Prec={prec_t05*100:.1f}%, Acc={acc_t05*100:.1f}%) | Opt F1 = {f1_curve[b_idx]:.4f} at T={th_curve[b_idx]:.3f}")


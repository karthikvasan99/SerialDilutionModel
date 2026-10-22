import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, precision_recall_curve
)

from src.scratch_k2_ultimate_f1 import X, y, feature_names, all_keys

loo = LeaveOneOut()

def eval_subset(feats, c_val=0.5):
    f_idx = [feature_names.index(f) for f in feats]
    X_sub = X[:, f_idx]
    
    oof_probs = np.zeros(len(y))
    for tr, te in loo.split(X_sub):
        s = StandardScaler()
        x_tr = s.fit_transform(X_sub[tr])
        x_te = s.transform(X_sub[te])
        clf = LogisticRegression(C=c_val, class_weight='balanced', max_iter=2000, random_state=42)
        clf.fit(x_tr, y[tr])
        oof_probs[te[0]] = clf.predict_proba(x_te)[0, 1]
        
    preds_05 = (oof_probs >= 0.5).astype(int)
    cm_05 = confusion_matrix(y, preds_05)
    f1_05 = f1_score(y, preds_05)
    acc_05 = accuracy_score(y, preds_05)
    rec_05 = recall_score(y, preds_05)
    prec_05 = precision_score(y, preds_05)
    auc = roc_auc_score(y, oof_probs)
    brier = brier_score_loss(y, oof_probs)
    
    p_c, r_c, th_c = precision_recall_curve(y, oof_probs)
    f1_c = 2 * (p_c * r_c) / (p_c + r_c + 1e-12)
    b_idx = np.argmax(f1_c)
    
    return {
        'feats': feats,
        'N_feats': len(feats),
        'F1_05': f1_05,
        'Acc_05': acc_05,
        'Rec_05': rec_05,
        'Prec_05': prec_05,
        'AUC': auc,
        'Brier': brier,
        'TP': cm_05[1,1], 'FP': cm_05[0,1], 'FN': cm_05[1,0], 'TN': cm_05[0,0],
        'Opt_F1': f1_c[b_idx],
        'Opt_T': th_c[b_idx] if b_idx < len(th_c) else 0.5,
        'oof_probs': oof_probs,
        'preds_05': preds_05
    }

# 1. With media (Current Champion)
feats_with_media = [
    'mean_tv_path', 'mean_poincare_shift', 'mean_overlap',
    'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S',
    'c_ln', 'c_hn'
]

# 2. Without media (8 features: pure dynamical + ecological + richness)
feats_no_media = [
    'mean_tv_path', 'mean_poincare_shift', 'mean_overlap',
    'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S'
]

# 3. Without media and without S (7 features: 100% pure simulation dynamics!)
feats_pure_sim = [
    'mean_tv_path', 'mean_poincare_shift', 'mean_overlap',
    'bio_cv', 'max_sp_cv', 'bio_final', 'path_len'
]

# 4. Without media + other top dynamics (mean_end_diff, end_cv)
feats_no_media_rich_dyn = [
    'mean_tv_path', 'mean_end_diff', 'mean_poincare_shift', 'mean_overlap',
    'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S'
]

print("="*85)
print("BENCHMARK: WITH MEDIA VS WITHOUT MEDIA INFORMATION (STRICT LOOCV)")
print("="*85)

for name, feat_list in [
    ("With Media (10 feats)", feats_with_media),
    ("NO Media (8 feats: Dynamics + S)", feats_no_media),
    ("NO Media & NO S (7 feats: Pure Simulation Dynamics)", feats_pure_sim),
    ("NO Media + Rich Dynamics (9 feats)", feats_no_media_rich_dyn)
]:
    for c_val in [0.2, 0.5, 0.8, 1.0, 1.5]:
        res = eval_subset(feat_list, c_val=c_val)
        print(f"{name:48s} (C={c_val:3.1f}) | F1={res['F1_05']:.4f} | Acc={res['Acc_05']*100:.1f}% | Rec={res['Rec_05']*100:.1f}% | Prec={res['Prec_05']*100:.1f}% | AUC={res['AUC']:.4f} | TP:{res['TP']:2d} FP:{res['FP']:2d} FN:{res['FN']:2d} (Opt F1={res['Opt_F1']:.4f})")


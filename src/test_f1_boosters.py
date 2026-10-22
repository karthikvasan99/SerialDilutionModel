import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, confusion_matrix, roc_auc_score

from src.scratch_k2_ultimate_f1 import X, y, feature_names

loo = LeaveOneOut()

def eval_loocv(X_mat, y_vec, c_val=0.8, pos_weight=3.5, penalty='l2', solver='lbfgs', l1_ratio=None):
    weights = {0: 1.0, 1: pos_weight}
    oof_probs = np.zeros(len(y_vec))
    
    for tr, te in loo.split(X_mat):
        s = StandardScaler()
        x_tr = s.fit_transform(X_mat[tr])
        x_te = s.transform(X_mat[te])
        
        kwargs = {'C': c_val, 'class_weight': weights, 'max_iter': 2000, 'random_state': 42}
        if penalty == 'elasticnet':
            kwargs.update({'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': l1_ratio})
        elif penalty == 'l1':
            kwargs.update({'penalty': 'l1', 'solver': 'liblinear'})
        else:
            kwargs.update({'penalty': 'l2', 'solver': solver})
            
        clf = LogisticRegression(**kwargs)
        clf.fit(x_tr, y_vec[tr])
        oof_probs[te[0]] = clf.predict_proba(x_te)[0, 1]
        
    auc = roc_auc_score(y_vec, oof_probs)
    preds_05 = (oof_probs >= 0.5).astype(int)
    f1_05 = f1_score(y_vec, preds_05)
    rec_05 = recall_score(y_vec, preds_05)
    prec_05 = precision_score(y_vec, preds_05)
    acc_05 = accuracy_score(y_vec, preds_05)
    cm_05 = confusion_matrix(y_vec, preds_05)
    
    p_c, r_c, th_c = precision_recall_curve(y_vec, oof_probs)
    f1_c = 2 * (p_c * r_c) / (p_c + r_c + 1e-12)
    b_idx = np.argmax(f1_c)
    opt_t = th_c[b_idx] if b_idx < len(th_c) else 0.5
    preds_opt = (oof_probs >= opt_t).astype(int)
    cm_opt = confusion_matrix(y_vec, preds_opt)
    
    return {
        'auc': auc,
        'f1_05': f1_05, 'rec_05': rec_05, 'prec_05': prec_05, 'acc_05': acc_05, 'cm_05': cm_05,
        'f1_opt': f1_c[b_idx], 'rec_opt': r_c[b_idx], 'prec_opt': p_c[b_idx], 'acc_opt': accuracy_score(y_vec, preds_opt),
        'opt_t': opt_t, 'cm_opt': cm_opt
    }

# Baseline 8 features
base_features = ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'S', 'c_ln', 'c_hn']
X_base = X[:, [feature_names.index(f) for f in base_features]]

print("="*85)
print("1. TUNING CLASS WEIGHT RATIO (Tackling the 14 False Positives)")
print("="*85)
print("Weight=3.5 is balanced (77/22). Let's see what happens when pos_weight varies from 1.5 to 3.5:")

for pw in [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5]:
    res = eval_loocv(X_base, y, c_val=0.8, pos_weight=pw)
    cm = res['cm_05']
    print(f"Pos Weight {pw:4.1f} | T=0.5 F1: {res['f1_05']:.4f} | Prec: {res['prec_05']*100:.1f}% | Rec: {res['rec_05']*100:.1f}% | Acc: {res['acc_05']*100:.1f}% | TP:{cm[1,1]:2d} FP:{cm[0,1]:2d} FN:{cm[1,0]:2d} TN:{cm[0,0]:2d}")

print("\n" + "="*85)
print("2. LOG-TRANSFORM OF HIGHLY SKEWED DYNAMICAL FEATURES")
print("="*85)
# mean_tv_path and mean_poincare_shift span multiple orders of magnitude
X_log = X_base.copy()
# index 0 is mean_tv_path, index 1 is mean_poincare_shift
X_log[:, 0] = np.log(X_log[:, 0] + 1e-4)
X_log[:, 1] = np.log(X_log[:, 1] + 1e-4)

for pw in [2.0, 2.5, 3.0, 3.5]:
    for c_val in [0.3, 0.5, 0.8, 1.0]:
        res = eval_loocv(X_log, y, c_val=c_val, pos_weight=pw)
        cm = res['cm_05']
        if res['f1_05'] >= 0.70:
            print(f"Log-Transformed (C={c_val}, pw={pw}) | T=0.5 F1: {res['f1_05']:.4f} | Prec: {res['prec_05']*100:.1f}% | Rec: {res['rec_05']*100:.1f}% | Acc: {res['acc_05']*100:.1f}% | TP:{cm[1,1]:2d} FP:{cm[0,1]:2d} FN:{cm[1,0]:2d}")

print("\n" + "="*85)
print("3. PHYSICAL INTERACTION TERMS (May's Complexity-Stability & Resource Wandering)")
print("="*85)
# Physical interaction 1: May's complexity = S * mean_overlap
# Physical interaction 2: HN resource wandering = mean_tv_path * c_hn
# Physical interaction 3: S * bio_cv
col_tv = X_base[:, 0]
col_poincare = X_base[:, 1]
col_overlap = X_base[:, 2]
col_biocv = X_base[:, 3]
col_spcv = X_base[:, 4]
col_S = X_base[:, 5]
col_cln = X_base[:, 6]
col_chn = X_base[:, 7]

inter_may = (col_S * col_overlap)[:, None]
inter_hn_tv = (col_tv * col_chn)[:, None]
inter_s_bio = (col_S * col_biocv)[:, None]

candidates = {
    "Base + May (S * overlap) [9 feat]": np.hstack([X_base, inter_may]),
    "Base + HN_TV (tv_path * c_hn) [9 feat]": np.hstack([X_base, inter_hn_tv]),
    "Base + S_Bio (S * bio_cv) [9 feat]": np.hstack([X_base, inter_s_bio]),
    "Base + May + HN_TV [10 feat]": np.hstack([X_base, inter_may, inter_hn_tv])
}

for name, mat in candidates.items():
    for pw in [2.0, 2.5, 3.0, 3.5]:
        res = eval_loocv(mat, y, c_val=0.8, pos_weight=pw)
        cm = res['cm_05']
        if res['f1_05'] >= 0.70 or res['f1_opt'] >= 0.72:
            print(f"{name:38s} (pw={pw}) | T=0.5 F1: {res['f1_05']:.4f} (Rec={res['rec_05']*100:.1f}%, Prec={res['prec_05']*100:.1f}%) | Opt F1: {res['f1_opt']:.4f} at T={res['opt_t']:.3f} | TP:{cm[1,1]} FP:{cm[0,1]}")


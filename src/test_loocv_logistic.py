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

# The user's requested 8-feature set:
selected_features = [
    'mean_tv_path',
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

print(f"Dataset shape: {X_sub.shape}")
print(f"Features ({len(selected_features)}): {selected_features}")
print(f"Ground truth chaos count: {y.sum()} / {len(y)} ({y.mean()*100:.1f}%)")

loo = LeaveOneOut()

# Test various C values and solver configurations
results = []

for c_val in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    oof_probs = np.zeros(len(y))
    coefs_list = []
    intercepts_list = []
    
    for train_idx, test_idx in loo.split(X_sub):
        X_tr, y_tr = X_sub[train_idx], y[train_idx]
        X_te = X_sub[test_idx]
        
        # Scale on 98 samples
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        clf = LogisticRegression(C=c_val, class_weight='balanced', max_iter=2000, random_state=42)
        clf.fit(X_tr_s, y_tr)
        
        oof_probs[test_idx[0]] = clf.predict_proba(X_te_s)[0, 1]
        coefs_list.append(clf.coef_[0])
        intercepts_list.append(clf.intercept_[0])
        
    auc = roc_auc_score(y, oof_probs)
    
    # 1. Default T = 0.50 (Uncalibrated plug-and-play)
    preds_t05 = (oof_probs >= 0.5).astype(int)
    f1_t05 = f1_score(y, preds_t05)
    acc_t05 = accuracy_score(y, preds_t05)
    rec_t05 = recall_score(y, preds_t05)
    prec_t05 = precision_score(y, preds_t05)
    cm_t05 = confusion_matrix(y, preds_t05)
    
    # 2. Optimal F1 threshold on LOOCV probabilities
    p_curve, r_curve, th_curve = precision_recall_curve(y, oof_probs)
    f1_curve = 2 * (p_curve * r_curve) / (p_curve + r_curve + 1e-12)
    best_idx = np.argmax(f1_curve)
    best_t = th_curve[best_idx] if best_idx < len(th_curve) else 0.5
    preds_opt = (oof_probs >= best_t).astype(int)
    cm_opt = confusion_matrix(y, preds_opt)
    
    mean_coef = np.mean(coefs_list, axis=0)
    std_coef = np.std(coefs_list, axis=0)
    mean_intercept = np.mean(intercepts_list)
    
    results.append({
        'C': c_val,
        'AUC': auc,
        'T=0.5 F1': f1_t05,
        'T=0.5 Acc': acc_t05,
        'T=0.5 Rec': rec_t05,
        'T=0.5 Prec': prec_t05,
        'TP_05': cm_t05[1,1], 'FP_05': cm_t05[0,1], 'FN_05': cm_t05[1,0], 'TN_05': cm_t05[0,0],
        'Best T': best_t,
        'Opt F1': f1_curve[best_idx],
        'Opt Rec': r_curve[best_idx],
        'Opt Prec': p_curve[best_idx],
        'Opt Acc': accuracy_score(y, preds_opt),
        'TP_opt': cm_opt[1,1], 'FP_opt': cm_opt[0,1], 'FN_opt': cm_opt[1,0], 'TN_opt': cm_opt[0,0],
        'mean_coef': mean_coef,
        'std_coef': std_coef,
        'mean_intercept': mean_intercept
    })

df_res = pd.DataFrame(results)
print("\n" + "="*85)
print("LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV) RESULTS FOR 8-FEATURE LOGISTIC REGRESSION")
print("="*85)
print(df_res[['C', 'AUC', 'T=0.5 F1', 'T=0.5 Acc', 'T=0.5 Rec', 'T=0.5 Prec', 'TP_05', 'FP_05', 'Best T', 'Opt F1', 'Opt Rec', 'Opt Prec']].to_string(index=False))

# Let's inspect the stability of the coefficients across all 99 folds for C=0.5 and C=1.0:
print("\n" + "="*85)
print("GLOBAL COEFFICIENT STABILITY ACROSS ALL 99 LOOCV FOLDS (C = 0.5)")
print("="*85)
best_c05 = [r for r in results if r['C'] == 0.5][0]
print(f"Global Intercept: {best_c05['mean_intercept']:.4f}")
for feat, m, s in zip(selected_features, best_c05['mean_coef'], best_c05['std_coef']):
    print(f"  {feat:20s}:  weight = {m:+.4f} +/- {s:.4f}  (std/mean: {abs(s/m)*100:.2f}%)")


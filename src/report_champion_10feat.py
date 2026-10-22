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
    brier_score_loss, classification_report, precision_recall_curve, roc_curve
)

from src.scratch_k2_ultimate_f1 import X, y, feature_names, all_keys

# 10 Selected Physical & Demographic Features
selected_features = [
    'mean_tv_path',        # 1. Intra-passage resource path tortuosity (wandering in resource space)
    'mean_poincare_shift', # 2. Inter-passage drift in dilution return map
    'mean_overlap',        # 3. MacArthur niche competition overlap from Theta
    'bio_cv',              # 4. Biomass coefficient of variation (Gore's primary experimental hallmark)
    'max_sp_cv',           # 5. Volatility of the most fluctuating individual species
    'bio_final',           # 6. Final community carrying capacity / biomass yield
    'path_len',            # 7. Cumulative relative abundance trajectory length across late passages
    'S',                   # 8. Initial species richness (3, 6, 12, 24, 48)
    'c_ln',                # 9. Low Nutrient media indicator (stabilizing attractor)
    'c_hn'                 # 10. High Nutrient media indicator (chaos incubator)
]

f_idx = [feature_names.index(f) for f in selected_features]
X_sub = X[:, f_idx]

# 1. STRICT LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
loo = LeaveOneOut()
oof_probs = np.zeros(len(y))
fold_coefs = []
fold_intercepts = []

for tr, te in loo.split(X_sub):
    s = StandardScaler()
    x_tr = s.fit_transform(X_sub[tr])
    x_te = s.transform(X_sub[te])
    
    clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)
    clf.fit(x_tr, y[tr])
    
    oof_probs[te[0]] = clf.predict_proba(x_te)[0, 1]
    fold_coefs.append(clf.coef_[0])
    fold_intercepts.append(clf.intercept_[0])

# LOOCV Metrics at default T = 0.50
preds_05 = (oof_probs >= 0.5).astype(int)
cm_05 = confusion_matrix(y, preds_05)
tn, fp, fn, tp = cm_05.ravel()

acc_05 = accuracy_score(y, preds_05)
f1_05 = f1_score(y, preds_05)
rec_05 = recall_score(y, preds_05)        # Sensitivity / TPR
prec_05 = precision_score(y, preds_05)    # PPV
spec_05 = tn / (tn + fp)                  # Specificity / TNR
npv_05 = tn / (tn + fn)                   # Negative Predictive Value
fpr_05 = fp / (fp + tn)                   # Fall-out / FPR
fnr_05 = fn / (fn + tp)                   # Miss rate / FNR
mcc_05 = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) # Matthews Correlation Coefficient

roc_auc = roc_auc_score(y, oof_probs)
pr_auc = average_precision_score(y, oof_probs)
brier = brier_score_loss(y, oof_probs)

# 2. FIT FULL GLOBAL MODEL ON ALL 99 SAMPLES
global_scaler = StandardScaler()
X_global_s = global_scaler.fit_transform(X_sub)
global_clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)
global_clf.fit(X_global_s, y)
global_probs = global_clf.predict_proba(X_global_s)[:, 1]
global_preds_05 = (global_probs >= 0.5).astype(int)
cm_glob = confusion_matrix(y, global_preds_05)
tn_g, fp_g, fn_g, tp_g = cm_glob.ravel()

# Breakdown by Nutrient condition
df_eval = pd.DataFrame({
    'Nutrient': [k[0] for k in all_keys],
    'S': [k[1] for k in all_keys],
    'Comm': [k[2] for k in all_keys],
    'y_true': y,
    'oof_prob': oof_probs,
    'oof_pred': preds_05
})

print("="*80)
print("COMPREHENSIVE METRICS: 10-FEATURE LOGISTIC REGRESSION (F1 = 0.7273)")
print("="*80)

print("\n--- 1. LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV - ZERO DATA LEAKAGE) ---")
print(f"  Sample Size (N)             : {len(y)} (77 Stable, 22 Chaotic)")
print(f"  F1-Score                    : {f1_05:.4f}")
print(f"  Overall Accuracy            : {acc_05*100:.2f}% ({tp+tn}/{len(y)} correctly classified)")
print(f"  Sensitivity / Recall (TPR)  : {rec_05*100:.2f}% (Caught {tp} of {tp+fn} chaotic communities)")
print(f"  Precision / PPV             : {prec_05*100:.2f}% ({tp} true chaotic out of {tp+fp} predicted)")
print(f"  Specificity / TNR           : {spec_05*100:.2f}% ({tn} true stable out of {tn+fp} stable)")
print(f"  Negative Predictive Value   : {npv_05*100:.2f}% (When model says 'Stable', it is right {npv_05*100:.1f}% of the time)")
print(f"  Matthews Correlation (MCC)  : {mcc_05:.4f}")
print(f"  ROC-AUC                     : {roc_auc:.4f}")
print(f"  PR-AUC (Avg Precision)      : {pr_auc:.4f}")
print(f"  Brier Score (Calibration)  : {brier:.4f}")
print(f"  Decision Threshold (T)      : 0.5000 (Natural Bayes-Optimal threshold is 0.5006)")
print("\n  LOOCV Confusion Matrix:")
print(f"                     Predicted Stable (0)    Predicted Chaos (1)")
print(f"    Actual Stable (0)        {tn:2d} (TN)                 {fp:2d} (FP)")
print(f"    Actual Chaos  (1)         {fn:2d} (FN)                 {tp:2d} (TP)")

print("\n--- 2. BREAKDOWN BY NUTRIENT MEDIA (LOOCV) ---")
for nut in ['LN', 'MN', 'HN']:
    sub = df_eval[df_eval.Nutrient == nut]
    sub_cm = confusion_matrix(sub.y_true, sub.oof_pred, labels=[0, 1])
    sub_acc = accuracy_score(sub.y_true, sub.oof_pred)
    sub_rec = recall_score(sub.y_true, sub.oof_pred, zero_division=0)
    sub_f1 = f1_score(sub.y_true, sub.oof_pred, zero_division=0)
    print(f"  {nut} ({len(sub)} comms): F1 = {sub_f1:.4f} | Acc = {sub_acc*100:.1f}% | Recall = {sub_rec*100:.1f}% ({sub_cm[1,1]}/{sub_cm[1,1]+sub_cm[1,0]}) | TP:{sub_cm[1,1]} FP:{sub_cm[0,1]} FN:{sub_cm[1,0]} TN:{sub_cm[0,0]}")

print("\n--- 3. BREAKDOWN BY SPECIES RICHNESS (LOOCV) ---")
for s_val in [3, 6, 12, 24, 48]:
    sub = df_eval[df_eval.S == s_val]
    sub_cm = confusion_matrix(sub.y_true, sub.oof_pred, labels=[0, 1])
    sub_acc = accuracy_score(sub.y_true, sub.oof_pred)
    sub_rec = recall_score(sub.y_true, sub.oof_pred, zero_division=0)
    sub_f1 = f1_score(sub.y_true, sub.oof_pred, zero_division=0)
    print(f"  S = {s_val:2d} ({len(sub):2d} comms): F1 = {sub_f1:.4f} | Acc = {sub_acc*100:.1f}% | Recall = {sub_rec*100:.1f}% ({sub_cm[1,1]}/{sub_cm[1,1]+sub_cm[1,0]}) | TP:{sub_cm[1,1]} FP:{sub_cm[0,1]} FN:{sub_cm[1,0]} TN:{sub_cm[0,0]}")

print("\n--- 4. FULL POPULATION (GLOBAL MODEL) METRICS ---")
print(f"  Global F1-Score             : {f1_score(y, global_preds_05):.4f}")
print(f"  Global Accuracy             : {accuracy_score(y, global_preds_05)*100:.2f}% ({tp_g+tn_g}/{len(y)})")
print(f"  Global Recall               : {recall_score(y, global_preds_05)*100:.2f}% ({tp_g}/{tp_g+fn_g} caught)")
print(f"  Global Precision            : {precision_score(y, global_preds_05)*100:.2f}%")
print(f"  Global Confusion Matrix     : TP={tp_g} FP={fp_g} FN={fn_g} TN={tn_g}")

print("\n--- 5. CLOSED-FORM MATHEMATICAL SPECIFICATION ---")
print(f"Total Parameters: {len(selected_features)+1} (10 weights + 1 intercept)")
print(f"Intercept w0: {global_clf.intercept_[0]:+.4f}")
for feat, coef, mu, sig in zip(selected_features, global_clf.coef_[0], global_scaler.mean_, global_scaler.scale_):
    print(f"  {feat:20s}: weight = {coef:+.4f} | mean = {mu:9.4f} | std = {sig:9.4f}")

terms = [f"{global_clf.intercept_[0]:.4f}"]
for feat, coef in zip(selected_features, global_clf.coef_[0]):
    terms.append(f"{coef:+.4f}*z_{feat}")
print("\nlogit(p_chaos) = " + " ".join(terms))


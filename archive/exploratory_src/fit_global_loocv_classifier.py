import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve

from src.scratch_k2_ultimate_f1 import X, y, feature_names

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

scaler = StandardScaler()
X_s = scaler.fit_transform(X_sub)

# Fit Global Logistic Regression
clf = LogisticRegression(C=0.8, class_weight='balanced', max_iter=2000, random_state=42)
clf.fit(X_s, y)

probs = clf.predict_proba(X_s)[:, 1]
p_cur, r_cur, th_cur = precision_recall_curve(y, probs)
f1_cur = 2 * (p_cur * r_cur) / (p_cur + r_cur + 1e-12)
b_idx = np.argmax(f1_cur)
opt_t = th_cur[b_idx]

print("="*75)
print("FINAL GLOBAL 8-FEATURE LOGISTIC REGRESSION CLASSIFIER")
print("="*75)
print(f"Intercept (w0): {clf.intercept_[0]:.4f}")
for feat, coef, mu, sig in zip(selected_features, clf.coef_[0], scaler.mean_, scaler.scale_):
    print(f"  {feat:20s}: weight = {coef:+.4f} | mean = {mu:10.4f} | std = {sig:10.4f}")

print("\nClosed-form formula:")
terms = [f"{clf.intercept_[0]:.4f}"]
for feat, coef in zip(selected_features, clf.coef_[0]):
    terms.append(f"{coef:+.4f}*z_{feat}")
print("logit(p_chaos) = " + " ".join(terms))

preds_05 = (probs >= 0.5).astype(int)
preds_opt = (probs >= opt_t).astype(int)

cm05 = confusion_matrix(y, preds_05)
cm_opt = confusion_matrix(y, preds_opt)

print("\n--- PERFORMANCE SUMMARY ---")
print(f"At Default Threshold T=0.50:")
print(f"  F1-Score: {f1_score(y, preds_05):.4f}")
print(f"  Accuracy: {accuracy_score(y, preds_05)*100:.1f}%")
print(f"  Recall  : {recall_score(y, preds_05)*100:.1f}% ({cm05[1,1]}/{cm05[1,1]+cm05[1,0]} caught)")
print(f"  Precision: {precision_score(y, preds_05)*100:.1f}%")
print(f"  Confusion Matrix: TP={cm05[1,1]} FP={cm05[0,1]} FN={cm05[1,0]} TN={cm05[0,0]}")

print(f"\nAt Optimal Threshold T={opt_t:.4f}:")
print(f"  F1-Score: {f1_score(y, preds_opt):.4f}")
print(f"  Accuracy: {accuracy_score(y, preds_opt)*100:.1f}%")
print(f"  Recall  : {recall_score(y, preds_opt)*100:.1f}% ({cm_opt[1,1]}/{cm_opt[1,1]+cm_opt[1,0]} caught)")
print(f"  Precision: {precision_score(y, preds_opt)*100:.1f}%")
print(f"  Confusion Matrix: TP={cm_opt[1,1]} FP={cm_opt[0,1]} FN={cm_opt[1,0]} TN={cm_opt[0,0]}")


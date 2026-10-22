import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.feature_selection import mutual_info_classif, SequentialFeatureSelector

from src.scratch_k2_ultimate_f1 import X, y, feature_names

# 1. Univariate metrics
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

univariate_results = []
for j, name in enumerate(feature_names):
    col = X[:, j]
    r_pearson = np.corrcoef(col, y)[0, 1]
    r_spearman = spearmanr(col, y).correlation
    
    # Univariate logistic regression 5-fold CV
    oof_probs = np.zeros(len(y))
    for tr, va in skf.split(col[:, None], y):
        s = StandardScaler()
        x_tr = s.fit_transform(col[tr, None])
        x_va = s.transform(col[va, None])
        clf = LogisticRegression(class_weight='balanced', random_state=42)
        clf.fit(x_tr, y[tr])
        oof_probs[va] = clf.predict_proba(x_va)[:, 1]
        
    auc = roc_auc_score(y, oof_probs)
    p, r, th = precision_recall_curve(y, oof_probs)
    f1_curve = 2 * (p * r) / (p + r + 1e-12)
    best_f1 = np.max(f1_curve)
    
    univariate_results.append({
        'Feature': name,
        'Pearson_r': r_pearson,
        'Spearman_r': r_spearman,
        'ROC_AUC': auc,
        'Max_F1': best_f1
    })

df_uni = pd.DataFrame(univariate_results).sort_values('ROC_AUC', ascending=False)
print("="*75)
print("ALL 26 FEATURES RANKED BY UNIVARIATE ROC-AUC & CORRELATION")
print("="*75)
print(df_uni.to_string(index=False))

# 2. Sequential Forward Selection to find the absolute best 5 features for Logistic Regression
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

print("\n" + "="*75)
print("GREEDY FORWARD SELECTION (BEST 1 TO 7 FEATURES BY 5-FOLD CV F1)")
print("="*75)

selected = []
remaining = list(range(len(feature_names)))

for step in range(7):
    best_step_f1 = -1
    best_feat_idx = -1
    
    for candidate in remaining:
        trial_indices = selected + [candidate]
        trial_X = X_s[:, trial_indices]
        
        # 5-fold CV
        oof_probs = np.zeros(len(y))
        for tr, va in skf.split(trial_X, y):
            clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)
            clf.fit(trial_X[tr], y[tr])
            oof_probs[va] = clf.predict_proba(trial_X[va])[:, 1]
            
        p, r, th = precision_recall_curve(y, oof_probs)
        f1_curve = 2 * (p * r) / (p + r + 1e-12)
        max_f1 = np.max(f1_curve)
        
        if max_f1 > best_step_f1:
            best_step_f1 = max_f1
            best_feat_idx = candidate
            
    selected.append(best_feat_idx)
    remaining.remove(best_feat_idx)
    feat_name = feature_names[best_feat_idx]
    print(f"Step {step+1}: Added '{feat_name:18s}' -> Subset F1 = {best_step_f1:.4f} (Features: {[feature_names[i] for i in selected]})")


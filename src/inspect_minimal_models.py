import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.scratch_k2_ultimate_f1 import X, y, feature_names

feature_idx = {name: i for i, name in enumerate(feature_names)}

# Selected 5 minimal physical features
selected_feats = ['mean_path_diff', 'bio_cv', 'bio_final', 'c_ln', 'c_hn']
idxs = [feature_idx[f] for f in selected_feats]
X_sub = X[:, idxs]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sub)

# Fit Logistic Regression
clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)
clf.fit(X_scaled, y)

print("="*60)
print("5-FEATURE LOGISTIC REGRESSION (6 TOTAL PARAMETERS)")
print("="*60)
print(f"Intercept (w0): {clf.intercept_[0]:.4f}")
for feat, coef in zip(selected_feats, clf.coef_[0]):
    print(f"  Weight for {feat:16s}: {coef:+.4f}")

print("\nFormula:")
eq_terms = [f"{clf.intercept_[0]:.3f}"]
for feat, coef in zip(selected_feats, clf.coef_[0]):
    eq_terms.append(f"{coef:+.3f} * z({feat})")
print("logit(p_chaos) = " + " ".join(eq_terms))


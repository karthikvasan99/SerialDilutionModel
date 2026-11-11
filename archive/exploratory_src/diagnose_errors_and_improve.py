import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, confusion_matrix, roc_auc_score

from src.scratch_k2_ultimate_f1 import X, y, feature_names, all_keys

# Features:
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
X_sub = X[:, f_idx].copy()

# Run baseline LOOCV
loo = LeaveOneOut()
oof_probs = np.zeros(len(y))
for tr, te in loo.split(X_sub):
    s = StandardScaler()
    x_tr = s.fit_transform(X_sub[tr])
    x_te = s.transform(X_sub[te])
    clf = LogisticRegression(C=0.8, class_weight='balanced', max_iter=2000, random_state=42)
    clf.fit(x_tr, y[tr])
    oof_probs[te[0]] = clf.predict_proba(x_te)[0, 1]

preds = (oof_probs >= 0.5).astype(int)

# Inspect errors
df_diag = pd.DataFrame({
    'idx': range(len(y)),
    'Nutrient': [k[0] for k in all_keys],
    'S': [k[1] for k in all_keys],
    'Comm': [k[2] for k in all_keys],
    'y_true': y,
    'prob_chaos': oof_probs,
    'pred_05': preds,
    'mean_tv_path': X_sub[:, 0],
    'bio_cv': X_sub[:, 3],
    'max_sp_cv': X_sub[:, 4]
})

fps = df_diag[(df_diag.y_true == 0) & (df_diag.pred_05 == 1)]
fns = df_diag[(df_diag.y_true == 1) & (df_diag.pred_05 == 0)]

print("="*80)
print(f"DIAGNOSIS: {len(fps)} FALSE POSITIVES (Model predicted Chaos, Gore called Stable)")
print("="*80)
print(fps[['Nutrient', 'S', 'Comm', 'prob_chaos', 'mean_tv_path', 'bio_cv', 'max_sp_cv']].to_string(index=False))

print("\n" + "="*80)
print(f"DIAGNOSIS: {len(fns)} FALSE NEGATIVES (Model predicted Stable, Gore called Chaos)")
print("="*80)
print(fns[['Nutrient', 'S', 'Comm', 'prob_chaos', 'mean_tv_path', 'bio_cv', 'max_sp_cv']].to_string(index=False))

# Notice that many FPs might be High Nutrient (HN) communities with S=12, 24, 48 where OD CV was ~0.11-0.13!
print("\nFP breakdown by Nutrient & S:")
print(fps.groupby(['Nutrient', 'S']).size())


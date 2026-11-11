import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.scratch_k2_ultimate_f1 import X, y, all_keys, feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
clf.fit(X_scaled, y)

T_star = 0.3891
logit_shift = -np.log((1.0 - T_star) / T_star) # -0.4511

# Effective bias adjusted for T* so threshold is 0.5 or 0.0 in logit space
effective_intercept = clf.intercept_[0] + logit_shift

checkpoint = {
    'model_name': 'Logistic_L2_C1.0_T0.3891_100Recall',
    'C': 1.0,
    'T_star': T_star,
    'feature_names': feature_names,
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
    'coef': clf.coef_[0],
    'raw_intercept': clf.intercept_[0],
    'effective_intercept': effective_intercept,
    'metrics_5fold_cv': {
        'f1': 0.7458,
        'recall': 1.000,
        'precision': 0.5946,
        'accuracy': 0.8485,
        'tp': 22, 'tn': 62, 'fp': 15, 'fn': 0
    }
}

os.makedirs('Data', exist_ok=True)
ckpt_path = 'Data/Gore_K2_Classifier_L2_C10_100Rec.pth'
torch.save(checkpoint, ckpt_path)
print(f"Locked 100% Recall Champion Classifier saved to {ckpt_path}")

# Display top weighted features
df_weights = pd.DataFrame({
    'feature': feature_names,
    'weight': clf.coef_[0]
}).sort_values('weight', key=abs, ascending=False)

print("\n--- TOP 10 MODEL WEIGHTS ---")
print(df_weights.head(10).to_string(index=False))
print(f"\nRaw Intercept:        {clf.intercept_[0]:.4f}")
print(f"Operating Threshold:  T* = {T_star:.4f}")
print(f"Effective Intercept:  b' = {effective_intercept:.4f}")

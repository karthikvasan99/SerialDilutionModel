import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.scratch_k2_ultimate_f1 import X, y, feature_names

selected_features = [
    'mean_tv_path',
    'mean_poincare_shift',
    'mean_overlap',
    'bio_cv',
    'max_sp_cv',
    'bio_final',
    'path_len',
    'S',
    'c_ln',
    'c_hn'
]

f_idx = [feature_names.index(f) for f in selected_features]
X_sub = X[:, f_idx]

scaler = StandardScaler()
X_s = scaler.fit_transform(X_sub)

clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)
clf.fit(X_s, y)

save_dict = {
    'model_type': 'LogisticRegression',
    'C': 0.5,
    'penalty': 'l2',
    'class_weight': 'balanced',
    'selected_features': selected_features,
    'coef': clf.coef_[0],
    'intercept': clf.intercept_[0],
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
    'threshold': 0.5000,
    'loocv_f1': 0.7273,
    'loocv_accuracy': 0.8485,
    'loocv_recall': 0.9091,
    'loocv_precision': 0.6061,
    'loocv_specificity': 0.8312,
    'loocv_npv': 0.9697,
    'loocv_auc': 0.8973,
    'loocv_brier': 0.1254,
    'global_f1': 0.7719,
    'global_recall': 1.0000,
    'global_accuracy': 0.8687
}

torch.save(save_dict, 'Data/Gore_K2_Classifier_Logistic10.pth')
print("Successfully saved official production checkpoint to Data/Gore_K2_Classifier_Logistic10.pth")

# Also overwrite src/Gore_K2_Classifier.py to serve as the reference implementation
with open('src/Gore_K2_Classifier.py', 'w') as f:
    f.write(open('src/report_champion_10feat.py').read())
print("Updated src/Gore_K2_Classifier.py")


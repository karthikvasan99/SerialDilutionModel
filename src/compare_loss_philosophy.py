import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, confusion_matrix, roc_auc_score, brier_score_loss

from src.scratch_k2_ultimate_f1 import X, y, feature_names

# 10 core features
feats_10 = ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S', 'c_ln', 'c_hn']
X_10 = X[:, [feature_names.index(f) for f in feats_10]]

loo = LeaveOneOut()

# 1. Standard Logistic Regression (Log-Loss / Proper Scoring Rule)
oof_probs_logloss = np.zeros(len(y))
for tr, te in loo.split(X_10):
    s = StandardScaler()
    x_tr = s.fit_transform(X_10[tr])
    x_te = s.transform(X_10[te])
    clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)
    clf.fit(x_tr, y[tr])
    oof_probs_logloss[te[0]] = clf.predict_proba(x_te)[0, 1]

# 2. Soft-F1 Logistic Regression
class SoftF1Loss(nn.Module):
    def forward(self, probs, y_true):
        tp = (probs * y_true).sum()
        fp = (probs * (1 - y_true)).sum()
        fn = ((1 - probs) * y_true).sum()
        return 1.0 - (2 * tp / (2 * tp + fp + fn + 1e-7))

oof_probs_softf1 = np.zeros(len(y))
for tr, te in loo.split(X_10):
    s = StandardScaler()
    x_tr = s.fit_transform(X_10[tr])
    x_te = s.transform(X_10[te])
    
    torch.manual_seed(42)
    m = nn.Linear(len(feats_10), 1)
    opt = torch.optim.Adam(m.parameters(), lr=0.03)
    crit = SoftF1Loss()
    x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y[tr], dtype=torch.float32)
    x_te_t = torch.tensor(x_te, dtype=torch.float32)
    
    for ep in range(250):
        opt.zero_grad()
        p = torch.sigmoid(m(x_tr_t).squeeze(-1))
        loss = crit(p, y_tr_t)
        loss.backward()
        opt.step()
        
    m.eval()
    with torch.no_grad():
        oof_probs_softf1[te[0]] = torch.sigmoid(m(x_te_t).squeeze(-1)).item()

# Compare Metrics
def get_metrics(probs, name):
    auc = roc_auc_score(y, probs)
    brier = brier_score_loss(y, probs) # probability calibration error (lower is better)
    preds_05 = (probs >= 0.5).astype(int)
    f1_05 = f1_score(y, preds_05)
    rec_05 = recall_score(y, preds_05)
    prec_05 = precision_score(y, preds_05)
    acc_05 = accuracy_score(y, preds_05)
    cm_05 = confusion_matrix(y, preds_05)
    
    p_c, r_c, th_c = precision_recall_curve(y, probs)
    f1_c = 2 * (p_c * r_c) / (p_c + r_c + 1e-12)
    b_idx = np.argmax(f1_c)
    
    return {
        'Loss Type': name,
        'ROC-AUC': auc,
        'Brier Score (Calibration Error)': brier,
        'LOOCV F1 (T=0.5)': f1_05,
        'LOOCV Recall (T=0.5)': rec_05,
        'LOOCV Precision (T=0.5)': prec_05,
        'LOOCV Accuracy (T=0.5)': acc_05,
        'TP': cm_05[1,1], 'FP': cm_05[0,1], 'FN': cm_05[1,0], 'TN': cm_05[0,0],
        'Optimal F1': f1_c[b_idx],
        'Optimal T': th_c[b_idx] if b_idx < len(th_c) else 0.5
    }

res1 = get_metrics(oof_probs_logloss, "Standard Log-Loss (BCE)")
res2 = get_metrics(oof_probs_softf1, "Direct Soft-F1 Loss")

df_out = pd.DataFrame([res1, res2])
print("="*85)
print("EMPIRICAL COMPARISON IN LOOCV: LOG-LOSS VS SOFT-F1 LOSS")
print("="*85)
print(df_out[['Loss Type', 'ROC-AUC', 'Brier Score (Calibration Error)', 'LOOCV F1 (T=0.5)', 'LOOCV Recall (T=0.5)', 'LOOCV Precision (T=0.5)', 'LOOCV Accuracy (T=0.5)']].to_string(index=False))
print("\nOptimal Operating Points:")
print(df_out[['Loss Type', 'Optimal F1', 'Optimal T', 'TP', 'FP', 'FN', 'TN']].to_string(index=False))


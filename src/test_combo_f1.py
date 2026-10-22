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

loo = LeaveOneOut()

# Candidate features
feats_9 = ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'path_len', 'S', 'c_ln', 'c_hn']
X_9 = X[:, [feature_names.index(f) for f in feats_9]].copy()

# Try log transforming the two path features
X_9_log = X_9.copy()
X_9_log[:, 0] = np.log(X_9_log[:, 0] + 1e-4) # log(mean_tv_path)
X_9_log[:, 1] = np.log(X_9_log[:, 1] + 1e-4) # log(mean_poincare_shift)
X_9_log[:, 5] = np.log(X_9_log[:, 5] + 1e-4) # log(path_len)

feats_10 = ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S', 'c_ln', 'c_hn']
X_10 = X[:, [feature_names.index(f) for f in feats_10]].copy()
X_10_log = X_10.copy()
X_10_log[:, 0] = np.log(X_10_log[:, 0] + 1e-4)
X_10_log[:, 1] = np.log(X_10_log[:, 1] + 1e-4)
X_10_log[:, 6] = np.log(X_10_log[:, 6] + 1e-4)

for name, mat in [("9 Feats (Raw)", X_9), ("9 Feats (Log)", X_9_log), ("10 Feats (Raw)", X_10), ("10 Feats (Log)", X_10_log)]:
    for c_val in [0.3, 0.5, 0.8, 1.0, 1.5]:
        oof_probs = np.zeros(len(y))
        for tr, te in loo.split(mat):
            s = StandardScaler()
            x_tr = s.fit_transform(mat[tr])
            x_te = s.transform(mat[te])
            clf = LogisticRegression(C=c_val, class_weight='balanced', max_iter=2000, random_state=42)
            clf.fit(x_tr, y[tr])
            oof_probs[te[0]] = clf.predict_proba(x_te)[0, 1]
            
        auc = roc_auc_score(y, oof_probs)
        preds_05 = (oof_probs >= 0.5).astype(int)
        f1_05 = f1_score(y, preds_05)
        rec_05 = recall_score(y, preds_05)
        prec_05 = precision_score(y, preds_05)
        acc_05 = accuracy_score(y, preds_05)
        cm = confusion_matrix(y, preds_05)
        
        p_c, r_c, th_c = precision_recall_curve(y, oof_probs)
        f1_c = 2 * (p_c * r_c) / (p_c + r_c + 1e-12)
        b_idx = np.argmax(f1_c)
        opt_f1 = f1_c[b_idx]
        opt_t = th_c[b_idx] if b_idx < len(th_c) else 0.5
        
        if f1_05 >= 0.70 or opt_f1 >= 0.74:
            print(f"{name:16s} (C={c_val}) | AUC={auc:.4f} | T=0.5 F1={f1_05:.4f} (Rec={rec_05*100:.1f}%, Prec={prec_05*100:.1f}%, Acc={acc_05*100:.1f}%) | Opt F1={opt_f1:.4f} at T={opt_t:.3f} | TP:{cm[1,1]} FP:{cm[0,1]} FN:{cm[1,0]}")


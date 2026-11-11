import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve

from src.scratch_k2_ultimate_f1 import X, y, feature_names

# In scratch_k2_ultimate_f1, all_N and all_R are stored for 10 passages
from src.test_later_passages_classifier import all_N, all_R, all_bio, all_rel, extract_features

y_true = y
loo = LeaveOneOut()

print("="*85)
print("TESTING PASSAGE WINDOWS WITHIN THE 10-PASSAGE EXPERIMENTAL HORIZON")
print("="*85)

for (p_s, p_e) in [(5, 10), (6, 10), (7, 10), (8, 10)]:
    X_mat = extract_features(p_s, p_e)
    for c_val in [0.3, 0.5, 0.8, 1.0]:
        oof_probs = np.zeros(len(y_true))
        for tr, te in loo.split(X_mat):
            s = StandardScaler()
            x_tr = s.fit_transform(X_mat[tr])
            x_te = s.transform(X_mat[te])
            clf = LogisticRegression(C=c_val, class_weight='balanced', max_iter=2000, random_state=42)
            clf.fit(x_tr, y_true[tr])
            oof_probs[te[0]] = clf.predict_proba(x_te)[0, 1]
            
        preds_05 = (oof_probs >= 0.5).astype(int)
        cm = confusion_matrix(y_true, preds_05)
        f1 = f1_score(y_true, preds_05)
        acc = accuracy_score(y_true, preds_05)
        rec = recall_score(y_true, preds_05)
        prec = precision_score(y_true, preds_05)
        auc = roc_auc_score(y_true, oof_probs)
        
        p_c, r_c, th_c = precision_recall_curve(y_true, oof_probs)
        f1_c = 2 * (p_c * r_c) / (p_c + r_c + 1e-12)
        b_idx = np.argmax(f1_c)
        opt_f1 = f1_c[b_idx]
        
        if f1 >= 0.68 or opt_f1 >= 0.72:
            print(f"Passages {p_s:2d} to {p_e:2d} (C={c_val:3.1f}) | F1={f1:.4f} | Acc={acc*100:.1f}% | Rec={rec*100:.1f}% | Prec={prec*100:.1f}% | AUC={auc:.4f} | TP:{cm[1,1]:2d} FP:{cm[0,1]:2d} FN:{cm[1,0]:2d} (Opt F1={opt_f1:.4f})")


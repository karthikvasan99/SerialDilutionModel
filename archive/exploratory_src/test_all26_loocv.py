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

print("ALL 26 FEATURES LOGISTIC REGRESSION UNDER STRICT LOOCV:")
for c_val in [0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    oof_probs = np.zeros(len(y))
    for tr, te in loo.split(X):
        s = StandardScaler()
        x_tr = s.fit_transform(X[tr])
        x_te = s.transform(X[te])
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
    
    print(f"C={c_val:4.1f} | AUC={auc:.4f} | T=0.5 F1={f1_05:.4f} (Rec={rec_05*100:.1f}%, Prec={prec_05*100:.1f}%, Acc={acc_05*100:.1f}%) | Opt F1={opt_f1:.4f} at T={opt_t:.3f} | TP:{cm[1,1]:2d} FP:{cm[0,1]:2d} FN:{cm[1,0]:2d}")


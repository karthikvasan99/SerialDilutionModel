import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from src.scratch_k2_ultimate_f1 import X, y, feature_names, all_keys

# Features WITH media (10 features)
feats_with = [
    'mean_tv_path', 'mean_poincare_shift', 'mean_overlap',
    'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S',
    'c_ln', 'c_hn'
]

# Features WITHOUT media (8 features: identical physical & demographic features, just dropping c_ln and c_hn)
feats_without = [
    'mean_tv_path', 'mean_poincare_shift', 'mean_overlap',
    'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S'
]

def run_loocv_probs(f_list, c_val=0.5):
    X_sub = X[:, [feature_names.index(f) for f in f_list]]
    loo = LeaveOneOut()
    oof_probs = np.zeros(len(y))
    for tr, te in loo.split(X_sub):
        s = StandardScaler()
        x_tr = s.fit_transform(X_sub[tr])
        x_te = s.transform(X_sub[te])
        clf = LogisticRegression(C=c_val, class_weight='balanced', max_iter=2000, random_state=42)
        clf.fit(x_tr, y[tr])
        oof_probs[te[0]] = clf.predict_proba(x_te)[0, 1]
    return oof_probs

probs_with = run_loocv_probs(feats_with, c_val=0.5)
probs_without = run_loocv_probs(feats_without, c_val=0.5) # evaluate at same C=0.5

# We also check C=0.8 for without media just in case
probs_without_08 = run_loocv_probs(feats_without, c_val=0.8)

df_all = pd.DataFrame({
    'Nutrient': [k[0] for k in all_keys],
    'S': [k[1] for k in all_keys],
    'Comm': [k[2] for k in all_keys],
    'y_true': y,
    'prob_with': probs_with,
    'pred_with': (probs_with >= 0.5).astype(int),
    'prob_without': probs_without,
    'pred_without': (probs_without >= 0.5).astype(int),
    'prob_without_08': probs_without_08,
    'pred_without_08': (probs_without_08 >= 0.5).astype(int)
})

def format_cm(cm):
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    return f"TN:{tn:2d} | FP:{fp:2d} | FN:{fn:2d} | TP:{tp:2d}"

print("="*90)
print("SIDE-BY-SIDE MEDIA-BY-MEDIA CONFUSION MATRIX COMPARISON (LOOCV at T=0.50)")
print("="*90)

for nut in ['LN', 'MN', 'HN']:
    sub = df_all[df_all.Nutrient == nut]
    y_sub = sub.y_true.values
    p_with = sub.pred_with.values
    p_without = sub.pred_without.values
    
    cm_w = confusion_matrix(y_sub, p_with, labels=[0, 1])
    cm_wo = confusion_matrix(y_sub, p_without, labels=[0, 1])
    
    acc_w = accuracy_score(y_sub, p_with)
    acc_wo = accuracy_score(y_sub, p_without)
    
    rec_w = recall_score(y_sub, p_with, zero_division=0)
    rec_wo = recall_score(y_sub, p_without, zero_division=0)
    
    prec_w = precision_score(y_sub, p_with, zero_division=0)
    prec_wo = precision_score(y_sub, p_without, zero_division=0)
    
    f1_w = f1_score(y_sub, p_with, zero_division=0)
    f1_wo = f1_score(y_sub, p_without, zero_division=0)
    
    print(f"\n==========================================================================")
    print(f"MEDIA: {nut} (Total: {len(sub)} communities | True Stable: {np.sum(y_sub==0)}, True Chaos: {np.sum(y_sub==1)})")
    print(f"==========================================================================")
    print(f"  MODEL WITH MEDIA CONDITIONS:")
    print(f"    Confusion Matrix  : {format_cm(cm_w)}")
    print(f"    Layout            : [Predicted Stable, Predicted Chaos]")
    print(f"      Actual Stable   : [ TN = {cm_w[0,0]:2d} ,  FP = {cm_w[0,1]:2d} ]")
    print(f"      Actual Chaos    : [ FN = {cm_w[1,0]:2d} ,  TP = {cm_w[1,1]:2d} ]")
    print(f"    Accuracy: {acc_w*100:.1f}% | Recall: {rec_w*100:.1f}% | Precision: {prec_w*100:.1f}% | F1: {f1_w:.4f}")
    
    print(f"\n  MODEL WITHOUT MEDIA CONDITIONS:")
    print(f"    Confusion Matrix  : {format_cm(cm_wo)}")
    print(f"    Layout            : [Predicted Stable, Predicted Chaos]")
    print(f"      Actual Stable   : [ TN = {cm_wo[0,0]:2d} ,  FP = {cm_wo[0,1]:2d} ]")
    print(f"      Actual Chaos    : [ FN = {cm_wo[1,0]:2d} ,  TP = {cm_wo[1,1]:2d} ]")
    print(f"    Accuracy: {acc_wo*100:.1f}% | Recall: {rec_wo*100:.1f}% | Precision: {prec_wo*100:.1f}% | F1: {f1_wo:.4f}")

# Overall Total Comparison
print("\n" + "="*90)
print("OVERALL TOTAL DATASET (N=99, 77 Stable, 22 Chaos)")
print("="*90)
cm_tot_w = confusion_matrix(df_all.y_true, df_all.pred_with, labels=[0, 1])
cm_tot_wo = confusion_matrix(df_all.y_true, df_all.pred_without, labels=[0, 1])
print(f"WITH MEDIA    : {format_cm(cm_tot_w)} | Acc={accuracy_score(df_all.y_true, df_all.pred_with)*100:.1f}% | Rec={recall_score(df_all.y_true, df_all.pred_with)*100:.1f}% | Prec={precision_score(df_all.y_true, df_all.pred_with)*100:.1f}% | F1={f1_score(df_all.y_true, df_all.pred_with):.4f}")
print(f"WITHOUT MEDIA : {format_cm(cm_tot_wo)} | Acc={accuracy_score(df_all.y_true, df_all.pred_without)*100:.1f}% | Rec={recall_score(df_all.y_true, df_all.pred_without)*100:.1f}% | Prec={precision_score(df_all.y_true, df_all.pred_without)*100:.1f}% | F1={f1_score(df_all.y_true, df_all.pred_without):.4f}")

# Show exact communities where predictions diverge
diffs = df_all[df_all.pred_with != df_all.pred_without]
print("\n" + "="*90)
print(f"COMMUNITIES WHERE PREDICTIONS DIVERGE ({len(diffs)} total):")
print("="*90)
print(diffs[['Nutrient', 'S', 'Comm', 'y_true', 'prob_with', 'pred_with', 'prob_without', 'pred_without']].to_string(index=False))


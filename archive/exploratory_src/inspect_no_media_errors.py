import sys, os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score

from src.scratch_k2_ultimate_f1 import X, y, feature_names, all_keys

# Compare error breakdown With Media vs Without Media
feats_with_media = ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S', 'c_ln', 'c_hn']
feats_no_media = ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S']

def run_loocv(f_list, c_val=0.5):
    X_sub = X[:, [feature_names.index(f) for f in f_list]]
    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    for tr, te in loo.split(X_sub):
        s = StandardScaler()
        x_tr = s.fit_transform(X_sub[tr])
        x_te = s.transform(X_sub[te])
        clf = LogisticRegression(C=c_val, class_weight='balanced', max_iter=2000, random_state=42)
        clf.fit(x_tr, y[tr])
        probs[te[0]] = clf.predict_proba(x_te)[0, 1]
    return (probs >= 0.5).astype(int)

preds_with = run_loocv(feats_with_media, c_val=0.5)
preds_without = run_loocv(feats_no_media, c_val=0.8)

df = pd.DataFrame({
    'Nutrient': [k[0] for k in all_keys],
    'S': [k[1] for k in all_keys],
    'Comm': [k[2] for k in all_keys],
    'y_true': y,
    'pred_with_media': preds_with,
    'pred_no_media': preds_without
})

print("="*80)
print("ERROR BREAKDOWN: WITH MEDIA VS WITHOUT MEDIA")
print("="*80)
for nut in ['LN', 'MN', 'HN']:
    sub = df[df.Nutrient == nut]
    cm_with = confusion_matrix(sub.y_true, sub.pred_with_media, labels=[0, 1])
    cm_without = confusion_matrix(sub.y_true, sub.pred_no_media, labels=[0, 1])
    print(f"\n{nut} Media ({len(sub)} communities, {sub.y_true.sum()} chaotic):")
    print(f"  WITH Media   : TP={cm_with[1,1]:2d} FP={cm_with[0,1]:2d} FN={cm_with[1,0]:2d} TN={cm_with[0,0]:2d} | Acc = {accuracy_score(sub.y_true, sub.pred_with_media)*100:.1f}%")
    print(f"  WITHOUT Media: TP={cm_without[1,1]:2d} FP={cm_without[0,1]:2d} FN={cm_without[1,0]:2d} TN={cm_without[0,0]:2d} | Acc = {accuracy_score(sub.y_true, sub.pred_no_media)*100:.1f}%")

# Communities where prediction flipped
df['flipped'] = df.pred_with_media != df.pred_no_media
print("\n" + "="*80)
print("COMMUNITIES WHOSE PREDICTION CHANGED WHEN REMOVING MEDIA:")
print("="*80)
print(df[df.flipped][['Nutrient', 'S', 'Comm', 'y_true', 'pred_with_media', 'pred_no_media']].to_string(index=False))


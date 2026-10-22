import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix

from src.scratch_k2_ultimate_f1 import X, y, feature_names, all_keys

# Base 8 features without media conditions
base_features = [
    'mean_tv_path', 'mean_poincare_shift', 'mean_overlap',
    'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S'
]

# Candidates to pick 2 from (excluding any media indicators: c_ln, c_mn, c_hn)
media_cols = {'c_ln', 'c_mn', 'c_hn'}
candidate_pool = [f for f in feature_names if f not in base_features and f not in media_cols]

print(f"Base 8 features: {base_features}")
print(f"Candidate pool ({len(candidate_pool)} features): {candidate_pool}")

loo = LeaveOneOut()

def evaluate_features_loocv(feat_list, C=0.5):
    f_idx = [feature_names.index(f) for f in feat_list]
    X_sub = X[:, f_idx]
    
    oof_probs = np.zeros(len(y))
    for tr, te in loo.split(X_sub):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(X_sub[tr])
        x_te = scaler.transform(X_sub[te])
        
        clf = LogisticRegression(C=C, class_weight='balanced', max_iter=1000, random_state=42)
        clf.fit(x_tr, y[tr])
        oof_probs[te[0]] = clf.predict_proba(x_te)[0, 1]
        
    preds = (oof_probs >= 0.5).astype(int)
    f1 = f1_score(y, preds)
    acc = accuracy_score(y, preds)
    rec = recall_score(y, preds)
    prec = precision_score(y, preds)
    auc = roc_auc_score(y, oof_probs)
    cm = confusion_matrix(y, preds)
    
    return {
        'features': feat_list,
        'f1': f1,
        'acc': acc,
        'rec': rec,
        'prec': prec,
        'auc': auc,
        'cm': cm,
        'oof_probs': oof_probs,
        'preds': preds
    }

# Benchmark Base 8 alone
res_base = evaluate_features_loocv(base_features, C=0.5)
cm_b = res_base['cm']
print(f"\n--- BASE 8 (NO MEDIA) ---")
print(f"F1 = {res_base['f1']:.4f} | Acc = {res_base['acc']*100:.1f}% | Rec = {res_base['rec']*100:.1f}% | Prec = {res_base['prec']*100:.1f}% | AUC = {res_base['auc']:.4f} | TP:{cm_b[1,1]} FP:{cm_b[0,1]} FN:{cm_b[1,0]} TN:{cm_b[0,0]}")

# Test adding each single feature (9 features)
results_9 = []
for f in candidate_pool:
    res = evaluate_features_loocv(base_features + [f], C=0.5)
    results_9.append((f, res))

results_9.sort(key=lambda x: (x[1]['f1'], x[1]['auc']), reverse=True)
print("\n--- TOP 5 SINGLE ADDITIONS (9 FEATURES) ---")
for f, r in results_9[:5]:
    cm = r['cm']
    print(f"+ {f:18s} | F1 = {r['f1']:.4f} | Acc = {r['acc']*100:.1f}% | Rec = {r['rec']*100:.1f}% | Prec = {r['prec']*100:.1f}% | AUC = {r['auc']:.4f} | TP:{cm[1,1]} FP:{cm[0,1]} FN:{cm[1,0]}")

# Test all pairs of additions (10 features)
results_10 = []
for pair in itertools.combinations(candidate_pool, 2):
    f_set = base_features + list(pair)
    # Test a couple C values
    for C in [0.3, 0.5, 1.0]:
        res = evaluate_features_loocv(f_set, C=C)
        results_10.append((pair, C, res))

results_10.sort(key=lambda x: (x[2]['f1'], x[2]['auc']), reverse=True)

print("\n" + "="*85)
print("TOP 10 FEATURE COMBINATIONS (EXACTLY 10 PHYSICAL FEATURES, ZERO MEDIA INDICATORS)")
print("="*85)
seen_pairs = set()
count = 0
for pair, C, r in results_10:
    if pair in seen_pairs:
        continue
    seen_pairs.add(pair)
    count += 1
    cm = r['cm']
    print(f"Pair: {pair[0]:16s} + {pair[1]:16s} (C={C}) | F1 = {r['f1']:.4f} | Acc = {r['acc']*100:.1f}% | Rec = {r['rec']*100:.1f}% | Prec = {r['prec']*100:.1f}% | AUC = {r['auc']:.4f} | TP:{cm[1,1]} FP:{cm[0,1]} FN:{cm[1,0]} TN:{cm[0,0]}")
    if count >= 10:
        break

# Pick the champion 10 features without media
best_pair, best_C, best_res = results_10[0]
best_10_features = base_features + list(best_pair)
print(f"\nCHAMPION 10-FEATURE SET (NO MEDIA): {best_10_features}")

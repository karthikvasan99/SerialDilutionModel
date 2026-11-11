import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, confusion_matrix, roc_auc_score

from src.scratch_k2_ultimate_f1 import X, y, feature_names

print(f"Dataset: N={len(y)}, Chaos={y.sum()} ({y.mean()*100:.1f}%), Features={len(feature_names)}")

# Let's inspect which features provide the most complementary information.
# Let's run L1-regularized logistic regression (Lasso) with different C values to get the exact natural sparsity path.
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

print("\n--- LASSO (L1) REGULARIZATION PATH (NATURAL FEATURE SELECTION) ---")
for c_val in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]:
    clf_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=c_val, class_weight='balanced', random_state=42)
    clf_l1.fit(X_s, y)
    active = np.where(clf_l1.coef_[0] != 0)[0]
    coefs = [(feature_names[i], clf_l1.coef_[0][i]) for i in active]
    coefs = sorted(coefs, key=lambda x: abs(x[1]), reverse=True)
    print(f"C={c_val:4.2f} -> {len(active):2d} non-zero features: {[c[0] for c in coefs]}")

# Let's test a variety of model families on these natural L1 subsets under strict nested CV:
def evaluate_nested(clf_fn, X_data, y_data):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    oof_preds_nested = np.zeros(len(y_data))
    oof_probs = np.zeros(len(y_data))
    
    for train_idx, val_idx in outer_cv.split(X_data, y_data):
        X_tr, y_tr = X_data[train_idx], y_data[train_idx]
        X_va, y_va = X_data[val_idx], y_data[val_idx]
        
        s = StandardScaler()
        X_tr_s = s.fit_transform(X_tr)
        X_va_s = s.transform(X_va)
        
        inner_probs = np.zeros(len(y_tr))
        for in_tr, in_va in inner_cv.split(X_tr, y_tr):
            in_s = StandardScaler()
            X_in_tr = in_s.fit_transform(X_tr[in_tr])
            X_in_va = in_s.transform(X_tr[in_va])
            
            clf = clf_fn()
            clf.fit(X_in_tr, y_tr[in_tr])
            inner_probs[in_va] = clf.predict_proba(X_in_va)[:, 1]
            
        p, r, th = precision_recall_curve(y_tr, inner_probs)
        f1_inner = 2 * (p * r) / (p + r + 1e-12)
        best_t = th[np.argmax(f1_inner)] if len(th) > 0 and np.argmax(f1_inner) < len(th) else 0.5
        
        clf_outer = clf_fn()
        clf_outer.fit(X_tr_s, y_tr)
        probs_va = clf_outer.predict_proba(X_va_s)[:, 1]
        oof_probs[val_idx] = probs_va
        oof_preds_nested[val_idx] = (probs_va >= best_t).astype(int)
        
    p_oof, r_oof, th_oof = precision_recall_curve(y_data, oof_probs)
    f1_curve = 2 * (p_oof * r_oof) / (p_oof + r_oof + 1e-12)
    best_oracle_idx = np.argmax(f1_curve)
    
    return {
        'nested_f1': f1_score(y_data, oof_preds_nested),
        'nested_acc': accuracy_score(y_data, oof_preds_nested),
        'nested_rec': recall_score(y_data, oof_preds_nested),
        'nested_prec': precision_score(y_data, oof_preds_nested),
        'cm_nested': confusion_matrix(y_data, oof_preds_nested),
        't05_f1': f1_score(y_data, (oof_probs >= 0.5).astype(int)),
        'oracle_f1': f1_curve[best_oracle_idx],
        'oracle_rec': r_oof[best_oracle_idx],
        'oracle_prec': p_oof[best_oracle_idx],
        'oracle_acc': accuracy_score(y_data, (oof_probs >= th_oof[best_oracle_idx]).astype(int)) if best_oracle_idx < len(th_oof) else 0.0,
        'auc': roc_auc_score(y_data, oof_probs)
    }

print("\n" + "="*85)
print("BENCHMARKING MODEL FAMILIES ACROSS FEATURE COUNTS (STRICT NESTED CV)")
print("="*85)

# Test candidate feature sets:
candidate_feature_sets = {}

# Sets derived from L1 regularization path:
for c_val in [0.1, 0.15, 0.2, 0.3, 0.5]:
    clf_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=c_val, class_weight='balanced', random_state=42)
    clf_l1.fit(X_s, y)
    active = np.where(clf_l1.coef_[0] != 0)[0]
    candidate_feature_sets[f"L1_C{c_val} ({len(active)} feat)"] = [feature_names[i] for i in active]

# Pure physical feature sets
candidate_feature_sets["Resource_Divergence_5"] = ['mean_path_diff', 'mean_end_diff', 'std_tv_path', 'bio_cv', 'c_hn']
candidate_feature_sets["Gore_Physical_6"] = ['mean_path_diff', 'bio_cv', 'bio_final', 'max_sp_cv', 'c_ln', 'c_hn']
candidate_feature_sets["All_26"] = feature_names

models_to_test = {
    "Logistic L2 (C=0.5)": lambda: LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42),
    "Logistic L2 (C=1.0)": lambda: LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42),
    "Logistic L1 (C=0.5)": lambda: LogisticRegression(penalty='l1', solver='liblinear', C=0.5, class_weight='balanced', random_state=42),
    "ExtraTrees (d=4)": lambda: ExtraTreesClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42),
    "RandomForest (d=4)": lambda: RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42),
}

benchmark_rows = []
for set_name, f_list in candidate_feature_sets.items():
    f_idx = [feature_names.index(f) for f in f_list]
    X_sub = X[:, f_idx]
    
    for m_name, m_fn in models_to_test.items():
        res = evaluate_nested(m_fn, X_sub, y)
        benchmark_rows.append({
            'Set': set_name,
            'N_Feats': len(f_list),
            'Model': m_name,
            'Nested_F1': res['nested_f1'],
            'Nested_Acc': res['nested_acc'],
            'Nested_Rec': res['nested_rec'],
            'Nested_Prec': res['nested_prec'],
            'AUC': res['auc'],
            'Oracle_F1': res['oracle_f1'],
            'Oracle_Rec': res['oracle_rec'],
            'Oracle_Prec': res['oracle_prec'],
            'Oracle_Acc': res['oracle_acc']
        })

df_bench = pd.DataFrame(benchmark_rows).sort_values('Nested_F1', ascending=False)
print(df_bench.head(15).to_string(index=False))


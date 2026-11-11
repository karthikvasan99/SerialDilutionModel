import sys, os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

from src.scratch_k2_resource_path_dynamics import X, y, feature_names, all_keys

print(f"X shape: {X.shape}, Features: {len(feature_names)}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Test various feature subsets & model families
# Specifically, we know:
# - Resource path features (mean_path_consec_diff, mean_end_consec_diff, end_cv, end_var)
# - Biomass features (bio_cv, bio_final)
# - Species volatility (max_sp_cv, mean_sp_cv)
# - Environment (c_hn, c_ln)

# Let's test non-linear kernels: RBF SVM, MLP, Gradient Boosting, ExtraTrees
models = {
    'RBF SVM (C=1.0, gamma=scale)': Pipeline([('s', StandardScaler()), ('clf', SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42))]),
    'RBF SVM (C=2.0, gamma=0.05)': Pipeline([('s', StandardScaler()), ('clf', SVC(kernel='rbf', C=2.0, gamma=0.05, probability=True, class_weight='balanced', random_state=42))]),
    'RBF SVM (C=5.0, gamma=0.02)': Pipeline([('s', StandardScaler()), ('clf', SVC(kernel='rbf', C=5.0, gamma=0.02, probability=True, class_weight='balanced', random_state=42))]),
    'MLP (32, 16)': Pipeline([('s', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, alpha=0.1, random_state=42))]),
    'MLP (16)': Pipeline([('s', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, alpha=0.5, random_state=42))]),
    'Logistic L1 (C=0.5)': Pipeline([('s', StandardScaler()), ('clf', LogisticRegression(penalty='l1', solver='liblinear', C=0.5, class_weight='balanced', random_state=42))]),
    'Logistic L1 (C=1.0)': Pipeline([('s', StandardScaler()), ('clf', LogisticRegression(penalty='l1', solver='liblinear', C=1.0, class_weight='balanced', random_state=42))]),
    'Logistic ElasticNet (C=0.5)': Pipeline([('s', StandardScaler()), ('clf', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.5, class_weight='balanced', max_iter=2000, random_state=42))]),
    'ExtraTrees (n=200, d=4, mf=0.5)': ExtraTreesClassifier(n_estimators=200, max_depth=4, max_features=0.5, class_weight='balanced', random_state=42),
    'Random Forest (n=200, d=4, mf=0.5)': RandomForestClassifier(n_estimators=200, max_depth=4, max_features=0.5, class_weight='balanced', random_state=42),
    'Gradient Boosting (n=50, lr=0.05, d=2)': GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=2, random_state=42)
}

results = []

for name, clf in models.items():
    try:
        y_prob = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]
    except:
        # Fallback for models without predict_proba
        y_prob = cross_val_predict(clf, X, y, cv=skf, method='decision_function')
        # Min-max scale
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
        
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_t = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_prob >= best_t).astype(int)
    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    prec = precisions[best_idx]
    rec = recalls[best_idx]
    
    results.append({
        'name': name,
        'f1': best_f1,
        'prec': prec,
        'rec': rec,
        'acc': acc,
        'tp': cm[1, 1],
        'fp': cm[0, 1],
        'fn': cm[1, 0],
        'tn': cm[0, 0]
    })
    print(f"{name:38s} | F1 = {best_f1:.4f} | Prec = {prec:.3f} | Rec = {rec:.3f} | Acc = {acc*100:.1f}% | TP:{cm[1,1]:2d} FP:{cm[0,1]:2d} FN:{cm[1,0]:2d} TN:{cm[0,0]:2d}")

df_res = pd.DataFrame(results).sort_values('f1', ascending=False)
print("\nTop 5 Models by F1 Score (Strictly K=2):")
print(df_res[['name', 'f1', 'prec', 'rec', 'acc', 'tp', 'fp', 'fn', 'tn']].head(5).to_string(index=False))

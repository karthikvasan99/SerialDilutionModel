import sys, os
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from src.compare_media_by_media_confusion import df_all

rows = []
for nut in ['LN', 'MN', 'HN']:
    for s_val in [3, 6, 12, 24, 48]:
        sub = df_all[(df_all.Nutrient == nut) & (df_all.S == s_val)]
        n_comms = len(sub)
        y_true = sub.y_true.values
        y_pred = sub.pred_with.values
        
        n_chaos = int(np.sum(y_true == 1))
        n_stable = int(np.sum(y_true == 0))
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, zero_division=0) if n_chaos > 0 else np.nan
        f1 = f1_score(y_true, y_pred, zero_division=0) if n_chaos > 0 else np.nan
        
        rows.append({
            'Nutrient': nut,
            'S': s_val,
            'Comms': n_comms,
            'Chaos': n_chaos,
            'Stable': n_stable,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Accuracy': f"{acc*100:.1f}%",
            'Recall': f"{rec*100:.1f}%" if not np.isnan(rec) else "N/A",
            'F1': f"{f1:.4f}" if not np.isnan(f1) else "N/A"
        })

df_res = pd.DataFrame(rows)
print("="*95)
print("FULL BREAKDOWN BY NUTRIENT MEDIA AND SPECIES RICHNESS (S) - CHAMPION 10-FEATURE MODEL")
print("="*95)
print(df_res.to_string(index=False))


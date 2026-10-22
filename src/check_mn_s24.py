import sys, os
sys.path.insert(0, '.')
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.compare_media_by_media_confusion import df_all

sub_mn24 = df_all[(df_all.Nutrient == 'MN') & (df_all.S == 24)]
print("--- ALL 8 COMMUNITIES FOR MN S=24 ---")
print(sub_mn24[['Comm', 'y_true', 'prob_with', 'pred_with']].to_string(index=False))

cm = confusion_matrix(sub_mn24.y_true, sub_mn24.pred_with, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion matrix for MN S=24:")
print(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
print(f"  Total: {len(sub_mn24)} communities")
print(f"  True Chaos: {sub_mn24.y_true.sum()}, True Stable: {len(sub_mn24) - sub_mn24.y_true.sum()}")

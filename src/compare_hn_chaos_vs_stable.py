import sys, os
sys.path.insert(0, '.')
import pandas as pd
from src.scratch_k2_ultimate_f1 import all_keys, y, X, feature_names

hn_indices = [i for i, k in enumerate(all_keys) if k[0] == 'HN']
df_hn = pd.DataFrame(X[hn_indices], columns=feature_names)
df_hn['S'] = [all_keys[i][1] for i in hn_indices]
df_hn['Comm'] = [all_keys[i][2] for i in hn_indices]
df_hn['y_true'] = y[hn_indices]

df_hn12 = df_hn[df_hn.S == 12]
print("HN S=12 COMMUNITIES COMPARISON:")
cols_to_compare = ['Comm', 'y_true', 'mean_tv_path', 'mean_path_diff', 'bio_cv', 'bio_diff_consec', 'max_sp_cv', 'mean_overlap', 'surv_count']
print(df_hn12[cols_to_compare].sort_values('y_true').to_string(index=False))

print("\nMean feature values for HN S=12 (Chaos vs Stable):")
print(df_hn12.groupby('y_true')[['mean_tv_path', 'mean_path_diff', 'bio_cv', 'bio_diff_consec', 'max_sp_cv', 'mean_overlap', 'surv_count']].mean())

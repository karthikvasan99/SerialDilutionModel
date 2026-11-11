import sys, os
sys.path.insert(0, '.')
import numpy as np
from src.test_extended_passages_classifier import extract_feature_matrix

for (p_s, p_e) in [(7, 10), (11, 20), (21, 30)]:
    X_mat, f_names = extract_feature_matrix(p_s, p_e)
    nan_count = np.isnan(X_mat).sum()
    inf_count = np.isinf(X_mat).sum()
    print(f"Window [{p_s}, {p_e}]: NaNs = {nan_count}, Infs = {inf_count}")
    if nan_count > 0 or inf_count > 0:
        bad_cols = [f_names[j] for j in range(len(f_names)) if np.isnan(X_mat[:, j]).any() or np.isinf(X_mat[:, j]).any()]
        print("  Bad columns:", bad_cols)
        for col_name in bad_cols:
            col = X_mat[:, f_names.index(col_name)]
            print(f"    {col_name}: has_nan={np.isnan(col).any()}, has_inf={np.isinf(col).any()}")


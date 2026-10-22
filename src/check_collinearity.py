import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from src.scratch_k2_ultimate_f1 import X, feature_names

top_feats = ['std_tv_path', 'mean_end_diff', 'mean_tv_path', 'end_cv', 'mean_path_diff', 'bio_final', 'bio_cv', 'c_hn']
feat_idx = [feature_names.index(f) for f in top_feats]
X_top = X[:, feat_idx]

df_corr = pd.DataFrame(np.corrcoef(X_top.T), index=top_feats, columns=top_feats)
print("Correlation matrix among top candidate features:")
print(df_corr.round(3).to_string())

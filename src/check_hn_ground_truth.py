import sys, os
sys.path.insert(0, '.')
import pandas as pd
from src.scratch_k2_ultimate_f1 import all_keys, y

df_hn = pd.DataFrame([
    {'Nutrient': k[0], 'S': k[1], 'Comm': k[2], 'y_true': y[i]}
    for i, k in enumerate(all_keys)
    if k[0] == 'HN'
])

print("Ground truth chaos in HN (High Nutrient):")
print(df_hn.groupby('S')['y_true'].value_counts())
print("\nAll HN communities:")
print(df_hn.to_string(index=False))

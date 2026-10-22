import pandas as pd
df = pd.read_csv('Results/Gore_PhysODE_LowKSweep_Summary.csv')
print(df[['name', 'K', 'mean_rel_r2', 'median_rel_r2', 'global_bio_r', 'auc_chaos', 'ln_rel_r2', 'mn_rel_r2', 'hn_rel_r2']].to_string())

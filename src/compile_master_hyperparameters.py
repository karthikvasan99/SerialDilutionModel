import os
import pandas as pd

PROJECT_ROOT = '/home/karthik/SerialDilutionModel'
data_dir = os.path.join(PROJECT_ROOT, 'Data')

# 1. First initial search
df_init = pd.read_csv(os.path.join(data_dir, 'LOOCV_Hyperparameter_Search_ScalarAttach.csv'))
df_init['search_stage'] = 'Stage1_Initial_Grid'
df_init['b0_mode'] = 'fixed'

# 2. Edge exploration
df_edge = pd.read_csv(os.path.join(data_dir, 'LOOCV_Edge_Exploration_ScalarAttach.csv'))
df_edge['search_stage'] = 'Stage2_Edge_Exploration'
df_edge['b0_mode'] = 'fixed'

# 3. Learnable b0 comparison
df_b0 = pd.read_csv(os.path.join(data_dir, 'Learnable_b0_LOOCV_Comparison.csv'))
df_b0['search_stage'] = 'Stage3_Learnable_b0'
df_b0['K'] = 4
df_b0['c_w'] = 0.70
df_b0['reg_alpha'] = 0.001
df_b0['reg_w'] = 0.001
df_b0['params'] = [325, 326, 327]
df_b0['b0_val'] = [1.60, 0.33, 0.46]
df_b0.rename(columns={'mode': 'b0_mode'}, inplace=True)

# 4. Learnable b0 K sweep
df_k = pd.read_csv(os.path.join(data_dir, 'Learnable_b0_K_Sweep.csv'))
df_k['search_stage'] = 'Stage4_K_Sweep_Learnable_b0'
df_k['K'] = [3, 4, 5]
df_k['c_w'] = 0.70
df_k['reg_alpha'] = 0.001
df_k['reg_w'] = 0.001
df_k['params'] = [268, 327, 388]
df_k['b0_val'] = [0.61, 0.46, 0.45]
df_k.rename(columns={'mode': 'b0_mode'}, inplace=True)

# Merge all into unified dataframe
cols = ['search_stage', 'K', 'c_w', 'b0_mode', 'b0_val', 'reg_alpha', 'reg_w', 'params', 'mean_val_kl', 'std_val_kl', 'val_kl_no', 'val_kl_mu', 'mean_b0_liq', 'mean_b0_sup']

for d in [df_init, df_edge]:
    d['mean_b0_liq'] = d['b0_val']
    d['mean_b0_sup'] = d['b0_val']

unified_df = pd.concat([df_init, df_edge, df_b0, df_k], ignore_index=True)
unified_df = unified_df[cols]

# Drop duplicate rows (e.g. baseline anchor rows run in both stage 1 and stage 2)
unified_df.drop_duplicates(subset=['K', 'c_w', 'b0_mode', 'b0_val', 'reg_alpha'], keep='last', inplace=True)
unified_df.sort_values(by='mean_val_kl', ascending=True, inplace=True)

master_path = os.path.join(data_dir, 'Master_LOOCV_Hyperparameter_Optimization.csv')
unified_df.to_csv(master_path, index=False)
print(f"Successfully compiled master hyperparameter search results into {master_path}")
print(f"Total evaluated configurations: {len(unified_df)}")
print("\nTop 5 performing models:")
print(unified_df[['K', 'c_w', 'b0_mode', 'mean_val_kl', 'std_val_kl', 'val_kl_no', 'val_kl_mu']].head(5).to_string())

import pandas as pd
df_a = pd.read_csv('Results/Gore_PhaseA_K_Sweep_Summary.csv')
print("=== Phase A K Sweep Summary ===")
print(df_a[['K', 'mean_r2_linear', 'median_r2_linear', 'mean_r2_log', 'ln_r2_linear', 'mn_r2_linear', 'hn_r2_linear']].to_string())

if os.path.exists('Results/Gore_PhaseC_Optimization_Detailed.csv'):
    df_c = pd.read_csv('Results/Gore_PhaseC_Optimization_Detailed.csv')
    print("\n=== Phase C Detailed Summary by Config ===")
    print(df_c.groupby('config')[['r2_linear', 'r2_log', 'r2_endpoint_linear']].agg(['mean', 'median']).to_string())

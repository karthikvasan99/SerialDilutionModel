import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

def prepare_data_averaged_p0(csv_path):
    df = pd.read_csv(csv_path)
    
    sums = df[df['sampleType'] != 'inoculum'].groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
    if sums.mean() > 50:
        df['Relative_Abundance'] = df['Read_Fraction'] / 100.0
    else:
        df['Relative_Abundance'] = df['Read_Fraction']
        
    df_non_inoc = df[df['sampleType'] != 'inoculum'].copy()
    
    sample_cols = ['sampleType', 'passage', 'replicate']
    num_samples = len(df_non_inoc.drop_duplicates(subset=sample_cols))
    mean_abundances = df_non_inoc.groupby('Strain_Name')['Relative_Abundance'].sum() / num_samples
    strains_keep = mean_abundances[mean_abundances >= 1e-3].index
    
    df_filtered = df[df['Strain_Name'].isin(strains_keep)].copy()
    
    df_p0 = df_filtered[df_filtered['sampleType'] == 'inoculum']
    p0_avg = df_p0.groupby('Strain_Name')['Relative_Abundance'].mean()
    
    df_envs = df_filtered[df_filtered['sampleType'] != 'inoculum']
    env_avg = df_envs.groupby(['sampleType', 'passage', 'Strain_Name'])['Relative_Abundance'].mean().reset_index()
    
    pivot_envs = env_avg.pivot_table(index=['sampleType', 'passage'], columns='Strain_Name', values='Relative_Abundance', fill_value=0.0)
    
    unique_envs = list(pivot_envs.index.get_level_values('sampleType').unique())
    passages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    num_species = len(strains_keep)
    
    p0_vec = np.zeros(num_species)
    for i, strain in enumerate(pivot_envs.columns):
        if strain in p0_avg.index:
            p0_vec[i] = p0_avg[strain]
            
    trajectories = []
    
    for env_idx, env in enumerate(unique_envs):
        seq = [p0_vec]
        for p in passages:
            if (env, p) in pivot_envs.index:
                seq.append(pivot_envs.loc[(env, p)].values)
            else:
                seq.append(np.zeros(num_species))
        
        seq_arr = np.array(seq) + 1e-6
        seq_arr = seq_arr / seq_arr.sum(axis=1, keepdims=True)
        trajectories.append(seq_arr)
        
    return np.array(trajectories), unique_envs

def main():
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv')
    trajectories, unique_envs = prepare_data_averaged_p0(data_path)
    
    null_r2 = []
    for idx, env in enumerate(unique_envs):
        seq = trajectories[idx]
        
        # P3-P6 are indices 3, 4, 5, 6. Their previous passages are 2, 3, 4, 5.
        target = seq[3:7].flatten()
        pred = seq[2:6].flatten()
        
        r_val, _ = pearsonr(target, pred)
        r2 = r_val**2 if not np.isnan(r_val) else 0.0
        null_r2.append(r2)
        
        print(f"{env}: {r2:.4f}")
        
    print(null_r2)

if __name__ == '__main__':
    main()

import os
import sys
import torch
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

data_path = os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv')
df = pd.read_csv(data_path)
df = df[df['sampleType'] != 'inoculum'].copy()

sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
df['Relative_Abundance'] = df['Read_Fraction'] / 100.0 if sums.mean() > 50 else df['Read_Fraction']

sample_cols = ['sampleType', 'passage', 'replicate']
num_samples = len(df.drop_duplicates(subset=sample_cols))
mean_abundances = df.groupby('Strain_Name')['Relative_Abundance'].sum() / num_samples
strains_keep = sorted(list(mean_abundances[mean_abundances >= 1e-3].index))
N = len(strains_keep)
print(f"Retained N = {N} strains across Pollard dataset.")

# Replicates: R1a, R1b, R1c, R2a, R2b, R2c, R3a, R3b, R3c
vessels = ['R1', 'R2', 'R3']
passages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
sub_reps = {'R1': ['R1a', 'R1b', 'R1c'],
            'R2': ['R2a', 'R2b', 'R2c'],
            'R3': ['R3a', 'R3b', 'R3c']}

def get_vessel_tensors(sample_type):
    sub = df[df['sampleType'] == sample_type].pivot_table(
        index=['replicate', 'passage'],
        columns='Strain_Name',
        values='Relative_Abundance',
        fill_value=0.0
    ).reindex(columns=strains_keep, fill_value=0.0)
    
    # Normalize per sample
    norm = (sub + 1e-6).div((sub + 1e-6).sum(axis=1), axis=0)
    
    vessel_means = []
    vessel_sems = []
    
    for v in vessels:
        reps = sub_reps[v]
        # shape: (3, 6, 50)
        v_data = np.array([norm.loc[rep].reindex(passages).values for rep in reps])
        
        # Mean across spatial replicates a, b, c
        v_mean = np.mean(v_data, axis=0) # shape (6, 50)
        # Re-normalize mean to strictly sum to 1
        v_mean = v_mean / v_mean.sum(axis=-1, keepdims=True)
        
        # Spatial SEM across a, b, c
        v_sem = np.std(v_data, axis=0, ddof=1) / np.sqrt(len(reps))
        
        vessel_means.append(v_mean)
        vessel_sems.append(v_sem)
        
    tensor_mean = torch.tensor(np.array(vessel_means), dtype=torch.float32) # (3, 6, 50)
    tensor_sem = torch.tensor(np.array(vessel_sems), dtype=torch.float32)   # (3, 6, 50)
    return tensor_mean, tensor_sem

print("Generating spatially averaged tensors...")
no_c_tensor, no_c_sem = get_vessel_tensors('no carriers')
plain_c_tensor, plain_c_sem = get_vessel_tensors('plain-agar carriers')
plain_s_tensor, plain_s_sem = get_vessel_tensors('plain-agar supernatant')
mucin_c_tensor, mucin_c_sem = get_vessel_tensors('mucin-agar carriers')
mucin_s_tensor, mucin_s_sem = get_vessel_tensors('mucin-agar supernatant')

print(f"no_c_tensor shape: {no_c_tensor.shape}")
print(f"plain_c_tensor shape: {plain_c_tensor.shape}")
print(f"mucin_c_tensor shape: {mucin_c_tensor.shape}")

out_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
torch.save({
    'strains': strains_keep,
    'vessels': vessels,
    'passages': passages,
    'no_c_tensor': no_c_tensor,
    'no_c_sem': no_c_sem,
    'plain_c_tensor': plain_c_tensor,
    'plain_c_sem': plain_c_sem,
    'plain_s_tensor': plain_s_tensor,
    'plain_s_sem': plain_s_sem,
    'mucin_c_tensor': mucin_c_tensor,
    'mucin_c_sem': mucin_c_sem,
    'mucin_s_tensor': mucin_s_tensor,
    'mucin_s_sem': mucin_s_sem
}, out_path)

print(f"Successfully saved clean biological vessel tensors to: {out_path}")

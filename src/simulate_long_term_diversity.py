import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.scratch_evaluate_k2_800ep import all_keys, gt_labels, all_m, all_N0, all_c
from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 1. Load K=2 Physical Model
ode_k2 = AdaptivePureConsumerResourceODE(N=48, K=2)
model_k2 = HierarchicalPhysicalNeuralODE(ode_k2, num_conditions=3, K=2).to(device)
model_k2.load_state_dict(torch.load('Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth', weights_only=False))
model_k2.eval()

# Clamp W to >= 0 for mass conservation
model_k2.ode_func.W.data = torch.clamp(model_k2.ode_func.W.data, min=0.0)

num_passages = 100
delta_t = 0.05
tau_steps = int(1.0 / delta_t) + 1

print(f"Simulating K=2 Physical Model for {num_passages} serial dilution passages (days)...")

with torch.no_grad():
    R0_table = model_k2.get_R0()
    R0 = R0_table[all_c]
    yield_scale = F.softplus(model_k2.yield_raw)[all_c].unsqueeze(-1)
    
    N_curr = all_N0.clone()
    all_N_passages = [N_curr.cpu().numpy()]
    
    for p in range(1, num_passages + 1):
        N0 = torch.clamp(N_curr / model_k2.dilution_factor + model_k2.dispersal_rate * all_m, min=0.0)
        S0 = torch.cat([N0, R0], dim=-1)
        
        from torchdiffeq import odeint
        t_span = torch.linspace(0.0, 1.0, tau_steps, device=device)
        S_t = odeint(model_k2.ode_func, S0, t_span, method='euler', options={'step_size': delta_t})
        
        N_final = torch.clamp(S_t[..., :48][-1] * all_m, min=0.0) * yield_scale
        N_curr = N_final
        all_N_passages.append(N_final.cpu().numpy())
        
        if p % 20 == 0:
            print(f"  Completed passage {p}/{num_passages}...")

all_N = np.stack(all_N_passages, axis=1) # (99, num_passages + 1, 48)
all_bio = all_N.sum(axis=-1)             # (99, num_passages + 1)
all_rel = all_N / np.where(all_bio[..., None] > 0, all_bio[..., None], 1.0)

threshold_rel = 1e-3

timepoints = [0, 1, 5, 10, 20, 30, 50, 75, 100]

richness_data = []
shannon_data = []

for p in range(num_passages + 1):
    rel_p = all_rel[:, p, :] # (99, 48)
    surv_p = (rel_p >= threshold_rel).sum(axis=-1) # (99,)
    
    p_safe = np.where(rel_p > 1e-12, rel_p, 1e-12)
    shannon_p = -np.sum(np.where(rel_p >= 1e-12, rel_p * np.log(p_safe), 0.0), axis=-1)
    
    richness_data.append(surv_p)
    shannon_data.append(shannon_p)

richness_arr = np.stack(richness_data, axis=1) # (99, num_passages + 1)
shannon_arr = np.stack(shannon_data, axis=1)   # (99, num_passages + 1)
effective_sp = np.exp(shannon_arr)             # (99, num_passages + 1)

df_communities = pd.DataFrame({
    'Nutrient': [k[0] for k in all_keys],
    'S_initial': [k[1] for k in all_keys],
    'Community': [k[2] for k in all_keys]
})

print("\n" + "="*85)
print("LONG-TERM DIVERSITY AUDIT: SURVIVING SPECIES COUNT ACROSS PASSAGES")
print("="*85)

for s in [3, 6, 12, 24, 48]:
    idx = df_communities[df_communities.S_initial == s].index
    print(f"\n--- INITIAL POOL S = {s} ({len(idx)} communities) ---")
    header = f"{'Nutrient':<8} " + " ".join([f"Day {t:<4}" for t in timepoints])
    print(header)
    print("-" * len(header))
    
    for nut in ['LN', 'MN', 'HN']:
        nut_idx = df_communities[(df_communities.S_initial == s) & (df_communities.Nutrient == nut)].index
        if len(nut_idx) == 0:
            continue
        vals = [f"{richness_arr[nut_idx, t].mean():.1f}±{richness_arr[nut_idx, t].std():.1f}" for t in timepoints]
        print(f"{nut:<8} " + " ".join([f"{v:<8}" for v in vals]))

print("\n" + "="*85)
print("EFFECTIVE NUMBER OF SPECIES (exp(Shannon H)) ACROSS PASSAGES")
print("="*85)
for s in [3, 6, 12, 24, 48]:
    idx = df_communities[df_communities.S_initial == s].index
    print(f"\n--- INITIAL POOL S = {s} (Effective Species exp(H)) ---")
    header = f"{'Nutrient':<8} " + " ".join([f"Day {t:<4}" for t in timepoints])
    print(header)
    print("-" * len(header))
    for nut in ['LN', 'MN', 'HN']:
        nut_idx = df_communities[(df_communities.S_initial == s) & (df_communities.Nutrient == nut)].index
        if len(nut_idx) == 0:
            continue
        vals = [f"{effective_sp[nut_idx, t].mean():.2f}" for t in timepoints]
        print(f"{nut:<8} " + " ".join([f"{v:<8}" for v in vals]))

day100_surv = richness_arr[:, 100]
crashed_to_1 = np.sum(day100_surv == 1)
crashed_to_2 = np.sum(day100_surv == 2)
crashed_le_2 = np.sum(day100_surv <= 2)
maintained_ge_3 = np.sum(day100_surv >= 3)
maintained_ge_4 = np.sum(day100_surv >= 4)
maintained_ge_6 = np.sum(day100_surv >= 6)

print("\n" + "="*85)
print("DAY 100 SURVIVAL DISTRIBUTION ACROSS ALL 99 COMMUNITIES")
print("="*85)
print(f"Total communities evaluated: {len(day100_surv)}")
print(f"Communities with <= 2 species at Day 100: {crashed_le_2} / 99 ({crashed_le_2/99*100:.1f}%)")
print(f"Communities maintaining >= 3 species at Day 100: {maintained_ge_3} / 99 ({maintained_ge_3/99*100:.1f}%)")
print(f"Communities maintaining >= 4 species at Day 100: {maintained_ge_4} / 99 ({maintained_ge_4/99*100:.1f}%)")
print(f"Communities maintaining >= 6 species at Day 100: {maintained_ge_6} / 99 ({maintained_ge_6/99*100:.1f}%)")

print("\n" + "="*85)
print("HIGH-RICHNESS COMMUNITIES AT DAY 100 (S=24 and S=48)")
print("="*85)
high_s_idx = df_communities[df_communities.S_initial >= 24].index
for i in high_s_idx:
    row = df_communities.iloc[i]
    print(f"{row.Nutrient} S={row.S_initial:<2d} {row.Community}: Initial={row.S_initial:<2d} -> Day 10={richness_arr[i, 10]:2d} -> Day 30={richness_arr[i, 30]:2d} -> Day 100={richness_arr[i, 100]:2d} species (exp(H)={effective_sp[i, 100]:.2f})")

# Save long term simulation array for plotting
np.save('Data/Gore_LongTerm_Abundances_100p.npy', all_N)
print("\nSaved long-term trajectories to Data/Gore_LongTerm_Abundances_100p.npy")

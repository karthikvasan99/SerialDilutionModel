import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torchdiffeq import odeint

from SerialDilutionModels import LatentAssemblyNeuralODE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running Cross-Dataset Diversity Audit on: {device}")

# =====================================================================
# 1. GERMAN'S DATASET (Rumen Microbiome, N=32, K=4, 12 Trajectories)
# =====================================================================
print("\n" + "="*80)
print("1. AUDITING GERMAN'S DATASET (Rumen Microbiome)")
print("="*80)

german_ckpt = torch.load('Results/German_ODE_full_model.pth', map_location='cpu', weights_only=False)
model_g = LatentAssemblyNeuralODE(num_species=32, num_latent_dims=4, num_conditions=6).to(device)
model_g.load_state_dict(german_ckpt['model_state_dict'])
model_g.eval()

# Clamp W >= 0 for mass conservation
model_g.ode_func.W.data = torch.clamp(model_g.ode_func.W.data, min=0.0)

# Load German T1 data
german_data = torch.load('Data/German_processed_data.pt', map_location='cpu', weights_only=False)
strains_g = german_ckpt['strains']
trajectories_g = german_data['trajectories']

# Filter to 32 modeled strains
keep_indices = [german_data['strains'].index(s) for s in strains_g]

initial_states_g = []
conditions_g = []
labels_g = []

env_map = german_ckpt['env_map']
for cond_key, seq in trajectories_g.items():
    filt_seq = seq[:, keep_indices]
    filt_seq = filt_seq / filt_seq.sum(dim=-1, keepdim=True)
    initial_states_g.append(filt_seq[0]) # T1 initial state
    conditions_g.append(env_map[cond_key])
    labels_g.append(f"{cond_key[0]}_{cond_key[1]}_{cond_key[2]}bar")

initial_states_g = torch.stack(initial_states_g).to(device) # (12, 32)
conditions_g = torch.tensor(conditions_g, dtype=torch.long, device=device) # (12,)

num_passages = 100
delta_t = 0.05
tau_steps = int(1.0 / delta_t) + 1

with torch.no_grad():
    a_batch = model_g.a[conditions_g]
    R0 = F.softplus(a_batch) # (12, 4)
    
    N_curr = initial_states_g.clone()
    german_passages = [N_curr.cpu().numpy()]
    
    for p in range(1, num_passages + 1):
        # 1:100 dilution (German default_dilution_factor is 100) + 1e-6 dispersal
        N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
        S0 = torch.cat([N0, R0], dim=-1)
        
        t_span = torch.linspace(0.0, 1.0, tau_steps, device=device)
        S_t = odeint(model_g.ode_func, S0, t_span, method='euler', options={'step_size': delta_t})
        
        N_final = torch.clamp(S_t[-1, :, :32], min=0.0)
        # Normalize to relative abundance
        rel_final = N_final / N_final.sum(dim=-1, keepdim=True)
        N_curr = rel_final
        german_passages.append(rel_final.cpu().numpy())

german_arr = np.stack(german_passages, axis=1) # (12, 101, 32)
surv_g = (german_arr >= 1e-3).sum(axis=-1)     # (12, 101)
p_g_safe = np.where(german_arr > 1e-12, german_arr, 1e-12)
eff_g = np.exp(-np.sum(np.where(german_arr >= 1e-12, german_arr * np.log(p_g_safe), 0.0), axis=-1))

print(f"German Communities (N=12 trajectories):")
print(f"  Day 0  (Initial): Surviving Species = {surv_g[:, 0].mean():.1f} ± {surv_g[:, 0].std():.1f} | exp(H) = {eff_g[:, 0].mean():.2f}")
print(f"  Day 6  (Exp End): Surviving Species = {surv_g[:, 6].mean():.1f} ± {surv_g[:, 6].std():.1f} | exp(H) = {eff_g[:, 6].mean():.2f}")
print(f"  Day 20 (Mid):     Surviving Species = {surv_g[:, 20].mean():.1f} ± {surv_g[:, 20].std():.1f} | exp(H) = {eff_g[:, 20].mean():.2f}")
print(f"  Day 50 (Late):    Surviving Species = {surv_g[:, 50].mean():.1f} ± {surv_g[:, 50].std():.1f} | exp(H) = {eff_g[:, 50].mean():.2f}")
print(f"  Day 100(Steady):  Surviving Species = {surv_g[:, 100].mean():.1f} ± {surv_g[:, 100].std():.1f} | exp(H) = {eff_g[:, 100].mean():.2f}")
print(f"  Communities with >= 3 species at Day 100: {(surv_g[:, 100] >= 3).sum()}/12 ({(surv_g[:, 100] >= 3).mean()*100:.1f}%)")
print(f"  Communities with >= 6 species at Day 100: {(surv_g[:, 100] >= 6).sum()}/12 ({(surv_g[:, 100] >= 6).mean()*100:.1f}%)")

# =====================================================================
# 2. POLLARD'S DATASET (50 Strains, K=5, 20 Trajectories)
# =====================================================================
print("\n" + "="*80)
print("2. AUDITING POLLARD'S DATASET (50-Strain Synthetic Gut Microbiome)")
print("="*80)

from Pollard_Train_ODE_50_Extended_v2 import prepare_data

data_path_p = 'Data/SupplementaryDataS2.csv'
trajectories_p, conditions_p, unique_envs_p, num_species_p = prepare_data(data_path_p)

model_p = LatentAssemblyNeuralODE(num_species=50, num_latent_dims=5, num_conditions=5).to(device)
model_p.load_state_dict(torch.load('Data/Pollard_50Strains_ODE_model_Extended_v2.pth', map_location='cpu', weights_only=False))
model_p.eval()
model_p.ode_func.W.data = torch.clamp(model_p.ode_func.W.data, min=0.0)

initial_states_p = trajectories_p[:, 0, :] # (20, 50)
num_trajs_p = trajectories_p.size(0)

with torch.no_grad():
    a_batch_p = model_p.a[conditions_p]
    R0_p = F.softplus(a_batch_p) # (20, 5)
    
    N_curr = initial_states_p.clone()
    pollard_passages = [N_curr.cpu().numpy()]
    
    for p in range(1, num_passages + 1):
        N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
        S0 = torch.cat([N0, R0_p], dim=-1)
        
        t_span = torch.linspace(0.0, 1.0, tau_steps, device=device)
        S_t = odeint(model_p.ode_func, S0, t_span, method='euler', options={'step_size': delta_t})
        
        N_final = torch.clamp(S_t[-1, :, :50], min=0.0)
        rel_final = N_final / N_final.sum(dim=-1, keepdim=True)
        N_curr = rel_final
        pollard_passages.append(rel_final.cpu().numpy())

pollard_arr = np.stack(pollard_passages, axis=1) # (20, 101, 50)
surv_p = (pollard_arr >= 1e-3).sum(axis=-1)      # (20, 101)
p_p_safe = np.where(pollard_arr > 1e-12, pollard_arr, 1e-12)
eff_p = np.exp(-np.sum(np.where(pollard_arr >= 1e-12, pollard_arr * np.log(p_p_safe), 0.0), axis=-1))

print(f"Pollard Communities (N={num_trajs_p} trajectories across 5 conditions):")
print(f"  Day 0  (Initial): Surviving Species = {surv_p[:, 0].mean():.1f} ± {surv_p[:, 0].std():.1f} | exp(H) = {eff_p[:, 0].mean():.2f}")
print(f"  Day 6  (Exp End): Surviving Species = {surv_p[:, 6].mean():.1f} ± {surv_p[:, 6].std():.1f} | exp(H) = {eff_p[:, 6].mean():.2f}")
print(f"  Day 20 (Mid):     Surviving Species = {surv_p[:, 20].mean():.1f} ± {surv_p[:, 20].std():.1f} | exp(H) = {eff_p[:, 20].mean():.2f}")
print(f"  Day 50 (Late):    Surviving Species = {surv_p[:, 50].mean():.1f} ± {surv_p[:, 50].std():.1f} | exp(H) = {eff_p[:, 50].mean():.2f}")
print(f"  Day 100(Steady):  Surviving Species = {surv_p[:, 100].mean():.1f} ± {surv_p[:, 100].std():.1f} | exp(H) = {eff_p[:, 100].mean():.2f}")
print(f"  Communities with >= 3 species at Day 100:  {(surv_p[:, 100] >= 3).sum()}/{num_trajs_p} ({(surv_p[:, 100] >= 3).mean()*100:.1f}%)")
print(f"  Communities with >= 10 species at Day 100: {(surv_p[:, 100] >= 10).sum()}/{num_trajs_p} ({(surv_p[:, 100] >= 10).mean()*100:.1f}%)")
print(f"  Communities with >= 20 species at Day 100: {(surv_p[:, 100] >= 20).sum()}/{num_trajs_p} ({(surv_p[:, 100] >= 20).mean()*100:.1f}%)")

# =====================================================================
# 3. OPHELIA'S DATASET (Venturelli 12 Species, K=2, 175 Communities)
# =====================================================================
print("\n" + "="*80)
print("3. AUDITING OPHELIA'S DATASET (Venturelli 12-Species Gut Microbiome)")
print("="*80)

from Ophelia_ODE_evaluation import load_venturelli_arrays

train_mat = loadmat('Data/train_data.mat')['train_data']
test_mat = loadmat('Data/test_data.mat')['test_data']
all_venturelli = np.concatenate([train_mat, test_mat], axis=0) # (175, 6, 12)

model_o = LatentAssemblyNeuralODE(num_species=12, num_latent_dims=2, num_conditions=1).to(device)
model_o.load_state_dict(torch.load('Data/Ophelia_ODE_model.pth', map_location='cpu', weights_only=False))
model_o.eval()
model_o.ode_func.W.data = torch.clamp(model_o.ode_func.W.data, min=0.0)

# Extract valid initial t0 relative abundances
ophelia_t0 = []
for sample_idx in range(all_venturelli.shape[0]):
    sample_seq = all_venturelli[sample_idx]
    if not np.isnan(sample_seq[0]).any():
        clean_t0 = np.clip(sample_seq[0], 0, None)
        tot = clean_t0.sum()
        if tot > 1e-8:
            ophelia_t0.append(clean_t0 / tot)

ophelia_t0 = torch.tensor(np.array(ophelia_t0), dtype=torch.float32, device=device) # (num_comms, 12)
num_comms_o = ophelia_t0.size(0)

with torch.no_grad():
    a_batch_o = model_o.a[torch.zeros(num_comms_o, dtype=torch.long, device=device)]
    R0_o = F.softplus(a_batch_o) # (num_comms, 2)
    
    N_curr = ophelia_t0.clone()
    ophelia_passages = [N_curr.cpu().numpy()]
    
    for p in range(1, num_passages + 1):
        N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
        S0 = torch.cat([N0, R0_o], dim=-1)
        
        t_span = torch.linspace(0.0, 1.0, tau_steps, device=device)
        S_t = odeint(model_o.ode_func, S0, t_span, method='euler', options={'step_size': delta_t})
        
        N_final = torch.clamp(S_t[-1, :, :12], min=0.0)
        rel_final = N_final / N_final.sum(dim=-1, keepdim=True)
        N_curr = rel_final
        ophelia_passages.append(rel_final.cpu().numpy())

ophelia_arr = np.stack(ophelia_passages, axis=1) # (N, 101, 12)
surv_o = (ophelia_arr >= 1e-3).sum(axis=-1)       # (N, 101)
p_o_safe = np.where(ophelia_arr > 1e-12, ophelia_arr, 1e-12)
eff_o = np.exp(-np.sum(np.where(ophelia_arr >= 1e-12, ophelia_arr * np.log(p_o_safe), 0.0), axis=-1))

print(f"Ophelia Communities (N={num_comms_o} synthetic communities):")
print(f"  Day 0  (Initial): Surviving Species = {surv_o[:, 0].mean():.1f} ± {surv_o[:, 0].std():.1f} | exp(H) = {eff_o[:, 0].mean():.2f}")
print(f"  Day 6  (Exp End): Surviving Species = {surv_o[:, 6].mean():.1f} ± {surv_o[:, 6].std():.1f} | exp(H) = {eff_o[:, 6].mean():.2f}")
print(f"  Day 20 (Mid):     Surviving Species = {surv_o[:, 20].mean():.1f} ± {surv_o[:, 20].std():.1f} | exp(H) = {eff_o[:, 20].mean():.2f}")
print(f"  Day 50 (Late):    Surviving Species = {surv_o[:, 50].mean():.1f} ± {surv_o[:, 50].std():.1f} | exp(H) = {eff_o[:, 50].mean():.2f}")
print(f"  Day 100(Steady):  Surviving Species = {surv_o[:, 100].mean():.1f} ± {surv_o[:, 100].std():.1f} | exp(H) = {eff_o[:, 100].mean():.2f}")
print(f"  Communities with >= 3 species at Day 100: {(surv_o[:, 100] >= 3).sum()}/{num_comms_o} ({(surv_o[:, 100] >= 3).mean()*100:.1f}%)")
print(f"  Communities with >= 4 species at Day 100: {(surv_o[:, 100] >= 4).sum()}/{num_comms_o} ({(surv_o[:, 100] >= 4).mean()*100:.1f}%)")
print(f"  Communities with >= 6 species at Day 100: {(surv_o[:, 100] >= 6).sum()}/{num_comms_o} ({(surv_o[:, 100] >= 6).mean()*100:.1f}%)")

# Save results for plotting
np.savez('Data/Cross_Dataset_LongTerm_Diversity.npz',
         german_arr=german_arr, surv_g=surv_g, eff_g=eff_g,
         pollard_arr=pollard_arr, surv_p=surv_p, eff_p=eff_p,
         ophelia_arr=ophelia_arr, surv_o=surv_o, eff_o=eff_o)
print("\nSaved cross-dataset long term diversity arrays to Data/Cross_Dataset_LongTerm_Diversity.npz")

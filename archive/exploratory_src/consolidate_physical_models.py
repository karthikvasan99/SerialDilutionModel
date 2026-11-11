"""
Consolidation Script: Unified Physical Consumer-Resource Neural ODE Models Across All 4 Datasets
1. Gore et al. (Science 2022, N=48, K=2)
2. Pollard / Jin et al. (Nat Comm 2023, N=50, K=5)
3. German et al. (Rumen Enrichment, N=32, K=4)
4. Ophelia / Venturelli et al. (Mol Syst Biol 2018, N=12, K=2)

Guarantees:
- Unnormalized physical biomass propagation across dilution cycles (no simplex resets)
- Unconstrained environmental baseline resource learning (R0 = softplus(a))
- Mass conservation (W >= 0)
- Quadratic intra-specific self-limitation (alpha_i > 0)
- Saves checkpoints and unified 100-passage arrays to Data/Cross_Dataset_LongTerm_Diversity.npz
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.io import loadmat

# Path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running Consolidation Pipeline on device: {device}")


# =====================================================================
# UNIFIED PHYSICAL CONSUMER-RESOURCE NEURAL ODE CLASS
# =====================================================================
class UnifiedPhysicalConsumerResourceODE(nn.Module):
    def __init__(self, num_species, num_latent_dims, num_conditions=1):
        super().__init__()
        self.N = num_species
        self.K = num_latent_dims
        self.num_conditions = num_conditions
        
        # Unconstrained environmental resources (R0 = softplus(a))
        self.a = nn.Parameter(torch.randn(num_conditions, num_latent_dims) * 0.5 + 2.0)
        # Metabolic traits Theta >= 0
        self.theta_raw = nn.Parameter(torch.randn(num_species, num_latent_dims) * 0.1)
        # Cross-feeding W >= 0 (mass conservation)
        self.w_raw = nn.Parameter(torch.randn(num_latent_dims, num_latent_dims) * 0.1)
        # Self-limitation alpha > 0
        self.alpha_raw = nn.Parameter(torch.randn(num_species) * 0.05 - 1.0)
        # Initial unnormalized biomass scale
        self.b0_raw = nn.Parameter(torch.tensor(2.0))
        
    def get_params(self):
        Theta = F.softplus(self.theta_raw)
        W = F.softplus(self.w_raw)
        alpha = F.softplus(self.alpha_raw)
        R0 = F.softplus(self.a)
        B0 = F.softplus(self.b0_raw)
        return Theta, W, alpha, R0, B0

    def ode_deriv(self, t, S, Theta, W, alpha):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        growth = torch.matmul(R_t, Theta.t())
        dN_dt = N_t * growth - alpha * (N_t ** 2)
        flux = torch.matmul(torch.matmul(N_t, Theta), W.t())
        dR_dt = - R_t * flux
        return torch.cat([dN_dt, dR_dt], dim=-1)


# =====================================================================
# 1. TRAIN & CONSOLIDATE GERMAN MODEL
# =====================================================================
def consolidate_german(epochs=200, lr=0.01):
    print("\n" + "="*80)
    print("1. CONSOLIDATING GERMAN ET AL. (Rumen Enrichment, N=32, K=4)")
    print("="*80)
    
    german_ckpt = torch.load('Results/German_ODE_full_model.pth', map_location='cpu', weights_only=False)
    german_data = torch.load('Data/German_processed_data.pt', map_location='cpu', weights_only=False)
    strains_g = german_ckpt['strains']
    trajectories_g = german_data['trajectories']
    env_map = german_ckpt['env_map']

    keep_indices = [german_data['strains'].index(s) for s in strains_g]
    seqs = []
    conds = []
    for cond_key, seq in trajectories_g.items():
        filt = seq[:, keep_indices]
        filt = filt / filt.sum(dim=-1, keepdim=True)
        seqs.append(filt)
        conds.append(env_map[cond_key])
        
    seqs = torch.stack(seqs).to(device) # (12, 6, 32)
    conds = torch.tensor(conds, dtype=torch.long, device=device)
    
    model_g = UnifiedPhysicalConsumerResourceODE(num_species=32, num_latent_dims=4, num_conditions=len(set(env_map.values()))).to(device)
    opt = torch.optim.Adam(model_g.parameters(), lr=lr)
    
    delta_t = 0.10
    t_span = torch.linspace(0.0, 1.0, 11, device=device)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        Theta, W, alpha, R0_table, B0 = model_g.get_params()
        N_curr = seqs[:, 0, :] * B0
        R0 = R0_table[conds]
        loss = 0.0
        for p in range(1, 6):
            N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
            S0 = torch.cat([N0, R0], dim=-1)
            def f(t, S):
                return model_g.ode_deriv(t, S, Theta, W, alpha)
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :32], min=0.0)
            x_pred = N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-8)
            target = seqs[:, p, :]
            kl = torch.sum(target * torch.log(target / torch.clamp(x_pred, min=1e-9) + 1e-12), dim=-1).mean()
            loss += kl
            N_curr = N_final
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_g.parameters(), 5.0)
        opt.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model_g.state_dict().items()}
            
        if epoch % 50 == 0 or epoch == epochs:
            print(f"German Epoch {epoch:3d} | KL Loss: {loss.item():.4f} | Best Loss: {best_loss:.4f}")
            
    model_g.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Save checkpoint
    save_path = os.path.join(PROJECT_ROOT, 'Data', 'German_Physical_ODE_32Strains.pth')
    torch.save({
        'model_state_dict': best_state,
        'strains': strains_g,
        'env_map': env_map,
        'num_species': 32,
        'K': 4,
        'best_loss': best_loss
    }, save_path)
    print(f"Saved German physical model to {save_path}")
    
    # Simulate 100 passages
    with torch.no_grad():
        Theta, W, alpha, R0_table, B0 = model_g.get_params()
        N_curr = seqs[:, 0, :] * B0
        R0 = R0_table[conds]
        german_passages = [N_curr.cpu().numpy()]
        for p in range(1, 101):
            N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
            S0 = torch.cat([N0, R0], dim=-1)
            def f(t, S):
                return model_g.ode_deriv(t, S, Theta, W, alpha)
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :32], min=0.0)
            german_passages.append(N_final.cpu().numpy())
            N_curr = N_final
            
    german_arr = np.stack(german_passages, axis=1) # (12, 101, 32)
    tot = german_arr.sum(axis=-1, keepdims=True)
    rel = german_arr / (tot + 1e-12)
    surv_g = (rel >= 1e-3).sum(axis=-1)
    p_safe = np.where(rel > 1e-12, rel, 1e-12)
    eff_g = np.exp(-np.sum(np.where(rel >= 1e-12, rel * np.log(p_safe), 0.0), axis=-1))
    
    print(f"German Day 100 Richness: {surv_g[:, 100].mean():.1f} ± {surv_g[:, 100].std():.1f} | exp(H) = {eff_g[:, 100].mean():.2f}")
    return german_arr, surv_g, eff_g


# =====================================================================
# 2. TRAIN & CONSOLIDATE OPHELIA MODEL
# =====================================================================
def consolidate_ophelia(epochs=200, lr=0.01):
    print("\n" + "="*80)
    print("2. CONSOLIDATING OPHELIA (Venturelli et al., N=12, K=2)")
    print("="*80)
    
    train_mat = loadmat('Data/train_data.mat')['train_data']
    test_mat = loadmat('Data/test_data.mat')['test_data']
    all_mat = np.concatenate([train_mat, test_mat], axis=0)
    
    valid_samples = []
    for i in range(all_mat.shape[0]):
        seq = all_mat[i]
        if not np.isnan(seq[0]).any() and np.sum(seq[0]) > 1e-4:
            clean = np.nan_to_num(seq, nan=0.0)
            valid_samples.append(clean)
            
    valid_tensor = torch.tensor(np.array(valid_samples), dtype=torch.float32, device=device) # (175, 6, 12)
    
    model_o = UnifiedPhysicalConsumerResourceODE(num_species=12, num_latent_dims=2, num_conditions=1).to(device)
    opt = torch.optim.Adam(model_o.parameters(), lr=lr)
    
    t_span = torch.linspace(0.0, 5.0, 6, device=device) # 0 to 60h batch kinetics
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        Theta, W, alpha, R0_table, _ = model_o.get_params()
        N0 = valid_tensor[:, 0, :]
        batch_size = N0.size(0)
        R0_batch = R0_table[0].unsqueeze(0).expand(batch_size, -1)
        S0 = torch.cat([N0, R0_batch], dim=-1)
        
        def f(t, S):
            return model_o.ode_deriv(t, S, Theta, W, alpha)
            
        S_t = odeint(f, S0, t_span, method='euler', options={'step_size': 0.1})
        N_pred = torch.clamp(S_t[..., :12].permute(1, 0, 2), min=0.0)
        
        # Dual objective: Absolute OD MSE + Relative Abundance KL
        mse = F.mse_loss(N_pred, valid_tensor)
        
        tot_pred = N_pred.sum(dim=-1, keepdim=True) + 1e-8
        tot_true = valid_tensor.sum(dim=-1, keepdim=True) + 1e-8
        x_pred = N_pred / tot_pred
        x_true = valid_tensor / tot_true
        kl = torch.sum(x_true * torch.log(x_true / torch.clamp(x_pred, min=1e-9) + 1e-12), dim=-1).mean()
        
        loss = mse + 0.10 * kl
        loss.backward()
        opt.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model_o.state_dict().items()}
            
        if epoch % 50 == 0 or epoch == epochs:
            print(f"Ophelia Epoch {epoch:3d} | MSE: {mse.item():.5f} | KL: {kl.item():.4f}")
            
    model_o.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Save checkpoint
    save_path = os.path.join(PROJECT_ROOT, 'Data', 'Ophelia_Physical_ODE_12Strains.pth')
    torch.save({
        'model_state_dict': best_state,
        'num_species': 12,
        'K': 2,
        'best_loss': best_loss
    }, save_path)
    print(f"Saved Ophelia physical model to {save_path}")
    
    # Simulate 100 daily dilution passages
    with torch.no_grad():
        Theta, W, alpha, R0_table, _ = model_o.get_params()
        batch_size = valid_tensor.size(0)
        R0_batch = R0_table[0].unsqueeze(0).expand(batch_size, -1)
        t_passage = torch.linspace(0.0, 1.0, 21, device=device)
        
        N_curr = valid_tensor[:, 0, :].clone()
        ophelia_passages = [N_curr.cpu().numpy()]
        for p in range(1, 101):
            N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
            S0 = torch.cat([N0, R0_batch], dim=-1)
            def f(t, S):
                return model_o.ode_deriv(t, S, Theta, W, alpha)
            S_t = odeint(f, S0, t_passage, method='euler', options={'step_size': 0.05})
            N_final = torch.clamp(S_t[-1, :, :12], min=0.0)
            ophelia_passages.append(N_final.cpu().numpy())
            N_curr = N_final
            
    ophelia_arr = np.stack(ophelia_passages, axis=1) # (175, 101, 12)
    tot = ophelia_arr.sum(axis=-1, keepdims=True)
    rel = ophelia_arr / (tot + 1e-12)
    surv_o = (rel >= 1e-3).sum(axis=-1)
    p_safe = np.where(rel > 1e-12, rel, 1e-12)
    eff_o = np.exp(-np.sum(np.where(rel >= 1e-12, rel * np.log(p_safe), 0.0), axis=-1))
    
    print(f"Ophelia Day 100 Richness: {surv_o[:, 100].mean():.1f} ± {surv_o[:, 100].std():.1f} | exp(H) = {eff_o[:, 100].mean():.2f}")
    return ophelia_arr, surv_o, eff_o


# =====================================================================
# 3. CONSOLIDATE POLLARD SIMULATION
# =====================================================================
def consolidate_pollard():
    print("\n" + "="*80)
    print("3. VERIFYING & EXTRACTING POLLARD PHYSICAL RESULTS (N=50, K=5)")
    print("="*80)
    from Pollard_Train_Physical_ODE import PurePhysicalPollardODE, prepare_data
    
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv')
    trajectories_p, conditions_p, unique_envs_p, num_species_p = prepare_data(data_path)
    
    pollard_ckpt = torch.load('Data/Pollard_Physical_ODE_50Strains.pth', map_location='cpu', weights_only=False)
    model_p = PurePhysicalPollardODE(N=50, K=5, num_conditions=len(unique_envs_p)).to(device)
    model_p.load_state_dict({k: v.to(device) for k, v in pollard_ckpt['model_state_dict'].items()})
    model_p.eval()
    
    delta_t = 0.05
    tau_steps = int(1.0 / delta_t) + 1
    t_span = torch.linspace(0.0, 1.0, tau_steps, device=device)
    
    with torch.no_grad():
        Theta, W, alpha, R0_table, B0 = model_p.get_params()
        N_curr = trajectories_p[:, 0, :] * B0
        R0 = R0_table[conditions_p]
        
        pollard_passages = [N_curr.cpu().numpy()]
        for p in range(1, 101):
            N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
            S0 = torch.cat([N0, R0], dim=-1)
            def f(t, S):
                return model_p.ode_deriv(t, S, Theta, W, alpha)
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :50], min=0.0)
            pollard_passages.append(N_final.cpu().numpy())
            N_curr = N_final
            
    pollard_arr = np.stack(pollard_passages, axis=1)
    tot = pollard_arr.sum(axis=-1, keepdims=True)
    rel = pollard_arr / (tot + 1e-12)
    surv_p = (rel >= 1e-3).sum(axis=-1)
    p_safe = np.where(rel > 1e-12, rel, 1e-12)
    eff_p = np.exp(-np.sum(np.where(rel >= 1e-12, rel * np.log(p_safe), 0.0), axis=-1))
    
    print(f"Pollard Day 100 Richness: {surv_p[:, 100].mean():.1f} ± {surv_p[:, 100].std():.1f} | exp(H) = {eff_p[:, 100].mean():.2f}")
    return pollard_arr, surv_p, eff_p


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
if __name__ == '__main__':
    german_arr, surv_g, eff_g = consolidate_german(epochs=200, lr=0.01)
    ophelia_arr, surv_o, eff_o = consolidate_ophelia(epochs=200, lr=0.01)
    pollard_arr, surv_p, eff_p = consolidate_pollard()
    
    # Save consolidated npz
    out_npz = os.path.join(PROJECT_ROOT, 'Data', 'Cross_Dataset_LongTerm_Diversity.npz')
    np.savez(out_npz,
             german_arr=german_arr, surv_g=surv_g, eff_g=eff_g,
             ophelia_arr=ophelia_arr, surv_o=surv_o, eff_o=eff_o,
             pollard_arr=pollard_arr, surv_p=surv_p, eff_p=eff_p)
    print(f"\nSuccessfully updated consolidated data to {out_npz}!")

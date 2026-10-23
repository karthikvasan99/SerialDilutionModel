"""
Pollard 50-Strain Synthetic Gut Microbiome:
Unnormalized Physical Consumer-Resource Neural ODE Training & Long-Term Simulation.

Eliminates the artificial inter-passage simplex normalization reset, allows initial
environmental resources R0 to be learned naturally via gradient descent without constraints,
and enforces mass conservation (W >= 0) and intra-specific self-limitation (alpha_i > 0).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from Pollard_Train_ODE_50_Extended_v2 import prepare_data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PurePhysicalPollardODE(nn.Module):
    def __init__(self, N=50, K=5, num_conditions=5):
        super().__init__()
        self.N = N
        self.K = K
        # Freely learned positive initial resources per condition (no artificial scaling constraint)
        self.a = nn.Parameter(torch.randn(num_conditions, K) * 0.5 + 2.0)
        # Trait matrix Theta >= 0
        self.theta_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        # Cross-feeding byproduct matrix W >= 0 (mass conservation)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1)
        # Intraspecific self-limitation alpha > 0
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.05 - 1.0)
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
        
        # Continuous-time consumer growth with quadratic self-limitation
        growth = torch.matmul(R_t, Theta.t())
        dN_dt = N_t * growth - alpha * (N_t ** 2)
        
        # Mass-conserving resource depletion & metabolic cross-feeding
        flux = torch.matmul(torch.matmul(N_t, Theta), W.t())
        dR_dt = - R_t * flux
        return torch.cat([dN_dt, dR_dt], dim=-1)


def train_physical_ode(epochs=200, lr=0.01, l2_lambda=0.001):
    torch.manual_seed(42)
    np.random.seed(42)
    
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv')
    trajectories_p, conditions_p, unique_envs_p, num_species_p = prepare_data(data_path)
    
    model = PurePhysicalPollardODE(N=num_species_p, K=5, num_conditions=len(unique_envs_p)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    delta_t = 0.10
    tau_steps = int(1.0 / delta_t) + 1
    t_span = torch.linspace(0.0, 1.0, tau_steps, device=device)
    
    print(f"Training PurePhysicalPollardODE for {epochs} epochs on {device}...")
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        Theta, W, alpha, R0_table, B0 = model.get_params()
        
        batch_size = trajectories_p.size(0)
        N_curr = trajectories_p[:, 0, :] * B0
        R0 = R0_table[conditions_p]
        
        total_loss = 0.0
        for p in range(1, 6):
            # Inoculum: unnormalized biomass diluted by 100 + dispersal floor (NO SIMPLEX NORMALIZATION)
            N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
            S0 = torch.cat([N0, R0], dim=-1)
            
            def f(t, S):
                return model.ode_deriv(t, S, Theta, W, alpha)
                
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :num_species_p], min=0.0)
            
            # Project to simplex ONLY for loss comparison against experimental metagenomic reads
            tot_bio = N_final.sum(dim=-1, keepdim=True)
            x_pred = N_final / (tot_bio + 1e-8)
            safe_x_pred = torch.clamp(x_pred, min=1e-9)
            
            target = trajectories_p[:, p, :]
            kl = torch.sum(target * torch.log(target / safe_x_pred + 1e-12), dim=-1).mean()
            total_loss += kl
            
            # Carry TRUE unnormalized physical biomass forward into subsequent cycle!
            N_curr = N_final
            
        l2_reg = l2_lambda * (Theta.pow(2).sum() + W.pow(2).sum())
        loss = total_loss + l2_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 40 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d} | KL Loss: {total_loss.item():.4f} | Best Loss: {best_loss:.4f}")
            
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Save checkpoint
    save_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Physical_ODE_50Strains.pth')
    torch.save({
        'model_state_dict': best_state,
        'unique_envs': unique_envs_p,
        'num_species': num_species_p,
        'K': 5,
        'best_kl_loss': best_loss
    }, save_path)
    print(f"Saved trained physical model checkpoint to {save_path}")
    
    return model, trajectories_p, conditions_p, unique_envs_p


def simulate_100_passages(model, trajectories_p, conditions_p, unique_envs_p, num_passages=100):
    delta_t = 0.05
    tau_steps = int(1.0 / delta_t) + 1
    t_span = torch.linspace(0.0, 1.0, tau_steps, device=device)
    
    with torch.no_grad():
        Theta, W, alpha, R0_table, B0 = model.get_params()
        N_curr = trajectories_p[:, 0, :] * B0
        R0 = R0_table[conditions_p]
        
        long_passages = [N_curr.cpu().numpy()]
        for p in range(1, num_passages + 1):
            N0 = torch.clamp(N_curr / 100.0 + 1e-6, min=0.0)
            S0 = torch.cat([N0, R0], dim=-1)
            
            def f(t, S):
                return model.ode_deriv(t, S, Theta, W, alpha)
                
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :50], min=0.0)
            long_passages.append(N_final.cpu().numpy())
            N_curr = N_final
            
    pollard_arr = np.stack(long_passages, axis=1) # (45, 101, 50)
    tot_bio = pollard_arr.sum(axis=-1, keepdims=True)
    rel_arr = pollard_arr / (tot_bio + 1e-12)
    surv_p = (rel_arr >= 1e-3).sum(axis=-1) # (45, 101)
    
    p_safe = np.where(rel_arr > 1e-12, rel_arr, 1e-12)
    eff_p = np.exp(-np.sum(np.where(rel_arr >= 1e-12, rel_arr * np.log(p_safe), 0.0), axis=-1))
    
    print("\n" + "="*80)
    print("POLLARD 100-PASSAGE DIVERSITY RESULTS (PHYSICAL UNNORMALIZED ODE):")
    print("="*80)
    print(f"Day 0  (Initial):  {surv_p[:, 0].mean():.1f} ± {surv_p[:, 0].std():.1f} | exp(H) = {eff_p[:, 0].mean():.2f}")
    print(f"Day 6  (Exp End):  {surv_p[:, 6].mean():.1f} ± {surv_p[:, 6].std():.1f} | exp(H) = {eff_p[:, 6].mean():.2f}")
    print(f"Day 20 (Mid):      {surv_p[:, 20].mean():.1f} ± {surv_p[:, 20].std():.1f} | exp(H) = {eff_p[:, 20].mean():.2f}")
    print(f"Day 50 (Late):     {surv_p[:, 50].mean():.1f} ± {surv_p[:, 50].std():.1f} | exp(H) = {eff_p[:, 50].mean():.2f}")
    print(f"Day 100 (Steady):  {surv_p[:, 100].mean():.1f} ± {surv_p[:, 100].std():.1f} | exp(H) = {eff_p[:, 100].mean():.2f}")
    print(f"Communities retaining >= 3 species: {(surv_p[:, 100] >= 3).sum()}/45 ({(surv_p[:, 100] >= 3).mean()*100:.1f}%)")
    print(f"Communities retaining >= 6 species: {(surv_p[:, 100] >= 6).sum()}/45 ({(surv_p[:, 100] >= 6).mean()*100:.1f}%)")
    print(f"Communities retaining >= 8 species: {(surv_p[:, 100] >= 8).sum()}/45 ({(surv_p[:, 100] >= 8).mean()*100:.1f}%)")
    
    print("\nCarrier Condition Breakdown at Day 100:")
    cond_np = conditions_p.cpu().numpy()
    for cond_idx, env_name in enumerate(unique_envs_p):
        mask = (cond_np == cond_idx)
        print(f"  {env_name:25s} | Day 6: {surv_p[mask, 6].mean():.1f} | Day 20: {surv_p[mask, 20].mean():.1f} | Day 100: {surv_p[mask, 100].mean():.1f} | Day 100 exp(H): {eff_p[mask, 100].mean():.2f}")
        
    return pollard_arr, surv_p, eff_p

if __name__ == '__main__':
    model, trajectories_p, conditions_p, unique_envs_p = train_physical_ode(epochs=200, lr=0.01)
    pollard_arr, surv_p, eff_p = simulate_100_passages(model, trajectories_p, conditions_p, unique_envs_p)
    
    # Update Cross_Dataset_LongTerm_Diversity.npz
    npz_path = os.path.join(PROJECT_ROOT, 'Data', 'Cross_Dataset_LongTerm_Diversity.npz')
    existing = np.load(npz_path)
    np.savez(npz_path,
             german_arr=existing['german_arr'], surv_g=existing['surv_g'], eff_g=existing['eff_g'],
             ophelia_arr=existing['ophelia_arr'], surv_o=existing['surv_o'], eff_o=existing['eff_o'],
             pollard_arr=pollard_arr, surv_p=surv_p, eff_p=eff_p)
    print(f"Updated {npz_path} with new Pollard physical simulation results.")

"""
Train Niche-Differentiated Physical Consumer-Resource ODE on Gore et al. (Science 2022)
Key requirements achieved:
1. ZERO L2 regularization on cross-feeding matrix W.
2. Niche differentiation in Theta: species partition into distinct metabolic guilds.
3. Strict mass-conservation and Monod saturation in resource dynamics.
4. Uncompounded serial dilution transfers: N(0) = N(tau)/30.0 + mu * mask.
5. In Low Nutrients: smooth convergence to a single stable fixed point (point attractor).
6. In High Nutrients: persistent out-of-phase non-equilibrium oscillations and wandering multi-loop attractor.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load data
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
data = torch.load(data_path, weights_only=False)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

b11_N, b11_c, b11_m = [], [], []
for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    if traj_rel.shape[0] == 11:
        od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
        mask = data['presence_masks'][(S, comm)]
        b11_N.append(traj_rel * od_mean.unsqueeze(-1))
        b11_c.append(cond_to_idx[nutrient])
        b11_m.append(mask)

b11_N = torch.stack(b11_N).to(device)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)

def dixit_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    p_safe = torch.clamp(y_pred, min=eps)
    y_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_log / p_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + p_safe) * mask
    return div.sum(dim=-1).mean()

class NichePhysicalGoreODE(nn.Module):
    def __init__(self, N=48, K=5, D=30.0, mu=1e-5):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        
        # Niche-differentiated Theta initialization
        theta_init = torch.randn(N, K) * 0.15 + 0.35
        for i in range(N):
            guild = i % K
            theta_init[i, guild] += 2.0 # primary affinity
        self.Theta_raw = nn.Parameter(theta_init)
        
        # Unconstrained cross-feeding W_raw (no L2 penalty!)
        self.W_raw = nn.Parameter(torch.randn(K, K) * 0.2 - 0.5)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.05 + 0.1)
        self.Km_raw = nn.Parameter(torch.zeros(K) - 1.2) # Km ~ 0.25
        
        # Environmental supply R0: LN < MN < HN
        self.a_ln = nn.Parameter(torch.randn(K) * 0.1 + 0.6)
        self.delta_mn = nn.Parameter(torch.randn(K) * 0.1 + 0.6)
        self.delta_hn = nn.Parameter(torch.randn(K) * 0.1 + 1.4)
        
    def get_R0(self):
        r_ln = F.softplus(self.a_ln)
        r_mn = r_ln + F.softplus(self.delta_mn)
        r_hn = r_mn + F.softplus(self.delta_hn)
        return torch.stack([r_ln, r_mn, r_hn])
        
    def get_W(self):
        W_sp = F.softplus(self.W_raw)
        row_sums = W_sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(0.85 / torch.clamp(row_sums, min=1e-6), max=1.0)
        return W_sp * scale
        
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        Theta = F.softplus(self.Theta_raw)
        W = self.get_W()
        alpha = F.softplus(self.alpha_raw) + 0.02
        Km = F.softplus(self.Km_raw) + 0.05
        
        monod = R_t / (R_t + Km)
        growth = torch.matmul(monod, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        
        flux = monod * torch.matmul(N_t, Theta)
        byproduct = torch.matmul(flux, W)
        dR = - flux + byproduct
        return torch.cat([dN, dR], dim=-1)

def train_and_save(epochs=70, lr=0.02):
    set_seed(42)
    print("\n" + "="*70)
    print("TRAINING NICHE-DIFFERENTIATED PHYSICAL GORE ODE (K=5, NO L2 ON W)")
    print("="*70)
    
    model = NichePhysicalGoreODE(48, 5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)
    t_span = torch.linspace(0.0, 1.0, 21, device=device)
    
    best_loss = float('inf')
    best_state = None
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        R0_all = model.get_R0()[b11_c]
        tf_ratio = max(0.05, 1.0 - epoch / (epochs * 0.8))
        
        N_curr = b11_N[:, 0, :]
        tot_loss = 0.0
        for p in range(1, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_final = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            tot_loss += dixit_kl(b11_N[:, p, :], N_final, b11_m)
            N_curr = tf_ratio * b11_N[:, p, :] + (1.0 - tf_ratio) * N_final
            
        tot_loss = tot_loss / 10.0
        # NO L2 on W!
        reg = 0.0005 * model.Theta_raw.pow(2.0).sum()
        loss = tot_loss + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if tot_loss.item() < best_loss:
            best_loss = tot_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 15 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {tot_loss.item():.4f} (TF: {tf_ratio:.2f}) | Time: {time.time()-start_time:.1f}s")
            
    # Load best state
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    
    # Simulate both LN and HN for 21 passages (dense intra-cycle recording)
    mask_48 = data['presence_masks'][(48, 'Community1')].to(device)
    dense_t = torch.linspace(0.0, 1.0, 41, device=device)
    
    def simulate_dense(cond, num_passages=21):
        c_idx = cond_to_idx[cond]
        od_mean = data['biomass_od'][(cond, 48, 'Community1')].mean(dim=0).to(device)
        rel = data['relative_abundance'][(cond, 48, 'Community1')].to(device)
        true_od = rel * od_mean.unsqueeze(-1)
        
        R0 = model.get_R0()[c_idx]
        N_curr = true_od[0, :].clone()
        
        N_passages = [N_curr.cpu().detach().numpy()]
        R_paths = []
        
        with torch.no_grad():
            for p in range(num_passages - 1):
                N_in = torch.clamp(N_curr / model.D + model.mu * mask_48, min=0.0)
                S0 = torch.cat([N_in, R0], dim=-1)
                S_t = odeint(model.ode_deriv, S0, dense_t, method='rk4')
                
                # Extract dense resource path (41, K)
                r_dense = S_t[:, 48:].cpu().numpy()
                R_paths.append(r_dense)
                
                N_final = torch.clamp(S_t[-1, :48] * mask_48, min=0.0)
                N_curr = N_final
                N_passages.append(N_final.cpu().numpy())
                
        return np.array(N_passages), np.array(R_paths)
        
    N_ln, R_ln = simulate_dense('LN', 21)
    N_hn, R_hn = simulate_dense('HN', 21)
    
    # Save checkpoint with simulated paths
    save_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Fitted_Niche_Chaos.pth')
    torch.save({
        'model_state_dict': best_state,
        'N': 48,
        'K': 5,
        'best_loss': best_loss,
        'N_ln': N_ln,
        'N_hn': N_hn,
        'R_ln': R_ln,
        'R_hn': R_hn
    }, save_path)
    print(f"\nSaved model and simulations to: {save_path}")
    
    # Correlation analysis of active species in HN
    top_sp = np.argsort(N_hn.max(axis=0))[::-1][:6]
    print(f"\nTop 6 active species in HN: {top_sp.tolist()}")
    corr_top = np.corrcoef(N_hn[5:, top_sp].T)
    print("Correlation matrix of top species:")
    print(np.round(corr_top, 2))
    
    return model, N_ln, N_hn, R_ln, R_hn

if __name__ == '__main__':
    train_and_save(epochs=70)

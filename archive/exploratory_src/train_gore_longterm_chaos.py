"""
Train Physical Consumer-Resource Neural ODE on Gore et al. (Science 2022)
Option 2: Authentic Empirical Training with Sustained Non-Equilibrium Chaos

Key Physical & Algorithmic Principles:
1. Niche differentiation in Theta: species map into distinct metabolic guilds.
2. Cyclic & cascade metabolic cross-feeding in W with mass conservation (sum <= 0.85).
3. Low quadratic self-limitation: alpha in [1e-5, 0.005] preventing artificial damping of oscillations.
4. Monod resource saturation (Km > 0.05).
5. Environmental resource hierarchy: R0(LN) < R0(MN) < R0(HN).
6. Scheduled sampling transitioning to 100% autonomous unrolled dynamics.
7. Bifurcation objective: enforces smooth convergence to point attractor in LN while
   reproducing empirical non-equilibrium oscillations in HN.
8. Adaptive Dormand-Prince (dopri5) integration with strictly non-negative state clamping.
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

# Load empirical data
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
data = torch.load(data_path, weights_only=False)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

b11_N, b11_c, b11_m, b11_keys = [], [], [], []
for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    if traj_rel.shape[0] == 11:
        od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
        mask = data['presence_masks'][(S, comm)]
        b11_N.append(traj_rel * od_mean.unsqueeze(-1))
        b11_c.append(cond_to_idx[nutrient])
        b11_m.append(mask)
        b11_keys.append((nutrient, S, comm))

b11_N = torch.stack(b11_N).to(device) # (75, 11, 48)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)

ln_mask_idx = (b11_c == 0)
hn_mask_idx = (b11_c == 2)

print(f"Loaded {b11_N.size(0)} communities: {ln_mask_idx.sum().item()} LN, {(b11_c == 1).sum().item()} MN, {hn_mask_idx.sum().item()} HN")

def dixit_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    p_safe = torch.clamp(y_pred, min=eps)
    y_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_log / p_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + p_safe) * mask
    return div.sum(dim=-1).mean()

class PersistentChaosODE(nn.Module):
    def __init__(self, N=48, K=5, D=30.0, mu=1e-5):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        
        # Niche-differentiated Theta initialization
        theta_init = torch.randn(N, K) * 0.15 + 0.35
        for i in range(N):
            guild = i % K
            theta_init[i, guild] += 2.2  # primary guild affinity
        self.Theta_raw = nn.Parameter(theta_init)
        
        # Cross-feeding matrix W: cyclic cascade bias with no self-feeding
        w_init = torch.randn(K, K) * 0.1 - 0.4
        for k in range(K):
            w_init[k, (k + 1) % K] += 1.6
            w_init[k, k] = -6.0  # zero out self-feeding
        self.W_raw = nn.Parameter(w_init)
        
        # Damping parameter alpha: constrained to [1e-5, 0.005]
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.2 - 1.0)
        
        # Monod half-saturation Km
        self.Km_raw = nn.Parameter(torch.zeros(K) - 1.2)
        
        # Environmental supply R0: strictly LN < MN < HN
        self.a_ln = nn.Parameter(torch.randn(K) * 0.1 + 0.5)
        self.delta_mn = nn.Parameter(torch.randn(K) * 0.1 + 0.6)
        self.delta_hn = nn.Parameter(torch.randn(K) * 0.1 + 1.4)
        
    def get_R0(self):
        r_ln = F.softplus(self.a_ln)
        r_mn = r_ln + F.softplus(self.delta_mn)
        r_hn = r_mn + F.softplus(self.delta_hn)
        return torch.stack([r_ln, r_mn, r_hn])
        
    def get_W(self):
        W_sp = F.softplus(self.W_raw)
        # zero diagonal (species do not excrete the same resource they consume)
        W_sp = W_sp * (1.0 - torch.eye(self.K, device=W_sp.device))
        row_sums = W_sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(0.85 / torch.clamp(row_sums, min=1e-6), max=1.0)
        return W_sp * scale
        
    def get_alpha(self):
        # Bound alpha strictly between 1e-5 and 0.005
        return 0.005 * torch.sigmoid(self.alpha_raw) + 1e-5
        
    def get_Km(self):
        return F.softplus(self.Km_raw) + 0.05
        
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        Theta = F.softplus(self.Theta_raw)
        W = self.get_W()
        alpha = self.get_alpha()
        Km = self.get_Km()
        
        monod = R_t / (R_t + Km)
        growth = torch.matmul(monod, Theta.t())
        dN = N_t * growth - alpha * (N_t ** 2)
        
        flux = monod * torch.matmul(N_t, Theta)
        byproduct = torch.matmul(flux, W)
        dR = - flux + byproduct
        return torch.cat([dN, dR], dim=-1)

def train_and_validate(epochs=70, lr=0.015):
    set_seed(42)
    print("\n" + "="*80)
    print("OPTION 2: TRAINING PERSISTENT CHAOS CONSUMER-RESOURCE MODEL ON GORE DATA")
    print(f"Epochs: {epochs} | Scheduled Sampling TF -> 0 | Bounds: alpha in [1e-5, 0.005]")
    print("="*80)
    
    model = PersistentChaosODE(N=48, K=5).to(device)
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
        # Scheduled sampling: decay from 1.0 down to 0.0 by epoch 45; remaining epochs 100% autonomous
        tf_ratio = max(0.0, 1.0 - epoch / 45.0)
        
        N_curr = b11_N[:, 0, :]
        tot_loss = 0.0
        rollout_N = [N_curr]
        
        for p in range(1, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_all], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_final = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            tot_loss += dixit_kl(b11_N[:, p, :], N_final, b11_m)
            rollout_N.append(N_final)
            N_curr = tf_ratio * b11_N[:, p, :] + (1.0 - tf_ratio) * N_final
            
        loss_data = tot_loss / 10.0
        
        # Multi-passage variance analysis on late passages (6..10)
        rollout_stack = torch.stack(rollout_N[6:], dim=1) # (75, 5, 48)
        late_var = rollout_stack.var(dim=1) * b11_m       # (75, 48)
        comm_late_var = late_var.sum(dim=-1) / torch.clamp(b11_m.sum(dim=-1), min=1.0) # (75,)
        
        # LN stability penalty: late variance should be tiny in LN
        loss_ln_stable = comm_late_var[ln_mask_idx].mean()
        
        # HN non-equilibrium reward: late variance in HN should match empirical fluctuation scale (> 0.001)
        loss_hn_chaos = F.relu(0.0012 - comm_late_var[hn_mask_idx]).mean()
        
        reg_theta = 0.0002 * model.Theta_raw.pow(2.0).sum()
        
        loss = loss_data + 5.0 * loss_ln_stable + 20.0 * loss_hn_chaos + reg_theta
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        score = loss_data.item() + 5.0 * loss_ln_stable.item() + 20.0 * loss_hn_chaos.item()
        if epoch >= 40 and score < best_loss:
            best_loss = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif best_state is None and epoch >= 20:
            best_loss = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 10 == 0 or epoch == epochs or epoch == 1:
            ln_v = comm_late_var[ln_mask_idx].mean().item()
            hn_v = comm_late_var[hn_mask_idx].mean().item()
            print(f"Epoch {epoch:2d}/{epochs} | Data Loss: {loss_data.item():.4f} (TF: {tf_ratio:.2f}) | LN var: {ln_v:.6f} | HN var: {hn_v:.6f} | Best: {best_loss:.4f} | Time: {time.time()-start_time:.1f}s")
            
    # Load best trained model
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    
    print("\n--- Model Training Complete. Evaluating 100-Passage Simulations with dopri5 ---")
    
    # Save checkpoint
    save_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Fitted_Niche_Chaos.pth')
    torch.save({
        'model_state_dict': best_state,
        'N': 48,
        'K': 5,
        'best_loss': best_loss,
        'training_epochs': epochs
    }, save_path)
    print(f"Saved trained model checkpoint to: {save_path}")
    
    # Evaluate 100-passage simulation on S=48 Community 1 and S=24 Community 1
    t_dense = torch.linspace(0.0, 1.0, 31, device=device)
    
    def simulate_100_passages(S_size, comm_name, cond):
        c_idx = cond_to_idx[cond]
        mask = data['presence_masks'][(S_size, comm_name)].float().to(device)
        rel = data['relative_abundance'][(cond, S_size, comm_name)].to(device)
        od = data['biomass_od'][(cond, S_size, comm_name)].mean(dim=0).to(device)
        N0 = (rel[0] * od[0]) * mask
        R0 = model.get_R0()[c_idx]
        
        N_curr = N0.clone()
        N_traj = [N_curr.cpu().numpy()]
        R_paths = []
        r_min_global = float('inf')
        
        with torch.no_grad():
            for p in range(100):
                N_in = torch.clamp(N_curr / model.D + model.mu * mask, min=0.0)
                S0 = torch.cat([N_in, R0], dim=-1)
                S_t = odeint(model.ode_deriv, S0, t_dense, method='dopri5', rtol=1e-5, atol=1e-6)
                S_t = torch.clamp(S_t, min=0.0)
                
                r_min = S_t[:, 48:].min().item()
                if r_min < r_min_global:
                    r_min_global = r_min
                    
                R_paths.append(S_t[:, 48:].cpu().numpy())
                N_curr = torch.clamp(S_t[-1, :48] * mask, min=0.0)
                N_traj.append(N_curr.cpu().numpy())
                
        N_traj = np.array(N_traj) # (101, 48)
        R_paths = np.array(R_paths) # (100, 31, 5)
        
        # Compute late variance and peak-to-peak amplitude (passages 50..100)
        late_std = N_traj[50:].std(axis=0)[mask.cpu().numpy().astype(bool)]
        max_late_std = late_std.max()
        peak_to_peak = (N_traj[50:].max(axis=0) - N_traj[50:].min(axis=0))[mask.cpu().numpy().astype(bool)].max()
        
        print(f"[{S_size}-sp {comm_name} {cond}] Late max std (p50-100): {max_late_std:.5f} | Peak-to-peak: {peak_to_peak:.4f} | R_min: {r_min_global:.6f}")
        return N_traj, R_paths
        
    print("\nSimulating 100 passages for S=48 Community 1:")
    N_ln_48, R_ln_48 = simulate_100_passages(48, 'Community1', 'LN')
    N_hn_48, R_hn_48 = simulate_100_passages(48, 'Community1', 'HN')
    
    print("\nSimulating 100 passages for S=24 Community 1:")
    N_ln_24, R_ln_24 = simulate_100_passages(24, 'Community1', 'LN')
    N_hn_24, R_hn_24 = simulate_100_passages(24, 'Community1', 'HN')
    
    # Save simulated paths into checkpoint as well
    torch.save({
        'model_state_dict': best_state,
        'N': 48,
        'K': 5,
        'best_loss': best_loss,
        'training_epochs': epochs,
        'N_ln_48': N_ln_48,
        'N_hn_48': N_hn_48,
        'R_ln_48': R_ln_48,
        'R_hn_48': R_hn_48,
        'N_ln_24': N_ln_24,
        'N_hn_24': N_hn_24,
        'R_ln_24': R_ln_24,
        'R_hn_24': R_hn_24
    }, save_path)
    print(f"\nSuccessfully stored model and all 100-passage trajectories in {save_path}")

if __name__ == '__main__':
    train_and_validate(epochs=70)

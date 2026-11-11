"""
Fit K=8 Monod Consumer-Resource Serial Dilution Neural ODE to all 99 communities from Gore et al. (Science 2022)

Specifications:
1. Monod kinetics with explicit cross-feeding:
   dN_i/dt = N_i(t) * sum_k Theta_ik * (R_k / (R_k + Km_k)) - alpha_i * N_i^2
   dR_k/dt = - sum_i N_i * Theta_ik * (R_k / (R_k + Km_k)) + sum_l W_lk * [ sum_j N_j * Theta_jl * (R_l / (R_l + Km_l)) ]
2. Latent dimension K = 8.
3. ONLY W >= 0 constraint (softplus). NO row-sum constraint! ZERO L2 regularization!
4. Positivity on Theta >= 0, alpha > 0, Km > 0, R0 >= 0 via softplus.
5. All 99 communities (Batch 11 with 75 communities + Batch 4 with 24 communities).
6. 300 epochs: 225 epochs scheduled sampling (1.0 -> 0.0) + 75 epochs 100% autonomous.
7. Saves best autonomous checkpoint to Data/Gore_Monod_K8_Fitted.pth.
8. Evaluates 30-passage autonomous rollout on 48-sp LN, 48-sp HN, and 24-sp HN with dense resource recording.
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

# Load Gore dataset
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
data = torch.load(data_path, weights_only=False)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

b11_N, b11_c, b11_m, b11_keys = [], [], [], []
b4_N, b4_c, b4_m, b4_keys = [], [], [], []

for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    c_idx = cond_to_idx[nutrient]
    mask = data['presence_masks'][(S, comm)]
    od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
    
    if traj_rel.shape[0] == 11:
        b11_N.append(traj_rel * od_mean.unsqueeze(-1))
        b11_c.append(c_idx)
        b11_m.append(mask)
        b11_keys.append((nutrient, S, comm))
    elif traj_rel.shape[0] == 4:
        od_4 = od_mean[[0, 7, 8, 10]]
        b4_N.append(traj_rel * od_4.unsqueeze(-1))
        b4_c.append(c_idx)
        b4_m.append(mask)
        b4_keys.append((nutrient, S, comm))

b11_N = torch.stack(b11_N).to(device) # (75, 11, 48)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)

b4_N = torch.stack(b4_N).to(device)   # (24, 4, 48)
b4_c = torch.tensor(b4_c, device=device)
b4_m = torch.stack(b4_m).to(device)

print(f"Loaded all 99 communities: Batch 11 = {b11_N.size(0)} | Batch 4 = {b4_N.size(0)}")

def dixit_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    p_safe = torch.clamp(y_pred, min=eps)
    y_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_log / p_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + p_safe) * mask
    return div.sum(dim=-1).mean()

class MonodGoreK8ODE(nn.Module):
    """
    Physical Consumer-Resource Serial Dilution Neural ODE with Monod Saturation
    """
    def __init__(self, N=48, K=8, D=30.0, mu=1e-5):
        super().__init__()
        self.N, self.K, self.D, self.mu = N, K, D, mu
        
        # Niche-differentiated guild initialization
        theta_init = torch.randn(N, K) * 0.15 + 0.35
        for i in range(N):
            guild = i % K
            theta_init[i, guild] += 1.8  # primary guild affinity
        self.Theta_raw = nn.Parameter(theta_init)
        
        # ONLY W >= 0 (NO row sum cap, NO L2 regularization!)
        self.W_raw = nn.Parameter(torch.randn(K, K) * 0.2 - 0.4)
        
        # Intraspecific competition alpha > 0
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.05 + 0.1)
        
        # Monod half-saturation Km > 0
        self.Km_raw = nn.Parameter(torch.zeros(K) - 1.0)
        
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
        # User constraint: ONLY W >= 0! No row sum cap!
        return F.softplus(self.W_raw)
        
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

def train_model(epochs=300, lr=0.015):
    set_seed(42)
    print("\n" + "="*80)
    print(f"TRAINING MONOD CONSUMER-RESOURCE SERIAL DILUTION MODEL (K=8, ONLY W >= 0)")
    print(f"Dataset: All 99 Communities | Epochs: {epochs} | Zero Regularization")
    print("="*80)
    
    model = MonodGoreK8ODE(N=48, K=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-4)
    t_span = torch.tensor([0.0, 1.0], device=device)
    d4_map = {7: 1, 8: 2, 10: 3}
    
    best_loss = float('inf')
    best_auto_loss = float('inf')
    best_state = None
    start_time = time.time()
    
    # Scheduled sampling: decay over first 75% of epochs (1 to 225)
    # Remaining 75 epochs (226 to 300) are 100% autonomous (TF = 0.0)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        R0_11 = model.get_R0()[b11_c]
        R0_4  = model.get_R0()[b4_c]
        
        raw_tf = 1.0 - (epoch / (epochs * 0.75))
        tf_ratio = max(0.0, min(1.0, raw_tf))
        
        # Batch 11 rollout
        N_curr = b11_N[:, 0, :]
        loss11 = 0.0
        for p in range(1, 11):
            N_in = torch.clamp(N_curr / model.D + model.mu * b11_m, min=0.0)
            S0 = torch.cat([N_in, R0_11], dim=-1)
            S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
            N_final = torch.clamp(S_t[-1, ..., :48] * b11_m, min=0.0)
            loss11 += dixit_kl(b11_N[:, p, :], N_final, b11_m)
            N_curr = tf_ratio * b11_N[:, p, :] + (1.0 - tf_ratio) * N_final
            
        # Batch 4 rollout
        N_curr_4 = b4_N[:, 0, :]
        loss4 = 0.0
        for day in range(1, 11):
            N_in_4 = torch.clamp(N_curr_4 / model.D + model.mu * b4_m, min=0.0)
            S0_4 = torch.cat([N_in_4, R0_4], dim=-1)
            S_t_4 = odeint(model.ode_deriv, S0_4, t_span, method='rk4')
            N_final_4 = torch.clamp(S_t_4[-1, ..., :48] * b4_m, min=0.0)
            if day in d4_map:
                t_idx = d4_map[day]
                loss4 += dixit_kl(b4_N[:, t_idx, :], N_final_4, b4_m)
                N_curr_4 = tf_ratio * b4_N[:, t_idx, :] + (1.0 - tf_ratio) * N_final_4
            else:
                N_curr_4 = N_final_4
                
        # ZERO REGULARIZATION
        total_loss = (loss11 / 10.0) + (loss4 / 3.0)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Track best autonomous checkpoint during the autonomous regime
        if epoch >= int(epochs * 0.70):
            if total_loss.item() < best_auto_loss:
                best_auto_loss = total_loss.item()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif best_state is None or total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 25 == 0 or epoch == epochs or epoch == 1:
            elapsed = time.time() - start_time
            w_max = model.get_W().max().item()
            w_sum = model.get_W().sum(dim=-1).max().item()
            curr_best = best_auto_loss if epoch >= int(epochs * 0.70) else best_loss
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {total_loss.item():.4f} (TF: {tf_ratio:.2f}) | W max: {w_max:.2f}, row_sum max: {w_sum:.2f} | Best: {curr_best:.4f} | Time: {elapsed:.1f}s")
            
    print(f"\nTraining completed in {(time.time() - start_time)/60.0:.2f} minutes.")
    
    # Load best autonomous state
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    
    # Dense intra-cycle simulation (41 points per passage) for Phase Portraits and 30-cycle dynamics
    dense_t = torch.linspace(0.0, 1.0, 41, device=device)
    
    def simulate_dense(nutrient, S, comm, num_passages=30):
        c_idx = cond_to_idx[nutrient]
        mask = data['presence_masks'][(S, comm)].to(device)
        od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0).to(device)
        rel = data['relative_abundance'][(nutrient, S, comm)].to(device)
        true_od = rel * od_mean.unsqueeze(-1)
        
        R0 = model.get_R0()[c_idx]
        N_curr = true_od[0, :].clone()
        
        N_passages = [N_curr.cpu().detach().numpy()]
        R_paths = []
        
        with torch.no_grad():
            for p in range(num_passages):
                N_in = torch.clamp(N_curr / model.D + model.mu * mask, min=0.0)
                S0 = torch.cat([N_in, R0], dim=-1)
                S_t = odeint(model.ode_deriv, S0, dense_t, method='rk4')
                
                # Dense resource path (41, K)
                r_dense = S_t[:, 48:].cpu().numpy()
                R_paths.append(r_dense)
                
                N_final = torch.clamp(S_t[-1, :48] * mask, min=0.0)
                N_curr = N_final
                N_passages.append(N_final.cpu().numpy())
                
        N_passages = np.array(N_passages) # (num_passages+1, 48)
        R_paths = np.array(R_paths)       # (num_passages, 41, K)
        
        late = N_passages[15:]
        std_sp = float(np.std(late, axis=0).max())
        diffs = np.diff(late, axis=0)
        signs = np.sign(diffs)
        flips = int(((signs[:-1] * signs[1:]) < 0).sum())
        
        top3 = np.argsort(late.max(axis=0))[-3:]
        corr_min = 1.0
        if len(top3) >= 2 and np.std(late[:, top3]).min() > 1e-4:
            corr_mat = np.corrcoef(late[:, top3].T)
            corr_min = float(np.nanmin(corr_mat))
            
        return N_passages, R_paths, std_sp, flips, corr_min
        
    print("\n--- Autonomous Dynamic Evaluation (Passages 15..30) ---")
    N_48_ln, R_48_ln, std_48_ln, flips_48_ln, corr_48_ln = simulate_dense('LN', 48, 'Community1', 30)
    print(f"48-sp Low Nutrients  : Max Std = {std_48_ln:.6f} | Flips = {flips_48_ln} | Attractor: {'Point Attractor' if std_48_ln < 0.005 else 'Oscillating'}")
    
    N_48_hn, R_48_hn, std_48_hn, flips_48_hn, corr_48_hn = simulate_dense('HN', 48, 'Community1', 30)
    print(f"48-sp High Nutrients : Max Std = {std_48_hn:.6f} | Flips = {flips_48_hn} | Min Pair Corr = {corr_48_hn:.2f} | Attractor: {'Persistent Chaotic Waves' if std_48_hn > 0.01 else 'Damped'}")
    
    N_24_hn, R_24_hn, std_24_hn, flips_24_hn, corr_24_hn = simulate_dense('HN', 24, 'Community1', 30)
    print(f"24-sp High Nutrients : Max Std = {std_24_hn:.6f} | Flips = {flips_24_hn} | Min Pair Corr = {corr_24_hn:.2f}")
    
    save_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Monod_K8_Fitted.pth')
    torch.save({
        'model_state_dict': best_state,
        'N': 48,
        'K': 8,
        'epochs': epochs,
        'best_loss': best_loss,
        'N_ln': N_48_ln[:21],
        'N_hn': N_48_hn[:21],
        'R_ln': R_48_ln[:20],
        'R_hn': R_48_hn[:20],
        'metrics': {
            'std_48_ln': std_48_ln,
            'std_48_hn': std_48_hn,
            'flips_48_hn': flips_48_hn,
            'corr_48_hn': corr_48_hn,
            'std_24_hn': std_24_hn,
            'flips_24_hn': flips_24_hn,
            'corr_24_hn': corr_24_hn
        }
    }, save_path)
    print(f"\nSaved winning fitted Monod checkpoint to: {save_path}")
    return save_path

if __name__ == '__main__':
    train_model(epochs=300, lr=0.015)

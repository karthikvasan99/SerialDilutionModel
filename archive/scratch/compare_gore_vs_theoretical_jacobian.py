import os
import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1. Theoretical Model
print("="*70)
print("ANALYZING THEORETICAL MODEL (HW Niche Preemption)")
print("="*70)

def theoretical_deriv(t, S):
    N = torch.clamp(S[:3], min=0.0)
    R = torch.clamp(S[3:], min=0.0)
    monod_R = R / (R + 0.2)
    g = 10.0 * monod_R
    dN = N * g
    C = torch.tensor([
        [0.2, 3.0, 0.05],
        [0.05, 0.2, 3.0],
        [3.0, 0.05, 0.2]
    ], device=device)
    depletion = monod_R * torch.matmul(N, C)
    dR = - depletion
    return torch.cat([dN, dR])

# 2. Model fitted to Gore data
print("\n" + "="*70)
print("ANALYZING GORE-FITTED MODEL (K=3, W=0.99, alpha > 0, Theta = Phi)")
print("="*70)

ckpt_path = 'Data/Gore_K3_FixedW99_LimitCycle_ODE.pth'
ckpt = torch.load(ckpt_path, map_location=device)
sd = ckpt['model_state_dict']

Theta = F.softplus(sd['Theta_raw'])
alpha = F.softplus(sd['alpha_raw']).item()
scale_hn = F.softplus(sd['scale_hn']).item()
r_dist = F.softmax(sd['r_dist'], dim=-1)
R0_gore = r_dist * scale_hn

W_gore = torch.zeros(3, 3, device=device)
W_gore[0, 1] = 0.99
W_gore[1, 2] = 0.99
W_gore[2, 0] = 0.99

def gore_deriv(t, S):
    N_t = torch.clamp(S[:48], min=0.0)
    R_t = torch.clamp(S[48:], min=0.0)
    monod = R_t / (R_t + 0.05)
    growth = torch.matmul(monod, Theta.t())
    dN = N_t * growth - alpha * (N_t ** 2)
    flux = monod * torch.matmul(N_t, Theta)
    byproduct = torch.matmul(flux, W_gore)
    dR = - flux + byproduct
    return torch.cat([dN, dR])

print(f"Gore Model alpha (self-limitation): {alpha:.6f}")
print(f"Gore Model Theta mean: {Theta.mean().item():.3f}, max: {Theta.max().item():.3f}")
print(f"Gore Model R0 total: {R0_gore.sum().item():.3f}")
print(f"Coupling symmetry: Theta_growth == Phi_depletion strictly: TRUE")

# Check resource depletion timing during Day in Gore model
data = torch.load('Data/Gore_processed_data.pt', weights_only=False)
mask = data['presence_masks'][(24, 'Community5')].to(device)
N_in = mask.clone().float() / mask.sum() * 0.5 / 10.0
S0 = torch.cat([N_in, R0_gore])
t_span = torch.linspace(0.0, 1.0, 50, device=device)
S_t = odeint(gore_deriv, S0, t_span, method='dopri5', rtol=1e-4, atol=1e-5)

R_traj = S_t[:, 48:].detach().cpu().numpy()
t_eval = t_span.cpu().numpy()

# Find when resources hit ~0 in Gore model
for k in range(3):
    depleted_idx = np.where(R_traj[:, k] < 0.005)[0]
    t_dep = t_eval[depleted_idx[0]] * 24.0 if len(depleted_idx) > 0 else 24.0
    print(f"Gore Model Resource {k} depleted at hour: {t_dep:.2f} / 24h")


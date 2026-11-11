import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Let's see: In a 3-species system, what exact Theta and Phi produces limit cycles under serial dilution with dopri5?
# In our theoretical model:
# Theta = diag(10, 10, 10)
# Phi = [[0.2, 3.0, 0.05], [0.05, 0.2, 3.0], [3.0, 0.05, 0.2]]
# R0 = [1.0, 1.0, 1.0], D = 30.0

def test_3sp(theta_diag, phi_own, phi_preempt, R0_val=1.0, D=30.0):
    Theta = torch.eye(3, device=device) * theta_diag
    Phi = torch.tensor([
        [phi_own, phi_preempt, 0.05],
        [0.05, phi_own, phi_preempt],
        [phi_preempt, 0.05, phi_own]
    ], device=device)
    
    def deriv(t, S):
        N = torch.clamp(S[:3], min=0.0)
        R = torch.clamp(S[3:], min=0.0)
        monod = R / (R + 0.2)
        growth = monod * 10.0 # diag(10, 10, 10)
        dN = N * growth
        flux = monod * torch.matmul(N, Phi)
        dR = - flux
        return torch.cat([dN, dR])
        
    t_span = torch.linspace(0.0, 1.0, 25, device=device)
    R0 = torch.tensor([R0_val, R0_val, R0_val], device=device)
    S = torch.tensor([0.2, 0.2, 0.2, R0_val, R0_val, R0_val], device=device)
    
    traj = []
    for p in range(60):
        N_in = S[:3] / D + 1e-4
        S_start = torch.cat([N_in, R0])
        S_t = odeint(deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        S = torch.clamp(S_t[-1], min=0.0)
        traj.append(S[:3].cpu().numpy())
    traj = np.array(traj)
    
    # Check if late passages oscillate
    last20 = traj[40:]
    diffs = np.diff(last20[:, 0])
    sign_flips = np.sum(diffs[:-1] * diffs[1:] < 0)
    p2p = (last20[:, 0].max() - last20[:, 0].min()) / last20[:, 0].mean()
    return sign_flips, p2p, last20[-1]

print("Testing cyclic preemption parameters (Phi_preempt vs Phi_own):")
for phi_preempt in [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]:
    flips, p2p, end_val = test_3sp(10.0, 0.2, phi_preempt)
    status = f"LIMIT CYCLE (flips={flips}, p2p={p2p*100:.1f}%)" if flips > 4 else f"FIXED POINT (p2p={p2p*100:.2f}%)"
    print(f"  phi_preempt = {phi_preempt:.1f}: {status}")


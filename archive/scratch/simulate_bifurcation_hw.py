import os
import torch
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def simulate_community(R0_val, ch=3.0, cl=0.2, passages=100, D=30.0):
    r = torch.tensor([10.0, 10.0, 10.0], device=device)
    K = torch.tensor([0.2, 0.2, 0.2], device=device)
    C = torch.tensor([
        [cl, ch, 0.05],
        [0.05, cl, ch],
        [ch, 0.05, cl]
    ], device=device)
    
    def deriv(t, S):
        N = torch.clamp(S[:3], min=0.0)
        R = torch.clamp(S[3:], min=0.0)
        
        monod_R = R / (R + K)
        g = r * monod_R
        dN = N * g
        
        # Strictly non-negative resource consumption
        depletion = monod_R * torch.matmul(N, C)
        dR = - depletion
        return torch.cat([dN, dR])
        
    t_span = torch.linspace(0.0, 1.0, 25, device=device)
    R0_pulse = torch.tensor([R0_val, R0_val, R0_val], device=device)
    S = torch.tensor([0.1, 0.15, 0.08, R0_val, R0_val, R0_val], device=device)
    
    traj = []
    min_R_seen = float('inf')
    for p in range(passages):
        N_in = S[:3] / D + 1e-4
        S_start = torch.cat([N_in, R0_pulse])
        S_t = odeint(deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        S = S_t[-1]
        min_R_seen = min(min_R_seen, S_t[:, 3:].min().item())
        traj.append(S[:3].detach().cpu().numpy())
        
    traj = np.array(traj)
    late = traj[50:]
    diffs0 = np.diff(late[:, 0])
    flips0 = np.sum(diffs0[:-1] * diffs0[1:] < -1e-8)
    p2p = (late.max(axis=0) - late.min(axis=0)) / np.maximum(late.mean(axis=0), 1e-6)
    return traj, p2p, flips0, min_R_seen

print("Running LN (R0=0.08)...", flush=True)
traj_ln, p2p_ln, flips_ln, min_r_ln = simulate_community(R0_val=0.08, ch=3.0)
print(f"LN: Flips={flips_ln}, P2P={p2p_ln.round(4)}, min_R={min_r_ln:.6e}, Last 3:\n{traj_ln[-3:].round(4)}", flush=True)

print("\nRunning HN (R0=1.0)...", flush=True)
traj_hn, p2p_hn, flips_hn, min_r_hn = simulate_community(R0_val=1.0, ch=3.0)
print(f"HN: Flips={flips_hn}, P2P={p2p_hn.round(4)}, min_R={min_r_hn:.6e}, Last 6:\n{traj_hn[-6:].round(4)}", flush=True)

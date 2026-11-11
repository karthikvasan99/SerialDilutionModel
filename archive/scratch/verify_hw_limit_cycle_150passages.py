import os
import torch
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def simulate_hw_model(R0_val, ch=3.0, cl=0.2, passages=150, D=30.0):
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
        
        # Monod saturation on each resource
        monod_R = R / (R + K)
        g = r * monod_R
        dN = N * g
        
        # Depletion rate of resource j
        depletion = monod_R * torch.matmul(N, C)
        dR = - depletion
        return torch.cat([dN, dR])
        
    t_span = torch.linspace(0.0, 1.0, 30, device=device)
    R0_pulse = torch.tensor([R0_val, R0_val, R0_val], device=device)
    S = torch.tensor([0.1, 0.15, 0.08, R0_val, R0_val, R0_val], device=device)
    
    traj = []
    min_R_seen = float('inf')
    for p in range(passages):
        N_in = S[:3] / D + 1e-4
        S_start = torch.cat([N_in, R0_pulse])
        S_t = odeint(deriv, S_start, t_span, method='dopri5', rtol=1e-5, atol=1e-6)
        S = torch.clamp(S_t[-1], min=0.0)
        min_R_seen = min(min_R_seen, S_t[:, 3:].min().item())
        traj.append(S[:3].detach().cpu().numpy())
        
    traj = np.array(traj)
    late = traj[80:]
    diffs = [np.diff(late[:, i]) for i in range(3)]
    flips = [np.sum(d[:-1] * d[1:] < -1e-8) for d in diffs]
    p2p = (late.max(axis=0) - late.min(axis=0)) / np.maximum(late.mean(axis=0), 1e-6)
    return traj, p2p, flips, min_R_seen

print("Running 150 Passages with DOPRI5 (rtol=1e-5, atol=1e-6):")
print("1. Simulating Low Nutrients (LN: R0=0.08)...", flush=True)
traj_ln, p2p_ln, flips_ln, min_r_ln = simulate_hw_model(R0_val=0.08)
print(f"   LN Complete: Flips={flips_ln}, P2P={p2p_ln.round(4)}, min_R={min_r_ln:.6e}")
print(f"   LN Asymptotic State: {traj_ln[-1].round(4)}")

print("2. Simulating High Nutrients (HN: R0=1.0)...", flush=True)
traj_hn, p2p_hn, flips_hn, min_r_hn = simulate_hw_model(R0_val=1.0)
print(f"   HN Complete: Flips={flips_hn}, P2P={p2p_hn.round(4)}, min_R={min_r_hn:.6e}")
print(f"   HN Peak-to-Peak: Sp0={p2p_hn[0]*100:.1f}%, Sp1={p2p_hn[1]*100:.1f}%, Sp2={p2p_hn[2]*100:.1f}%")
print(f"   HN Sample Passages (140-150):\n{traj_hn[140:150].round(4)}")

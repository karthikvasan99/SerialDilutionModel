import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_theta_phi(N_sp=3, K_res=3, asym_strength=2.0, D=30.0, R0_val=1.0, passages=80):
    # Setup Theta (growth benefit) and Phi (depletion rate)
    # Ring-like non-reciprocal coupling
    # Species i gets high benefit from resource i, but depletes resource (i+1)%K heavily
    Theta = torch.tensor([
        [5.0, 0.5, 0.5],
        [0.5, 5.0, 0.5],
        [0.5, 0.5, 5.0]
    ], device=device)
    
    # Decoupled depletion Phi: Species i depletes resource (i+1)%K heavily (preemption / sink)
    Phi = torch.tensor([
        [2.0, asym_strength, 0.2],
        [0.2, 2.0, asym_strength],
        [asym_strength, 0.2, 2.0]
    ], device=device)
    
    # Cross feeding W: each consumed resource yields byproduct
    # Consuming R_k produces R_{(k+1)%K} with yield w
    w = 0.5
    W = torch.tensor([
        [0.0, w, 0.0],
        [0.0, 0.0, w],
        [w, 0.0, 0.0]
    ], device=device)
    
    Km = 0.1
    m = 0.5
    
    def deriv(t, S):
        N_t = torch.clamp(S[:N_sp], min=0.0)
        R_t = torch.clamp(S[N_sp:], min=0.0)
        
        monod = R_t / (R_t + Km)
        growth = torch.matmul(monod, Theta.t()) - m
        dN = N_t * growth
        
        depletion = monod * torch.matmul(N_t, Phi)
        byproduct = torch.matmul(depletion, W)
        dR = - depletion + byproduct
        return torch.cat([dN, dR])
        
    t_span = torch.linspace(0.0, 1.0, 25, device=device)
    S = torch.tensor([0.1, 0.1, 0.1, R0_val, 0.0, 0.0], device=device)
    R0_pulse = torch.tensor([R0_val, 0.0, 0.0], device=device)
    
    N_hist = []
    for p in range(passages):
        N_in = S[:N_sp] / D + 1e-4
        S_start = torch.cat([N_in, R0_pulse])
        S_t = odeint(deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        S = S_t[-1]
        N_hist.append(S[:N_sp].detach().cpu().numpy())
        
    N_hist = np.array(N_hist)
    late = N_hist[40:]
    p2p = (late.max(axis=0) - late.min(axis=0)) / np.maximum(late.mean(axis=0), 1e-6)
    diffs = np.diff(late[:, 0])
    flips = np.sum(diffs[:-1] * diffs[1:] < -1e-8)
    return p2p, flips, late[-5:]

for asym in [0.5, 2.0, 5.0, 10.0, 20.0]:
    for r0 in [0.2, 1.0, 5.0]:
        p2p, flips, sample = test_theta_phi(asym_strength=asym, R0_val=r0)
        if flips > 5 or np.any(p2p > 0.05):
            print(f"FOUND OSCILLATIONS! asym={asym}, R0={r0} -> P2P={p2p.max():.3f}, flips={flips}")
        else:
            print(f"Fixed point: asym={asym}, R0={r0} -> P2P={p2p.max():.4f}, flips={flips}")

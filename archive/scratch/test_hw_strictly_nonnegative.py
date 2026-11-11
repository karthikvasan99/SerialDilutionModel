import torch
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def run_hw_strictly_nonneg(D=30.0, passages=100, R0_val=2.0, ch=5.0, cl=0.5):
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
        monod_R = R / (R + K) # (3,)
        
        # Species growth: needs resource i to grow
        g = r * monod_R
        dN = N * g
        
        # Resource consumption: each resource j is consumed proportional to monod_R[j]!
        # This guarantees dR_j/dt >= 0 as R_j -> 0!
        # Depletion rate of resource j is monod_R[j] * sum_i (N_i * C_{i, j})
        depletion = monod_R * torch.matmul(N, C)
        dR = - depletion
        return torch.cat([dN, dR])
        
    t_span = torch.linspace(0.0, 1.0, 20, device=device)
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
    return flips0, p2p, min_R_seen, late[-5:]

for r0 in [0.5, 1.0, 2.0, 5.0, 10.0]:
    for ch in [2.0, 5.0, 10.0, 20.0]:
        flips, p2p, min_r, sample = run_hw_strictly_nonneg(R0_val=r0, ch=ch)
        if flips >= 10 and np.all(p2p > 0.05):
            print(f"--> SUSTAINED LIMIT CYCLE! R0={r0}, ch={ch} | Flips={flips}, P2P={p2p.round(3)} | min_R={min_r:.6e}")
        elif flips >= 5:
            print(f"Weak cycle: R0={r0}, ch={ch} | Flips={flips}, P2P={p2p.round(3)} | min_R={min_r:.6e}")
        else:
            print(f"Fixed pt: R0={r0}, ch={ch} | Flips={flips}, P2P={p2p.max():.4f} | min_R={min_r:.6e}")

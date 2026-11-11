import torch
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def run_hw(D=30.0, passages=120, R0_val=0.5, ch=0.5, cl=0.1):
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
        g = r * R / (R + K)
        dN = N * g
        dR = - torch.matmul(N * g, C)
        return torch.cat([dN, dR])
        
    t_span = torch.linspace(0.0, 1.0, 20, device=device)
    R0_pulse = torch.tensor([R0_val, R0_val, R0_val], device=device)
    S = torch.tensor([0.1, 0.15, 0.08, R0_val, R0_val, R0_val], device=device)
    
    traj = []
    for p in range(passages):
        N_in = S[:3] / D + 1e-4
        S_start = torch.cat([N_in, R0_pulse])
        S_t = odeint(deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        S = S_t[-1]
        traj.append(S[:3].detach().cpu().numpy())
        
    traj = np.array(traj)
    print("Passages 80-100:")
    for i in range(80, 100):
        print(f"P{i:3d}: Sp0={traj[i, 0]:.4f}, Sp1={traj[i, 1]:.4f}, Sp2={traj[i, 2]:.4f}")
    
    late = traj[60:]
    diffs0 = np.diff(late[:, 0])
    flips0 = np.sum(diffs0[:-1] * diffs0[1:] < -1e-8)
    p2p = (late.max(axis=0) - late.min(axis=0)) / np.maximum(late.mean(axis=0), 1e-6)
    print(f"\nLate passages (60-120): Flips={flips0}, P2P={p2p.round(3)}")

run_hw()

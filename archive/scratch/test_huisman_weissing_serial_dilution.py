import torch
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_huisman_weissing(D=30.0, passages=80, R0_val=2.0, c_heavy=0.8, c_light=0.1):
    # 3 species, 3 resources
    # Species i needs resource i for growth, but consumes resource (i+1)%3 heavily!
    # Growth rates:
    # mu_i(R) = r_i * R_i / (K_i + R_i)
    # Resource consumption matrix C:
    # C[i, j] = rate at which species i consumes resource j
    
    r = torch.tensor([10.0, 10.0, 10.0], device=device)
    K = torch.tensor([0.2, 0.2, 0.2], device=device)
    
    # Consumption matrix: Species 0 consumes R0 (light) and R1 (heavy)
    # Species 1 consumes R1 (light) and R2 (heavy)
    # Species 2 consumes R2 (light) and R0 (heavy)
    C = torch.tensor([
        [c_light, c_heavy, 0.05],
        [0.05, c_light, c_heavy],
        [c_heavy, 0.05, c_light]
    ], device=device)
    
    # Growth yield: species i grows on resource i
    def deriv(t, S):
        N = torch.clamp(S[:3], min=0.0)
        R = torch.clamp(S[3:], min=0.0)
        
        # Growth determined by essential resource i
        # Monod on primary resource
        g = r * R / (R + K)
        dN = N * g
        
        # Resource consumption: sum_i N_i * C_{i, j} * g_i
        dR = - torch.matmul(N * g, C)
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
    late = traj[40:]
    p2p = (late.max(axis=0) - late.min(axis=0)) / np.maximum(late.mean(axis=0), 1e-6)
    diffs = np.diff(late[:, 0])
    flips = np.sum(diffs[:-1] * diffs[1:] < -1e-8)
    return traj, p2p, flips, min_R_seen

print("Testing Huisman-Weissing Cyclic Dominance under DOPRI5:")
for r0 in [0.5, 1.0, 2.0, 4.0]:
    for ch in [0.5, 1.0, 2.0, 4.0]:
        for cl in [0.1, 0.3]:
            traj, p2p, flips, min_r = test_huisman_weissing(R0_val=r0, c_heavy=ch, c_light=cl)
            if flips >= 10 and np.all(p2p > 0.05):
                print(f"--> LIMIT CYCLE! R0={r0:3.1f}, ch={ch:3.1f}, cl={cl:3.1f} | Flips: {flips:2d} | P2P: {p2p.round(3)} | min_R: {min_r:.6f}")
            elif flips >= 5:
                print(f"Weak/Damped: R0={r0:3.1f}, ch={ch:3.1f}, cl={cl:3.1f} | Flips: {flips:2d} | P2P: {p2p.round(3)}")

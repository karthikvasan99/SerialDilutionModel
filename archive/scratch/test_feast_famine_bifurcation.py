import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def simulate_feast_famine(R0_val, delta, mu_max=12.0, Km=0.1, Kd=0.01, D=30.0, passages=80):
    def deriv(t, S):
        N = torch.clamp(S[0], min=0.0)
        R = torch.clamp(S[1], min=0.0)
        
        monod = R / (R + Km)
        growth = mu_max * monod
        
        # Starvation death gate: active when R is depleted
        starv = Kd / (R + Kd)
        death = delta * starv
        
        dN = N * (growth - death)
        dR = - N * growth
        return torch.stack([dN, dR])
        
    t_span = torch.linspace(0.0, 1.0, 20, device=device)
    S = torch.tensor([0.1, R0_val], device=device)
    
    N_hist = []
    min_R = float('inf')
    for p in range(passages):
        N_in = S[0] / D + 1e-5
        S_start = torch.stack([N_in, torch.tensor(R0_val, device=device)])
        S_t = odeint(deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        S = S_t[-1]
        min_R = min(min_R, S_t[:, 1].min().item())
        N_hist.append(S[0].item())
        
    N_arr = np.array(N_hist)
    late = N_arr[40:]
    p2p = (late.max() - late.min()) / max(late.mean(), 1e-6)
    diffs = np.diff(late)
    flips = np.sum(diffs[:-1] * diffs[1:] < -1e-8)
    return p2p, flips, min_R, late[-6:]

print("Feast-Famine Starvation Parameter Sweep under DOPRI5:")
print(f"{'R0':>5} | {'delta':>5} | {'Flips':>5} | {'P2P':>7} | {'min_R':>10} | {'Trajectory (last 6)'}")
print("-" * 65)

for r0 in [0.2, 0.5, 1.0, 2.0, 5.0]:
    for d in [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]:
        p2p, flips, min_r, sample = simulate_feast_famine(r0, d)
        if flips >= 10 and p2p > 0.05:
            print(f"{r0:5.1f} | {d:5.1f} | {flips:5d} | {p2p:7.3f} | {min_r:10.2e} | LIMIT CYCLE: {sample.round(3)}")
        elif flips >= 5:
            print(f"{r0:5.1f} | {d:5.1f} | {flips:5d} | {p2p:7.3f} | {min_r:10.2e} | Weak cycle: {sample.round(3)}")
        else:
            if d in [2.0, 8.0]:
                print(f"{r0:5.1f} | {d:5.1f} | {flips:5d} | {p2p:7.4f} | {min_r:10.2e} | Fixed pt: {sample[-1]:.4f}")

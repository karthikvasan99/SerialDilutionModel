import torch
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def run_trajectory(R0_val, delta=3.0, mu_max=12.0, Km=0.1, Kd=0.01, D=30.0, passages=50):
    def deriv(t, S):
        N = torch.clamp(S[0], min=0.0)
        R = torch.clamp(S[1], min=0.0)
        g = mu_max * R / (R + Km)
        starv = Kd / (R + Kd)
        death = delta * starv
        return torch.stack([N * (g - death), - N * g])
        
    t_span = torch.linspace(0.0, 1.0, 20, device=device)
    S = torch.tensor([0.1, R0_val], device=device)
    
    traj = []
    min_R_seen = float('inf')
    for p in range(passages):
        N_in = S[0] / D + 1e-4
        S_start = torch.stack([N_in, torch.tensor(R0_val, device=device)])
        S_t = odeint(deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        S = S_t[-1]
        min_R_seen = min(min_R_seen, S_t[:, 1].min().item())
        traj.append(S[0].item())
        
    traj = np.array(traj)
    late = traj[25:]
    p2p = (late.max() - late.min()) / max(late.mean(), 1e-6)
    diffs = np.diff(late)
    flips = np.sum(diffs[:-1] * diffs[1:] < -1e-8)
    return traj, p2p, flips, min_R_seen

print("Running Trajectory Test with dopri5:")
for r0 in [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]:
    for delta in [1.0, 2.0, 3.0, 4.0]:
        traj, p2p, flips, min_r = run_trajectory(R0_val=r0, delta=delta)
        status = "FIXED POINT"
        if flips >= 10 and p2p > 0.05:
            status = "SUSTAINED LIMIT CYCLE"
        elif flips >= 5:
            status = "DAMPED / WEAK OSCILLATION"
        print(f"R0={r0:4.1f}, delta={delta:4.1f} | Status: {status:23s} | Flips: {flips:2d} | P2P: {p2p*100:5.1f}% | min_R: {min_r:.6f} | Last 4: {traj[-4:].round(4)}")

import torch
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

r = torch.tensor([10.0, 10.0, 10.0], device=device)
K = torch.tensor([0.2, 0.2, 0.2], device=device)
C = torch.tensor([
    [0.1, 0.5, 0.05],
    [0.05, 0.1, 0.5],
    [0.5, 0.05, 0.1]
], device=device)

def deriv(t, S):
    N = torch.clamp(S[:3], min=0.0)
    R = torch.clamp(S[3:], min=0.0)
    g = r * R / (R + K)
    dN = N * g
    dR = - torch.matmul(N * g, C)
    return torch.cat([dN, dR])
    
t_span = torch.linspace(0.0, 1.0, 50, device=device)
R0_pulse = torch.tensor([0.5, 0.5, 0.5], device=device)
S = torch.tensor([0.1, 0.15, 0.08, 0.5, 0.5, 0.5], device=device)

min_R_overall = float('inf')
for p in range(50):
    N_in = S[:3] / 30.0 + 1e-4
    S_start = torch.cat([N_in, R0_pulse])
    S_t = odeint(deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
    S = S_t[-1]
    min_R_overall = min(min_R_overall, S_t[:, 3:].min().item())

print(f"Overall minimum resource concentration: {min_R_overall:.6e}")

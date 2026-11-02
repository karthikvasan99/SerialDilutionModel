import os
import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1. Theoretical Model Return Map Jacobian at Fixed Point
def theoretical_deriv(t, S):
    N = torch.clamp(S[:3], min=0.0)
    R = torch.clamp(S[3:], min=0.0)
    monod_R = R / (R + 0.2)
    g = 10.0 * monod_R
    dN = N * g
    C = torch.tensor([
        [0.2, 3.0, 0.05],
        [0.05, 0.2, 3.0],
        [3.0, 0.05, 0.2]
    ], device=device)
    depletion = monod_R * torch.matmul(N, C)
    dR = - depletion
    return torch.cat([dN, dR])

t_span = torch.linspace(0.0, 1.0, 25, device=device)

# In Theoretical Model:
# At LN (R0 = 0.08):
R0_ln = torch.tensor([0.08, 0.08, 0.08], device=device)
N_star_ln = torch.tensor([0.0041, 0.0041, 0.0041], device=device)

# Numerical Jacobian of return map F(N) = N_end / D
J_th_ln = np.zeros((3, 3))
eps = 1e-5
for j in range(3):
    pert = torch.zeros(3, device=device)
    pert[j] = eps
    S0_pert = torch.cat([(N_star_ln + pert) / 30.0, R0_ln])
    S_end_pert = odeint(theoretical_deriv, S0_pert, t_span, method='dopri5', rtol=1e-4, atol=1e-5)[-1, :3]
    
    S0_base = torch.cat([N_star_ln / 30.0, R0_ln])
    S_end_base = odeint(theoretical_deriv, S0_base, t_span, method='dopri5', rtol=1e-4, atol=1e-5)[-1, :3]
    J_th_ln[:, j] = ((S_end_pert - S_end_base) / eps).cpu().numpy()

eigs_th_ln = np.linalg.eigvals(J_th_ln)
print("Theoretical Model LN Return Map Eigenvalues:")
print(f"  Eigenvalues: {eigs_th_ln}")
print(f"  Max |Eigenvalue|: {np.max(np.abs(eigs_th_ln)):.4f} (< 1.0 -> STABLE FIXED POINT)")

# At HN (R0 = 1.0):
R0_hn = torch.tensor([1.0, 1.0, 1.0], device=device)
N_guess_hn = torch.tensor([1.0, 1.0, 1.0], device=device)
# Run 100 passages to see where it goes
N_curr = N_guess_hn
for _ in range(80):
    S0 = torch.cat([N_curr / 30.0, R0_hn])
    N_curr = odeint(theoretical_deriv, S0, t_span, method='dopri5', rtol=1e-4, atol=1e-5)[-1, :3]

# Compute Jacobian along the orbit
J_th_hn = np.zeros((3, 3))
for j in range(3):
    pert = torch.zeros(3, device=device)
    pert[j] = eps
    S0_pert = torch.cat([(N_curr + pert) / 30.0, R0_hn])
    S_end_pert = odeint(theoretical_deriv, S0_pert, t_span, method='dopri5', rtol=1e-4, atol=1e-5)[-1, :3]
    
    S0_base = torch.cat([N_curr / 30.0, R0_hn])
    S_end_base = odeint(theoretical_deriv, S0_base, t_span, method='dopri5', rtol=1e-4, atol=1e-5)[-1, :3]
    J_th_hn[:, j] = ((S_end_pert - S_end_base) / eps).cpu().numpy()

eigs_th_hn = np.linalg.eigvals(J_th_hn)
print("\nTheoretical Model HN Return Map Eigenvalues (on orbit):")
print(f"  Eigenvalues: {eigs_th_hn}")
print(f"  Max |Eigenvalue|: {np.max(np.abs(eigs_th_hn)):.4f}")


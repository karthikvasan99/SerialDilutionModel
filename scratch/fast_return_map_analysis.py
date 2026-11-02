import torch
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def compute_return_map_derivative(deriv_fn, N_guess, R0_val, D=30.0, eps=1e-4):
    t_span = torch.linspace(0.0, 1.0, 20, device=device)
    
    # Find fixed point by iterating 50 passages
    N_curr = torch.tensor(N_guess, device=device)
    for _ in range(50):
        N_in = N_curr / D
        S_start = torch.cat([N_in, torch.tensor(R0_val, device=device)])
        S_end = odeint(deriv_fn, S_start, t_span, method='dopri5', rtol=1e-5, atol=1e-6)[-1]
        N_curr = S_end[:len(N_guess)]
        
    N_star = N_curr
    
    # Perturb each species by eps to get Jacobian of return map F(N) = N_end / D
    dim = len(N_guess)
    J = torch.zeros((dim, dim), device=device)
    
    for j in range(dim):
        delta = torch.zeros(dim, device=device)
        delta[j] = eps
        
        N_pert = (N_star + delta) / D
        S_pert = torch.cat([N_pert, torch.tensor(R0_val, device=device)])
        S_end_pert = odeint(deriv_fn, S_pert, t_span, method='dopri5', rtol=1e-5, atol=1e-6)[-1]
        F_pert = S_end_pert[:dim]
        
        N_unpert = N_star / D
        S_unpert = torch.cat([N_unpert, torch.tensor(R0_val, device=device)])
        S_end_unpert = odeint(deriv_fn, S_unpert, t_span, method='dopri5', rtol=1e-5, atol=1e-6)[-1]
        F_unpert = S_end_unpert[:dim]
        
        J[:, j] = (F_pert - F_unpert) / (eps / D)
        
    eigenvalues = torch.linalg.eigvals(J).cpu().numpy()
    max_abs_eig = np.max(np.abs(eigenvalues))
    return N_star.cpu().numpy(), eigenvalues, max_abs_eig

print("1. Testing Standard Monod Batch Model:")
def std_deriv(t, S):
    N, R = torch.clamp(S[0], min=0.0), torch.clamp(S[1], min=0.0)
    g = 12.0 * R / (R + 0.1)
    return torch.stack([N * g, - N * g])

N_star, eigs, max_eig = compute_return_map_derivative(std_deriv, [1.0], [2.0])
print(f"Standard Monod: N* = {N_star[0]:.4f}, Eigenvalues = {eigs}, max |eig| = {max_eig:.4f}")

print("\n2. Testing Feast-Famine Starvation Model (delta = 5.0, 10.0, 15.0):")
for delta_val in [2.0, 5.0, 8.0, 12.0, 15.0, 20.0]:
    def starv_deriv(t, S, d=delta_val):
        N, R = torch.clamp(S[0], min=0.0), torch.clamp(S[1], min=0.0)
        g = 12.0 * R / (R + 0.1)
        death = d * (0.01 / (R + 0.01))
        return torch.stack([N * (g - death), - N * g])
    N_star, eigs, max_eig = compute_return_map_derivative(starv_deriv, [1.0], [2.0])
    print(f"Starvation delta={delta_val:4.1f}: N* = {N_star[0]:.4f}, Eigenvalues = {eigs.round(4)}, max |eig| = {max_eig:.4f}")

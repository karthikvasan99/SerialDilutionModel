import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Candidate 1: Blumenthal, Rocks, Mehta (PRL 2024) - Non-reciprocal Theta != Phi
# Candidate 2: Trophic Cross-Inhibition (Toxins / Acidification)
# Candidate 3: Substrate Inhibition (Haldane) + Cross-feeding
# Candidate 4: Asymmetric Diauxic / Essential resources (Huisman & Weissing)

def run_serial_dilution(deriv_fn, S0_init, R0_pulse, D, passages=100, t_steps=20):
    t_span = torch.linspace(0.0, 1.0, t_steps, device=device)
    S = S0_init.clone()
    N_history = []
    
    # Store dimensions
    dim_N = S0_init.shape[-1] - R0_pulse.shape[-1]
    
    for p in range(passages):
        # Dilution and resource replenishment
        N_in = S[:dim_N] / D
        S_start = torch.cat([N_in, R0_pulse], dim=-1)
        
        # Adaptive dopri5 solver
        S_t = odeint(deriv_fn, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        S = S_t[-1]
        N_history.append(S[:dim_N].detach().cpu().numpy())
        
    N_arr = np.array(N_history) # (passages, dim_N)
    # Check last 50 passages for oscillations
    late = N_arr[50:]
    std = late.std(axis=0)
    p2p = (late.max(axis=0) - late.min(axis=0)) / np.maximum(late.mean(axis=0), 1e-6)
    
    # Count direction flips
    diffs = np.diff(late[:, 0])
    flips = np.sum(diffs[:-1] * diffs[1:] < -1e-8)
    
    return N_arr, p2p, flips

print("Ready to test candidates.")

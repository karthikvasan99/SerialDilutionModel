import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Theoretical Model: Classical Microbial Rock-Paper-Scissors / Colicin Consumer-Resource Model
# Under Serial Dilution with dopri5
# 3 Strains:
# Strain 0 (Producer C): Consumes nutrient R0, secretes toxin R1 (colicin) at rate w_tox, immune to toxin. Growth rate slightly lower due to metabolic cost.
# Strain 1 (Sensitive S): Consumes nutrient R0 with highest growth rate, sensitive to toxin R1.
# Strain 2 (Resistant R): Consumes nutrient R0 with intermediate growth rate, immune to toxin R1, does not produce toxin.
#
# Primary Nutrient: R0 (supplied at each passage, e.g. R0_LN = 0.5, R0_HN = 5.0)
# Toxin: R1 (secreted by Strain 0 as a byproduct of growth/consumption)

def test_colicin_rps(R0_val=2.0, tox_potency=5.0, tox_yield=0.5, D=30.0, passages=100):
    # Parameters
    # Growth rates on primary nutrient R0: S > R > C
    mu_S = 10.0
    mu_R = 8.5
    mu_C = 7.0
    
    Km = 0.2
    Ki = 0.1 # half-inhibition concentration of toxin for S
    
    def deriv(t, S_vec):
        N_C = torch.clamp(S_vec[0], min=0.0)
        N_S = torch.clamp(S_vec[1], min=0.0)
        N_R = torch.clamp(S_vec[2], min=0.0)
        R0 = torch.clamp(S_vec[3], min=0.0)
        Tox = torch.clamp(S_vec[4], min=0.0)
        
        monod = R0 / (R0 + Km)
        
        # Growth
        g_C = mu_C * monod
        # S is inhibited by toxin (or killed by toxin)
        # Using standard non-competitive inhibition or direct killing:
        # g_S = mu_S * monod * (Ki / (Ki + Tox))
        # Or toxic mortality:
        g_S = mu_S * monod - tox_potency * (Tox / (Ki + Tox))
        g_R = mu_R * monod
        
        dN_C = N_C * g_C
        dN_S = N_S * g_S
        dN_R = N_R * g_R
        
        # Resource consumption
        # 1 unit biomass consumes 1 unit R0
        dR0 = - (N_C * g_C + N_S * torch.clamp(g_S, min=0.0) + N_R * g_R)
        # Toxin production: C produces toxin proportional to growth
        dTox = tox_yield * N_C * g_C - 0.5 * Tox # degradation
        
        return torch.stack([dN_C, dN_S, dN_R, dR0, dTox])

    t_span = torch.linspace(0.0, 1.0, 20, device=device)
    # Initial state
    S = torch.tensor([0.1, 0.1, 0.1, R0_val, 0.0], device=device)
    
    N_hist = []
    min_R0, min_Tox = float('inf'), float('inf')
    for p in range(passages):
        N_in = S[:3] / D + 1e-4
        # At dilution, culture is diluted by D into fresh medium with R0_val nutrient
        # Toxin is also diluted by D
        Tox_in = S[4] / D
        S_start = torch.cat([N_in, torch.tensor([R0_val, Tox_in], device=device)])
        
        S_t = odeint(deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        S = S_t[-1]
        
        min_R0 = min(min_R0, S_t[:, 3].min().item())
        min_Tox = min(min_Tox, S_t[:, 4].min().item())
        N_hist.append(S[:3].detach().cpu().numpy())
        
    N_hist = np.array(N_hist)
    late = N_hist[50:]
    p2p = (late.max(axis=0) - late.min(axis=0)) / np.maximum(late.mean(axis=0), 1e-6)
    diffs = np.diff(late[:, 0])
    flips = np.sum(diffs[:-1] * diffs[1:] < -1e-8)
    return p2p, flips, min_R0, min_Tox, late[-10:]

print("Testing Colicin RPS model across parameter grid...")
for r0 in [0.5, 1.0, 2.0, 5.0, 10.0]:
    for tox_p in [1.0, 3.0, 5.0, 10.0, 15.0]:
        p2p, flips, min_r, min_t, sample = test_colicin_rps(R0_val=r0, tox_potency=tox_p)
        if flips >= 10 and np.all(p2p > 0.05):
            print(f"--> SUSTAINED LIMIT CYCLE FOUND! R0={r0}, tox_p={tox_p} | Flips={flips}, P2P={p2p.round(3)} | min_R={min_r:.6f}")
        elif flips >= 5:
            print(f"Transient/weak cycle: R0={r0}, tox_p={tox_p} | Flips={flips}, P2P={p2p.round(3)}")

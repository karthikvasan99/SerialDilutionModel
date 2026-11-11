"""
Rigorous Chaos Diagnostics for Fitted Serial Dilution Model on Gore et al. (Science 2022)
Evaluates:
1. Long-term asymptotic behavior (Passages 50 to 200) across LN, MN, HN.
2. Maximal Lyapunov Exponent (lambda_max) via Benettin's algorithm on the discrete Poincare dilution map.
3. Butterfly Effect / Sensitivity to initial conditions (exponential trajectory divergence delta(m)).
4. Fourier Power Spectral Density (PSD) on late passages (discrete harmonics vs broadband continuum).
5. Richness-dependent bifurcation across pool sizes S in {3, 6, 12, 24, 48}.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load fitted model
ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Fitted_Niche_Chaos.pth')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
sd = ckpt['model_state_dict']

# Load experimental data
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
data = torch.load(data_path, weights_only=False)

# Model parameters
Theta = F.softplus(sd['Theta_raw']).to(device)
W_sp = F.softplus(sd['W_raw']).to(device)
W_sp = W_sp * (1.0 - torch.eye(5, device=device))
row_sums = W_sp.sum(dim=-1, keepdim=True)
scale = torch.clamp(0.85 / torch.clamp(row_sums, min=1e-6), max=1.0)
W = (W_sp * scale).to(device)
alpha = (0.005 * torch.sigmoid(sd['alpha_raw']) + 1e-5).to(device)
Km = (F.softplus(sd['Km_raw']) + 0.05).to(device)

R0_ln = F.softplus(sd['a_ln']).to(device)
R0_mn = (R0_ln + F.softplus(sd['delta_mn'].to(device)))
R0_hn = (R0_mn + F.softplus(sd['delta_hn'].to(device)))

print(f"R0 LN: {R0_ln.cpu().numpy().round(3)}")
print(f"R0 MN: {R0_mn.cpu().numpy().round(3)}")
print(f"R0 HN: {R0_hn.cpu().numpy().round(3)}")
print(f"Alpha range: [{alpha.min().item():.5f}, {alpha.max().item():.5f}]")
print(f"Km range: [{Km.min().item():.3f}, {Km.max().item():.3f}]")

def ode_deriv(t, S):
    N_t = torch.clamp(S[..., :48], min=0.0)
    R_t = torch.clamp(S[..., 48:], min=0.0)
    monod = R_t / (R_t + Km)
    growth = torch.matmul(monod, Theta.t())
    dN = N_t * growth - alpha * (N_t ** 2)
    flux = monod * torch.matmul(N_t, Theta)
    byproduct = torch.matmul(flux, W)
    dR = - flux + byproduct
    return torch.cat([dN, dR], dim=-1)

t_span = torch.linspace(0.0, 1.0, 25, device=device)

def one_passage_step(N_in, R0_vec, mask):
    # Serial dilution step: within-passage ODE integration
    S0 = torch.cat([N_in, R0_vec], dim=-1)
    S_t = odeint(ode_deriv, S0, t_span, method='dopri5', rtol=1e-5, atol=1e-6)
    S_t = torch.clamp(S_t, min=0.0)
    N_final = torch.clamp(S_t[-1, ..., :48] * mask, min=0.0)
    # Next dilution inoculum
    N_next_in = torch.clamp(N_final / 30.0 + 1e-5 * mask, min=0.0)
    return N_final, N_next_in

def simulate_passages(N0, R0_vec, mask, num_passages=150):
    N_traj = [N0.cpu().numpy()]
    N_curr = N0.clone()
    for p in range(num_passages):
        N_in = torch.clamp(N_curr / 30.0 + 1e-5 * mask, min=0.0)
        N_final, _ = one_passage_step(N_in, R0_vec, mask)
        N_traj.append(N_final.cpu().numpy())
        N_curr = N_final
    return np.array(N_traj) # (num_passages + 1, 48)

# 1. Simulate S=48 Community 1 under LN, MN, HN
mask_48 = data['presence_masks'][(48, 'Community1')].float().to(device)
rel_48 = data['relative_abundance'][('HN', 48, 'Community1')].to(device)
od_48 = data['biomass_od'][('HN', 48, 'Community1')].mean(dim=0).to(device)
N0_48 = (rel_48[0] * od_48[0]) * mask_48

print("\n--- 1. Long-Term 150-Passage Autonomous Trajectory Simulation (S=48) ---")
traj_ln = simulate_passages(N0_48, R0_ln, mask_48, num_passages=150)
traj_mn = simulate_passages(N0_48, R0_mn, mask_48, num_passages=150)
traj_hn = simulate_passages(N0_48, R0_hn, mask_48, num_passages=150)

# Check asymptotic variance in passages 100..150
std_late_ln = np.std(traj_ln[100:], axis=0).max()
std_late_mn = np.std(traj_mn[100:], axis=0).max()
std_late_hn = np.std(traj_hn[100:], axis=0).max()

p2p_late_ln = (traj_ln[100:].max(axis=0) - traj_ln[100:].min(axis=0)).max()
p2p_late_mn = (traj_mn[100:].max(axis=0) - traj_mn[100:].min(axis=0)).max()
p2p_late_hn = (traj_hn[100:].max(axis=0) - traj_hn[100:].min(axis=0)).max()

print(f"Low Nutrients (LN)  : Late Max Std = {std_late_ln:.6f} | Peak-to-Peak Amplitude = {p2p_late_ln:.6f}")
print(f"Medium Nutrients(MN): Late Max Std = {std_late_mn:.6f} | Peak-to-Peak Amplitude = {p2p_late_mn:.6f}")
print(f"High Nutrients (HN) : Late Max Std = {std_late_hn:.6f} | Peak-to-Peak Amplitude = {p2p_late_hn:.6f}")

# 2. Maximal Lyapunov Exponent via Benettin Algorithm
def compute_lyapunov(N0, R0_vec, mask, num_passages=80, burn_in=40, eps=1e-6):
    """
    Computes largest Lyapunov exponent lambda_max per serial dilution passage.
    """
    # Burn-in onto attractor
    N_curr = N0.clone()
    for _ in range(burn_in):
        N_in = torch.clamp(N_curr / 30.0 + 1e-5 * mask, min=0.0)
        N_curr, _ = one_passage_step(N_in, R0_vec, mask)
        
    # Orthogonal initial perturbation on active species
    active_mask = (mask > 0).float()
    torch.manual_seed(123)
    p_rand = torch.randn_like(N_curr) * active_mask
    delta = (p_rand / torch.norm(p_rand)) * eps
    
    lyap_sum = 0.0
    expansions = []
    
    for p in range(num_passages):
        N_in_base = torch.clamp(N_curr / 30.0 + 1e-5 * mask, min=0.0)
        N_in_pert = torch.clamp((N_curr + delta) / 30.0 + 1e-5 * mask, min=0.0)
        
        N_next_base, _ = one_passage_step(N_in_base, R0_vec, mask)
        N_next_pert, _ = one_passage_step(N_in_pert, R0_vec, mask)
        
        diff = (N_next_pert - N_next_base) * active_mask
        d1 = torch.norm(diff).item()
        
        expansion = d1 / eps
        expansions.append(expansion)
        lyap_sum += np.log(max(expansion, 1e-12))
        
        # Re-scale perturbation back to eps along the divergence direction
        delta = (diff / (torch.norm(diff) + 1e-12)) * eps
        N_curr = N_next_base
        
    lambda_max = lyap_sum / float(num_passages)
    return lambda_max, expansions

print("\n--- 2. Maximal Lyapunov Exponent (lambda_max per passage) ---")
lam_ln, _ = compute_lyapunov(N0_48, R0_ln, mask_48, num_passages=60, burn_in=40)
lam_mn, _ = compute_lyapunov(N0_48, R0_mn, mask_48, num_passages=60, burn_in=40)
lam_hn, _ = compute_lyapunov(N0_48, R0_hn, mask_48, num_passages=60, burn_in=40)

print(f"Low Nutrients (LN)  : lambda_max = {lam_ln:+.4f} (Negative -> Point Attractor / Damped)")
print(f"Medium Nutrients(MN): lambda_max = {lam_mn:+.4f}")
print(f"High Nutrients (HN) : lambda_max = {lam_hn:+.4f} ({'Positive -> CHAOTIC ATTRACTOR' if lam_hn > 0 else 'Non-positive'})")

# 3. Trajectory Divergence Test (Butterfly Effect)
print("\n--- 3. Trajectory Divergence (Sensitivity to Initial Conditions) ---")
def trajectory_divergence(N0, R0_vec, mask, eps=1e-6, num_passages=40):
    active_mask = (mask > 0).float()
    torch.manual_seed(999)
    p_rand = torch.randn_like(N0) * active_mask
    delta0 = (p_rand / torch.norm(p_rand)) * eps
    
    N1 = N0.clone()
    N2 = torch.clamp(N0 + delta0, min=0.0)
    
    divergences = [torch.norm((N2 - N1) * active_mask).item()]
    
    for p in range(num_passages):
        N1_in = torch.clamp(N1 / 30.0 + 1e-5 * mask, min=0.0)
        N2_in = torch.clamp(N2 / 30.0 + 1e-5 * mask, min=0.0)
        
        N1_next, _ = one_passage_step(N1_in, R0_vec, mask)
        N2_next, _ = one_passage_step(N2_in, R0_vec, mask)
        
        diff_norm = torch.norm((N2_next - N1_next) * active_mask).item()
        divergences.append(diff_norm)
        
        N1 = N1_next
        N2 = N2_next
        
    return np.array(divergences)

div_ln = trajectory_divergence(N0_48, R0_ln, mask_48, eps=1e-6, num_passages=30)
div_hn = trajectory_divergence(N0_48, R0_hn, mask_48, eps=1e-6, num_passages=30)

print(f"LN Divergence: Start = {div_ln[0]:.2e} -> Pass 5 = {div_ln[5]:.2e} -> Pass 15 = {div_ln[15]:.2e} -> Pass 30 = {div_ln[30]:.2e}")
print(f"HN Divergence: Start = {div_hn[0]:.2e} -> Pass 5 = {div_hn[5]:.2e} -> Pass 15 = {div_hn[15]:.2e} -> Pass 30 = {div_hn[30]:.2e}")

# 4. Richness-Dependent Bifurcation Analysis (S in {3, 6, 12, 24, 48})
print("\n--- 4. Emergent Phase Transition vs Pool Size S in High Nutrients ---")
richness_levels = [3, 6, 12, 24, 48]
s_results = {}

for S in richness_levels:
    comm_name = 'Community1'
    mask_s = data['presence_masks'][(S, comm_name)].float().to(device)
    rel_s = data['relative_abundance'][('HN', S, comm_name)].to(device)
    od_s = data['biomass_od'][('HN', S, comm_name)].mean(dim=0).to(device)
    N0_s = (rel_s[0] * od_s[0]) * mask_s
    
    # Run 100 passages
    traj_s = simulate_passages(N0_s, R0_hn, mask_s, num_passages=100)
    std_s = np.std(traj_s[50:], axis=0)[mask_s.cpu().numpy().astype(bool)].max()
    p2p_s = (traj_s[50:].max(axis=0) - traj_s[50:].min(axis=0))[mask_s.cpu().numpy().astype(bool)].max()
    lam_s, _ = compute_lyapunov(N0_s, R0_hn, mask_s, num_passages=50, burn_in=30)
    
    s_results[S] = {'std': std_s, 'p2p': p2p_s, 'lambda': lam_s}
    status = "CHAOS" if lam_s > 0.05 and p2p_s > 0.5 else ("OSCILLATING" if p2p_s > 0.1 else "STABLE FIXED POINT")
    print(f"S={S:2d} (HN): Late Std = {std_s:.4f} | P2P = {p2p_s:.4f} | lambda_max = {lam_s:+.4f} | Regime: {status}")

# 5. Fourier Power Spectral Density (FFT) on late passages for dominant species in HN
print("\n--- 5. Power Spectral Density (PSD) Analysis ---")
dominant_sp = np.argsort(traj_hn[50:].mean(axis=0))[-1]
signal = traj_hn[50:, dominant_sp]
signal_detrend = signal - np.mean(signal)
fft_vals = np.abs(np.fft.rfft(signal_detrend)) ** 2
freqs = np.fft.rfftfreq(len(signal_detrend), d=1.0) # cycles per passage

# Spectral entropy: measure of broadband complexity (high = chaos, 0 = pure sine)
psd_norm = fft_vals / (np.sum(fft_vals) + 1e-12)
spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12)) / np.log(len(psd_norm))
print(f"Dominant species {dominant_sp} in HN (Passages 50..150): Spectral Entropy = {spectral_entropy:.4f} (0=pure periodic, 1=white noise)")
top_freqs = freqs[np.argsort(fft_vals)[-3:]]
print(f"Top 3 spectral peak frequencies: {top_freqs.round(3)} cycles/passage")

print("\nDiagnostic complete.")

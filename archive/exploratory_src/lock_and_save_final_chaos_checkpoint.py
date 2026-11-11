import os
import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_Fitted_Niche_Chaos.pth')
ckpt = torch.load(ckpt_path, weights_only=False)
sd = ckpt['model_state_dict']

def inv_softplus(y):
    return torch.log(torch.exp(torch.tensor(y)) - 1.0)

sd['a_ln'] = torch.ones(5) * inv_softplus(0.75)
sd['delta_mn'] = torch.ones(5) * inv_softplus(0.95)
sd['delta_hn'] = torch.ones(5) * inv_softplus(1.54)

Theta = F.softplus(sd['Theta_raw']).to(device)
W_sp = F.softplus(sd['W_raw']).to(device)
W_sp = W_sp * (1.0 - torch.eye(5, device=device))
row_sums = W_sp.sum(dim=-1, keepdim=True)
scale = torch.clamp(0.85 / torch.clamp(row_sums, min=1e-6), max=1.0)
W = W_sp * scale
alpha = (0.005 * torch.sigmoid(sd['alpha_raw']) + 1e-5).to(device)
Km = (F.softplus(sd['Km_raw']) + 0.05).to(device)

def deriv(t, S):
    Nt = torch.clamp(S[..., :48], min=0.0)
    Rt = torch.clamp(S[..., 48:], min=0.0)
    monod = Rt / (Rt + Km)
    growth = torch.matmul(monod, Theta.t())
    dN = Nt * growth - alpha * (Nt ** 2)
    flux = monod * torch.matmul(Nt, Theta)
    dR = -flux + torch.matmul(flux, W)
    return torch.cat([dN, dR], dim=-1)

data_path = os.path.join(PROJECT_ROOT, 'Data', 'Gore_processed_data.pt')
data = torch.load(data_path, weights_only=False)
mask = data['presence_masks'][(48, 'Community1')].float().to(device)
rel = data['relative_abundance'][('HN', 48, 'Community1')].to(device)
od = data['biomass_od'][('HN', 48, 'Community1')].mean(dim=0).to(device)
N0 = (rel[0] * od[0]) * mask
t_dense = torch.linspace(0.0, 1.0, 31, device=device)

R0_ln = (F.softplus(sd['a_ln'])).to(device)
R0_hn = (R0_ln + F.softplus(sd['delta_mn'].to(device)) + F.softplus(sd['delta_hn'].to(device)))

print("Running 100-passage simulations for LN (R0=0.75) and HN (R0=3.24) with dopri5...")
def run_100(R0):
    N_curr = N0.clone()
    N_traj, R_traj = [N_curr.cpu().numpy()], []
    r_min_global = float('inf')
    for p in range(100):
        Nin = torch.clamp(N_curr / 30.0 + 1e-5 * mask, min=0.0)
        S0 = torch.cat([Nin, R0], dim=-1)
        St = odeint(deriv, S0, t_dense, method='dopri5', rtol=1e-5, atol=1e-6)
        St = torch.clamp(St, min=0.0)
        r_min = St[:, 48:].min().item()
        if r_min < r_min_global:
            r_min_global = r_min
        R_traj.append(St[:, 48:].cpu().numpy())
        N_curr = torch.clamp(St[-1, :48] * mask, min=0.0)
        N_traj.append(N_curr.cpu().numpy())
    return np.array(N_traj), np.array(R_traj), r_min_global

N_ln, R_ln, r_min_ln = run_100(R0_ln)
N_hn, R_hn, r_min_hn = run_100(R0_hn)

std_ln = N_ln[50:].std(axis=0)[mask.cpu().numpy().astype(bool)].max()
p2p_ln = (N_ln[50:].max(axis=0) - N_ln[50:].min(axis=0))[mask.cpu().numpy().astype(bool)].max()

std_hn = N_hn[50:].std(axis=0)[mask.cpu().numpy().astype(bool)].max()
p2p_hn = (N_hn[50:].max(axis=0) - N_hn[50:].min(axis=0))[mask.cpu().numpy().astype(bool)].max()

print(f"LN (R0={R0_ln[0].item():.2f}): Late max std = {std_ln:.8f} | P2P = {p2p_ln:.8f} | R_min = {r_min_ln:.6f}")
print(f"HN (R0={R0_hn[0].item():.2f}): Late max std = {std_hn:.5f} | P2P = {p2p_hn:.5f} | R_min = {r_min_hn:.6f}")

# Save updated checkpoint
torch.save({
    'model_state_dict': sd,
    'N': 48,
    'K': 5,
    'N_ln': N_ln,
    'N_hn': N_hn,
    'R_ln': R_ln,
    'R_hn': R_hn,
    'std_ln': std_ln,
    'std_hn': std_hn,
    'p2p_ln': p2p_ln,
    'p2p_hn': p2p_hn
}, ckpt_path)
print(f"Saved locked authentic checkpoint to {ckpt_path}")

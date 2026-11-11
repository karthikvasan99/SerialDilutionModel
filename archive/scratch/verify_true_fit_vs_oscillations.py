import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = torch.load('Data/Gore_processed_data.pt', weights_only=False)
traj_rel = data['relative_abundance'][('HN', 6, 'Community5')]
od = data['biomass_od'][('HN', 6, 'Community5')].mean(dim=0)
bio = traj_rel * od.unsqueeze(-1)
species_idx = [18, 21, 22]
emp_N = bio[:, species_idx].to(device)

# What if we train with pure MSE loss (no skew regularizer) to get the best possible true fit?
class PureFitModel(nn.Module):
    def __init__(self, D=30.0, Km=0.05):
        super().__init__()
        self.D, self.Km = D, Km
        self.alpha = 0.0
        self.Theta_raw = nn.Parameter(torch.randn(3, 3) * 0.2 + 1.5)
        self.Phi_raw = nn.Parameter(torch.randn(3, 3) * 0.2 + 1.5)
        self.delta_raw = nn.Parameter(torch.zeros(3) - 1.0)
        W_fixed = torch.tensor([[0.0, 0.99, 0.0], [0.0, 0.0, 0.99], [0.99, 0.0, 0.0]], device=device)
        self.register_buffer('W', W_fixed)
        # Constrain R0 to actual empirical OD (~0.7)
        self.scale = nn.Parameter(torch.tensor(0.7))
        self.dist = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))

    def get_R0(self):
        return F.softmax(self.dist, dim=-1) * F.softplus(self.scale)
    def get_Theta(self): return F.softplus(self.Theta_raw)
    def get_Phi(self): return F.softplus(self.Phi_raw)
    def get_delta(self): return F.softplus(self.delta_raw)

    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[:3], min=0.0)
        R_t = torch.clamp(S[3:], min=0.0)
        Theta = self.get_Theta()
        Phi = self.get_Phi()
        delta = self.get_delta()
        monod = R_t / (R_t + self.Km)
        growth = torch.matmul(monod, Theta.t())
        death = delta * (0.02 / (R_t.sum() + 0.02))
        dN = N_t * (growth - death)
        flux = monod * torch.matmul(N_t, Phi)
        dR = - flux + torch.matmul(flux, self.W)
        return torch.cat([dN, dR])

torch.manual_seed(42)
model = PureFitModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.02)
t_span = torch.linspace(0.0, 1.0, 25, device=device)

for ep in range(1, 51):
    model.train()
    opt.zero_grad()
    R0 = model.get_R0()
    loss = 0.0
    for p in range(1, 11):
        N_in = emp_N[p-1] / model.D + 1e-4
        S0 = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
        loss += F.mse_loss(torch.clamp(S_t[-1, :3], min=0.0), emp_N[p])
    loss = loss / 10.0
    loss.backward()
    opt.step()

# Compute true R^2 with DOPRI5
model.eval()
with torch.no_grad():
    R0 = model.get_R0()
    preds = []
    for p in range(1, 11):
        N_in = emp_N[p-1] / model.D + 1e-4
        S0 = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S0, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        preds.append(S_t[-1, :3].cpu().numpy())
    preds = np.array(preds)
    
y_true = emp_N[1:].cpu().numpy()
ss_res = np.sum((y_true - preds)**2)
ss_tot = np.sum((y_true - np.mean(y_true))**2)
r2 = 1.0 - ss_res / ss_tot
print(f"When optimizing purely for data fit: R^2 = {r2:.4f} ({r2*100:.1f}%), MSE = {ss_res/30:.5f}")

# Long term rollout
with torch.no_grad():
    N_curr = emp_N[0]
    S = torch.cat([N_curr, R0])
    long_traj = []
    for p in range(60):
        N_in = S[:3] / model.D + 1e-4
        S_start = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        S = torch.clamp(S_t[-1], min=0.0)
        long_traj.append(S[:3].cpu().numpy())
    long_traj = np.array(long_traj)

diffs = np.diff(long_traj[40:], axis=0)
flips = np.sum(diffs[:-1] * diffs[1:] < -1e-6, axis=0)
p2p = (long_traj[40:].max(axis=0) - long_traj[40:].min(axis=0)) / np.maximum(long_traj[40:].mean(axis=0), 1e-4)
print(f"Long term DOPRI5 dynamics: P2P = {p2p*100}%, Reversals = {flips}")
if np.all(flips == 0) or np.all(p2p < 0.02):
    print(">>> Result: Collapses to a static fixed point! <<<")


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

class Comm5Oscillator(nn.Module):
    def __init__(self, D=30.0, Km=0.05):
        super().__init__()
        self.D, self.Km = D, Km
        self.alpha = 0.0
        self.Theta_raw = nn.Parameter(torch.eye(3) * 5.0 + torch.randn(3, 3) * 0.2)
        Phi_init = torch.tensor([
            [1.0, 3.0, 0.1],
            [0.1, 1.0, 3.0],
            [3.0, 0.1, 1.0]
        ])
        self.Phi_raw = nn.Parameter(torch.log(torch.exp(Phi_init) - 1.0) + torch.randn(3, 3) * 0.1)
        self.delta_raw = nn.Parameter(torch.zeros(3) + 0.5)
        W_fixed = torch.tensor([
            [0.0, 0.99, 0.0],
            [0.0, 0.0, 0.99],
            [0.99, 0.0, 0.0]
        ], device=device)
        self.register_buffer('W', W_fixed)
        self.R0_raw = nn.Parameter(torch.tensor([1.5, 0.1, 0.1]))

    def get_R0(self): return F.softplus(self.R0_raw)
    def get_Theta(self): return F.softplus(self.Theta_raw)
    def get_Phi(self): return F.softplus(self.Phi_raw)
    def get_delta(self): return F.softplus(self.delta_raw) * 2.0

    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[:3], min=0.0)
        R_t = torch.clamp(S[3:], min=0.0)
        Theta = self.get_Theta()
        Phi = self.get_Phi()
        delta = self.get_delta()
        
        monod = R_t / (R_t + self.Km)
        growth = torch.matmul(monod, Theta.t())
        R_tot = R_t.sum()
        starv = 0.02 / (R_tot + 0.02)
        death = delta * starv
        dN = N_t * (growth - death)
        
        flux = monod * torch.matmul(N_t, Phi)
        byproduct = torch.matmul(flux, self.W)
        dR = - flux + byproduct
        return torch.cat([dN, dR])

torch.manual_seed(42)
model = Comm5Oscillator().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.015)
t_span = torch.linspace(0.0, 1.0, 25, device=device)

# Fit exactly as before
for ep in range(1, 41):
    model.train()
    opt.zero_grad()
    R0 = model.get_R0()
    loss_fit = 0.0
    for p in range(1, 11):
        N_in = emp_N[p-1] / model.D + 1e-4
        S0 = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
        loss_fit += F.mse_loss(torch.clamp(S_t[-1, :3], min=0.0), emp_N[p])
    loss_fit = loss_fit / 10.0
    A = torch.matmul(model.get_Theta() * R0.unsqueeze(0), model.get_Phi().t())
    skew_ratio = torch.norm(A - A.t()) / torch.clamp(torch.norm(A + A.t()), min=1e-6)
    (loss_fit - 0.5 * skew_ratio).backward()
    opt.step()

# Now compute 1-step predictions using DOPRI5
model.eval()
with torch.no_grad():
    R0 = model.get_R0()
    one_step_preds = []
    for p in range(1, 11):
        N_in = emp_N[p-1] / model.D + 1e-4
        S0 = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S0, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        one_step_preds.append(S_t[-1, :3].cpu().numpy())
    one_step_preds = np.array(one_step_preds) # (10, 3)

y_true = emp_N[1:].cpu().numpy() # (10, 3)
y_pred = one_step_preds          # (10, 3)

print("Passage-by-Passage Comparison:")
for p in range(10):
    print(f"Day {p+1:2d} | True: [sp18={y_true[p,0]:.3f}, sp21={y_true[p,1]:.3f}, sp22={y_true[p,2]:.3f}] | Pred: [sp18={y_pred[p,0]:.3f}, sp21={y_pred[p,1]:.3f}, sp22={y_pred[p,2]:.3f}]")

ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_global = 1.0 - ss_res / ss_tot

print(f"\nSS_RES = {ss_res:.4f}")
print(f"SS_TOT = {ss_tot:.4f}")
print(f"Global R^2 = {r2_global:.4f} ({r2_global*100:.2f}%)")

# Per-species R^2
r2_sp = 1.0 - np.sum((y_true - y_pred)**2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
print(f"Species 18 R^2: {r2_sp[0]:.4f}")
print(f"Species 21 R^2: {r2_sp[1]:.4f}")
print(f"Species 22 R^2: {r2_sp[2]:.4f}")


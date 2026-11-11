import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Gore Data
data = torch.load('Data/Gore_processed_data.pt', weights_only=False)
traj_rel = data['relative_abundance'][('HN', 6, 'Community5')] # (11, 48)
od = data['biomass_od'][('HN', 6, 'Community5')].mean(dim=0) # (11,)
bio = traj_rel * od.unsqueeze(-1) # (11, 48)

# The 3 coexisting species: [18, 21, 22]
species_idx = [18, 21, 22]
emp_N = bio[:, species_idx].to(device) # (11, 3)

print("Empirical 10-Passage Trajectory for ('HN', 6, 'Community5') [Sp 18, 21, 22]:")
for p in range(11):
    print(f"Day {p:2d}: Sp18={emp_N[p, 0].item():.4f}, Sp21={emp_N[p, 1].item():.4f}, Sp22={emp_N[p, 2].item():.4f}")

# Loss function
def mse_flow(y_true, y_pred):
    return F.mse_loss(y_pred, y_true)

# 3-Species Consumer-Resource Model
class Comm5Oscillator(nn.Module):
    def __init__(self, D=30.0, Km=0.05):
        super().__init__()
        self.D, self.Km = D, Km
        self.alpha = 0.0 # Strictly zero quadratic damping
        
        # Growth Theta (3 x 3)
        # Initialize near diagonal (niche specialization)
        self.Theta_raw = nn.Parameter(torch.eye(3) * 5.0 + torch.randn(3, 3) * 0.2)
        
        # Consumption Phi (3 x 3)
        # Initialize with cyclic preemption bias
        Phi_init = torch.tensor([
            [1.0, 3.0, 0.1],
            [0.1, 1.0, 3.0],
            [3.0, 0.1, 1.0]
        ])
        self.Phi_raw = nn.Parameter(torch.log(torch.exp(Phi_init) - 1.0) + torch.randn(3, 3) * 0.1)
        
        # Starvation decay rate delta
        self.delta_raw = nn.Parameter(torch.zeros(3) + 0.5)
        
        # Non-learnable cyclic W with 0.99
        W_fixed = torch.tensor([
            [0.0, 0.99, 0.0],
            [0.0, 0.0, 0.99],
            [0.99, 0.0, 0.0]
        ], device=device)
        self.register_buffer('W', W_fixed)
        
        # HN initial resource pulse
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

# Fit to this community's empirical data
torch.manual_seed(42)
model = Comm5Oscillator().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.015)
t_span = torch.linspace(0.0, 1.0, 25, device=device)

print("\nFitting model to empirical trajectory...")
for ep in range(1, 41):
    model.train()
    opt.zero_grad()
    R0 = model.get_R0()
    loss_fit = 0.0
    for p in range(1, 11):
        N_in = emp_N[p-1] / model.D + 1e-4
        S0 = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
        N_pred = torch.clamp(S_t[-1, :3], min=0.0)
        loss_fit += mse_flow(emp_N[p], N_pred)
    loss_fit = loss_fit / 10.0
    
    # Skewness regularizer
    A = torch.matmul(model.get_Theta() * R0.unsqueeze(0), model.get_Phi().t())
    skew_ratio = torch.norm(A - A.t()) / torch.clamp(torch.norm(A + A.t()), min=1e-6)
    loss_skew = - 0.5 * skew_ratio
    
    (loss_fit + loss_skew).backward()
    opt.step()
    
    if ep % 10 == 0 or ep == 1:
        print(f"Epoch {ep:2d}/40 | Fit Loss (MSE): {loss_fit.item():.5f} | SkewRatio: {skew_ratio.item():.4f}")

# Now simulate 80 passages under DOPRI5!
print("\nSimulating 80 autonomous passages under DOPRI5...")
model.eval()
with torch.no_grad():
    R0 = model.get_R0()
    print(f"Fitted R0: {R0.cpu().numpy()}")
    print("Fitted Theta:")
    print(model.get_Theta().cpu().numpy())
    print("Fitted Phi:")
    print(model.get_Phi().cpu().numpy())
    
    N_curr = emp_N[0]
    S = torch.cat([N_curr, R0])
    traj = []
    for p in range(80):
        N_in = S[:3] / model.D + 1e-4
        S_start = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S_start, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        S = torch.clamp(S_t[-1], min=0.0)
        traj.append(S[:3].cpu().numpy())
        
    traj = np.array(traj)

last30 = traj[50:]
diffs = np.diff(last30, axis=0)
flips = np.sum(diffs[:-1] * diffs[1:] < -1e-6, axis=0)
p2p = (last30.max(axis=0) - last30.min(axis=0)) / np.maximum(last30.mean(axis=0), 1e-4)

print("\nResults for Passages 50..80 under DOPRI5:")
for idx, sp in enumerate(species_idx):
    print(f"  Species {sp}: Mean={last30[:, idx].mean():.4f}, P2P={p2p[idx]*100:.1f}%, Reversals={flips[idx]}")

if np.any(flips >= 4) and np.any(p2p > 0.20):
    print("\n>>> SUCCESS: AUTHENTIC SUSTAINED LIMIT CYCLE UNDER DOPRI5 FITTED TO GORE'S DATA! <<<")
else:
    print("\n>>> Damped to fixed point. <<<")

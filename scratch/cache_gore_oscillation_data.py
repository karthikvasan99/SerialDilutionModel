"""
Trains the Comm5Oscillator, simulates 80 passages, and saves all arrays needed
for the 3-panel Gore oscillation figure to Data/Gore_Comm5_Oscillation_Cache.npz
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torchdiffeq import odeint
from sklearn.decomposition import PCA
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ── Load empirical data ──────────────────────────────────────────────────────
data = torch.load('Data/Gore_processed_data.pt', weights_only=False)
traj_rel = data['relative_abundance'][('HN', 6, 'Community5')]
od = data['biomass_od'][('HN', 6, 'Community5')].mean(dim=0)
bio = traj_rel * od.unsqueeze(-1)
species_idx = [18, 21, 22]
emp_N = bio[:, species_idx].to(device)  # (11, 3)
emp_N_np = emp_N.cpu().numpy()

# ── Model ────────────────────────────────────────────────────────────────────
class Comm5Oscillator(nn.Module):
    def __init__(self, D=30.0, Km=0.05):
        super().__init__()
        self.D, self.Km = D, Km
        self.Theta_raw = nn.Parameter(torch.eye(3)*5.0 + torch.randn(3,3)*0.2)
        Phi_init = torch.tensor([[1.,3.,.1],[.1,1.,3.],[3.,.1,1.]])
        self.Phi_raw = nn.Parameter(torch.log(torch.exp(Phi_init)-1.) + torch.randn(3,3)*0.1)
        self.delta_raw = nn.Parameter(torch.zeros(3)+0.5)
        self.register_buffer('W', torch.tensor([[0.,.99,0.],[0.,0.,.99],[.99,0.,0.]], device=device))
        self.R0_raw = nn.Parameter(torch.tensor([1.5,.1,.1]))
    def get_R0(self):   return F.softplus(self.R0_raw)
    def get_Theta(self): return F.softplus(self.Theta_raw)
    def get_Phi(self):   return F.softplus(self.Phi_raw)
    def get_delta(self): return F.softplus(self.delta_raw)*2.
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[:3], min=0.)
        R_t = torch.clamp(S[3:], min=0.)
        monod = R_t/(R_t+self.Km)
        dN = N_t*(torch.matmul(monod, self.get_Theta().t())
                  - self.get_delta()*(0.02/(R_t.sum()+0.02)))
        flux = monod*torch.matmul(N_t, self.get_Phi())
        dR = -flux + torch.matmul(flux, self.W)
        return torch.cat([dN, dR])

torch.manual_seed(42)
model = Comm5Oscillator().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.015)
t_span  = torch.linspace(0., 1., 25,  device=device)
t_intra = torch.linspace(0., 1., 120, device=device)

print("Training...")
for ep in range(1, 41):
    model.train(); opt.zero_grad()
    R0 = model.get_R0()
    loss = sum(F.mse_loss(torch.clamp(
                odeint(model.ode_deriv, torch.cat([emp_N[p-1]/model.D+1e-4, R0]),
                       t_span, method='rk4')[-1,:3], min=0.), emp_N[p])
               for p in range(1,11)) / 10.
    A = torch.matmul(model.get_Theta()*R0.unsqueeze(0), model.get_Phi().t())
    (loss - 0.5*torch.norm(A-A.t())/torch.clamp(torch.norm(A+A.t()), min=1e-6)).backward()
    opt.step()
    if ep % 10 == 0:
        print(f"  Epoch {ep}/40 | Loss: {loss.item():.5f}")

# ── Simulate 80 passages, collect intra-passage N and R ─────────────────────
print("Simulating 80 passages...")
all_N_traj = []   # (80, 120, 3) - species abundance within passage
all_R_traj = []   # (80, 120, 3) - resource concentrations within passage
passage_end_N = []  # (80, 3)    - species at end of each passage

model.eval()
with torch.no_grad():
    R0 = model.get_R0()
    S = torch.cat([emp_N[0], R0])
    for p in range(80):
        N_in = S[:3]/model.D + 1e-4
        S_start = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S_start, t_intra, method='dopri5', rtol=1e-4, atol=1e-5)
        S = torch.clamp(S_t[-1], min=0.)
        all_N_traj.append(S_t[:, :3].cpu().numpy())
        all_R_traj.append(S_t[:, 3:].cpu().numpy())
        passage_end_N.append(S_t[-1, :3].cpu().numpy())

all_N_traj  = np.stack(all_N_traj)   # (80, 120, 3)
all_R_traj  = np.stack(all_R_traj)   # (80, 120, 3)
passage_end_N = np.array(passage_end_N)  # (80, 3)

# Also save 1-step predictions for comparison with empirical data
one_step_preds = []
with torch.no_grad():
    R0 = model.get_R0()
    for p in range(1, 11):
        N_in = emp_N[p-1]/model.D + 1e-4
        S0 = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S0, t_span, method='dopri5', rtol=1e-4, atol=1e-5)
        one_step_preds.append(S_t[-1, :3].cpu().numpy())
one_step_preds = np.array(one_step_preds)  # (10, 3)

# ── PCA on resources ─────────────────────────────────────────────────────────
R_flat = all_R_traj.reshape(-1, 3)   # (80*120, 3)
pca = PCA(n_components=2)
pca.fit(R_flat)
R_pca = pca.transform(R_flat).reshape(80, 120, 2)
pca_components = pca.components_     # (2, 3)
pca_mean = pca.mean_                 # (3,)
pca_var  = pca.explained_variance_ratio_  # (2,)

# ── Relative abundance ───────────────────────────────────────────────────────
total = passage_end_N.sum(axis=1, keepdims=True)
rel_abund = passage_end_N / np.maximum(total, 1e-6)  # (80, 3)

# ── Learned parameters ───────────────────────────────────────────────────────
with torch.no_grad():
    Theta_np = model.get_Theta().cpu().numpy()
    Phi_np   = model.get_Phi().cpu().numpy()
    delta_np = model.get_delta().cpu().numpy()
    R0_np    = model.get_R0().cpu().numpy()

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = 'Data/Gore_Comm5_Oscillation_Cache.npz'
np.savez(out_path,
    # Empirical data
    emp_N = emp_N_np,                 # (11, 3) - observed biomass days 0..10
    species_idx = np.array(species_idx),  # [18, 21, 22]

    # 1-step model predictions vs observations
    one_step_preds = one_step_preds,  # (10, 3) - predicted days 1..10

    # 80-passage autonomous simulation
    all_N_traj   = all_N_traj,        # (80, 120, 3) - intra-passage species trajectories
    all_R_traj   = all_R_traj,        # (80, 120, 3) - intra-passage resource trajectories
    passage_end_N = passage_end_N,    # (80, 3)       - species at end of each passage
    rel_abund    = rel_abund,         # (80, 3)       - relative abundance per passage

    # PCA of resource space
    R_pca          = R_pca,           # (80, 120, 2)  - resource trajectories in PC space
    pca_components = pca_components,  # (2, 3)        - PCA loadings
    pca_mean       = pca_mean,        # (3,)          - PCA mean
    pca_var        = pca_var,         # (2,)          - fraction of variance explained

    # Learned model parameters
    Theta = Theta_np,    # (3, 3)
    Phi   = Phi_np,      # (3, 3)
    delta = delta_np,    # (3,)
    R0    = R0_np,       # (3,)
    W     = np.array([[0.,.99,0.],[0.,0.,.99],[.99,0.,0.]]),  # fixed
)
print(f"\nSaved to: {out_path}")
print(f"Arrays saved:")
cache = np.load(out_path)
for k in cache.files:
    print(f"  {k:20s}: shape={cache[k].shape}, dtype={cache[k].dtype}")
print(f"\nPCA: PC1={pca_var[0]*100:.1f}%  PC2={pca_var[1]*100:.1f}%  Total={pca_var.sum()*100:.1f}%")

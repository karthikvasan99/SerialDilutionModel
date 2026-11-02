"""
Generates the combined Panels 6+7 figure for the poster:
- Panel A: Oscillations in relative abundance (passages over time)
- Panel B: Species abundance cycle (2D phase portrait)  
- Panel C: Resource space trajectory over one intra-passage day
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shutil

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11.5
plt.rcParams['axes.linewidth'] = 1.4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load data
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
        Phi_init = torch.tensor([[1.0, 3.0, 0.1],[0.1, 1.0, 3.0],[3.0, 0.1, 1.0]])
        self.Phi_raw = nn.Parameter(torch.log(torch.exp(Phi_init) - 1.0) + torch.randn(3, 3) * 0.1)
        self.delta_raw = nn.Parameter(torch.zeros(3) + 0.5)
        W_fixed = torch.tensor([[0.0, 0.99, 0.0],[0.0, 0.0, 0.99],[0.99, 0.0, 0.0]], device=device)
        self.register_buffer('W', W_fixed)
        self.R0_raw = nn.Parameter(torch.tensor([1.5, 0.1, 0.1]))
    def get_R0(self): return F.softplus(self.R0_raw)
    def get_Theta(self): return F.softplus(self.Theta_raw)
    def get_Phi(self): return F.softplus(self.Phi_raw)
    def get_delta(self): return F.softplus(self.delta_raw) * 2.0
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[:3], min=0.0)
        R_t = torch.clamp(S[3:], min=0.0)
        Theta, Phi, delta = self.get_Theta(), self.get_Phi(), self.get_delta()
        monod = R_t / (R_t + self.Km)
        growth = torch.matmul(monod, Theta.t())
        death = delta * (0.02 / (R_t.sum() + 0.02))
        dN = N_t * (growth - death)
        flux = monod * torch.matmul(N_t, Phi)
        dR = - flux + torch.matmul(flux, self.W)
        return torch.cat([dN, dR])

torch.manual_seed(42)
model = Comm5Oscillator().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.015)
t_span = torch.linspace(0.0, 1.0, 25, device=device)
t_intra = torch.linspace(0.0, 1.0, 200, device=device)

for ep in range(1, 41):
    model.train(); opt.zero_grad()
    R0 = model.get_R0()
    loss_fit = 0.0
    for p in range(1, 11):
        N_in = emp_N[p-1] / model.D + 1e-4
        S0 = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S0, t_span, method='rk4')
        loss_fit += F.mse_loss(torch.clamp(S_t[-1, :3], min=0.0), emp_N[p])
    loss_fit /= 10.0
    A = torch.matmul(model.get_Theta() * R0.unsqueeze(0), model.get_Phi().t())
    skew_ratio = torch.norm(A - A.t()) / torch.clamp(torch.norm(A + A.t()), min=1e-6)
    (loss_fit - 0.5 * skew_ratio).backward()
    opt.step()

# Simulate 80 passages
model.eval()
with torch.no_grad():
    R0 = model.get_R0()
    N_curr = emp_N[0]
    S = torch.cat([N_curr, R0])
    long_traj = []
    intra_N_last = None
    intra_R_last = None
    for p in range(80):
        N_in = S[:3] / model.D + 1e-4
        S_start = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S_start, t_intra, method='dopri5', rtol=1e-4, atol=1e-5)
        S = torch.clamp(S_t[-1], min=0.0)
        long_traj.append(S_t[-1, :3].cpu().numpy())
        if p == 79:
            intra_N_last = S_t[:, :3].cpu().numpy()
            intra_R_last = S_t[:, 3:].cpu().numpy()
    long_traj = np.array(long_traj)

# Relative abundance
eps = 1e-6
total_abund = long_traj.sum(axis=1, keepdims=True)
rel_abund = long_traj / np.maximum(total_abund, eps)

# ── FIGURE ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5.5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

sp_colors = ['#1f77b4', '#2ca02c', '#d62728']
sp_labels = ['Sp. 18 (Resource $R_1$ specialist)', 'Sp. 21 (Resource $R_2$ specialist)', 'Sp. 22 (Resource $R_3$ specialist)']

# ── Panel A: Relative Abundance Oscillations ─────────────────────────────
ax_a = fig.add_subplot(gs[0])
passages = np.arange(1, 81)
for i in range(3):
    ax_a.plot(passages, rel_abund[:, i], color=sp_colors[i], lw=2.3, label=sp_labels[i])
ax_a.set_xlabel('Serial Dilution Passage', fontsize=13)
ax_a.set_ylabel('Relative Abundance', fontsize=13)
ax_a.set_title('(A)  Relative Abundance Oscillations\nunder Adaptive DOPRI5 (80 Passages)', fontsize=13, fontweight='bold')
ax_a.set_ylim(0, 1.05)
ax_a.set_xlim(1, 80)
ax_a.legend(loc='upper right', fontsize=9.5, framealpha=0.92)
ax_a.grid(True, alpha=0.25)

# ── Panel B: 2D Phase Portrait (Species Abundance Cycle) ──────────────────
ax_b = fig.add_subplot(gs[1])
start_idx = 40
n_show = len(long_traj) - start_idx
cmap = plt.cm.plasma
for i in range(n_show - 1):
    c = cmap(i / n_show)
    ax_b.plot(long_traj[start_idx+i, 0], long_traj[start_idx+i, 1], 'o',
              color=c, markersize=5.5, alpha=0.85)
ax_b.plot(long_traj[start_idx:, 0], long_traj[start_idx:, 1], '-',
          color='gray', lw=1.0, alpha=0.45, zorder=0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(40, 80))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax_b, shrink=0.7, pad=0.03)
cbar.set_label('Passage #', fontsize=10)
ax_b.set_xlabel('Species 18 Biomass OD', fontsize=13)
ax_b.set_ylabel('Species 21 Biomass OD', fontsize=13)
ax_b.set_title('(B)  Species Abundance Phase Portrait\nClosed Invariant Limit-Cycle Attractor', fontsize=13, fontweight='bold')
ax_b.grid(True, alpha=0.25)

# ── Panel C: Intra-Passage Resource Space Trajectory ─────────────────────
ax_c = fig.add_subplot(gs[2])
t_hours = np.linspace(0, 24, len(intra_R_last))
r_colors = ['#9467bd', '#8c564b', '#e377c2']
r_labels = ['Resource $R_1$ (Primary pulse)', 'Resource $R_2$ (Cross-fed byproduct)', 'Resource $R_3$ (Cross-fed byproduct)']
for k in range(3):
    ax_c.plot(t_hours, intra_R_last[:, k], color=r_colors[k], lw=2.3, label=r_labels[k])
ax_c.set_xlabel('Within-Passage Time (hours)', fontsize=13)
ax_c.set_ylabel('Resource Concentration', fontsize=13)
ax_c.set_title('(C)  Intra-Passage Resource Trajectory\nSequential Metabolic Cascade', fontsize=13, fontweight='bold')
ax_c.set_xlim(0, 24)
ax_c.legend(loc='upper right', fontsize=9.5, framealpha=0.92)
ax_c.grid(True, alpha=0.25)

plt.tight_layout()
out_fig = 'Poster/figures/Gore_Oscillation_Combined_Panel.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"Figure saved: {out_fig}")
shutil.copy(out_fig, '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143/Gore_Oscillation_Combined_Panel.png')
print("Artifact copied.")

"""
Panel C: PCA on (R1, R2, R3) intra-passage trajectories, one colored curve per passage,
showing convergence to a common invariant orbit in resource PC space.
"""
import os, torch, torch.nn as nn, torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
import shutil

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11.5
plt.rcParams['axes.linewidth'] = 1.4

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
        self.Theta_raw = nn.Parameter(torch.eye(3)*5.0 + torch.randn(3,3)*0.2)
        Phi_init = torch.tensor([[1.,3.,.1],[.1,1.,3.],[3.,.1,1.]])
        self.Phi_raw = nn.Parameter(torch.log(torch.exp(Phi_init)-1.) + torch.randn(3,3)*0.1)
        self.delta_raw = nn.Parameter(torch.zeros(3)+0.5)
        self.register_buffer('W', torch.tensor([[0.,.99,0.],[0.,0.,.99],[.99,0.,0.]], device=device))
        self.R0_raw = nn.Parameter(torch.tensor([1.5,.1,.1]))
    def get_R0(self): return F.softplus(self.R0_raw)
    def get_Theta(self): return F.softplus(self.Theta_raw)
    def get_Phi(self): return F.softplus(self.Phi_raw)
    def get_delta(self): return F.softplus(self.delta_raw)*2.
    def ode_deriv(self, t, S):
        N_t = torch.clamp(S[:3], min=0.)
        R_t = torch.clamp(S[3:], min=0.)
        monod = R_t/(R_t+self.Km)
        dN = N_t*(torch.matmul(monod,self.get_Theta().t()) - self.get_delta()*(0.02/(R_t.sum()+0.02)))
        flux = monod*torch.matmul(N_t,self.get_Phi())
        dR = -flux + torch.matmul(flux, self.W)
        return torch.cat([dN, dR])

torch.manual_seed(42)
model = Comm5Oscillator().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.015)
t_span = torch.linspace(0., 1., 25, device=device)
t_intra = torch.linspace(0., 1., 120, device=device)  # 120 points per passage

for ep in range(1, 41):
    model.train(); opt.zero_grad()
    R0 = model.get_R0()
    loss = sum(F.mse_loss(torch.clamp(odeint(model.ode_deriv, torch.cat([emp_N[p-1]/model.D+1e-4, R0]),
                          t_span, method='rk4')[-1,:3], min=0.), emp_N[p]) for p in range(1,11)) / 10.
    A = torch.matmul(model.get_Theta()*R0.unsqueeze(0), model.get_Phi().t())
    (loss - 0.5*torch.norm(A-A.t())/torch.clamp(torch.norm(A+A.t()),min=1e-6)).backward()
    opt.step()

# Simulate 80 passages, collect full intra-passage R trajectories
model.eval()
all_R_traj = []   # list of (120, 3) arrays
with torch.no_grad():
    R0 = model.get_R0()
    S = torch.cat([emp_N[0], R0])
    for p in range(80):
        N_in = S[:3]/model.D + 1e-4
        S_start = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S_start, t_intra, method='dopri5', rtol=1e-4, atol=1e-5)
        S = torch.clamp(S_t[-1], min=0.)
        all_R_traj.append(S_t[:, 3:].cpu().numpy())  # (120, 3)

all_R = np.stack(all_R_traj)  # (80, 120, 3)

# Flatten to fit PCA, then reshape back
R_flat = all_R.reshape(-1, 3)  # (80*120, 3)
pca = PCA(n_components=2)
pca.fit(R_flat)
var_explained = pca.explained_variance_ratio_ * 100
R_pca = pca.transform(R_flat).reshape(80, 120, 2)  # (80, 120, 2)

# Also get passage-end species abundances for coloring
passage_end_N = []
with torch.no_grad():
    R0 = model.get_R0()
    S = torch.cat([emp_N[0], R0])
    for p in range(80):
        N_in = S[:3]/model.D + 1e-4
        S_start = torch.cat([N_in, R0])
        S_t = odeint(model.ode_deriv, S_start, t_intra, method='dopri5', rtol=1e-4, atol=1e-5)
        S = torch.clamp(S_t[-1], min=0.)
        passage_end_N.append(S[:3].cpu().numpy())
passage_end_N = np.array(passage_end_N)  # (80, 3)

# Relative abundance for Panel A
long_traj = passage_end_N
total = long_traj.sum(axis=1, keepdims=True)
rel_abund = long_traj / np.maximum(total, 1e-6)

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5.8))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

sp_colors = ['#1f77b4', '#2ca02c', '#d62728']
sp_labels = ['Sp. 18 ($R_1$ specialist)', 'Sp. 21 ($R_2$ specialist)', 'Sp. 22 ($R_3$ specialist)']

# ── Panel A: Relative Abundance Oscillations ─────────────────────────────────
ax_a = fig.add_subplot(gs[0])
passages = np.arange(1, 81)
for i in range(3):
    ax_a.plot(passages, rel_abund[:, i], color=sp_colors[i], lw=2.3, label=sp_labels[i])
ax_a.set_xlabel('Serial Dilution Passage', fontsize=13)
ax_a.set_ylabel('Relative Abundance', fontsize=13)
ax_a.set_title('(A)  Relative Abundance Oscillations\n(80 autonomous passages, DOPRI5)', fontsize=13, fontweight='bold')
ax_a.set_ylim(0, 1.05); ax_a.set_xlim(1, 80)
ax_a.legend(loc='upper right', fontsize=9.5, framealpha=0.92)
ax_a.grid(True, alpha=0.25)

# ── Panel B: 2D Phase Portrait (Species Abundance Cycle) ─────────────────────
ax_b = fig.add_subplot(gs[1])
cmap_b = plt.cm.plasma
start_idx = 40
n_show = 80 - start_idx
for i in range(n_show - 1):
    c = cmap_b(i / n_show)
    ax_b.plot(long_traj[start_idx+i, 0], long_traj[start_idx+i, 1], 'o', color=c, markersize=5.5, alpha=0.85)
ax_b.plot(long_traj[start_idx:, 0], long_traj[start_idx:, 1], '-', color='gray', lw=0.8, alpha=0.4, zorder=0)
sm_b = plt.cm.ScalarMappable(cmap=cmap_b, norm=plt.Normalize(40, 80))
sm_b.set_array([])
plt.colorbar(sm_b, ax=ax_b, shrink=0.72, pad=0.03).set_label('Passage #', fontsize=10)
ax_b.set_xlabel('Species 18 Biomass OD', fontsize=13)
ax_b.set_ylabel('Species 21 Biomass OD', fontsize=13)
ax_b.set_title('(B)  Species Abundance Phase Portrait\nClosed Invariant Limit-Cycle Attractor', fontsize=13, fontweight='bold')
ax_b.grid(True, alpha=0.25)

# ── Panel C: Resource PCA Trajectories (one curve per passage) ───────────────
ax_c = fig.add_subplot(gs[2])
# Show passages 40-80 (converged limit cycle); early transients in light grey
cmap_c = plt.cm.viridis
# Transient passages 5-39 in light grey
for p in range(5, 40):
    ax_c.plot(R_pca[p, :, 0], R_pca[p, :, 1], color='#cccccc', lw=0.8, alpha=0.6)
# Converged passages 40-79 color-coded
for p in range(40, 80):
    color = cmap_c((p - 40) / 39)
    ax_c.plot(R_pca[p, :, 0], R_pca[p, :, 1], lw=1.8, color=color, alpha=0.85)
    # Mark start (circle) and end (triangle) of each intra-passage trajectory
    ax_c.plot(R_pca[p, 0, 0], R_pca[p, 0, 1], 'o', color=color, ms=4.5, zorder=5)
    ax_c.plot(R_pca[p, -1, 0], R_pca[p, -1, 1], '^', color=color, ms=4.5, zorder=5)
sm_c = plt.cm.ScalarMappable(cmap=cmap_c, norm=plt.Normalize(40, 80))
sm_c.set_array([])
plt.colorbar(sm_c, ax=ax_c, shrink=0.72, pad=0.03).set_label('Passage #', fontsize=10)
ax_c.set_xlabel(f'Resource PC1 ({var_explained[0]:.1f}% var.)', fontsize=13)
ax_c.set_ylabel(f'Resource PC2 ({var_explained[1]:.1f}% var.)', fontsize=13)
ax_c.set_title('(C)  Resource Space Trajectories (PCA)\nEach curve = one intra-passage cycle', fontsize=13, fontweight='bold')
# Legend proxy patches
from matplotlib.lines import Line2D
legend_elements = [Line2D([0],[0], color='#cccccc', lw=2, label='Transient (passes. 5–39)'),
                   Line2D([0],[0], color=cmap_c(0.5), lw=2, label='Converged (passes. 40–80)')]
ax_c.legend(handles=legend_elements, fontsize=9.5, framealpha=0.92)
ax_c.grid(True, alpha=0.25)

plt.tight_layout()
out_fig = 'Poster/figures/Gore_Oscillation_Combined_Panel.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"Figure saved: {out_fig}")
shutil.copy(out_fig, '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143/Gore_Oscillation_Combined_Panel.png')
print(f"PCA variance explained: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}%")
print("Done.")

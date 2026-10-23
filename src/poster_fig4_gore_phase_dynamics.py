import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchdiffeq import odeint

from src.scratch_evaluate_k2_800ep import all_keys, all_m, all_N0, all_c
from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ode_k2 = AdaptivePureConsumerResourceODE(N=48, K=2)
model_k2 = HierarchicalPhysicalNeuralODE(ode_k2, num_conditions=3, K=2).to(device)
model_k2.load_state_dict(torch.load('Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth', weights_only=False))
model_k2.eval()

idx_ln = [i for i, k in enumerate(all_keys) if k[0]=='LN' and k[1]==12 and k[2]=='Community3'][0]
idx_hn = [i for i, k in enumerate(all_keys) if k[0]=='HN' and k[1]==12 and k[2]=='Community3'][0]

delta_t = 0.02
tau_steps = int(1.0 / delta_t) + 1
t_span = torch.linspace(0.0, 1.0, tau_steps, device=device)

def simulate_community(idx):
    m = all_m[idx:idx+1]
    c = all_c[idx:idx+1]
    N_curr = all_N0[idx:idx+1].clone()
    
    R0_table = model_k2.get_R0()
    R0 = R0_table[c]
    yield_scale = torch.nn.functional.softplus(model_k2.yield_raw)[c].unsqueeze(-1)
    
    N_hist = [N_curr.cpu().numpy()[0]]
    R_paths = []
    
    with torch.no_grad():
        for p in range(1, 11):
            N0 = N_curr / model_k2.dilution_factor + model_k2.dispersal_rate * m
            N0_safe = torch.clamp(N0, min=0.0)
            S0 = torch.cat([N0_safe, R0], dim=-1)
            
            S_t = odeint(model_k2.ode_func, S0, t_span, method='euler', options={'step_size': delta_t})
            S_t = S_t.permute(1, 0, 2)
            
            N_final = torch.clamp(S_t[..., :48] * m.unsqueeze(1), min=0.0)[:, -1, :] * yield_scale
            N_curr = N_final
            
            N_hist.append(N_final.cpu().numpy()[0])
            R_paths.append(S_t[0, :, 48:].cpu().numpy())
            
    return np.array(N_hist), np.array(R_paths), m[0].cpu().numpy().astype(bool)

N_ln, R_ln, mask = simulate_community(idx_ln)
N_hn, R_hn, _    = simulate_community(idx_hn)

present_sp = np.where(mask)[0]
num_sp = len(present_sp)
sp_palette = plt.cm.tab20(np.linspace(0, 1, num_sp))

plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig = plt.figure(figsize=(15.0, 3.8), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.1], hspace=0.32, wspace=0.22)
days = np.arange(11)

# Panel A1
ax0 = fig.add_subplot(gs[0, 0])
for s_idx, sp in enumerate(present_sp):
    ax0.plot(days, N_ln[:, sp], 'o-', color=sp_palette[s_idx], lw=2.2, ms=5, alpha=0.85)
ax0.set_title(r'$\mathbf{(A1)\;Biomass:\;Low\;Nutrients\;(Stable)}$', fontsize=13.0, fontweight='bold', color='#1A365D', pad=6)
ax0.set_ylabel('Biomass (OD)', fontsize=13.0, fontweight='bold', color='#1A365D')
ax0.set_xlim(-0.3, 10.3)
ax0.set_xticks(range(11))
ax0.tick_params(labelsize=11.5)
ax0.grid(True, linestyle='--', alpha=0.5)

# Panel A2
ax1 = fig.add_subplot(gs[0, 1])
for s_idx, sp in enumerate(present_sp):
    ax1.plot(days, N_hn[:, sp], 'o-', color=sp_palette[s_idx], lw=2.2, ms=5, alpha=0.85)
ax1.set_title(r'$\mathbf{(A2)\;Biomass:\;High\;Nutrients\;(Chaotic)}$', fontsize=13.0, fontweight='bold', color='#802010', pad=6)
ax1.set_ylabel('Biomass (OD)', fontsize=13.0, fontweight='bold', color='#1A365D')
ax1.set_xlim(-0.3, 10.3)
ax1.set_xticks(range(11))
ax1.tick_params(labelsize=11.5)
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel B1
ax2 = fig.add_subplot(gs[1, 0])
cycle_colors = plt.cm.Blues(np.linspace(0.4, 1.0, 10))
for p in range(10):
    r_path = R_ln[p]
    lw = 1.4 if p < 6 else 2.6
    alpha = 0.55 if p < 6 else 0.95
    label = f'P{p+1}' if p in [0, 4, 9] else None
    ax2.plot(r_path[:, 0], r_path[:, 1], color=cycle_colors[p], lw=lw, alpha=alpha, label=label)
    ax2.scatter(r_path[0, 0], r_path[0, 1], color=cycle_colors[p], s=25, zorder=5)
    ax2.scatter(r_path[-1, 0], r_path[-1, 1], marker='s', color=cycle_colors[p], s=30, zorder=5)

ax2.set_title(r'$\mathbf{(B1)\;Stable\;Resource\;Limit\;Cycle}\;(\|\Delta\mathbf{R}\| \to \mathbf{0.0003})$', fontsize=13.0, fontweight='bold', color='#1A365D', pad=6)
ax2.set_xlabel('Resource 1 ($R_1$)', fontsize=13.0, fontweight='bold', color='#1A365D')
ax2.set_ylabel('Resource 2 ($R_2$)', fontsize=13.0, fontweight='bold', color='#1A365D')
ax2.tick_params(labelsize=11.5)
ax2.legend(loc='lower left', fontsize=11.5, framealpha=0.95, edgecolor='#0F2942')
ax2.grid(True, linestyle='--', alpha=0.5)

# Panel B2
ax3 = fig.add_subplot(gs[1, 1])
cycle_colors_red = plt.cm.YlOrRd(np.linspace(0.4, 1.0, 10))
for p in range(10):
    r_path = R_hn[p]
    lw = 1.4 if p < 6 else 2.6
    alpha = 0.55 if p < 6 else 0.95
    label = f'P{p+1}' if p in [0, 4, 9] else None
    ax3.plot(r_path[:, 0], r_path[:, 1], color=cycle_colors_red[p], lw=lw, alpha=alpha, label=label)
    ax3.scatter(r_path[0, 0], r_path[0, 1], color=cycle_colors_red[p], s=25, zorder=5)
    ax3.scatter(r_path[-1, 0], r_path[-1, 1], marker='s', color=cycle_colors_red[p], s=30, zorder=5)

ax3.set_title(r'$\mathbf{(B2)\;Chaotic\;Resource\;Wandering}\;(\|\Delta\mathbf{R}\| \approx \mathbf{0.0214})$', fontsize=13.0, fontweight='bold', color='#802010', pad=6)
ax3.set_xlabel('Resource 1 ($R_1$)', fontsize=13.0, fontweight='bold', color='#1A365D')
ax3.set_ylabel('Resource 2 ($R_2$)', fontsize=13.0, fontweight='bold', color='#1A365D')
ax3.tick_params(labelsize=11.5)
ax3.legend(loc='lower left', fontsize=11.5, framealpha=0.95, edgecolor='#0F2942')
ax3.grid(True, linestyle='--', alpha=0.5)

out_path = 'Poster/figures/Gore_Phase_Space_Dynamics.png'
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved large font Gore phase dynamics figure to {out_path}")

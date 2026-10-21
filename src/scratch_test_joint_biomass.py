import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
from scipy.stats import pearsonr, spearmanr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

data = torch.load('Data/Gore_processed_data.pt', weights_only=False)

N_species = 48
K = 7
dt = 0.10
dilution = 30.0
dispersal = 1e-6
t_span = torch.tensor([0.0, 1.0], device=device)

# Prepare data including true OD biomass!
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

# For 11-day communities:
b11_rel, b11_od, b11_c, b11_m = [], [], [], []
b4_rel, b4_od, b4_c, b4_m = [], [], [], []

for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    cond_idx = cond_to_idx[nutrient]
    mask = data['presence_masks'][(S, comm)]
    # OD has shape (3, 11) - average over 3 replicates
    od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0) # (11,)
    
    if traj_rel.shape[0] == 11:
        b11_rel.append(traj_rel)
        b11_od.append(od_mean)
        b11_c.append(cond_idx)
        b11_m.append(mask)
    elif traj_rel.shape[0] == 4:
        # 4 days: 0, 7, 8, 10
        od_4 = od_mean[[0, 7, 8, 10]]
        b4_rel.append(traj_rel)
        b4_od.append(od_4)
        b4_c.append(cond_idx)
        b4_m.append(mask)

b11_rel = torch.stack(b11_rel).to(device) # (N11, 11, 48)
b11_od = torch.stack(b11_od).to(device)   # (N11, 11)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)

b4_rel = torch.stack(b4_rel).to(device)   # (N4, 4, 48)
b4_od = torch.stack(b4_od).to(device)     # (N4, 4)
b4_c = torch.tensor(b4_c, device=device)
b4_m = torch.stack(b4_m).to(device)

class JointConsumerResourceModel(nn.Module):
    def __init__(self, N, K, num_conds=3):
        super().__init__()
        self.N, self.K = N, K
        self.Theta = nn.Parameter(torch.randn(N, K) * 0.1)
        self.W = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.01)
        self.a = nn.Parameter(torch.randn(num_conds, K) * 0.1)
        
    def ode_func(self, t, S):
        N_t = S[..., :self.N]
        R_t = S[..., self.N:]
        alpha = F.softplus(self.alpha_raw)
        N_safe = torch.clamp(N_t, min=0.0)
        R_safe = torch.clamp(R_t, min=0.0)
        growth = torch.matmul(R_safe, self.Theta.t())
        dN = (N_safe * growth) - (alpha * (N_safe ** 2))
        sink = torch.matmul(N_safe, torch.matmul(self.Theta, self.W))
        dR = -R_safe * sink
        return torch.cat([dN, dR], dim=-1)

    def forward_passage(self, N_prev, cond_idx, mask):
        R0 = F.softplus(self.a[cond_idx])
        # Inoculum based on actual previous biomass!
        N0 = N_prev / dilution + dispersal * mask
        N0_safe = torch.clamp(N0, min=0.0)
        S0 = torch.cat([N0_safe, R0], dim=-1)
        
        S_t = odeint(self.ode_func, S0, t_span, method='euler', options={'step_size': dt})
        S_final = S_t[-1]
        N_final = torch.clamp(S_final[..., :self.N] * mask, min=0.0)
        
        # Absolute predicted biomass:
        pred_biomass = N_final.sum(dim=-1) # (batch,)
        
        # Relative abundance:
        tot = pred_biomass.unsqueeze(-1)
        tot_safe = torch.where(tot > 0, tot, torch.ones_like(tot))
        x_pred = N_final / tot_safe
        log_pred = torch.log(torch.clamp(x_pred, min=1e-9))
        
        return N_final, pred_biomass, log_pred, x_pred

model = JointConsumerResourceModel(N_species, K, num_conds=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
kl_loss = nn.KLDivLoss(reduction='batchmean')
mse_loss = nn.MSELoss()

epochs = 250
print("--- Training Joint Biomass + Relative Abundance Model ---")
for epoch in range(1, epochs + 1):
    model.train()
    tf_ratio = max(0.10, 1.0 - 0.90 * (epoch / epochs))
    
    # Batch 11
    # Day 0 initialization: true relative * true OD
    N_curr = b11_rel[:, 0, :] * b11_od[:, 0:1]
    loss11 = 0.0
    for t in range(1, 11):
        N_pred, pred_bio, log_x, x_pred = model.forward_passage(N_curr, b11_c, b11_m)
        target_x = b11_rel[:, t, :]
        target_bio = b11_od[:, t]
        
        loss_x = kl_loss(log_x, target_x)
        loss_bio = mse_loss(pred_bio, target_bio)
        loss11 += loss_x + 2.0 * loss_bio
        
        # Teacher forcing on absolute abundance:
        target_N = target_x * target_bio.unsqueeze(-1)
        N_curr = tf_ratio * target_N + (1.0 - tf_ratio) * N_pred
        
    loss = loss11 / 10.0
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}/{epochs}: Loss = {loss.item():.4f} (TF={tf_ratio:.2f})")

# Evaluate correlation between model unrolled biomass CV and data biomass CV!
model.eval()
with torch.no_grad():
    N_curr = b11_rel[:, 0, :] * b11_od[:, 0:1]
    bio_traj = [N_curr.sum(dim=-1).cpu().numpy()]
    x_traj = [b11_rel[:, 0, :].cpu().numpy()]
    for t in range(1, 11):
        N_pred, pred_bio, _, x_pred = model.forward_passage(N_curr, b11_c, b11_m)
        bio_traj.append(pred_bio.cpu().numpy())
        x_traj.append(x_pred.cpu().numpy())
        N_curr = N_pred # 100% autonomous unrolling!
        
    bio_traj = np.array(bio_traj).T # (N11, 11)
    actual_bio = b11_od.cpu().numpy() # (N11, 11)
    
    # Days 7 to 10 CV:
    model_cv = np.std(bio_traj[:, 7:11], axis=1) / np.mean(bio_traj[:, 7:11], axis=1)
    actual_cv = np.std(actual_bio[:, 7:11], axis=1) / np.mean(actual_bio[:, 7:11], axis=1)
    
    r_bio, p_bio = pearsonr(model_cv, actual_cv)
    rho_bio, sp_bio = spearmanr(model_cv, actual_cv)
    print(f"\nAutonomous Unrolled Biomass CV Correlation on b11 (N={len(model_cv)}):")
    print(f"Pearson r = {r_bio:.4f} (p = {p_bio:.2e})")
    print(f"Spearman rho = {rho_bio:.4f} (p = {sp_bio:.2e})")

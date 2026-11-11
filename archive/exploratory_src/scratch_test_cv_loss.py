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

cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

b11_rel, b11_c, b11_m, b11_cv = [], [], [], []
b4_rel, b4_c, b4_m, b4_cv = [], [], [], []

for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    cond_idx = cond_to_idx[nutrient]
    mask = data['presence_masks'][(S, comm)]
    
    # Target CV over days 7 to 10
    days = data['timepoints'][nutrient][(S, comm)]
    day_nums = np.array([int(d.replace('Day', '')) for d in days])
    late_idx = np.where(day_nums >= 7)[0]
    
    # Calculate target empirical CV
    sub_late = traj_rel[late_idx][:, mask.bool()] # (T_late, S_alive)
    m = sub_late.mean(dim=0)
    s = sub_late.std(dim=0)
    surv = m >= 1e-3
    if surv.any():
        targ_cv = torch.mean(s[surv] / (0.01 + m[surv]))
    else:
        targ_cv = torch.tensor(0.0)
        
    if traj_rel.shape[0] == 11:
        b11_rel.append(traj_rel)
        b11_c.append(cond_idx)
        b11_m.append(mask)
        b11_cv.append(targ_cv)
    elif traj_rel.shape[0] == 4:
        b4_rel.append(traj_rel)
        b4_c.append(cond_idx)
        b4_m.append(mask)
        b4_cv.append(targ_cv)

b11_rel = torch.stack(b11_rel).to(device) # (N11, 11, 48)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)
b11_cv = torch.tensor(b11_cv, device=device, dtype=torch.float32) # (N11,)

b4_rel = torch.stack(b4_rel).to(device)
b4_c = torch.tensor(b4_c, device=device)
b4_m = torch.stack(b4_m).to(device)
b4_cv = torch.tensor(b4_cv, device=device, dtype=torch.float32)

class ModelContainer(nn.Module):
    def __init__(self, N, K, num_conditions=3):
        super().__init__()
        self.N, self.K = N, K
        self.Theta = nn.Parameter(torch.randn(N, K) * 0.1)
        self.W = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.01)
        self.a = nn.Parameter(torch.randn(num_conditions, K) * 0.1)
        
    def ode_func(self, t, S):
        N_t, R_t = S[..., :self.N], S[..., self.N:]
        alpha = F.softplus(self.alpha_raw)
        N_safe, R_safe = torch.clamp(N_t, min=0.0), torch.clamp(R_t, min=0.0)
        growth = torch.matmul(R_safe, self.Theta.t())
        dN = (N_safe * growth) - (alpha * (N_safe ** 2))
        sink = torch.matmul(N_safe, torch.matmul(self.Theta, self.W))
        dR = -R_safe * sink
        return torch.cat([dN, dR], dim=-1)
        
    def step_forward(self, x_curr, cond_idx, mask):
        R0 = F.softplus(self.a[cond_idx])
        N0 = x_curr / dilution + dispersal * mask
        N0_safe = torch.clamp(N0, min=0.0)
        S0 = torch.cat([N0_safe, R0], dim=-1)
        
        S_t = odeint(self.ode_func, S0, t_span, method='euler', options={'step_size': dt})
        S_final = S_t[-1]
        N_final = S_final[..., :self.N] * mask
        N_safe = torch.clamp(N_final, min=0.0)
        tot = N_safe.sum(dim=-1, keepdim=True)
        tot = torch.where(tot > 0, tot, torch.ones_like(tot))
        x_pred = N_safe / tot
        log_pred = torch.log(torch.clamp(x_pred, min=1e-9))
        return log_pred, x_pred

model = ModelContainer(N_species, K).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
kl_loss = nn.KLDivLoss(reduction='batchmean')
mse_loss = nn.MSELoss()

epochs = 250
lambda_cv = 10.0 # Weight on stability matching loss!

print(f"--- Training with Stability-Aware Multi-Task Loss (lambda_cv={lambda_cv}) ---")
for epoch in range(1, epochs + 1):
    model.train()
    tf_ratio = max(0.10, 1.0 - 0.90 * (epoch / epochs))
    
    # Rollout batch 11
    x_curr = b11_rel[:, 0, :]
    loss_kl = 0.0
    preds_11 = [x_curr]
    
    for t in range(1, 11):
        log_p, x_p = model.step_forward(x_curr, b11_c, b11_m)
        target = b11_rel[:, t, :]
        loss_kl += kl_loss(log_p, target)
        x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_p
        preds_11.append(x_p)
        
    preds_11 = torch.stack(preds_11, dim=1) # (N11, 11, 48)
    
    # Compute model simulated CV over days 7 to 10 (indices 7, 8, 9, 10)
    late_preds = preds_11[:, 7:11, :] # (N11, 4, 48)
    # per-community CV:
    m_cvs = []
    for i in range(b11_s_len := b11_rel.size(0)):
        mask_i = b11_m[i].bool()
        sub = late_preds[i, :, mask_i] # (4, S_alive)
        mu = sub.mean(dim=0)
        sigma = sub.std(dim=0)
        # soft regularized CV
        cv_i = torch.mean(sigma / (0.01 + mu))
        m_cvs.append(cv_i)
    m_cvs = torch.stack(m_cvs) # (N11,)
    
    loss_cv = mse_loss(m_cvs, b11_cv)
    loss = (loss_kl / 10.0) + lambda_cv * loss_cv
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if epoch % 50 == 0:
        with torch.no_grad():
            r_val, _ = pearsonr(m_cvs.cpu().numpy(), b11_cv.cpu().numpy())
            print(f"Epoch {epoch:3d}/{epochs}: Loss={loss.item():.4f} (KL={loss_kl.item()/10.0:.4f}, CV_loss={loss_cv.item():.4f}) | Train CV Pearson r = {r_val:.4f}")

# Evaluate autonomous unrolling across all 75 communities
model.eval()
with torch.no_grad():
    x_curr = b11_rel[:, 0, :]
    preds = [x_curr.cpu().numpy()]
    for t in range(1, 11):
        _, x_curr = model.step_forward(x_curr, b11_c, b11_m)
        preds.append(x_curr.cpu().numpy())
    preds = np.stack(preds, axis=1)
    
    late_m = preds[:, 7:11, :]
    model_cv_list = []
    for i in range(len(preds)):
        mask_i = b11_m[i].cpu().numpy().astype(bool)
        sub = late_m[i, :, mask_i]
        mu = sub.mean(axis=0)
        sigma = sub.std(axis=0)
        model_cv_list.append(np.mean(sigma / (0.01 + mu)))
    model_cv = np.array(model_cv_list)
    actual_cv = b11_cv.cpu().numpy()
    
    r_auto, p_auto = pearsonr(model_cv, actual_cv)
    rho_auto, sp_auto = spearmanr(model_cv, actual_cv)
    print(f"\nAutonomous Unrolled Stability Correlation on b11 (N={len(model_cv)}):")
    print(f"Pearson r = {r_auto:.4f} (p = {p_auto:.2e})")
    print(f"Spearman rho = {rho_auto:.4f} (p = {sp_auto:.2e})")

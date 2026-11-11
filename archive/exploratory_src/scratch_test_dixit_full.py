import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import time
from scipy.stats import pearsonr, spearmanr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing Dixit Generalized KL Full Test on: {device}")

data = torch.load('Data/Gore_processed_data.pt', weights_only=False)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}
N_species = 48
K = 7
dilution_factor = 30.0
dispersal_rate = 1e-6
t_span = torch.tensor([0.0, 1.0], device=device)

# Prepare batches
b11_N, b11_od, b11_rel, b11_c, b11_m, b11_keys = [], [], [], [], [], []
b4_N, b4_od, b4_rel, b4_c, b4_m, b4_keys = [], [], [], [], [], []

for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    c_idx = cond_to_idx[nutrient]
    mask = data['presence_masks'][(S, comm)]
    od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
    
    if traj_rel.shape[0] == 11:
        b11_N.append(traj_rel * od_mean.unsqueeze(-1))
        b11_od.append(od_mean)
        b11_rel.append(traj_rel)
        b11_c.append(c_idx)
        b11_m.append(mask)
        b11_keys.append((nutrient, S, comm))
    elif traj_rel.shape[0] == 4:
        od_4 = od_mean[[0, 7, 8, 10]]
        b4_N.append(traj_rel * od_4.unsqueeze(-1))
        b4_od.append(od_mean)
        b4_rel.append(traj_rel)
        b4_c.append(c_idx)
        b4_m.append(mask)
        b4_keys.append((nutrient, S, comm))

b11_N = torch.stack(b11_N).to(device)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)

b4_N = torch.stack(b4_N).to(device)
b4_c = torch.tensor(b4_c, device=device)
b4_m = torch.stack(b4_m).to(device)

all_keys = b11_keys + b4_keys
all_N0 = torch.cat([b11_N[:, 0, :], b4_N[:, 0, :]], dim=0)
all_c = torch.cat([b11_c, b4_c], dim=0)
all_m = torch.cat([b11_m, b4_m], dim=0)

class PureConsumerResourceODE(nn.Module):
    def __init__(self, N, K):
        super().__init__()
        self.N = N
        self.K = K
        # Initial growth rate ~ 3.5 -> Theta ~ 0.5, R0 ~ 7 -> Theta*R ~ 3.5
        self.Theta = nn.Parameter(torch.randn(N, K) * 0.1 + 0.3)
        self.W = nn.Parameter(torch.randn(K, K) * 0.05 + torch.eye(K) * 0.2)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.01)
        
    def forward(self, t, S):
        N_t = torch.clamp(S[..., :self.N], min=0.0)
        R_t = torch.clamp(S[..., self.N:], min=0.0)
        alpha = F.softplus(self.alpha_raw)
        growth = torch.matmul(R_t, self.Theta.t())
        dN = (N_t * growth) - (alpha * (N_t ** 2))
        effective_uptake = torch.matmul(self.Theta, self.W)
        sink = torch.matmul(N_t, effective_uptake)
        dR = -R_t * sink
        return torch.cat([dN, dR], dim=-1)

class PhysicalAssemblyNeuralODE(nn.Module):
    def __init__(self, ode_func, num_conditions=3, K=7):
        super().__init__()
        self.ode_func = ode_func
        self.K = K
        # Hierarchical initial resources: LN ~ 3.3, MN ~ 7.5, HN ~ 9.0
        init_a = torch.tensor([
            [-0.3, -1.3,  0.0, -0.2, -1.3, -0.2, -0.5], # LN sum ~ 3.3
            [ 0.2,  1.3,  0.9,  1.0,  0.5, -1.0,  0.9], # MN sum ~ 7.6
            [ 0.8,  0.8,  0.0,  1.7,  1.5,  1.7, -0.3]  # HN sum ~ 9.1
        ])
        self.a = nn.Parameter(init_a)
        
    def step_forward(self, N_prev, cond_idx, mask, delta_t=0.05):
        R0 = F.softplus(self.a[cond_idx])
        N0 = N_prev / dilution_factor + dispersal_rate * mask
        N0_safe = torch.clamp(N0, min=0.0)
        S0 = torch.cat([N0_safe, R0], dim=-1)
        
        S_t = odeint(self.ode_func, S0, t_span, method='euler', options={'step_size': delta_t})
        S_final = S_t[-1]
        N_final = torch.clamp(S_final[..., :N_species] * mask, min=0.0)
        return N_final

def dixit_kl_batch(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    pred_safe = torch.clamp(y_pred, min=eps)
    y_for_log = torch.clamp(y_safe, min=eps)
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_for_log / pred_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + pred_safe) * mask
    return div.sum(dim=-1).mean()

ode_func = PureConsumerResourceODE(N_species, K)
model = PhysicalAssemblyNeuralODE(ode_func, num_conditions=3, K=K).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

epochs = 250
print(f"\n--- Starting 250 Epochs Training with Dr. Dixit's Generalized KL Loss ---")
start_time = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    tf_ratio = max(0.10, 1.0 - (epoch / (epochs * 0.75)))
    
    # Batch 11
    N_curr = b11_N[:, 0, :]
    loss11 = 0.0
    for t in range(1, 11):
        N_pred = model.step_forward(N_curr, b11_c, b11_m, delta_t=0.05)
        target = b11_N[:, t, :]
        loss11 += dixit_kl_batch(target, N_pred, b11_m)
        N_curr = tf_ratio * target + (1.0 - tf_ratio) * N_pred
        
    # Batch 4
    N_curr_4 = b4_N[:, 0, :]
    loss4 = 0.0
    d4_map = {7: 1, 8: 2, 10: 3}
    for day in range(1, 11):
        N_pred_4 = model.step_forward(N_curr_4, b4_c, b4_m, delta_t=0.05)
        if day in d4_map:
            t_idx = d4_map[day]
            target_4 = b4_N[:, t_idx, :]
            loss4 += dixit_kl_batch(target_4, N_pred_4, b4_m)
            N_curr_4 = tf_ratio * target_4 + (1.0 - tf_ratio) * N_pred_4
        else:
            N_curr_4 = N_pred_4
            
    # Regularization
    l2_reg = 0.005 * (model.ode_func.Theta.pow(2).sum() + model.ode_func.W.pow(2).sum())
    loss = (loss11 / 10.0) + (loss4 / 3.0) + l2_reg
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if epoch % 25 == 0 or epoch == epochs:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} (TF: {tf_ratio:.2f}) | Time: {elapsed:.1f}s")

# Autonomous Unrolled Evaluation
model.eval()
print("\n--- Running 100% Autonomous Unrolled Evaluation (Zero Teacher Forcing) ---")
with torch.no_grad():
    N_curr = all_N0.clone()
    pred_trajs_N = [N_curr.cpu().numpy()]
    for day in range(1, 11):
        N_pred = model.step_forward(N_curr, all_c, all_m, delta_t=0.05)
        pred_trajs_N.append(N_pred.cpu().numpy())
        N_curr = N_pred
        
    pred_trajs_N = np.stack(pred_trajs_N, axis=1) # (99, 11, 48)
    pred_biomass = pred_trajs_N.sum(axis=-1)       # (99, 11)
    tot_safe = np.where(pred_biomass[..., None] > 0, pred_biomass[..., None], 1.0)
    pred_rel = pred_trajs_N / tot_safe
    
    r2_rel_list, r2_bio_list, model_cvs, exp_cvs, od_cvs = [], [], [], [], []
    for i, (nutrient, S, comm) in enumerate(all_keys):
        mask_np = all_m[i].cpu().numpy().astype(bool)
        true_od = data['biomass_od'][(nutrient, S, comm)].mean(dim=0).numpy()
        pred_od = pred_biomass[i]
        ss_res_b = np.sum((true_od - pred_od)**2)
        ss_tot_b = np.sum((true_od - np.mean(true_od))**2)
        r2_bio_list.append(1.0 - ss_res_b / (ss_tot_b + 1e-9))
        
        days = data['timepoints'][nutrient][(S, comm)]
        day_nums = np.array([int(d.replace('Day', '')) for d in days])
        true_rel = data['relative_abundance'][(nutrient, S, comm)].numpy()
        sub_pred_rel = pred_rel[i, day_nums]
        y_t = true_rel[:, mask_np].flatten()
        y_p = sub_pred_rel[:, mask_np].flatten()
        ss_res_r = np.sum((y_t - y_p)**2)
        ss_tot_r = np.sum((y_t - np.mean(y_t))**2)
        r2_rel_list.append(1.0 - ss_res_r / (ss_tot_r + 1e-9))
        
        # Late-stage species CV (Days 7 to 10)
        late_m = pred_rel[i, 7:11][:, mask_np]
        mu_m = late_m.mean(axis=0)
        sig_m = late_m.std(axis=0)
        surv_m = mu_m >= 1e-3
        m_cv = np.mean(sig_m[surv_m] / (0.01 + mu_m[surv_m])) if surv_m.any() else 0.0
        model_cvs.append(m_cv)
        
        late_idx = np.where(day_nums >= 7)[0]
        late_e = true_rel[late_idx][:, mask_np]
        mu_e = late_e.mean(axis=0)
        sig_e = late_e.std(axis=0)
        surv_e = mu_e >= 1e-3
        e_cv = np.mean(sig_e[surv_e] / (0.01 + mu_e[surv_e])) if surv_e.any() else 0.0
        exp_cvs.append(e_cv)
        
        od_7_10 = data['biomass_od'][(nutrient, S, comm)].numpy()[:, 7:11]
        od_cv = np.mean(np.std(od_7_10, axis=1) / np.mean(od_7_10, axis=1))
        od_cvs.append(od_cv)
        
    model_cvs = np.array(model_cvs)
    exp_cvs = np.array(exp_cvs)
    od_cvs = np.array(od_cvs)
    
    r_log, p_log = pearsonr(np.log(model_cvs + 1e-4), np.log(exp_cvs + 1e-4))
    rho, p_rho = spearmanr(model_cvs, exp_cvs)
    
    # Global Biomass metrics
    all_true_od = np.array([data['biomass_od'][(n, s, c)].mean(dim=0).numpy() for (n, s, c) in all_keys]).flatten()
    all_pred_od = pred_biomass.flatten()
    r_bio_global, _ = pearsonr(all_true_od, all_pred_od)
    ss_res_global = np.sum((all_true_od - all_pred_od)**2)
    ss_tot_global = np.sum((all_true_od - np.mean(all_true_od))**2)
    r2_bio_global = 1.0 - ss_res_global / ss_tot_global
    
    print("\n================ FULL EVALUATION SUMMARY ================")
    print(f"Mean Unrolled Relative Abundance Linear R^2: {np.mean(r2_rel_list):.4f}")
    print(f"Median Unrolled Relative Abundance Linear R^2: {np.median(r2_rel_list):.4f}")
    print(f"Global Biomass OD Pearson r: {r_bio_global:.4f}")
    print(f"Global Biomass OD Linear R^2: {r2_bio_global:.4f}")
    print(f"Mean True OD: {all_true_od.mean():.4f} | Mean Pred OD: {all_pred_od.mean():.4f}")
    print(f"Log-Scale Pearson r on CV: {r_log:.4f} (p = {p_log:.2e})")
    print(f"Spearman rho on CV:        {rho:.4f} (p = {p_rho:.2e})")
    
    # Nutrient Breakdown
    for nut in ['LN', 'MN', 'HN']:
        sub_idxs = [k for k, key in enumerate(all_keys) if key[0] == nut]
        r_sub, _ = pearsonr(np.log(model_cvs[sub_idxs] + 1e-4), np.log(exp_cvs[sub_idxs] + 1e-4))
        rho_sub, _ = spearmanr(model_cvs[sub_idxs], exp_cvs[sub_idxs])
        rel_sub = [r2_rel_list[k] for k in sub_idxs]
        bio_sub_t = all_true_od.reshape(99, 11)[sub_idxs].flatten()
        bio_sub_p = all_pred_od.reshape(99, 11)[sub_idxs].flatten()
        r_bio_sub, _ = pearsonr(bio_sub_t, bio_sub_p)
        print(f"  {nut}: Rel R^2 = {np.mean(rel_sub):.3f} | Bio r = {r_bio_sub:.3f} | Log r = {r_sub:.3f} | Spearman rho = {rho_sub:.3f}")

# Save checkpoint
torch.save(model.state_dict(), 'Data/Gore_Global_ODE_Model_GeneralizedKL.pth')
print("\nSaved checkpoint to Data/Gore_Global_ODE_Model_GeneralizedKL.pth")

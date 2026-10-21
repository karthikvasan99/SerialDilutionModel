import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# -------------------------------------------------------------
# 1. Device and Reproducibility Setup
# -------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------------------------------------------
# 2. Pure Consumer-Resource ODE Architecture
# -------------------------------------------------------------
class PureConsumerResourceODE(nn.Module):
    """
    Pure Continuous-Time Consumer-Resource ODE governing bacterial growth and
    latent resource depletion. Strictly no direct species-species interaction matrix.
    
    Equations:
      dN_i/dt = N_i * sum_k(Theta_ki * R_k) - alpha_i * N_i^2
      dR_k/dt = -R_k * sum_l sum_j(W_kl * Theta_jl * N_j)
    """
    def __init__(self, N=48, K=7):
        super().__init__()
        self.N = N
        self.K = K
        # Trait matrices
        self.Theta = nn.Parameter(torch.randn(N, K) * 0.1 + 0.3)
        self.W = nn.Parameter(torch.randn(K, K) * 0.05 + torch.eye(K) * 0.2)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.01)
        
    def forward(self, t, S):
        N_t = S[..., :self.N]
        R_t = S[..., self.N:]
        alpha = F.softplus(self.alpha_raw)
        N_safe = torch.clamp(N_t, min=0.0)
        R_safe = torch.clamp(R_t, min=0.0)
        
        growth = torch.matmul(R_safe, self.Theta.t())
        dN = (N_safe * growth) - (alpha * (N_safe ** 2))
        
        effective_uptake = torch.matmul(self.Theta, self.W)
        sink = torch.matmul(N_safe, effective_uptake)
        dR = -R_safe * sink
        return torch.cat([dN, dR], dim=-1)

class HierarchicalPhysicalNeuralODE(nn.Module):
    """
    Physical-Assembly Neural ODE:
    1. Preserves the physical media nutrient hierarchy: R0(LN) < R0(MN) < R0(HN).
    2. Accounts for environmental biomass conversion efficiency / overflow metabolism.
    3. Propagates unbroken physical dilution: N0 = N_prev / 30.0 + 10^-6 * mask.
    """
    def __init__(self, ode_func, num_conditions=3, K=7, dilution_factor=30.0, dispersal_rate=1e-6):
        super().__init__()
        self.ode_func = ode_func
        self.K = K
        self.dilution_factor = dilution_factor
        self.dispersal_rate = dispersal_rate
        
        # Monotonic resource parameters
        self.a_ln = nn.Parameter(torch.randn(K) * 0.1 + 0.5)
        self.delta_mn = nn.Parameter(torch.randn(K) * 0.1 + 0.5)
        self.delta_hn = nn.Parameter(torch.randn(K) * 0.1 + 0.5)
        
        # Environmental yield / conversion scale (overflow metabolism in HN)
        self.yield_raw = nn.Parameter(torch.zeros(num_conditions))
        
    def get_R0(self):
        r_ln = F.softplus(self.a_ln)
        r_mn = r_ln + F.softplus(self.delta_mn)
        r_hn = r_mn + F.softplus(self.delta_hn)
        return torch.stack([r_ln, r_mn, r_hn])
        
    def step_forward(self, N_prev, cond_idx, mask, delta_t=0.05):
        R0_table = self.get_R0()
        R0 = R0_table[cond_idx]
        yield_scale = F.softplus(self.yield_raw)[cond_idx].unsqueeze(-1)
        
        N0 = N_prev / self.dilution_factor + self.dispersal_rate * mask
        N0_safe = torch.clamp(N0, min=0.0)
        S0 = torch.cat([N0_safe, R0], dim=-1)
        
        t_span = torch.tensor([0.0, 1.0], device=S0.device)
        S_t = odeint(self.ode_func, S0, t_span, method='euler', options={'step_size': delta_t})
        S_final = S_t[-1]
        N_final = torch.clamp(S_final[..., :self.ode_func.N] * mask, min=0.0)
        return N_final * yield_scale

# -------------------------------------------------------------
# 3. Dr. Purushottam Dixit's Generalized KL Loss Function
# -------------------------------------------------------------
def dixit_generalized_kl(y_true, y_pred, mask, eps=1e-8):
    """
    Computes Dr. Purushottam Dixit's Generalized KL Divergence:
      D_genKL = sum_{i in mask} ( y_true * ln((y_true + eps)/(y_pred + eps)) - y_true + y_pred )
    Averaged over communities in the batch.
    """
    y_safe = torch.clamp(y_true, min=0.0)
    pred_safe = torch.clamp(y_pred, min=eps)
    y_for_log = torch.clamp(y_safe, min=eps)
    
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_for_log / pred_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + pred_safe) * mask
    return div.sum(dim=-1).mean()

# -------------------------------------------------------------
# 4. Training Pipeline
# -------------------------------------------------------------
def train_generalized_kl(data_path='Data/Gore_processed_data.pt', epochs=250, lr=0.005,
                         l2_reg=0.005, delta_t=0.05, seed=42):
    set_seed(seed)
    print(f"Executing Generalized KL Consumer-Resource Training on: {device}")
    data = torch.load(data_path, weights_only=False)
    
    cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}
    N_species = 48
    K = 7
    
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
    
    ode_func = PureConsumerResourceODE(N_species, K)
    model = HierarchicalPhysicalNeuralODE(ode_func, num_conditions=3, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Batch 11: {b11_N.size(0)} communities | Batch 4: {b4_N.size(0)} communities")
    print(f"--- Starting {epochs} Epochs Training with Dr. Dixit's Generalized KL Loss ---")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Scheduled sampling decay to 0.10
        raw_tf = 1.0 - (epoch / (epochs * 0.75))
        tf_ratio = max(0.10, min(1.0, raw_tf))
        
        # Batch 11: All 10 passages
        N_curr = b11_N[:, 0, :]
        loss11 = 0.0
        for t in range(1, 11):
            N_pred = model.step_forward(N_curr, b11_c, b11_m, delta_t=delta_t)
            target = b11_N[:, t, :]
            loss11 += dixit_generalized_kl(target, N_pred, b11_m)
            N_curr = tf_ratio * target + (1.0 - tf_ratio) * N_pred
            
        # Batch 4 (LN): Unroll 10 passages, compute loss at days 7, 8, 10
        N_curr_4 = b4_N[:, 0, :]
        loss4 = 0.0
        d4_map = {7: 1, 8: 2, 10: 3}
        for day in range(1, 11):
            N_pred_4 = model.step_forward(N_curr_4, b4_c, b4_m, delta_t=delta_t)
            if day in d4_map:
                t_idx = d4_map[day]
                target_4 = b4_N[:, t_idx, :]
                loss4 += dixit_generalized_kl(target_4, N_pred_4, b4_m)
                N_curr_4 = tf_ratio * target_4 + (1.0 - tf_ratio) * N_pred_4
            else:
                N_curr_4 = N_pred_4
                
        # Decoupled L2 regularization on biological parameters
        theta_reg = model.ode_func.Theta.pow(2.0).sum()
        w_reg = model.ode_func.W.pow(2.0).sum()
        reg_penalty = l2_reg * (theta_reg + w_reg)
        
        loss = (loss11 / 10.0) + (loss4 / 3.0) + reg_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 25 == 0 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} (TF: {tf_ratio:.2f}) | Time: {elapsed:.1f}s")
            
    # Learned nutrient parameters
    R0_res = model.get_R0().detach().cpu().numpy()
    yield_res = F.softplus(model.yield_raw).detach().cpu().numpy()
    print("\nLearned Initial Nutrient Pools R(0) sum:")
    for nut, s_val in zip(['LN', 'MN', 'HN'], R0_res.sum(axis=1)):
        print(f"  {nut}: sum = {s_val:.3f}")
    print("Learned Environmental Yield Scale:")
    for nut, y_val in zip(['LN', 'MN', 'HN'], yield_res):
        print(f"  {nut}: yield = {y_val:.3f}")
        
    # -------------------------------------------------------------
    # 5. Autonomous Multi-step Evaluation (Zero Teacher Forcing)
    # -------------------------------------------------------------
    model.eval()
    print("\n--- Running 100% Autonomous Unrolled Evaluation (Zero Teacher Forcing) ---")
    
    with torch.no_grad():
        N_curr = all_N0.clone()
        pred_trajs_N = [N_curr.cpu().numpy()]
        for day in range(1, 11):
            N_pred = model.step_forward(N_curr, all_c, all_m, delta_t=delta_t)
            pred_trajs_N.append(N_pred.cpu().numpy())
            N_curr = N_pred
            
        pred_trajs_N = np.stack(pred_trajs_N, axis=1) # (99, 11, 48)
        pred_biomass = pred_trajs_N.sum(axis=-1)       # (99, 11)
        tot_safe = np.where(pred_biomass[..., None] > 0, pred_biomass[..., None], 1.0)
        pred_rel = pred_trajs_N / tot_safe
        
    records = []
    for i, (nutrient, S, comm) in enumerate(all_keys):
        mask_np = all_m[i].cpu().numpy().astype(bool)
        
        # 1. Biomass OD fit
        true_od = data['biomass_od'][(nutrient, S, comm)].mean(dim=0).numpy()
        pred_od = pred_biomass[i]
        ss_res_b = np.sum((true_od - pred_od)**2)
        ss_tot_b = np.sum((true_od - np.mean(true_od))**2)
        r2_b = 1.0 - ss_res_b / (ss_tot_b + 1e-9)
        
        # 2. Composition Linear R^2 across measured days
        days = data['timepoints'][nutrient][(S, comm)]
        day_nums = np.array([int(d.replace('Day', '')) for d in days])
        true_rel = data['relative_abundance'][(nutrient, S, comm)].numpy()
        sub_pred_rel = pred_rel[i, day_nums]
        y_t = true_rel[:, mask_np].flatten()
        y_p = sub_pred_rel[:, mask_np].flatten()
        ss_res_r = np.sum((y_t - y_p)**2)
        ss_tot_r = np.sum((y_t - np.mean(y_t))**2)
        r2_r = 1.0 - ss_res_r / (ss_tot_r + 1e-9)
        
        # 3. Model Late-Stage Regularized CV (Days 7 to 10)
        late_model = pred_rel[i, 7:11][:, mask_np]
        mu_m = late_model.mean(axis=0)
        sig_m = late_model.std(axis=0)
        surv_m = mu_m >= 1e-3
        model_cv = np.mean(sig_m[surv_m] / (0.01 + mu_m[surv_m])) if surv_m.any() else 0.0
        
        # 4. Empirical Late-Stage Species CV
        late_idx = np.where(day_nums >= 7)[0]
        late_exp = true_rel[late_idx][:, mask_np]
        mu_e = late_exp.mean(axis=0)
        sig_e = late_exp.std(axis=0)
        surv_e = mu_e >= 1e-3
        exp_cv = np.mean(sig_e[surv_e] / (0.01 + mu_e[surv_e])) if surv_e.any() else 0.0
        
        # 5. Empirical Biomass OD CV (CV >= 0.14)
        od_7_10 = data['biomass_od'][(nutrient, S, comm)].numpy()[:, 7:11]
        od_cv = np.mean(np.std(od_7_10, axis=1) / np.mean(od_7_10, axis=1))
        
        gt_chaotic = bool(exp_cv >= 0.25 or od_cv >= 0.14)
        records.append({
            'nutrient': nutrient,
            'S': S,
            'comm': comm,
            'r2_rel': r2_r,
            'r2_bio': r2_b,
            'model_cv': model_cv,
            'exp_cv': exp_cv,
            'od_cv': od_cv,
            'gt_chaotic': gt_chaotic
        })
        
    eval_df = pd.DataFrame(records)
    
    # Global Biomass stats
    all_true_od = np.array([data['biomass_od'][(n, s, c)].mean(dim=0).numpy() for (n, s, c) in all_keys]).flatten()
    all_pred_od = pred_biomass.flatten()
    r_bio_global, p_bio_global = pearsonr(all_true_od, all_pred_od)
    ss_res_glob = np.sum((all_true_od - all_pred_od)**2)
    ss_tot_glob = np.sum((all_true_od - np.mean(all_true_od))**2)
    r2_bio_global = 1.0 - ss_res_glob / ss_tot_glob
    
    log_m = np.log(eval_df['model_cv'] + 1e-4)
    log_e = np.log(eval_df['exp_cv'] + 1e-4)
    r_log, p_log = pearsonr(log_m, log_e)
    rho, p_rho = spearmanr(eval_df['model_cv'], eval_df['exp_cv'])
    r_lin, p_lin = pearsonr(eval_df['model_cv'], eval_df['exp_cv'])
    
    # Classification AUC and accuracy
    auc = roc_auc_score(eval_df['gt_chaotic'], eval_df['model_cv'])
    
    print("\n================ FULL EVALUATION SUMMARY ================")
    print(f"Mean Unrolled Relative Abundance Linear R^2: {eval_df['r2_rel'].mean():.4f}")
    print(f"Median Unrolled Relative Abundance Linear R^2: {eval_df['r2_rel'].median():.4f}")
    print(f"Global Biomass OD Pearson r: {r_bio_global:.4f} (p = {p_bio_global:.2e})")
    print(f"Global Biomass OD Linear R^2: {r2_bio_global:.4f}")
    print(f"Mean True OD: {all_true_od.mean():.4f} | Mean Pred OD: {all_pred_od.mean():.4f}")
    print(f"\nStability Dynamics Correlation (Day 0 -> 10 Autonomous Unroll):")
    print(f"  Log-Scale Pearson r: {r_log:.4f} (p = {p_log:.2e})")
    print(f"  Spearman rho:        {rho:.4f} (p = {p_rho:.2e})")
    print(f"  Linear Pearson r:    {r_lin:.4f} (p = {p_lin:.2e})")
    print(f"  ROC AUC for Chaotic Communities: {auc:.4f}")
    
    print("\nBreakdown by Nutrient Condition:")
    for nut in ['LN', 'MN', 'HN']:
        sub = eval_df[eval_df['nutrient'] == nut]
        r_sub, _ = pearsonr(np.log(sub['model_cv'] + 1e-4), np.log(sub['exp_cv'] + 1e-4))
        rho_sub, _ = spearmanr(sub['model_cv'], sub['exp_cv'])
        sub_idxs = [k for k, key in enumerate(all_keys) if key[0] == nut]
        bio_sub_t = all_true_od.reshape(99, 11)[sub_idxs].flatten()
        bio_sub_p = all_pred_od.reshape(99, 11)[sub_idxs].flatten()
        r_bio_sub, _ = pearsonr(bio_sub_t, bio_sub_p)
        print(f"  {nut}: Rel R^2 = {sub['r2_rel'].mean():.3f} | Bio r = {r_bio_sub:.3f} | Log r = {r_sub:.3f} | Spearman rho = {rho_sub:.3f}")
        
    # Save checkpoint and results
    os.makedirs('Data', exist_ok=True)
    os.makedirs('Results', exist_ok=True)
    torch.save(model.state_dict(), 'Data/Gore_Global_ODE_Model_GeneralizedKL.pth')
    eval_df.to_csv('Results/Gore_GeneralizedKL_Summary.csv', index=False)
    print("\nSaved checkpoint to Data/Gore_Global_ODE_Model_GeneralizedKL.pth")
    print("Saved evaluation table to Results/Gore_GeneralizedKL_Summary.csv")
    return model, eval_df

if __name__ == '__main__':
    train_generalized_kl()

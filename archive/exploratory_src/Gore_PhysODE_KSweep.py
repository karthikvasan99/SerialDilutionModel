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
from sklearn.metrics import roc_auc_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------------------------------------------
# 1. Pure Consumer-Resource ODE with K-Adaptive Scaling
# -------------------------------------------------------------
class AdaptivePureConsumerResourceODE(nn.Module):
    def __init__(self, N=48, K=7):
        super().__init__()
        self.N = N
        self.K = K
        # Scale initial Theta by 2.0 / K to keep initial growth rate ~2.0 across all K
        init_scale = 2.0 / float(K)
        self.Theta = nn.Parameter(torch.randn(N, K) * 0.05 + init_scale)
        self.W = nn.Parameter(torch.randn(K, K) * 0.02 + torch.eye(K) * 0.2)
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
    def __init__(self, ode_func, num_conditions=3, K=7, dilution_factor=30.0, dispersal_rate=1e-6):
        super().__init__()
        self.ode_func = ode_func
        self.K = K
        self.dilution_factor = dilution_factor
        self.dispersal_rate = dispersal_rate
        
        # Monotonic resource pools: LN < MN < HN
        self.a_ln = nn.Parameter(torch.randn(K) * 0.1 + 0.5)
        self.delta_mn = nn.Parameter(torch.randn(K) * 0.1 + 0.5)
        self.delta_hn = nn.Parameter(torch.randn(K) * 0.1 + 0.5)
        self.yield_raw = nn.Parameter(torch.zeros(num_conditions))
        
    def get_R0(self):
        r_ln = F.softplus(self.a_ln)
        r_mn = r_ln + F.softplus(self.delta_mn)
        r_hn = r_mn + F.softplus(self.delta_hn)
        return torch.stack([r_ln, r_mn, r_hn])
        
    def step_forward(self, N_prev, cond_idx, mask, delta_t=0.10):
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
# 2. Dr. Dixit's Generalized KL Loss (lambda_bio = 1.0)
# -------------------------------------------------------------
def dixit_generalized_kl(y_true, y_pred, mask, eps=1e-8):
    y_safe = torch.clamp(y_true, min=0.0)
    pred_safe = torch.clamp(y_pred, min=eps)
    y_for_log = torch.clamp(y_safe, min=eps)
    
    term1 = torch.where(y_safe > 0, y_safe * torch.log(y_for_log / pred_safe), torch.zeros_like(y_safe))
    div = (term1 - y_safe + pred_safe) * mask
    return div.sum(dim=-1).mean()

# -------------------------------------------------------------
# 3. Training and Evaluation Function
# -------------------------------------------------------------
def train_and_eval_cfg(cfg, dataset):
    set_seed(42)
    name = cfg['name']
    K = cfg['K']
    l2_reg = cfg.get('l2_reg', 0.002)
    delta_t = cfg.get('delta_t', 0.10)
    epochs = cfg.get('epochs', 200)
    lr = cfg.get('lr', 0.05)
    tf_min = cfg.get('tf_min', 0.10)
    
    b11_N = dataset['b11_N']
    b11_c = dataset['b11_c']
    b11_m = dataset['b11_m']
    b4_N = dataset['b4_N']
    b4_c = dataset['b4_c']
    b4_m = dataset['b4_m']
    all_N0 = dataset['all_N0']
    all_c = dataset['all_c']
    all_m = dataset['all_m']
    all_keys = dataset['all_keys']
    raw_data = dataset['raw_data']
    
    ode_func = AdaptivePureConsumerResourceODE(N=48, K=K)
    model = HierarchicalPhysicalNeuralODE(ode_func, num_conditions=3, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        raw_tf = 1.0 - (epoch / (epochs * 0.75))
        tf_ratio = max(tf_min, min(1.0, raw_tf))
        
        # Batch 11
        N_curr = b11_N[:, 0, :]
        loss11 = 0.0
        for t in range(1, 11):
            N_pred = model.step_forward(N_curr, b11_c, b11_m, delta_t=delta_t)
            target = b11_N[:, t, :]
            loss11 += dixit_generalized_kl(target, N_pred, b11_m)
            N_curr = tf_ratio * target + (1.0 - tf_ratio) * N_pred
            
        # Batch 4
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
                
        if l2_reg > 0:
            reg_penalty = l2_reg * (model.ode_func.Theta.pow(2.0).sum() + model.ode_func.W.pow(2.0).sum())
        else:
            reg_penalty = 0.0
            
        loss = (loss11 / 10.0) + (loss4 / 3.0) + reg_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    train_time = time.time() - start_time
    
    # Autonomous Unrolled Evaluation (Zero Teacher Forcing)
    model.eval()
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
        
    r2_rel_list, r2_bio_list, model_cvs, exp_cvs, od_cvs = [], [], [], [], []
    for i, (nutrient, S, comm) in enumerate(all_keys):
        mask_np = all_m[i].cpu().numpy().astype(bool)
        true_od = raw_data['biomass_od'][(nutrient, S, comm)].mean(dim=0).numpy()
        pred_od = pred_biomass[i]
        ss_res_b = np.sum((true_od - pred_od)**2)
        ss_tot_b = np.sum((true_od - np.mean(true_od))**2)
        r2_bio_list.append(1.0 - ss_res_b / (ss_tot_b + 1e-9))
        
        days = raw_data['timepoints'][nutrient][(S, comm)]
        day_nums = np.array([int(d.replace('Day', '')) for d in days])
        true_rel = raw_data['relative_abundance'][(nutrient, S, comm)].numpy()
        sub_pred_rel = pred_rel[i, day_nums]
        y_t = true_rel[:, mask_np].flatten()
        y_p = sub_pred_rel[:, mask_np].flatten()
        ss_res_r = np.sum((y_t - y_p)**2)
        ss_tot_r = np.sum((y_t - np.mean(y_t))**2)
        r2_rel_list.append(1.0 - ss_res_r / (ss_tot_r + 1e-9))
        
        late_model = pred_rel[i, 7:11][:, mask_np]
        mu_m = late_model.mean(axis=0)
        sig_m = late_model.std(axis=0)
        surv_m = mu_m >= 1e-3
        m_cv = np.mean(sig_m[surv_m] / (0.01 + mu_m[surv_m])) if surv_m.any() else 0.0
        model_cvs.append(m_cv)
        
        late_idx = np.where(day_nums >= 7)[0]
        late_exp = true_rel[late_idx][:, mask_np]
        mu_e = late_exp.mean(axis=0)
        sig_e = late_exp.std(axis=0)
        surv_e = mu_e >= 1e-3
        e_cv = np.mean(sig_e[surv_e] / (0.01 + mu_e[surv_e])) if surv_e.any() else 0.0
        exp_cvs.append(e_cv)
        
        od_7_10 = raw_data['biomass_od'][(nutrient, S, comm)].numpy()[:, 7:11]
        od_cv = np.mean(np.std(od_7_10, axis=1) / np.mean(od_7_10, axis=1))
        od_cvs.append(od_cv)
        
    model_cvs = np.array(model_cvs)
    exp_cvs = np.array(exp_cvs)
    od_cvs = np.array(od_cvs)
    
    all_true_od = np.array([raw_data['biomass_od'][(n, s, c)].mean(dim=0).numpy() for (n, s, c) in all_keys]).flatten()
    all_pred_od = pred_biomass.flatten()
    r_bio_global, _ = pearsonr(all_true_od, all_pred_od)
    
    log_m = np.log(model_cvs + 1e-4)
    log_e = np.log(exp_cvs + 1e-4)
    r_log, _ = pearsonr(log_m, log_e)
    rho, _ = spearmanr(model_cvs, exp_cvs)
    
    gt_chaotic = (exp_cvs >= 0.25) | (od_cvs >= 0.14)
    try:
        auc = roc_auc_score(gt_chaotic, model_cvs)
    except Exception:
        auc = 0.5
        
    cond_metrics = {}
    for nut in ['LN', 'MN', 'HN']:
        idxs = [k for k, key in enumerate(all_keys) if key[0] == nut]
        r_sub, _ = pearsonr(log_m[idxs], log_e[idxs])
        rho_sub, _ = spearmanr(model_cvs[idxs], exp_cvs[idxs])
        rel_sub = np.mean([r2_rel_list[k] for k in idxs])
        bio_sub_t = all_true_od.reshape(99, 11)[idxs].flatten()
        bio_sub_p = all_pred_od.reshape(99, 11)[idxs].flatten()
        r_bio_sub, _ = pearsonr(bio_sub_t, bio_sub_p)
        cond_metrics[nut] = {
            'rel_r2': rel_sub,
            'bio_r': r_bio_sub,
            'log_r': r_sub,
            'spearman_rho': rho_sub
        }
        
    summary = {
        'name': name,
        'K': K,
        'l2_reg': l2_reg,
        'delta_t': delta_t,
        'epochs': epochs,
        'lr': lr,
        'tf_min': tf_min,
        'train_time': round(train_time, 1),
        'mean_rel_r2': round(float(np.mean(r2_rel_list)), 4),
        'median_rel_r2': round(float(np.median(r2_rel_list)), 4),
        'global_bio_r': round(float(r_bio_global), 4),
        'log_r_stability': round(float(r_log), 4),
        'spearman_rho': round(float(rho), 4),
        'auc_chaos': round(float(auc), 4),
        'ln_rel_r2': round(cond_metrics['LN']['rel_r2'], 3),
        'mn_rel_r2': round(cond_metrics['MN']['rel_r2'], 3),
        'hn_rel_r2': round(cond_metrics['HN']['rel_r2'], 3),
        'ln_log_r': round(cond_metrics['LN']['log_r'], 3),
        'mn_log_r': round(cond_metrics['MN']['log_r'], 3),
        'hn_log_r': round(cond_metrics['HN']['log_r'], 3),
        'hn_rho': round(cond_metrics['HN']['spearman_rho'], 3),
    }
    return summary, model

# -------------------------------------------------------------
# 4. Main K, L2, Delta_t, Epochs Grid Sweep
# -------------------------------------------------------------
def run_k_hyperparameter_sweep():
    print(f"=== Starting Phys-ODE Grid Sweep over K, L2, delta_t, epochs ===")
    print(f"Parameters fixed per user instruction: lambda_bio=1.0, tf_min=0.10, LR=0.05")
    print(f"Device: {device}")
    
    data_path = 'Data/Gore_processed_data.pt'
    data = torch.load(data_path, weights_only=False)
    cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}
    
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
            
    dataset = {
        'b11_N': torch.stack(b11_N).to(device),
        'b11_c': torch.tensor(b11_c, device=device),
        'b11_m': torch.stack(b11_m).to(device),
        'b4_N': torch.stack(b4_N).to(device),
        'b4_c': torch.tensor(b4_c, device=device),
        'b4_m': torch.stack(b4_m).to(device),
        'all_N0': torch.cat([torch.stack(b11_N)[:, 0, :], torch.stack(b4_N)[:, 0, :]], dim=0).to(device),
        'all_c': torch.cat([torch.tensor(b11_c), torch.tensor(b4_c)], dim=0).to(device),
        'all_m': torch.cat([torch.stack(b11_m), torch.stack(b4_m)], dim=0).to(device),
        'all_keys': b11_keys + b4_keys,
        'raw_data': data
    }
    
    # 14 systematic configurations spanning K, L2, delta_t, epochs
    configs = [
        # --- Axis 1: K Dimension Sweep (K = 3, 5, 7, 10, 14) ---
        {'name': '1_K3',         'K': 3,  'l2_reg': 0.002,  'delta_t': 0.10, 'epochs': 200, 'lr': 0.05, 'tf_min': 0.10},
        {'name': '2_K5',         'K': 5,  'l2_reg': 0.002,  'delta_t': 0.10, 'epochs': 200, 'lr': 0.05, 'tf_min': 0.10},
        {'name': '3_K7_Base',    'K': 7,  'l2_reg': 0.002,  'delta_t': 0.10, 'epochs': 200, 'lr': 0.05, 'tf_min': 0.10},
        {'name': '4_K10',        'K': 10, 'l2_reg': 0.002,  'delta_t': 0.10, 'epochs': 200, 'lr': 0.05, 'tf_min': 0.10},
        {'name': '5_K14',        'K': 14, 'l2_reg': 0.002,  'delta_t': 0.10, 'epochs': 200, 'lr': 0.05, 'tf_min': 0.10},
        
        # --- Axis 2: L2 Regularization Penalty Sweep ---
        {'name': '6_L2_0.0',     'K': 7,  'l2_reg': 0.000,  'delta_t': 0.10, 'epochs': 200, 'lr': 0.05, 'tf_min': 0.10},
        {'name': '7_L2_0.0005',  'K': 7,  'l2_reg': 0.0005, 'delta_t': 0.10, 'epochs': 200, 'lr': 0.05, 'tf_min': 0.10},
        {'name': '8_L2_0.01',    'K': 7,  'l2_reg': 0.010,  'delta_t': 0.10, 'epochs': 200, 'lr': 0.05, 'tf_min': 0.10},
        
        # --- Axis 3: Delta_t (ODE Step Size) Sweep ---
        {'name': '9_Step_0.05',  'K': 7,  'l2_reg': 0.002,  'delta_t': 0.05, 'epochs': 200, 'lr': 0.05, 'tf_min': 0.10},
        {'name': '10_Step_0.20', 'K': 7,  'l2_reg': 0.002,  'delta_t': 0.20, 'epochs': 200, 'lr': 0.05, 'tf_min': 0.10},
        
        # --- Axis 4: Epochs Horizon Sweep ---
        {'name': '11_Ep_150',    'K': 7,  'l2_reg': 0.002,  'delta_t': 0.10, 'epochs': 150, 'lr': 0.05, 'tf_min': 0.10},
        {'name': '12_Ep_250',    'K': 7,  'l2_reg': 0.002,  'delta_t': 0.10, 'epochs': 250, 'lr': 0.05, 'tf_min': 0.10},
        {'name': '13_Ep_350',    'K': 7,  'l2_reg': 0.002,  'delta_t': 0.10, 'epochs': 350, 'lr': 0.05, 'tf_min': 0.10},
        
        # --- Combinations ---
        {'name': '14_K10_L2_0_Ep250', 'K': 10, 'l2_reg': 0.000, 'delta_t': 0.10, 'epochs': 250, 'lr': 0.05, 'tf_min': 0.10},
    ]
    
    results = []
    best_score = -999.0
    best_model = None
    best_cfg_name = ""
    
    for idx, cfg in enumerate(configs, 1):
        print(f"\n[{idx:2d}/{len(configs)}] Evaluating Config: {cfg['name']}")
        print(f"     Params: K={cfg['K']}, l2={cfg['l2_reg']}, dt={cfg['delta_t']}, epochs={cfg['epochs']}, lr={cfg['lr']}")
        summary, model = train_and_eval_cfg(cfg, dataset)
        results.append(summary)
        
        composite_score = summary['mean_rel_r2'] + 0.5 * summary['log_r_stability'] + 0.5 * summary['hn_log_r']
        print(f"     => Rel R^2: {summary['mean_rel_r2']:.4f} | Bio r: {summary['global_bio_r']:.4f} | Log r: {summary['log_r_stability']:.4f} | HN Log r: {summary['hn_log_r']:.3f} | Score: {composite_score:.4f}")
        
        if composite_score > best_score:
            best_score = composite_score
            best_model = model
            best_cfg_name = cfg['name']
            torch.save(model.state_dict(), 'Data/Gore_Global_ODE_Model_PhysODE_BestK.pth')
            print(f"     *** NEW BEST Phys-ODE MODEL: {best_cfg_name} (Saved to Data/Gore_Global_ODE_Model_PhysODE_BestK.pth) ***")
            
    df_results = pd.DataFrame(results)
    os.makedirs('Results', exist_ok=True)
    df_results.to_csv('Results/Gore_PhysODE_KSweep_Summary.csv', index=False)
    print("\n================ SWEEP COMPLETE ================")
    print(f"Saved full results to Results/Gore_PhysODE_KSweep_Summary.csv")
    print(f"Best configuration: {best_cfg_name} with score {best_score:.4f}")
    print(df_results[['name', 'K', 'l2_reg', 'delta_t', 'epochs', 'mean_rel_r2', 'global_bio_r', 'log_r_stability', 'hn_log_r']].to_string())

if __name__ == '__main__':
    run_k_hyperparameter_sweep()

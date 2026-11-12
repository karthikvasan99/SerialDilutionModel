import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}")

data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location=device, weights_only=False)

no_c_tensor = data['no_c_tensor'].to(device)       # (3, 6, 50)
plain_c_tensor = data['plain_c_tensor'].to(device) # (3, 6, 50)
plain_s_tensor = data['plain_s_tensor'].to(device) # (3, 6, 50)
mucin_c_tensor = data['mucin_c_tensor'].to(device) # (3, 6, 50)
mucin_s_tensor = data['mucin_s_tensor'].to(device) # (3, 6, 50)

N = no_c_tensor.size(-1) # 50 strains
vessels = data['vessels']

class DualThetaUnifiedCRM(nn.Module):
    def __init__(self, N, K, c_w=0.85, eta_mode='fixed_1_30'):
        super().__init__()
        self.N, self.K = N, K
        self.c_w = c_w
        self.eta_mode = eta_mode
        
        self.theta_c_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        self.theta_s_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.1)
        self.k_shed_raw = nn.Parameter(torch.randn(N) * 0.1 - 1.0)
        self.k_attach_raw = nn.Parameter(torch.randn(2, N) * 0.1 - 1.0)
        self.a = nn.Parameter(torch.randn(3, K) * 0.5 + 2.0)
        self.b0_liq_raw = nn.Parameter(torch.tensor(2.0))
        self.b0_c_raw = nn.Parameter(torch.tensor(2.0))
        self.b0_s_raw = nn.Parameter(torch.tensor(2.0))

    def get_params(self, enforce_w_zero=False):
        Theta_c = F.softplus(self.theta_c_raw)
        Theta_s = F.softplus(self.theta_s_raw)
        alpha = F.softplus(self.alpha_raw)
        k_shed = F.softplus(self.k_shed_raw)
        k_attach = F.softplus(self.k_attach_raw)
        R0 = F.softplus(self.a)
        b0_liq = F.softplus(self.b0_liq_raw)
        b0_c = F.softplus(self.b0_c_raw)
        b0_s = F.softplus(self.b0_s_raw)
        eta = torch.tensor(1.0 / 30.0, device=self.theta_c_raw.device)
        
        if enforce_w_zero:
            W = torch.zeros(self.K, self.K, device=self.w_raw.device)
        else:
            W_sp = F.softplus(self.w_raw) * (1.0 - torch.eye(self.K, device=self.w_raw.device))
            row_sums = W_sp.sum(dim=-1, keepdim=True)
            scale = torch.clamp(self.c_w / torch.clamp(row_sums, min=1e-6), max=1.0)
            W = W_sp * scale
        return Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta

    def rollout_no_carrier(self, n_c_sub, Theta_s, W, alpha, R0_liq, b0_liq, delta_t=0.1, tf_ratio=0.0):
        N_curr = n_c_sub[:, 0, :] * b0_liq
        R0_batch = R0_liq.unsqueeze(0).expand(n_c_sub.size(0), -1)
        preds = []
        t_span = torch.linspace(0.0, 1.0, int(1.0 / delta_t) + 1, device=device)
        for p in range(1, 6):
            if tf_ratio > 0.0 and self.training:
                N_transferred = tf_ratio * (n_c_sub[:, p-1, :] * N_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_curr
            else:
                N_transferred = N_curr
            N0 = torch.clamp(N_transferred / 100.0 + 1e-6, min=0.0)
            S0 = torch.cat([N0, R0_batch], dim=-1)
            def f(t, S):
                Nt = torch.clamp(S[..., :self.N], min=0.0)
                Rt = torch.clamp(S[..., self.N:], min=0.0)
                dNt = Nt * torch.matmul(Rt, Theta_s.t()) - alpha * (Nt**2)
                flux = Rt * torch.matmul(Nt, Theta_s)
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNt, dRt], dim=-1)
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :self.N], min=0.0)
            preds.append(N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-8))
            N_curr = N_final
        return torch.stack(preds, dim=1)

    def rollout_carrier(self, c_sub, s_sub, Theta_c, Theta_s, W, alpha, k_shed, k_att, R0_cond, b0_c, b0_s, eta, delta_t=0.1, tf_ratio=0.0):
        N_c_curr = c_sub[:, 0, :] * b0_c
        N_s_curr = s_sub[:, 0, :] * b0_s
        R0_batch = R0_cond.unsqueeze(0).expand(c_sub.size(0), -1)
        preds_c, preds_s = [], []
        t_span = torch.linspace(0.0, 1.0, int(1.0 / delta_t) + 1, device=device)
        for p in range(1, 6):
            if tf_ratio > 0.0 and self.training:
                N_c_transferred = tf_ratio * (c_sub[:, p-1, :] * N_c_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_c_curr
                N_s_transferred = tf_ratio * (s_sub[:, p-1, :] * N_s_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_s_curr
            else:
                N_c_transferred = N_c_curr
                N_s_transferred = N_s_curr
            N_c_0 = torch.clamp(0.2 * N_c_transferred + 1e-6, min=0.0)
            N_s_0 = torch.clamp(eta * N_s_transferred + 1e-6, min=0.0)
            S0 = torch.cat([N_c_0, N_s_0, R0_batch], dim=-1)
            def f(t, S):
                Nc = torch.clamp(S[..., :self.N], min=0.0)
                Ns = torch.clamp(S[..., self.N:2*self.N], min=0.0)
                Rt = torch.clamp(S[..., 2*self.N:], min=0.0)
                growth_c = torch.matmul(Rt, Theta_c.t())
                growth_s = torch.matmul(Rt, Theta_s.t())
                dNc = Nc * growth_c - alpha * (Nc**2) - k_shed * Nc + k_att * Ns
                dNs = Ns * growth_s - alpha * (Ns**2) + k_shed * Nc - k_att * Ns
                flux = Rt * (torch.matmul(Nc, Theta_c) + torch.matmul(Ns, Theta_s))
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNc, dNs, dRt], dim=-1)
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            Nc_fin = torch.clamp(S_t[-1, :, :self.N], min=0.0)
            Ns_fin = torch.clamp(S_t[-1, :, self.N:2*self.N], min=0.0)
            preds_c.append(Nc_fin / (Nc_fin.sum(dim=-1, keepdim=True) + 1e-8))
            preds_s.append(Ns_fin / (Ns_fin.sum(dim=-1, keepdim=True) + 1e-8))
            N_c_curr = Nc_fin
            N_s_curr = Ns_fin
        return torch.stack(preds_c, dim=1), torch.stack(preds_s, dim=1)

def evaluate_fold(model, test_idx):
    with torch.no_grad():
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params()
        p_no = model.rollout_no_carrier(no_c_tensor[test_idx:test_idx+1], Theta_s, W, alpha, R0[0], b0_liq, tf_ratio=0.0).cpu().numpy().flatten()
        t_no = no_c_tensor[test_idx:test_idx+1, 1:, :].cpu().numpy().flatten()
        
        p_pl_c, p_pl_s = model.rollout_carrier(plain_c_tensor[test_idx:test_idx+1], plain_s_tensor[test_idx:test_idx+1], Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta, tf_ratio=0.0)
        p_pl = np.concatenate([p_pl_c.cpu().numpy().flatten(), p_pl_s.cpu().numpy().flatten()])
        t_pl = np.concatenate([plain_c_tensor[test_idx:test_idx+1, 1:, :].cpu().numpy().flatten(), plain_s_tensor[test_idx:test_idx+1, 1:, :].cpu().numpy().flatten()])
        
        p_mu_c, p_mu_s = model.rollout_carrier(mucin_c_tensor[test_idx:test_idx+1], mucin_s_tensor[test_idx:test_idx+1], Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta, tf_ratio=0.0)
        p_mu = np.concatenate([p_mu_c.cpu().numpy().flatten(), p_mu_s.cpu().numpy().flatten()])
        t_mu = np.concatenate([mucin_c_tensor[test_idx:test_idx+1, 1:, :].cpu().numpy().flatten(), mucin_s_tensor[test_idx:test_idx+1, 1:, :].cpu().numpy().flatten()])
        
        all_pred = np.concatenate([p_no, p_pl, p_mu])
        all_true = np.concatenate([t_no, t_pl, t_mu])
        r2_global = 1.0 - np.sum((all_true - all_pred)**2) / np.sum((all_true - np.mean(all_true))**2)
        mae = np.mean(np.abs(all_true - all_pred))
        
        r_mucin = (k_attach[1] / k_shed).cpu().numpy()
        log_rm = np.log10(r_mucin)
        th_c_norm = torch.norm(Theta_c, dim=-1).cpu().numpy()
        th_s_norm = torch.norm(Theta_s, dim=-1).cpu().numpy()
        th_ratio = th_c_norm / th_s_norm
        from scipy.stats import pearsonr
        r_tradeoff, _ = pearsonr(log_rm, th_ratio)
        
        return r2_global, mae, r_tradeoff

def train_crm_fold_opt(K, c_w, lr, weight_decay, epochs, tf_decay, train_indices):
    torch.manual_seed(42)
    model = DualThetaUnifiedCRM(N=N, K=K, c_w=c_w, eta_mode='fixed_1_30').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    no_tr = no_c_tensor[train_indices]
    pl_c_tr = plain_c_tensor[train_indices]
    pl_s_tr = plain_s_tensor[train_indices]
    mu_c_tr = mucin_c_tensor[train_indices]
    mu_s_tr = mucin_s_tensor[train_indices]
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params()
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * tf_decay)))
        
        p_no = model.rollout_no_carrier(no_tr, Theta_s, W, alpha, R0[0], b0_liq, tf_ratio=tf_ratio)
        loss_no = torch.sum(no_tr[:, 1:, :] * torch.log(no_tr[:, 1:, :] / torch.clamp(p_no, min=1e-9) + 1e-12), dim=-1).mean()
        
        p_pl_c, p_pl_s = model.rollout_carrier(pl_c_tr, pl_s_tr, Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta, tf_ratio=tf_ratio)
        loss_pl = torch.sum(pl_c_tr[:, 1:, :] * torch.log(pl_c_tr[:, 1:, :] / torch.clamp(p_pl_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                  torch.sum(pl_s_tr[:, 1:, :] * torch.log(pl_s_tr[:, 1:, :] / torch.clamp(p_pl_s, min=1e-9) + 1e-12), dim=-1).mean()
                  
        p_mu_c, p_mu_s = model.rollout_carrier(mu_c_tr, mu_s_tr, Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta, tf_ratio=tf_ratio)
        loss_mu = torch.sum(mu_c_tr[:, 1:, :] * torch.log(mu_c_tr[:, 1:, :] / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                  torch.sum(mu_s_tr[:, 1:, :] * torch.log(mu_s_tr[:, 1:, :] / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
                  
        loss = loss_no + loss_pl + loss_mu
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    return model

print("="*80)
print("STAGE 2: OPTIMIZATION & SCHEDULE FINE-TUNING (3-FOLD CV)")
print("="*80)

folds = [
    ('Fold 1 (Test R3)', [0, 1], 2),
    ('Fold 2 (Test R2)', [0, 2], 1),
    ('Fold 3 (Test R1)', [1, 2], 0),
]

opt_configs = []
for K in [3, 4]:
    for c_w in [0.5, 0.85]:
        for lr in [0.010, 0.015]:
            for epochs in [200, 250]:
                opt_configs.append({
                    'K': K, 'c_w': c_w, 'lr': lr, 'weight_decay': 0.0,
                    'epochs': epochs, 'tf_decay': 0.75
                })

results = []

for i, cfg in enumerate(opt_configs, 1):
    t0 = time.time()
    fold_r2 = []
    fold_mae = []
    fold_tradeoffs = []
    
    for fold_name, tr_idx, te_idx in folds:
        model = train_crm_fold_opt(
            cfg['K'], cfg['c_w'], cfg['lr'], cfg['weight_decay'],
            cfg['epochs'], cfg['tf_decay'], tr_idx
        )
        r2_glob, mae, r_to = evaluate_fold(model, te_idx)
        fold_r2.append(r2_glob)
        fold_mae.append(mae)
        fold_tradeoffs.append(r_to)
        
    mean_r2 = np.mean(fold_r2)
    std_r2 = np.std(fold_r2)
    mean_mae = np.mean(fold_mae)
    mean_to = np.mean(fold_tradeoffs)
    elapsed = time.time() - t0
    
    print(f"[{i:2d}/{len(opt_configs):2d}] K={cfg['K']} | c_w={cfg['c_w']} | lr={cfg['lr']:.3f} | ep={cfg['epochs']} --> "
          f"Held-Out R2: {mean_r2:.4f} +/- {std_r2:.4f} | MAE: {mean_mae:.4f} | Tradeoff r: {mean_to:+.2f} ({elapsed:.1f}s)")
          
    results.append({
        'K': cfg['K'],
        'c_w': cfg['c_w'],
        'lr': cfg['lr'],
        'epochs': cfg['epochs'],
        'Mean_HeldOut_R2': mean_r2,
        'Std_HeldOut_R2': std_r2,
        'Mean_HeldOut_MAE': mean_mae,
        'Mean_Tradeoff_r': mean_to
    })

df_opt = pd.DataFrame(results).sort_values(by='Mean_HeldOut_R2', ascending=False)
print("\n" + "="*80)
print("STAGE 2 TOP OPTIMIZATION CONFIGURATIONS:")
print("="*80)
print(df_opt.head(10).to_string(index=False))

out_csv = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Stage2_Search_Results.csv')
df_opt.to_csv(out_csv, index=False)
print(f"\nSaved full results to {out_csv}")

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}", flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location=device, weights_only=False)

no_c_tensor = data['no_c_tensor'].to(device)       # (3, 6, 50)
mucin_c_tensor = data['mucin_c_tensor'].to(device) # (3, 6, 50)
mucin_s_tensor = data['mucin_s_tensor'].to(device) # (3, 6, 50)
strains = data['strains']
N = 50

# Load inoculum P0
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv'))
sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
df['Relative_Abundance'] = df['Read_Fraction'] / 100.0 if sums.mean() > 50 else df['Read_Fraction']
inoc = df[df['sampleType'] == 'inoculum'].pivot_table(index='replicate', columns='Strain_Name', values='Relative_Abundance', fill_value=0.0).reindex(columns=strains, fill_value=0.0)
inoc_norm = inoc.div(inoc.sum(axis=1), axis=0)
p0 = torch.tensor(inoc_norm.mean(axis=0).values, dtype=torch.float32, device=device).unsqueeze(0).expand(3, -1)

def compute_kl(t, p, eps=1e-12, clamp_min=1e-9):
    ratio = torch.clamp(t, min=eps) / torch.clamp(p, min=clamp_min)
    kl = torch.sum(torch.where(t > 0, t * torch.log(ratio), torch.zeros_like(t)), dim=-1)
    return kl.mean().item()

class ModularCRM_FixedB0(nn.Module):
    def __init__(self, N=50, K=3, c_w=0.50, b0_val=1.60):
        super().__init__()
        self.N, self.K = N, K
        self.c_w = c_w
        self.register_buffer('b0', torch.tensor(float(b0_val)))
        
        self.theta_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.1)
        
        # Strain-specific attachment and detachment rates (50 each)
        self.k_attach_raw = nn.Parameter(torch.randn(N) * 0.1 - 1.0)
        self.k_detach_raw = nn.Parameter(torch.randn(N) * 0.1 - 1.0)
        
        self.a_base = nn.Parameter(torch.randn(K) * 0.5 + 2.0)
        self.a_mucin = nn.Parameter(torch.randn(K) * 0.5 + 2.0)

    def get_params(self):
        Theta = F.softplus(self.theta_raw)
        alpha = F.softplus(self.alpha_raw)
        k_attach = F.softplus(self.k_attach_raw)
        k_detach = F.softplus(self.k_detach_raw)
        R0_base = F.softplus(self.a_base)
        R0_mucin = F.softplus(self.a_mucin)
        W_sp = F.softplus(self.w_raw) * (1.0 - torch.eye(self.K, device=device))
        scale = torch.clamp(self.c_w / torch.clamp(W_sp.sum(dim=-1, keepdim=True), min=1e-6), max=1.0)
        W = W_sp * scale
        return Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin

    def forward_carrier(self, p0_vec, is_mucin=True, tf_targets=None, tf_ratio=0.0):
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin = self.get_params()
        R0_cond = R0_mucin if is_mucin else R0_base
        R0_batch = R0_cond.unsqueeze(0).expand(p0_vec.size(0), -1)
        
        preds_c, preds_s = [], []
        t_span = torch.linspace(0.0, 1.0, 11, device=device)
        Nc_curr = torch.zeros_like(p0_vec)
        Ns_curr = p0_vec * self.b0
        eta = 1.0 / 30.0
        
        for p in range(6):
            if p == 0:
                Nc_0 = Nc_curr
                Ns_0 = Ns_curr
            else:
                if tf_ratio > 0.0 and tf_targets is not None:
                    target_c, target_s = tf_targets
                    Nc_tf = tf_ratio * (target_c[:, p-1, :] * Nc_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * Nc_curr
                    Ns_tf = tf_ratio * (target_s[:, p-1, :] * Ns_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * Ns_curr
                else:
                    Nc_tf = Nc_curr
                    Ns_tf = Ns_curr
                Nc_0 = torch.clamp(0.2 * Nc_tf + 1e-6, min=0.0)
                Ns_0 = torch.clamp(eta * Ns_tf + 1e-6, min=0.0)
                
            S0 = torch.cat([Nc_0, Ns_0, R0_batch], dim=-1)
            def f(t, S):
                Nc = torch.clamp(S[..., :50], min=0.0)
                Ns = torch.clamp(S[..., 50:100], min=0.0)
                Rt = torch.clamp(S[..., 100:], min=0.0)
                dNc = Nc * torch.matmul(Rt, Theta.t()) - alpha * (Nc**2) - k_detach * Nc + k_attach * Ns
                dNs = Ns * torch.matmul(Rt, Theta.t()) - alpha * (Ns**2) + k_detach * Nc - k_attach * Ns
                flux = Rt * torch.matmul(Nc + Ns, Theta)
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNc, dNs, dRt], dim=-1)
                
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': 0.1})
            Nc_curr = torch.clamp(S_t[-1, :, :50], min=0.0)
            Ns_curr = torch.clamp(S_t[-1, :, 50:100], min=0.0)
            preds_c.append(Nc_curr / (Nc_curr.sum(dim=-1, keepdim=True) + 1e-8))
            preds_s.append(Ns_curr / (Ns_curr.sum(dim=-1, keepdim=True) + 1e-8))
            
        return torch.stack(preds_c, dim=1), torch.stack(preds_s, dim=1)

    def forward_no_carrier(self, p0_vec, tf_target=None, tf_ratio=0.0):
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin = self.get_params()
        R0_batch = R0_base.unsqueeze(0).expand(p0_vec.size(0), -1)
        
        preds = []
        t_span = torch.linspace(0.0, 1.0, 11, device=device)
        N_curr = p0_vec * self.b0
        eta = 1.0 / 100.0
        
        for p in range(6):
            if p == 0:
                N_0 = N_curr
            else:
                if tf_ratio > 0.0 and tf_target is not None:
                    N_tf = tf_ratio * (tf_target[:, p-1, :] * N_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_curr
                else:
                    N_tf = N_curr
                N_0 = torch.clamp(eta * N_tf + 1e-6, min=0.0)
                
            S0 = torch.cat([N_0, R0_batch], dim=-1)
            def f(t, S):
                Nt = torch.clamp(S[..., :50], min=0.0)
                Rt = torch.clamp(S[..., 50:], min=0.0)
                dNt = Nt * torch.matmul(Rt, Theta.t()) - alpha * (Nt**2)
                flux = Rt * torch.matmul(Nt, Theta)
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNt, dRt], dim=-1)
                
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': 0.1})
            N_curr = torch.clamp(S_t[-1, :, :50], min=0.0)
            preds.append(N_curr / (N_curr.sum(dim=-1, keepdim=True) + 1e-8))
            
        return torch.stack(preds, dim=1)

def evaluate_loocv(config, epochs=200, lr=0.015):
    K = config['K']
    c_w = config['c_w']
    b0_val = config['b0_val']
    reg_alpha = config['reg_alpha']
    reg_w = config['reg_w']
    
    val_kls, val_kls_no, val_kls_mu = [], [], []
    
    for val_vessel in range(3):
        train_vessels = [v for v in range(3) if v != val_vessel]
        
        p0_train = p0[train_vessels]
        p0_val = p0[val_vessel:val_vessel+1]
        
        no_train = no_c_tensor[train_vessels]
        no_val = no_c_tensor[val_vessel:val_vessel+1]
        
        mu_c_train = mucin_c_tensor[train_vessels]
        mu_s_train = mucin_s_tensor[train_vessels]
        mu_c_val = mucin_c_tensor[val_vessel:val_vessel+1]
        mu_s_val = mucin_s_tensor[val_vessel:val_vessel+1]
        
        torch.manual_seed(42 + val_vessel)
        model = ModularCRM_FixedB0(N=50, K=K, c_w=c_w, b0_val=b0_val).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
            
            p_no = model.forward_no_carrier(p0_train, tf_target=no_train, tf_ratio=tf_ratio)
            ratio_no = torch.clamp(no_train, min=1e-12) / torch.clamp(p_no, min=1e-9)
            loss = torch.sum(torch.where(no_train > 0, no_train * torch.log(ratio_no), torch.zeros_like(no_train)), dim=-1).mean()
            
            p_mu_c, p_mu_s = model.forward_carrier(p0_train, is_mucin=True, tf_targets=(mu_c_train, mu_s_train), tf_ratio=tf_ratio)
            ratio_mu_c = torch.clamp(mu_c_train, min=1e-12) / torch.clamp(p_mu_c, min=1e-9)
            ratio_mu_s = torch.clamp(mu_s_train, min=1e-12) / torch.clamp(p_mu_s, min=1e-9)
            loss += torch.sum(torch.where(mu_c_train > 0, mu_c_train * torch.log(ratio_mu_c), torch.zeros_like(mu_c_train)), dim=-1).mean() + \
                    torch.sum(torch.where(mu_s_train > 0, mu_s_train * torch.log(ratio_mu_s), torch.zeros_like(mu_s_train)), dim=-1).mean()
                    
            Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin = model.get_params()
            loss += reg_w * (Theta.pow(2).sum() + W.pow(2).sum()) + reg_alpha * alpha.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            pred_no_val = model.forward_no_carrier(p0_val, tf_ratio=0.0)
            pred_mu_c_val, pred_mu_s_val = model.forward_carrier(p0_val, is_mucin=True, tf_ratio=0.0)
            
            kl_no = compute_kl(no_val, pred_no_val)
            kl_mu = 0.5 * (compute_kl(mu_c_val, pred_mu_c_val) + compute_kl(mu_s_val, pred_mu_s_val))
            val_kls.append(0.5 * (kl_no + kl_mu))
            val_kls_no.append(kl_no)
            val_kls_mu.append(kl_mu)
            
    return {
        'K': K,
        'c_w': c_w,
        'b0_val': b0_val,
        'reg_alpha': reg_alpha,
        'reg_w': reg_w,
        'params': 50*K + K*K + 50 + 50 + 50 + 2*K,
        'mean_val_kl': float(np.mean(val_kls)),
        'std_val_kl': float(np.std(val_kls)),
        'val_kl_no': float(np.mean(val_kls_no)),
        'val_kl_mu': float(np.mean(val_kls_mu))
    }

if __name__ == '__main__':
    grid = [
        # --- Axis 1: Model Order K Sweep (b0=1.6, c_w=0.50, reg_alpha=0.010, reg_w=0.001) ---
        {'K': 2, 'c_w': 0.50, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 3, 'c_w': 0.50, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001}, # The classic R2=0.425 model baseline
        {'K': 4, 'c_w': 0.50, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 5, 'c_w': 0.50, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 6, 'c_w': 0.50, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
        
        # --- Axis 2: Inoculum Biomass b0 Sweep on K=3 and K=4 ---
        {'K': 3, 'c_w': 0.50, 'b0_val': 1.00, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 3, 'c_w': 0.50, 'b0_val': 1.30, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 3, 'c_w': 0.50, 'b0_val': 1.90, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 3, 'c_w': 0.50, 'b0_val': 2.20, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 4, 'c_w': 0.50, 'b0_val': 1.00, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 4, 'c_w': 0.50, 'b0_val': 1.30, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 4, 'c_w': 0.50, 'b0_val': 1.90, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 4, 'c_w': 0.50, 'b0_val': 2.20, 'reg_alpha': 0.010, 'reg_w': 0.001},
        
        # --- Axis 3: Coupling Budget c_w Sweep on K=3 and K=4 (b0=1.60) ---
        {'K': 3, 'c_w': 0.35, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 3, 'c_w': 0.65, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 3, 'c_w': 0.75, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 4, 'c_w': 0.35, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 4, 'c_w': 0.65, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
        {'K': 4, 'c_w': 0.75, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.001},
        
        # --- Axis 4: Self-Inhibition Regularization reg_alpha Sweep ---
        {'K': 3, 'c_w': 0.50, 'b0_val': 1.60, 'reg_alpha': 0.003, 'reg_w': 0.001},
        {'K': 3, 'c_w': 0.50, 'b0_val': 1.60, 'reg_alpha': 0.030, 'reg_w': 0.001},
        {'K': 4, 'c_w': 0.50, 'b0_val': 1.60, 'reg_alpha': 0.003, 'reg_w': 0.001},
        {'K': 4, 'c_w': 0.50, 'b0_val': 1.60, 'reg_alpha': 0.030, 'reg_w': 0.001},
        
        # --- Axis 5: Network Regularization reg_w Sweep ---
        {'K': 3, 'c_w': 0.50, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.0003},
        {'K': 3, 'c_w': 0.50, 'b0_val': 1.60, 'reg_alpha': 0.010, 'reg_w': 0.0030},
    ]
    
    print("\n" + "="*120, flush=True)
    print("THOROUGH LOOCV SEARCH: STRAIN-SPECIFIC MODEL WITH SHARED HYPERPARAMETER b0")
    print("Architecture: Strain-specific k_attach (50) + Strain-specific k_detach (50) + Fixed Scalar b0")
    print("Goal: Minimum LOOCV KL on No Carrier + Mucin Agar with NO hyperparameters at search boundaries")
    print("="*120, flush=True)
    print(f"{'Idx':>3} | {'K':>2} | {'c_w':>4} | {'b0':>4} | {'reg_a':>6} | {'reg_w':>6} | {'Params':>6} | {'LOOCV Comb KL':>14} | {'Std':>6} | {'Val NoCar':>10} | {'Val Mucin':>10}", flush=True)
    print("-"*120, flush=True)
    
    results = []
    csv_path = os.path.join(PROJECT_ROOT, 'Data', 'LOOCV_StrainSpecific_FixedB0_Search.csv')
    
    for idx, cfg in enumerate(grid, 1):
        res = evaluate_loocv(cfg, epochs=200, lr=0.015)
        results.append(res)
        print(f"{idx:3d} | {res['K']:2d} | {res['c_w']:4.2f} | {res['b0_val']:4.1f} | {res['reg_alpha']:6.3f} | {res['reg_w']:6.4f} | {res['params']:6d} | {res['mean_val_kl']:14.4f} | {res['std_val_kl']:6.4f} | {res['val_kl_no']:10.4f} | {res['val_kl_mu']:10.4f}", flush=True)
        pd.DataFrame(results).to_csv(csv_path, index=False)
        
    print("="*120, flush=True)
    df_res = pd.DataFrame(results)
    best_idx = df_res['mean_val_kl'].idxmin()
    best_row = df_res.loc[best_idx]
    print(f"\nOVERALL WINNING CONFIGURATION (FIXED HYPERPARAMETER b0):", flush=True)
    print(f"K = {int(best_row['K'])}, c_w = {best_row['c_w']:.2f}, b0 = {best_row['b0_val']:.2f}, reg_alpha = {best_row['reg_alpha']:.3f}, reg_w = {best_row['reg_w']:.4f}", flush=True)
    print(f"Validation KL = {best_row['mean_val_kl']:.4f} +/- {best_row['std_val_kl']:.4f} nats (NoCar: {best_row['val_kl_no']:.4f}, Mucin: {best_row['val_kl_mu']:.4f})", flush=True)
    print(f"Full results saved to {csv_path}", flush=True)

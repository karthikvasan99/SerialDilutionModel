import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}", flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location=device, weights_only=False)

no_c_tensor = data['no_c_tensor'].to(device)       # (3, 6, 50)
plain_c_tensor = data['plain_c_tensor'].to(device) # (3, 6, 50)
plain_s_tensor = data['plain_s_tensor'].to(device) # (3, 6, 50)
mucin_c_tensor = data['mucin_c_tensor'].to(device) # (3, 6, 50)
mucin_s_tensor = data['mucin_s_tensor'].to(device) # (3, 6, 50)
N = no_c_tensor.size(-1)
strains_keep = data['strains']

# Load inoculum P0
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv'))
sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
df['Relative_Abundance'] = df['Read_Fraction'] / 100.0 if sums.mean() > 50 else df['Read_Fraction']
inoc = df[df['sampleType'] == 'inoculum'].pivot_table(index='replicate', columns='Strain_Name', values='Relative_Abundance', fill_value=0.0).reindex(columns=strains_keep, fill_value=0.0)
inoc_norm = inoc.div(inoc.sum(axis=1), axis=0)
p0 = torch.tensor(inoc_norm.mean(axis=0).values, dtype=torch.float32, device=device).unsqueeze(0).expand(3, -1)

def compute_r2(true_vals, pred_vals):
    diff_sq = np.sum((true_vals - pred_vals)**2)
    tot_sq = np.sum((true_vals - np.mean(true_vals))**2)
    return 1.0 - diff_sq / tot_sq

def compute_kl(t, p, eps=1e-12, clamp_min=1e-9):
    ratio = torch.clamp(t, min=eps) / torch.clamp(p, min=clamp_min)
    kl = torch.sum(torch.where(t > 0, t * torch.log(ratio), torch.zeros_like(t)), dim=-1)
    return kl.mean().item()

class UnifiedTransportCRM(nn.Module):
    """
    Even more constrained model:
    - SAME k_attach for both plain and mucin
    - SAME k_detach for both plain and mucin
    - SAME Theta for carrier and supernatant (universal)
    - Tied R0:
        a_base -> for No Carrier and Plain Agar
        a_mucin -> for Mucin Agar
    - Zero diagonal on W, bounded by c_w
    """
    def __init__(self, N=50, K=5, c_w=0.50, fix_b0=None):
        super().__init__()
        self.N, self.K = N, K
        self.c_w = c_w
        self.fix_b0 = fix_b0
        
        self.theta_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.1)
        
        # Single universal attach & detach for BOTH carrier types!
        self.k_attach_raw = nn.Parameter(torch.randn(N) * 0.1 - 1.0)
        self.k_detach_raw = nn.Parameter(torch.randn(N) * 0.1 - 1.0)
        
        self.a_base = nn.Parameter(torch.randn(K) * 0.5 + 2.0)
        self.a_mucin = nn.Parameter(torch.randn(K) * 0.5 + 2.0)
        
        if fix_b0 is not None:
            self.register_buffer('b0_fixed', torch.tensor(float(fix_b0)))
        else:
            self.b0_raw = nn.Parameter(torch.tensor(1.6))
        
    def get_params(self):
        Theta = F.softplus(self.theta_raw)
        alpha = F.softplus(self.alpha_raw)
        k_attach = F.softplus(self.k_attach_raw)
        k_detach = F.softplus(self.k_detach_raw)
        
        R0_base = F.softplus(self.a_base)
        R0_mucin = F.softplus(self.a_mucin)
        b0 = self.b0_fixed if self.fix_b0 is not None else F.softplus(self.b0_raw)
        
        W_sp = F.softplus(self.w_raw) * (1.0 - torch.eye(self.K, device=device))
        scale = torch.clamp(self.c_w / torch.clamp(W_sp.sum(dim=-1, keepdim=True), min=1e-6), max=1.0)
        W = W_sp * scale
        
        return Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin, b0

    def rollout_carrier(self, p0_vec, condition='plain', delta_t=0.1, tf_ratio=0.0):
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin, b0 = self.get_params()
        R0_cond = R0_base if condition == 'plain' else R0_mucin
        R0_batch = R0_cond.unsqueeze(0).expand(p0_vec.size(0), -1)
        
        target_c = plain_c_tensor if condition == 'plain' else mucin_c_tensor
        target_s = plain_s_tensor if condition == 'plain' else mucin_s_tensor
        
        preds_c, preds_s = [], []
        t_span = torch.linspace(0.0, 1.0, 11, device=device)
        
        Nc_curr = torch.zeros_like(p0_vec)
        Ns_curr = p0_vec * b0
        eta = 1.0 / 30.0
        
        for p in range(6):
            if p == 0:
                Nc_0 = Nc_curr
                Ns_0 = Ns_curr
            else:
                if tf_ratio > 0.0 and self.training:
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
                
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            Nc_curr = torch.clamp(S_t[-1, :, :50], min=0.0)
            Ns_curr = torch.clamp(S_t[-1, :, 50:100], min=0.0)
            preds_c.append(Nc_curr / (Nc_curr.sum(dim=-1, keepdim=True) + 1e-8))
            preds_s.append(Ns_curr / (Ns_curr.sum(dim=-1, keepdim=True) + 1e-8))
            
        return torch.stack(preds_c, dim=1), torch.stack(preds_s, dim=1)

    def rollout_no_carrier(self, p0_vec, delta_t=0.1, tf_ratio=0.0):
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin, b0 = self.get_params()
        R0_batch = R0_base.unsqueeze(0).expand(p0_vec.size(0), -1)
        preds = []
        t_span = torch.linspace(0.0, 1.0, 11, device=device)
        N_curr = p0_vec * b0
        
        for p in range(6):
            if p == 0:
                N0 = N_curr
            else:
                if tf_ratio > 0.0 and self.training:
                    N_tf = tf_ratio * (no_c_tensor[:, p-1, :] * N_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_curr
                else:
                    N_tf = N_curr
                N0 = torch.clamp(N_tf / 100.0 + 1e-6, min=0.0)
                
            S0 = torch.cat([N0, R0_batch], dim=-1)
            def f(t, S):
                Nt = torch.clamp(S[..., :50], min=0.0)
                Rt = torch.clamp(S[..., 50:], min=0.0)
                dNt = Nt * torch.matmul(Rt, Theta.t()) - alpha * (Nt**2)
                flux = Rt * torch.matmul(Nt, Theta)
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNt, dRt], dim=-1)
                
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :50], min=0.0)
            preds.append(N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-8))
            N_curr = N_final
            
        return torch.stack(preds, dim=1)

def run_experiment(lambda_alpha=0.01, fix_b0=None, include_no_carrier=False, epochs=350, lr=0.015, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = UnifiedTransportCRM(N=50, K=5, c_w=0.50, fix_b0=fix_b0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        loss = 0.0
        
        p_pl_c, p_pl_s = model.rollout_carrier(p0, 'plain', tf_ratio=tf_ratio)
        loss += torch.sum(plain_c_tensor * torch.log(plain_c_tensor / torch.clamp(p_pl_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                torch.sum(plain_s_tensor * torch.log(plain_s_tensor / torch.clamp(p_pl_s, min=1e-9) + 1e-12), dim=-1).mean()
                
        p_mu_c, p_mu_s = model.rollout_carrier(p0, 'mucin', tf_ratio=tf_ratio)
        loss += torch.sum(mucin_c_tensor * torch.log(mucin_c_tensor / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                torch.sum(mucin_s_tensor * torch.log(mucin_s_tensor / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
                
        if include_no_carrier:
            p_no = model.rollout_no_carrier(p0, tf_ratio=tf_ratio)
            loss += torch.sum(no_c_tensor * torch.log(no_c_tensor / torch.clamp(p_no, min=1e-9) + 1e-12), dim=-1).mean()
            
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin, b0 = model.get_params()
        # Regularization: Theta, W, and alpha penalty!
        loss += 0.001 * (Theta.pow(2).sum() + W.pow(2).sum())
        if lambda_alpha > 0.0:
            # Penalize deviation from median alpha or L2 penalty
            loss += lambda_alpha * alpha.pow(2).mean()
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        p_pl_c, p_pl_s = model.rollout_carrier(p0, 'plain', tf_ratio=0.0)
        p_mu_c, p_mu_s = model.rollout_carrier(p0, 'mucin', tf_ratio=0.0)
        p_no = model.rollout_no_carrier(p0, tf_ratio=0.0)
        
        t_pl = torch.cat([plain_c_tensor, plain_s_tensor], dim=-1).cpu().numpy().flatten()
        pred_pl = torch.cat([p_pl_c, p_pl_s], dim=-1).cpu().numpy().flatten()
        r2_pl = compute_r2(t_pl, pred_pl)
        kl_pl = 0.5 * (compute_kl(plain_c_tensor, p_pl_c) + compute_kl(plain_s_tensor, p_pl_s))
        
        t_mu = torch.cat([mucin_c_tensor, mucin_s_tensor], dim=-1).cpu().numpy().flatten()
        pred_mu = torch.cat([p_mu_c, p_mu_s], dim=-1).cpu().numpy().flatten()
        r2_mu = compute_r2(t_mu, pred_mu)
        kl_mu = 0.5 * (compute_kl(mucin_c_tensor, p_mu_c) + compute_kl(mucin_s_tensor, p_mu_s))
        
        t_no = no_c_tensor.cpu().numpy().flatten()
        pred_no = p_no.cpu().numpy().flatten()
        r2_no = compute_r2(t_no, pred_no)
        r_no, _ = pearsonr(t_no, pred_no)
        kl_no = compute_kl(no_c_tensor, p_no)
        
    return {
        'include_no_c': include_no_carrier,
        'lambda_alpha': lambda_alpha,
        'fix_b0': fix_b0,
        'plain_r2': r2_pl, 'plain_kl': kl_pl,
        'mucin_r2': r2_mu, 'mucin_kl': kl_mu,
        'nocarrier_r2': r2_no, 'nocarrier_r': r_no, 'nocarrier_kl': kl_no,
        'b0': b0.item(),
        'k_attach_mean': k_attach.mean().item(),
        'k_detach_mean': k_detach.mean().item()
    }

if __name__ == '__main__':
    print("\n" + "="*80)
    print("TESTING SINGLE ATTACH & DETACH (SHARED FOR BOTH CARRIERS) + ALPHA PENALTY")
    print("="*80)
    
    # 1. Joint fit test (baseline sanity check)
    res_joint = run_experiment(lambda_alpha=0.01, fix_b0=None, include_no_carrier=True)
    print(f"[Joint Fit] Plain R2: {res_joint['plain_r2']:.4f} | Mucin R2: {res_joint['mucin_r2']:.4f} | No Carrier R2: {res_joint['nocarrier_r2']:.4f} (r={res_joint['nocarrier_r']:.4f}, KL={res_joint['nocarrier_kl']:.4f})")
    
    # 2. Agar-only with lambda_alpha = 0.0 (no penalty)
    res_zero_0 = run_experiment(lambda_alpha=0.0, fix_b0=None, include_no_carrier=False)
    print(f"[Agar-only, lambda_alpha=0.0] Plain R2: {res_zero_0['plain_r2']:.4f} | Mucin R2: {res_zero_0['mucin_r2']:.4f} | No Carrier Zero-Shot R2: {res_zero_0['nocarrier_r2']:.4f} (r={res_zero_0['nocarrier_r']:.4f}, KL={res_zero_0['nocarrier_kl']:.4f}) | b0={res_zero_0['b0']:.4f}")
    
    # 3. Agar-only with lambda_alpha = 0.01
    res_zero_1 = run_experiment(lambda_alpha=0.01, fix_b0=None, include_no_carrier=False)
    print(f"[Agar-only, lambda_alpha=0.01] Plain R2: {res_zero_1['plain_r2']:.4f} | Mucin R2: {res_zero_1['mucin_r2']:.4f} | No Carrier Zero-Shot R2: {res_zero_1['nocarrier_r2']:.4f} (r={res_zero_1['nocarrier_r']:.4f}, KL={res_zero_1['nocarrier_kl']:.4f}) | b0={res_zero_1['b0']:.4f}")
    
    # 4. Agar-only with lambda_alpha = 0.05
    res_zero_2 = run_experiment(lambda_alpha=0.05, fix_b0=None, include_no_carrier=False)
    print(f"[Agar-only, lambda_alpha=0.05] Plain R2: {res_zero_2['plain_r2']:.4f} | Mucin R2: {res_zero_2['mucin_r2']:.4f} | No Carrier Zero-Shot R2: {res_zero_2['nocarrier_r2']:.4f} (r={res_zero_2['nocarrier_r']:.4f}, KL={res_zero_2['nocarrier_kl']:.4f}) | b0={res_zero_2['b0']:.4f}")
    
    # 5. Agar-only with lambda_alpha = 0.01 AND fix_b0 = 1.6
    res_zero_b0 = run_experiment(lambda_alpha=0.01, fix_b0=1.6, include_no_carrier=False)
    print(f"[Agar-only, lambda_alpha=0.01, fix_b0=1.6] Plain R2: {res_zero_b0['plain_r2']:.4f} | Mucin R2: {res_zero_b0['mucin_r2']:.4f} | No Carrier Zero-Shot R2: {res_zero_b0['nocarrier_r2']:.4f} (r={res_zero_b0['nocarrier_r']:.4f}, KL={res_zero_b0['nocarrier_kl']:.4f}) | b0={res_zero_b0['b0']:.4f}")
    print("="*80)

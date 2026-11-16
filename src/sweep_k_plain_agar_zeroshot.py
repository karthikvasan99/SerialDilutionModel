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
strains = data['strains']
N = 50

# Load inoculum P0
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv'))
sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
df['Relative_Abundance'] = df['Read_Fraction'] / 100.0 if sums.mean() > 50 else df['Read_Fraction']
inoc = df[df['sampleType'] == 'inoculum'].pivot_table(index='replicate', columns='Strain_Name', values='Relative_Abundance', fill_value=0.0).reindex(columns=strains, fill_value=0.0)
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

class ModularCRM(nn.Module):
    def __init__(self, N=50, K=5, c_w=0.50, b0_val=1.6):
        super().__init__()
        self.N, self.K = N, K
        self.c_w = c_w
        self.register_buffer('b0', torch.tensor(float(b0_val)))
        
        self.theta_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.1)
        
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
        
        for p in range(6):
            if p == 0:
                N0 = N_curr
            else:
                if tf_ratio > 0.0 and tf_target is not None:
                    N_tf = tf_ratio * (tf_target[:, p-1, :] * N_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_curr
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
                
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': 0.1})
            N_final = torch.clamp(S_t[-1, :, :50], min=0.0)
            preds.append(N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-8))
            N_curr = N_final
            
        return torch.stack(preds, dim=1)

def evaluate_k(K, epochs=350, lr=0.015, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ModularCRM(N=50, K=K, c_w=0.50, b0_val=1.6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        # 1. No Carrier
        p_no = model.forward_no_carrier(p0, tf_target=no_c_tensor, tf_ratio=tf_ratio)
        loss = torch.sum(no_c_tensor * torch.log(no_c_tensor / torch.clamp(p_no, min=1e-9) + 1e-12), dim=-1).mean()
        
        # 2. Mucin Agar
        p_mu_c, p_mu_s = model.forward_carrier(p0, is_mucin=True, tf_targets=(mucin_c_tensor, mucin_s_tensor), tf_ratio=tf_ratio)
        loss += torch.sum(mucin_c_tensor * torch.log(mucin_c_tensor / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                torch.sum(mucin_s_tensor * torch.log(mucin_s_tensor / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
                
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin = model.get_params()
        loss += 0.001 * (Theta.pow(2).sum() + W.pow(2).sum()) + 0.01 * alpha.pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        # Evaluate on training conditions
        pred_no = model.forward_no_carrier(p0, tf_ratio=0.0)
        pred_mu_c, pred_mu_s = model.forward_carrier(p0, is_mucin=True, tf_ratio=0.0)
        
        t_no = no_c_tensor.cpu().numpy().flatten()
        p_no_np = pred_no.cpu().numpy().flatten()
        train_r2_no = compute_r2(t_no, p_no_np)
        
        t_mu = np.concatenate([mucin_s_tensor.cpu().numpy().flatten(), mucin_c_tensor.cpu().numpy().flatten()])
        p_mu_np = np.concatenate([pred_mu_s.cpu().numpy().flatten(), pred_mu_c.cpu().numpy().flatten()])
        train_r2_mu = compute_r2(t_mu, p_mu_np)
        
        # Evaluate 100% ZERO-SHOT ON PLAIN AGAR (is_mucin=False, NO TEACHER FORCING)
        pred_pl_c, pred_pl_s = model.forward_carrier(p0, is_mucin=False, tf_ratio=0.0)
        
        # Plain Supernatant
        t_pl_s = plain_s_tensor.cpu().numpy().flatten()
        p_pl_s_np = pred_pl_s.cpu().numpy().flatten()
        r2_s = compute_r2(t_pl_s, p_pl_s_np)
        r_s, _ = pearsonr(t_pl_s, p_pl_s_np)
        kl_s = compute_kl(plain_s_tensor, pred_pl_s)
        
        # Plain Carrier
        t_pl_c = plain_c_tensor.cpu().numpy().flatten()
        p_pl_c_np = pred_pl_c.cpu().numpy().flatten()
        r2_c = compute_r2(t_pl_c, p_pl_c_np)
        r_c, _ = pearsonr(t_pl_c, p_pl_c_np)
        kl_c = compute_kl(plain_c_tensor, pred_pl_c)
        
        # Plain Combined
        t_comb = np.concatenate([t_pl_s, t_pl_c])
        p_comb = np.concatenate([p_pl_s_np, p_pl_c_np])
        r2_comb = compute_r2(t_comb, p_comb)
        r_comb, _ = pearsonr(t_comb, p_comb)
        kl_comb = 0.5 * (kl_s + kl_c)
        
    return {
        'K': K,
        'params': 50*K + K*K + 50 + 50 + 50 + 2*K,
        'train_r2_no': train_r2_no,
        'train_r2_mu': train_r2_mu,
        'zero_shot_r2_s': r2_s,
        'zero_shot_r_s': r_s,
        'zero_shot_kl_s': kl_s,
        'zero_shot_r2_c': r2_c,
        'zero_shot_r_c': r_c,
        'zero_shot_kl_c': kl_c,
        'zero_shot_r2_comb': r2_comb,
        'zero_shot_r_comb': r_comb,
        'zero_shot_kl_comb': kl_comb
    }

if __name__ == '__main__':
    k_candidates = [3, 4, 5, 6, 7, 8, 10, 12]
    print(f"\n{'='*110}", flush=True)
    print(f"BENCHMARKING LATENT RESOURCE DIMENSIONALITY K in {k_candidates}", flush=True)
    print(f"TASK: Train strictly on No Carrier + Mucin Agar -> Test 100% Zero-Shot on Plain Agar", flush=True)
    print(f"{'='*110}", flush=True)
    print(f"{'K':>2} | {'Params':>6} | {'Train No':>8} | {'Train Mu':>8} | {'ZS Sup R2':>9} | {'ZS Sup r':>8} | {'ZS Car R2':>9} | {'ZS Car r':>8} | {'ZS Comb R2':>10} | {'ZS Comb r':>9} | {'ZS KL':>7}", flush=True)
    print(f"{'-'*110}", flush=True)
    
    results = []
    for K in k_candidates:
        res = evaluate_k(K, epochs=350, lr=0.015, seed=42)
        results.append(res)
        print(f"{res['K']:2d} | {res['params']:6d} | {res['train_r2_no']:8.4f} | {res['train_r2_mu']:8.4f} | {res['zero_shot_r2_s']:9.4f} | {res['zero_shot_r_s']:8.4f} | {res['zero_shot_r2_c']:9.4f} | {res['zero_shot_r_c']:8.4f} | {res['zero_shot_r2_comb']:10.4f} | {res['zero_shot_r_comb']:9.4f} | {res['zero_shot_kl_comb']:7.4f}", flush=True)
        
    print(f"{'='*110}", flush=True)
    df_res = pd.DataFrame(results)
    csv_out = os.path.join(PROJECT_ROOT, 'Data', 'K_Dimensionality_ZeroShot_PlainAgar.csv')
    df_res.to_csv(csv_out, index=False)
    print(f"Results saved to {csv_out}", flush=True)

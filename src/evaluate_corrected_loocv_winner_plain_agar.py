import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from torchdiffeq import odeint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = torch.load(os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt'), map_location=device, weights_only=False)
no_c_tensor = data['no_c_tensor'].to(device)
mucin_c_tensor = data['mucin_c_tensor'].to(device)
mucin_s_tensor = data['mucin_s_tensor'].to(device)
plain_c_tensor = data['plain_c_tensor'].to(device)
plain_s_tensor = data['plain_s_tensor'].to(device)

from thorough_loocv_fixed_b0 import ModularCRM_FixedB0, p0

def compute_r2(true_vals, pred_vals):
    diff_sq = np.sum((true_vals - pred_vals)**2)
    tot_sq = np.sum((true_vals - np.mean(true_vals))**2)
    return float(1.0 - diff_sq / tot_sq)

def compute_kl(t, p, eps=1e-12, clamp_min=1e-9):
    ratio = torch.clamp(t, min=eps) / torch.clamp(p, min=clamp_min)
    kl = torch.sum(torch.where(t > 0, t * torch.log(ratio), torch.zeros_like(t)), dim=-1)
    return float(kl.mean().item())

def train_and_eval(K=4, c_w=0.75, b0_val=1.60, reg_alpha=0.001, reg_w=0.001, epochs=350, lr=0.015, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ModularCRM_FixedB0(N=50, K=K, c_w=c_w, b0_val=b0_val).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        # 1. No Carrier (with corrected 1:100 dilution!)
        p_no = model.forward_no_carrier(p0, tf_target=no_c_tensor, tf_ratio=tf_ratio)
        ratio_no = torch.clamp(no_c_tensor, min=1e-12) / torch.clamp(p_no, min=1e-9)
        loss = torch.sum(torch.where(no_c_tensor > 0, no_c_tensor * torch.log(ratio_no), torch.zeros_like(no_c_tensor)), dim=-1).mean()
        
        # 2. Mucin Agar
        p_mu_c, p_mu_s = model.forward_carrier(p0, is_mucin=True, tf_targets=(mucin_c_tensor, mucin_s_tensor), tf_ratio=tf_ratio)
        ratio_mu_c = torch.clamp(mucin_c_tensor, min=1e-12) / torch.clamp(p_mu_c, min=1e-9)
        ratio_mu_s = torch.clamp(mucin_s_tensor, min=1e-12) / torch.clamp(p_mu_s, min=1e-9)
        loss += torch.sum(torch.where(mucin_c_tensor > 0, mucin_c_tensor * torch.log(ratio_mu_c), torch.zeros_like(mucin_c_tensor)), dim=-1).mean() + \
                torch.sum(torch.where(mucin_s_tensor > 0, mucin_s_tensor * torch.log(ratio_mu_s), torch.zeros_like(mucin_s_tensor)), dim=-1).mean()
                
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin = model.get_params()
        loss += reg_w * (Theta.pow(2).sum() + W.pow(2).sum()) + reg_alpha * alpha.pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        # Evaluate 100% Zero-shot on Plain Agar
        p_pl_c, p_pl_s = model.forward_carrier(p0, is_mucin=False, tf_ratio=0.0)
        
        # Supernatant
        t_s = plain_s_tensor.cpu().numpy().flatten()
        p_s = p_pl_s.cpu().numpy().flatten()
        r2_s = compute_r2(t_s, p_s)
        r_s, _ = pearsonr(t_s, p_s)
        rho_s, _ = spearmanr(t_s, p_s)
        kl_s = compute_kl(plain_s_tensor, p_pl_s)
        
        # Carrier
        t_c = plain_c_tensor.cpu().numpy().flatten()
        p_c = p_pl_c.cpu().numpy().flatten()
        r2_c = compute_r2(t_c, p_c)
        r_c, _ = pearsonr(t_c, p_c)
        rho_c, _ = spearmanr(t_c, p_c)
        kl_c = compute_kl(plain_c_tensor, p_pl_c)
        
        # Combined
        t_comb = np.concatenate([t_s, t_c])
        p_comb = np.concatenate([p_s, p_c])
        r2_comb = compute_r2(t_comb, p_comb)
        r_comb, _ = pearsonr(t_comb, p_comb)
        rho_comb, _ = spearmanr(t_comb, p_comb)
        kl_comb = 0.5 * (kl_s + kl_c)
        
    return {
        'K': K, 'c_w': c_w, 'b0_val': b0_val, 'reg_alpha': reg_alpha,
        'Comb R2': float(r2_comb), 'Comb r': float(r_comb), 'Comb rho': float(rho_comb), 'Comb KL': float(kl_comb),
        'Sup R2': float(r2_s), 'Sup r': float(r_s), 'Sup rho': float(rho_s), 'Sup KL': float(kl_s),
        'Car R2': float(r2_c), 'Car r': float(r_c), 'Car rho': float(rho_c), 'Car KL': float(kl_c)
    }

if __name__ == '__main__':
    configs = [
        ('LOOCV Winner (K=4, cw=0.75, b0=1.60, reg_a=0.001)', {'K': 4, 'c_w': 0.75, 'b0_val': 1.60, 'reg_alpha': 0.001}),
        ('LOOCV Winner (K=4, cw=0.75, b0=2.00, reg_a=0.001)', {'K': 4, 'c_w': 0.75, 'b0_val': 2.00, 'reg_alpha': 0.001}),
        ('Classic Winner (K=3, cw=0.50, b0=1.60, reg_a=0.010)', {'K': 3, 'c_w': 0.50, 'b0_val': 1.60, 'reg_alpha': 0.010}),
    ]
    
    print(f"{'Configuration':<52} | {'Comb R2':<8} | {'Comb r':<7} | {'Comb rho':<8} | {'Comb KL':<8} | {'Sup R2':<7} | {'Car R2':<7}", flush=True)
    print('-'*110, flush=True)
    
    records = []
    for name, c in configs:
        res = train_and_eval(K=c['K'], c_w=c['c_w'], b0_val=c['b0_val'], reg_alpha=c['reg_alpha'])
        res['Configuration'] = name
        records.append(res)
        print(f"{name:<52} | {res['Comb R2']:8.4f} | {res['Comb r']:7.4f} | {res['Comb rho']:8.4f} | {res['Comb KL']:8.4f} | {res['Sup R2']:7.4f} | {res['Car R2']:7.4f}", flush=True)
        
    df = pd.DataFrame(records)
    out_csv = os.path.join(PROJECT_ROOT, 'Data', 'Corrected_LOOCV_Winner_PlainAgar_Transfer.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}", flush=True)

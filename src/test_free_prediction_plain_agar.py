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

from test_unified_transport_nocarrier import UnifiedTransportCRM

def train_no_and_mucin(lambda_alpha=0.01, fix_b0=1.6, epochs=350, lr=0.015, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = UnifiedTransportCRM(N=50, K=5, c_w=0.50, fix_b0=fix_b0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("\n--- Training on No Carrier + Mucin Agar (Plain Agar is 100% Held Out) ---", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        loss = 0.0
        
        # 1. No Carrier: trains Theta, W, alpha, and R0_base!
        p_no = model.rollout_no_carrier(p0, tf_ratio=tf_ratio)
        loss += torch.sum(no_c_tensor * torch.log(no_c_tensor / torch.clamp(p_no, min=1e-9) + 1e-12), dim=-1).mean()
        
        # 2. Mucin Agar: trains k_attach, k_detach, R0_mucin, and refines Theta, W, alpha!
        p_mu_c, p_mu_s = model.rollout_carrier(p0, 'mucin', tf_ratio=tf_ratio)
        loss += torch.sum(mucin_c_tensor * torch.log(mucin_c_tensor / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                torch.sum(mucin_s_tensor * torch.log(mucin_s_tensor / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
                
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin, b0 = model.get_params()
        loss += 0.001 * (Theta.pow(2).sum() + W.pow(2).sum())
        if lambda_alpha > 0.0:
            loss += lambda_alpha * alpha.pow(2).mean()
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f}", flush=True)
            
    # Evaluation on Plain Agar ZERO-SHOT!
    model.eval()
    with torch.no_grad():
        p_pl_c, p_pl_s = model.rollout_carrier(p0, 'plain', tf_ratio=0.0)
        p_mu_c, p_mu_s = model.rollout_carrier(p0, 'mucin', tf_ratio=0.0)
        p_no = model.rollout_no_carrier(p0, tf_ratio=0.0)
        
        # Evaluated on Plain Agar (100% UNSEEN)
        t_pl_c = plain_c_tensor.cpu().numpy().flatten()
        pred_pl_c = p_pl_c.cpu().numpy().flatten()
        r2_pl_c = compute_r2(t_pl_c, pred_pl_c)
        r_pl_c, _ = pearsonr(t_pl_c, pred_pl_c)
        kl_pl_c = compute_kl(plain_c_tensor, p_pl_c)
        
        t_pl_s = plain_s_tensor.cpu().numpy().flatten()
        pred_pl_s = p_pl_s.cpu().numpy().flatten()
        r2_pl_s = compute_r2(t_pl_s, pred_pl_s)
        r_pl_s, _ = pearsonr(t_pl_s, pred_pl_s)
        kl_pl_s = compute_kl(plain_s_tensor, p_pl_s)
        
        t_pl = torch.cat([plain_c_tensor, plain_s_tensor], dim=-1).cpu().numpy().flatten()
        pred_pl = torch.cat([p_pl_c, p_pl_s], dim=-1).cpu().numpy().flatten()
        r2_pl = compute_r2(t_pl, pred_pl)
        r_pl, _ = pearsonr(t_pl, pred_pl)
        kl_pl = 0.5 * (kl_pl_c + kl_pl_s)
        
        # Training conditions evaluation
        t_mu = torch.cat([mucin_c_tensor, mucin_s_tensor], dim=-1).cpu().numpy().flatten()
        r2_mu = compute_r2(t_mu, torch.cat([p_mu_c, p_mu_s], dim=-1).cpu().numpy().flatten())
        
        t_no = no_c_tensor.cpu().numpy().flatten()
        r2_no = compute_r2(t_no, p_no.cpu().numpy().flatten())
        
    print("\n" + "="*80, flush=True)
    print("RESULTS: FREE ZERO-SHOT PREDICTION ON PLAIN AGAR", flush=True)
    print("="*80, flush=True)
    print(f"  Training Sets: No Carrier R2 = {r2_no:.4f} | Mucin Agar R2 = {r2_mu:.4f}", flush=True)
    print(f"  PLAIN AGAR (100% Zero-Shot Out-of-Sample):", flush=True)
    print(f"    Overall Trajectory: R2 = {r2_pl:.4f} | Pearson r = {r_pl:.4f} | KL = {kl_pl:.4f} nats", flush=True)
    print(f"    Carrier Biofilm:    R2 = {r2_pl_c:.4f} | Pearson r = {r_pl_c:.4f} | KL = {kl_pl_c:.4f} nats", flush=True)
    print(f"    Supernatant Broth:  R2 = {r2_pl_s:.4f} | Pearson r = {r_pl_s:.4f} | KL = {kl_pl_s:.4f} nats", flush=True)
    print("="*80, flush=True)
    return r2_pl, r_pl, kl_pl

if __name__ == '__main__':
    # Test with fixed b0 = 1.6 and lambda_alpha = 0.01
    train_no_and_mucin(lambda_alpha=0.01, fix_b0=1.6)
    # Test with learned b0
    train_no_and_mucin(lambda_alpha=0.01, fix_b0=None)

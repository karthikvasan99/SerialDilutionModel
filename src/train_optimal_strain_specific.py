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
mucin_c_tensor = data['mucin_c_tensor'].to(device) # (3, 6, 50)
mucin_s_tensor = data['mucin_s_tensor'].to(device) # (3, 6, 50)
plain_c_tensor = data['plain_c_tensor'].to(device) # (3, 6, 50) - TEST SET
plain_s_tensor = data['plain_s_tensor'].to(device) # (3, 6, 50) - TEST SET
strains = data['strains']
N = 50

# Load inoculum P0
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv'))
sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
df['Relative_Abundance'] = df['Read_Fraction'] / 100.0 if sums.mean() > 50 else df['Read_Fraction']
inoc = df[df['sampleType'] == 'inoculum'].pivot_table(index='replicate', columns='Strain_Name', values='Relative_Abundance', fill_value=0.0).reindex(columns=strains, fill_value=0.0)
inoc_norm = inoc.div(inoc.sum(axis=1), axis=0)
p0 = torch.tensor(inoc_norm.mean(axis=0).values, dtype=torch.float32, device=device).unsqueeze(0).expand(3, -1)

from thorough_loocv_strain_specific import ModularCRM_StrainSpecific, compute_kl

def compute_r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1.0 - (ss_res / ss_tot)

def train_and_eval(epochs=300, lr=0.015, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("\n" + "="*90, flush=True)
    print("TRAINING FINAL OPTIMAL STRAIN-SPECIFIC MODEL (K=4, c_w=0.75, reg_alpha=0.0010, reg_w=0.0010)")
    print("Training across all 3 biological vessels of No Carrier + Mucin Agar...")
    print("="*90, flush=True)
    
    model = ModularCRM_StrainSpecific(N=50, K=4, c_w=0.75).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        # 1. No Carrier
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
        loss += 0.0010 * (Theta.pow(2).sum() + W.pow(2).sum()) + 0.0010 * alpha.pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == epochs:
            b0_l, b0_s = model.get_b0()
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} | b0_liq: {b0_l.item():.2f} | b0_sup: {b0_s.item():.2f}", flush=True)

    model.eval()
    with torch.no_grad():
        b0_l, b0_s = model.get_b0()
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin = model.get_params()
        
        # Training evaluation
        p_no_eval = model.forward_no_carrier(p0, tf_ratio=0.0)
        p_mu_c_eval, p_mu_s_eval = model.forward_carrier(p0, is_mucin=True, tf_ratio=0.0)
        
        t_no = no_c_tensor.cpu().numpy().flatten()
        p_no_np = p_no_eval.cpu().numpy().flatten()
        train_r2_no = compute_r2(t_no, p_no_np)
        train_r_no, _ = pearsonr(t_no, p_no_np)
        train_kl_no = compute_kl(no_c_tensor, p_no_eval)
        
        t_mu_s = mucin_s_tensor.cpu().numpy().flatten()
        p_mu_s_np = p_mu_s_eval.cpu().numpy().flatten()
        t_mu_c = mucin_c_tensor.cpu().numpy().flatten()
        p_mu_c_np = p_mu_c_eval.cpu().numpy().flatten()
        t_mu_comb = np.concatenate([t_mu_s, t_mu_c])
        p_mu_comb = np.concatenate([p_mu_s_np, p_mu_c_np])
        train_r2_mu = compute_r2(t_mu_comb, p_mu_comb)
        train_r_mu, _ = pearsonr(t_mu_comb, p_mu_comb)
        train_kl_mu = 0.5 * (compute_kl(mucin_s_tensor, p_mu_s_eval) + compute_kl(mucin_c_tensor, p_mu_c_eval))
        
        print("\n" + "="*90, flush=True)
        print("TRAINING PERFORMANCE (NO CARRIER + MUCIN AGAR)")
        print(f"No Carrier Liquid: R^2 = {train_r2_no:6.4f} | Pearson r = {train_r_no:6.4f} | KL = {train_kl_no:6.4f} nats")
        print(f"Mucin Agar Comb:   R^2 = {train_r2_mu:6.4f} | Pearson r = {train_r_mu:6.4f} | KL = {train_kl_mu:6.4f} nats")
        print("="*90, flush=True)
        
        # Evaluate on Plain Agar Zero-Shot (with b0_sup)
        pred_pl_c, pred_pl_s = model.forward_carrier(p0, is_mucin=False, tf_ratio=0.0)
        t_pl_s = plain_s_tensor.cpu().numpy().flatten()
        p_pl_s_np = pred_pl_s.cpu().numpy().flatten()
        t_pl_c = plain_c_tensor.cpu().numpy().flatten()
        p_pl_c_np = pred_pl_c.cpu().numpy().flatten()
        t_comb = np.concatenate([t_pl_s, t_pl_c])
        p_comb = np.concatenate([p_pl_s_np, p_pl_c_np])
        
        print("\nZERO-SHOT ON PLAIN AGAR (using b0_sup = 0.36):")
        print(f"Supernatant: R^2 = {compute_r2(t_pl_s, p_pl_s_np):6.4f} | r = {pearsonr(t_pl_s, p_pl_s_np)[0]:6.4f} | KL = {compute_kl(plain_s_tensor, pred_pl_s):6.4f} nats")
        print(f"Carrier:     R^2 = {compute_r2(t_pl_c, p_pl_c_np):6.4f} | r = {pearsonr(t_pl_c, p_pl_c_np)[0]:6.4f} | KL = {compute_kl(plain_c_tensor, pred_pl_c):6.4f} nats")
        print(f"Combined:    R^2 = {compute_r2(t_comb, p_comb):6.4f} | r = {pearsonr(t_comb, p_comb)[0]:6.4f} | KL = {0.5*(compute_kl(plain_s_tensor, pred_pl_s)+compute_kl(plain_c_tensor, pred_pl_c)):6.4f} nats")
        
        # Save model checkpoint
        ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'OptimalCRM_StrainSpecific_Model.pt')
        torch.save({
            'state_dict': model.state_dict(),
            'k_attach': k_attach.cpu().numpy(),
            'k_detach': k_detach.cpu().numpy(),
            'strains': strains,
            'b0_liq': b0_l.item(),
            'b0_sup': b0_s.item(),
            'Theta': Theta.cpu().numpy(),
            'W': W.cpu().numpy(),
            'R0_base': R0_base.cpu().numpy(),
            'R0_mucin': R0_mucin.cpu().numpy(),
        }, ckpt_path)
        print(f"\nModel checkpoint saved to {ckpt_path}", flush=True)

if __name__ == '__main__':
    train_and_eval(epochs=300, lr=0.015, seed=42)

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

def compute_kl(t, p, eps=1e-12, clamp_min=1e-9):
    ratio = torch.clamp(t, min=eps) / torch.clamp(p, min=clamp_min)
    kl = torch.sum(torch.where(t > 0, t * torch.log(ratio), torch.zeros_like(t)), dim=-1)
    return kl.mean().item()

def compute_r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1.0 - (ss_res / ss_tot)

class OptimalCRM(nn.Module):
    def __init__(self, N=50, K=4, c_w=0.70):
        super().__init__()
        self.N, self.K = N, K
        self.c_w = c_w
        
        self.theta_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.1)
        
        # Scalar k_attach, strain-specific k_detach
        self.k_attach_raw = nn.Parameter(torch.tensor([-1.0]))
        self.k_detach_raw = nn.Parameter(torch.randn(N) * 0.1 - 1.0)
        
        self.a_base = nn.Parameter(torch.randn(K) * 0.5 + 2.0)
        self.a_mucin = nn.Parameter(torch.randn(K) * 0.5 + 2.0)
        
        # Learnable condition-specific initial biomass
        self.b0_liq_raw = nn.Parameter(torch.tensor(1.25))
        self.b0_sup_raw = nn.Parameter(torch.tensor(1.25))

    def get_b0(self):
        b0_liq = F.softplus(self.b0_liq_raw) + 0.1
        b0_sup = F.softplus(self.b0_sup_raw) + 0.1
        return b0_liq, b0_sup

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

    def forward_carrier(self, p0_vec, is_mucin=True, tf_targets=None, tf_ratio=0.0, gamma_detach=1.0):
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin = self.get_params()
        _, b0_sup = self.get_b0()
        R0_cond = R0_mucin if is_mucin else R0_base
        R0_batch = R0_cond.unsqueeze(0).expand(p0_vec.size(0), -1)
        
        k_det = k_detach * gamma_detach
        
        preds_c, preds_s = [], []
        t_span = torch.linspace(0.0, 1.0, 11, device=device)
        Nc_curr = torch.zeros_like(p0_vec)
        Ns_curr = p0_vec * b0_sup
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
                dNc = Nc * torch.matmul(Rt, Theta.t()) - alpha * (Nc**2) - k_det * Nc + k_attach * Ns
                dNs = Ns * torch.matmul(Rt, Theta.t()) - alpha * (Ns**2) + k_det * Nc - k_attach * Ns
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
        b0_liq, _ = self.get_b0()
        R0_batch = R0_base.unsqueeze(0).expand(p0_vec.size(0), -1)
        
        preds = []
        t_span = torch.linspace(0.0, 1.0, 11, device=device)
        N_curr = p0_vec * b0_liq
        eta = 1.0 / 30.0
        
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

def run_experiment(epochs=300, lr=0.015, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("\n" + "="*90, flush=True)
    print("TRAINING OPTIMAL CRM (K=4, c_w=0.70, Learnable b0, lambda_alpha=0.001)")
    print("Training strictly on No Carrier + Mucin Agar across all vessels...")
    print("="*90, flush=True)
    
    model = OptimalCRM(N=50, K=4, c_w=0.70).to(device)
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
        loss += 0.001 * (Theta.pow(2).sum() + W.pow(2).sum()) + 0.001 * alpha.pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == epochs:
            b0_l, b0_s = model.get_b0()
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} | k_attach: {k_attach.item():.3f} | b0_liq: {b0_l.item():.2f} | b0_sup: {b0_s.item():.2f}", flush=True)

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
        
        # 100% ZERO-SHOT TRANSFER EVALUATION ON PLAIN AGAR (gamma = 1.0)
        pred_pl_c, pred_pl_s = model.forward_carrier(p0, is_mucin=False, tf_ratio=0.0, gamma_detach=1.0)
        
        t_pl_s = plain_s_tensor.cpu().numpy().flatten()
        p_pl_s_np = pred_pl_s.cpu().numpy().flatten()
        r2_s = compute_r2(t_pl_s, p_pl_s_np)
        r_s, _ = pearsonr(t_pl_s, p_pl_s_np)
        kl_s = compute_kl(plain_s_tensor, pred_pl_s)
        
        t_pl_c = plain_c_tensor.cpu().numpy().flatten()
        p_pl_c_np = pred_pl_c.cpu().numpy().flatten()
        r2_c = compute_r2(t_pl_c, p_pl_c_np)
        r_c, _ = pearsonr(t_pl_c, p_pl_c_np)
        kl_c = compute_kl(plain_c_tensor, pred_pl_c)
        
        t_comb = np.concatenate([t_pl_s, t_pl_c])
        p_comb = np.concatenate([p_pl_s_np, p_pl_c_np])
        r2_comb = compute_r2(t_comb, p_comb)
        r_comb, _ = pearsonr(t_comb, p_comb)
        kl_comb = 0.5 * (kl_s + kl_c)
        
        print("\n" + "="*90, flush=True)
        print("100% PURE ZERO-SHOT TRANSFER ON PLAIN AGAR (NO FINE-TUNING, gamma=1.0)")
        print("="*90, flush=True)
        print(f"Plain Supernatant: R^2 = {r2_s:6.4f} | Pearson r = {r_s:6.4f} | KL = {kl_s:6.4f} nats")
        print(f"Plain Carrier:     R^2 = {r2_c:6.4f} | Pearson r = {r_c:6.4f} | KL = {kl_c:6.4f} nats")
        print(f"Plain Combined:    R^2 = {r2_comb:6.4f} | Pearson r = {r_comb:6.4f} | KL = {kl_comb:6.4f} nats")
        print("="*90, flush=True)
        
        # Per-passage breakdown on Plain Agar
        print("\nPER-PASSAGE ZERO-SHOT R^2 AND KL BREAKDOWN (PLAIN AGAR):")
        print(f"{'Passage':<8} | {'Supernatant R^2':<15} | {'Supernatant r':<14} | {'Carrier R^2':<12} | {'Carrier r':<10} | {'Combined R^2':<12} | {'Comb KL (nats)':<14}")
        print("-" * 95)
        passage_records = []
        for p in range(6):
            t_s_p = plain_s_tensor[:, p, :].cpu().numpy().flatten()
            p_s_p = pred_pl_s[:, p, :].cpu().numpy().flatten()
            t_c_p = plain_c_tensor[:, p, :].cpu().numpy().flatten()
            p_c_p = pred_pl_c[:, p, :].cpu().numpy().flatten()
            
            r2_s_p = compute_r2(t_s_p, p_s_p)
            r_s_p, _ = pearsonr(t_s_p, p_s_p)
            r2_c_p = compute_r2(t_c_p, p_c_p)
            r_c_p, _ = pearsonr(t_c_p, p_c_p)
            
            t_comb_p = np.concatenate([t_s_p, t_c_p])
            p_comb_p = np.concatenate([p_s_p, p_c_p])
            r2_comb_p = compute_r2(t_comb_p, p_comb_p)
            r_comb_p, _ = pearsonr(t_comb_p, p_comb_p)
            kl_comb_p = 0.5 * (compute_kl(plain_s_tensor[:, p:p+1, :], pred_pl_s[:, p:p+1, :]) + 
                               compute_kl(plain_c_tensor[:, p:p+1, :], pred_pl_c[:, p:p+1, :]))
            
            passage_records.append({
                'passage': f"P{p+1}",
                'r2_supernatant': r2_s_p,
                'r_supernatant': r_s_p,
                'r2_carrier': r2_c_p,
                'r_carrier': r_c_p,
                'r2_combined': r2_comb_p,
                'r_combined': r_comb_p,
                'kl_combined': kl_comb_p
            })
            print(f"P{p+1:<7} | {r2_s_p:15.4f} | {r_s_p:14.4f} | {r2_c_p:12.4f} | {r_c_p:10.4f} | {r2_comb_p:12.4f} | {kl_comb_p:14.4f}")
        print("-" * 95)
        
        # Inoculum Baseline comparison
        p0_static = p0.unsqueeze(1).expand(-1, 6, -1)
        p0_s_np = p0_static.cpu().numpy().flatten()
        null_r2_s = compute_r2(t_pl_s, p0_s_np)
        null_r2_c = compute_r2(t_pl_c, p0_s_np)
        null_r2_comb = compute_r2(t_comb, np.concatenate([p0_s_np, p0_s_np]))
        null_kl = 0.5 * (compute_kl(plain_s_tensor, p0_static) + compute_kl(plain_c_tensor, p0_static))
        print(f"\nSTATIC INOCULUM NULL BASELINE (P0 predicted at all passages):")
        print(f"Combined Plain R^2 = {null_r2_comb:.4f} | KL = {null_kl:.4f} nats")
        
        # Test surface detachment hypothesis (gamma sweep on Plain Agar)
        print("\n" + "="*90, flush=True)
        print("TESTING DETACHMENT SCALING HYPOTHESIS: k_detach_plain = gamma * k_detach")
        print("="*90, flush=True)
        gamma_records = []
        for gamma in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            p_c_g, p_s_g = model.forward_carrier(p0, is_mucin=False, tf_ratio=0.0, gamma_detach=gamma)
            p_s_g_np = p_s_g.cpu().numpy().flatten()
            p_c_g_np = p_c_g.cpu().numpy().flatten()
            p_comb_g = np.concatenate([p_s_g_np, p_c_g_np])
            r2_g = compute_r2(t_comb, p_comb_g)
            r_g, _ = pearsonr(t_comb, p_comb_g)
            kl_g = 0.5 * (compute_kl(plain_s_tensor, p_s_g) + compute_kl(plain_c_tensor, p_c_g))
            gamma_records.append({'gamma': gamma, 'r2': r2_g, 'r': r_g, 'kl': kl_g})
            print(f"gamma = {gamma:4.1f} | Combined Plain R^2: {r2_g:6.4f} | Pearson r: {r_g:6.4f} | KL: {kl_g:6.4f} nats", flush=True)
        print("="*90, flush=True)
        
        # Save records
        pd.DataFrame(passage_records).to_csv(os.path.join(PROJECT_ROOT, 'Data', 'PlainAgar_Passage_ZeroShot_Results.csv'), index=False)
        pd.DataFrame(gamma_records).to_csv(os.path.join(PROJECT_ROOT, 'Data', 'PlainAgar_Gamma_Detachment_Sweep.csv'), index=False)
        
        # Save model checkpoint
        ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'OptimalCRM_K4_Model.pt')
        torch.save({
            'state_dict': model.state_dict(),
            'k_attach': k_attach.item(),
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
    run_experiment(epochs=300, lr=0.015, seed=42)

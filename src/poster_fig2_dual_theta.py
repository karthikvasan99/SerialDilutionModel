"""
Dual-Theta Unified Consumer-Resource Model (K=3) for Pollard Gut Microbiome:
Attached Biofilm Traits Theta^(c) vs Floating Planktonic Traits Theta^(s).

Conditions on X-Axis (Exactly 3 real experimental tube conditions):
1. No Carrier (Liquid)
2. Plain Agar Microcosm (Carrier Biofilm + Liquid Supernatant simultaneously)
3. Mucin Agar Microcosm (Carrier Biofilm + Liquid Supernatant simultaneously)

Rows in Figure:
Row 0: Mathematical Equations showing separate Theta^(c) and Theta^(s)
Row 1: Two learned Theta matrices side by side:
       - Panel A: Attached Biofilm Traits Theta^(c) (50 x 3)
       - Panel B: Floating Planktonic Traits Theta^(s) (50 x 3)
       - Inset: Cross-Feeding W (3 x 3)
Row 2: Benchmark comparison across the 3 conditions (with error bars over 50 shuffles,
       omitting Shuffled Adhesion for No Carrier).
"""

import sys
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.cluster.hierarchy import linkage, leaves_list

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_path = os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv')

df = pd.read_csv(data_path)
df = df[df['sampleType'] != 'inoculum'].copy()
sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
df['Relative_Abundance'] = df['Read_Fraction'] / 100.0 if sums.mean() > 50 else df['Read_Fraction']

sample_cols = ['sampleType', 'passage', 'replicate']
num_samples = len(df.drop_duplicates(subset=sample_cols))
mean_abundances = df.groupby('Strain_Name')['Relative_Abundance'].sum() / num_samples
strains_keep = sorted(list(mean_abundances[mean_abundances >= 1e-3].index))
N, K = len(strains_keep), 3

replicates = sorted(list(df['replicate'].unique()))
passages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

def get_pivot(st):
    sub = df[df['sampleType'] == st].pivot_table(index=['replicate', 'passage'], columns='Strain_Name', values='Relative_Abundance', fill_value=0.0).reindex(columns=strains_keep, fill_value=0.0)
    norm = (sub + 1e-6).div((sub + 1e-6).sum(axis=1), axis=0)
    return torch.tensor(np.array([norm.loc[rep].reindex(passages).values for rep in replicates]), dtype=torch.float32).to(device)

no_c_tensor = get_pivot('no carriers')
plain_c_tensor = get_pivot('plain-agar carriers')
plain_s_tensor = get_pivot('plain-agar supernatant')
mucin_c_tensor = get_pivot('mucin-agar carriers')
mucin_s_tensor = get_pivot('mucin-agar supernatant')

class DualThetaUnifiedCRM(nn.Module):
    def __init__(self, N=50, K=3):
        super().__init__()
        self.N = N
        self.K = K
        # 1. Attached / Biofilm Theta_c
        self.theta_c_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        # 2. Floating / Planktonic Theta_s
        self.theta_s_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        
        # 3. Byproduct cross-feeding W (K x K = 3 x 3)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1 - 1.0)
        # 4. Self-limitation alpha
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.05 - 1.0)
        
        # 5. Adhesion traits
        self.k_shed_raw = nn.Parameter(torch.randn(N) * 0.1 - 1.0)
        self.k_attach_raw = nn.Parameter(torch.randn(2, N) * 0.1 - 1.0)
        
        # 6. Environmental media resources: 3 conditions (0: No Carriers, 1: Plain Agar, 2: Mucin Agar)
        self.a = nn.Parameter(torch.randn(3, K) * 0.5 + 2.0)
        
        # 7. Biomass scales
        self.b0_liq_raw = nn.Parameter(torch.tensor(2.0))
        self.b0_c_raw = nn.Parameter(torch.tensor(2.0))
        self.b0_s_raw = nn.Parameter(torch.tensor(2.0))
        self.eta_raw = nn.Parameter(torch.tensor(-4.0))

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
        eta = torch.sigmoid(self.eta_raw) * 0.02
        
        if enforce_w_zero:
            W = torch.zeros(self.K, self.K, device=self.w_raw.device)
        else:
            W_sp = F.softplus(self.w_raw) * (1.0 - torch.eye(self.K, device=self.w_raw.device))
            row_sums = W_sp.sum(dim=-1, keepdim=True)
            scale = torch.clamp(0.5 / torch.clamp(row_sums, min=1e-6), max=1.0)
            W = W_sp * scale
        return Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta

    def rollout_no_carrier(self, Theta_s, W, alpha, R0_liq, b0_liq, delta_t=0.1, tf_ratio=0.0):
        # In No Carrier, all cells are floating -> use Theta_s
        N_curr = no_c_tensor[:, 0, :] * b0_liq
        R0_batch = R0_liq.unsqueeze(0).expand(no_c_tensor.size(0), -1)
        preds = []
        t_span = torch.linspace(0.0, 1.0, int(1.0 / delta_t) + 1, device=device)
        for p in range(1, 6):
            if tf_ratio > 0.0 and self.training:
                N_transferred = tf_ratio * (no_c_tensor[:, p-1, :] * N_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_curr
            else:
                N_transferred = N_curr
            N0 = torch.clamp(N_transferred / 100.0 + 1e-6, min=0.0)
            S0 = torch.cat([N0, R0_batch], dim=-1)
            def f(t, S):
                Nt = torch.clamp(S[..., :N], min=0.0)
                Rt = torch.clamp(S[..., N:], min=0.0)
                dNt = Nt * torch.matmul(Rt, Theta_s.t()) - alpha * (Nt**2)
                flux = Rt * torch.matmul(Nt, Theta_s)
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNt, dRt], dim=-1)
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :N], min=0.0)
            preds.append(N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-8))
            N_curr = N_final
        return torch.stack(preds, dim=1)

    def rollout_carrier(self, c_tensor, s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_att, R0_cond, b0_c, b0_s, eta, delta_t=0.1, tf_ratio=0.0):
        # Attached cells use Theta_c, Floating cells use Theta_s
        N_c_curr = c_tensor[:, 0, :] * b0_c
        N_s_curr = s_tensor[:, 0, :] * b0_s
        R0_batch = R0_cond.unsqueeze(0).expand(c_tensor.size(0), -1)
        preds_c, preds_s = [], []
        t_span = torch.linspace(0.0, 1.0, int(1.0 / delta_t) + 1, device=device)
        for p in range(1, 6):
            if tf_ratio > 0.0 and self.training:
                N_c_transferred = tf_ratio * (c_tensor[:, p-1, :] * N_c_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_c_curr
                N_s_transferred = tf_ratio * (s_tensor[:, p-1, :] * N_s_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_s_curr
            else:
                N_c_transferred = N_c_curr
                N_s_transferred = N_s_curr
            N_c_0 = torch.clamp(0.2 * N_c_transferred + 1e-6, min=0.0)
            N_s_0 = torch.clamp(eta * N_s_transferred + 1e-6, min=0.0)
            S0 = torch.cat([N_c_0, N_s_0, R0_batch], dim=-1)
            def f(t, S):
                Nc = torch.clamp(S[..., :N], min=0.0)
                Ns = torch.clamp(S[..., N:2*N], min=0.0)
                Rt = torch.clamp(S[..., 2*N:], min=0.0)
                growth_c = torch.matmul(Rt, Theta_c.t())
                growth_s = torch.matmul(Rt, Theta_s.t())
                dNc = Nc * growth_c - alpha * (Nc**2) - k_shed * Nc + k_att * Ns
                dNs = Ns * growth_s - alpha * (Ns**2) + k_shed * Nc - k_att * Ns
                # Combined consumption of shared resources
                flux = Rt * (torch.matmul(Nc, Theta_c) + torch.matmul(Ns, Theta_s))
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNc, dNs, dRt], dim=-1)
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            Nc_fin = torch.clamp(S_t[-1, :, :N], min=0.0)
            Ns_fin = torch.clamp(S_t[-1, :, N:2*N], min=0.0)
            preds_c.append(Nc_fin / (Nc_fin.sum(dim=-1, keepdim=True) + 1e-8))
            preds_s.append(Ns_fin / (Ns_fin.sum(dim=-1, keepdim=True) + 1e-8))
            N_c_curr = Nc_fin
            N_s_curr = Ns_fin
        return torch.stack(preds_c, dim=1), torch.stack(preds_s, dim=1)


def train_dual_theta_model(model, epochs=250, lr=0.015, enforce_w_zero=False, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    best_loss = float('inf')
    best_state = None
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params(enforce_w_zero=enforce_w_zero)
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        # 1. No Carrier Rollout (uses Theta_s)
        p_no_c = model.rollout_no_carrier(Theta_s, W, alpha, R0[0], b0_liq, tf_ratio=tf_ratio)
        loss_no_c = torch.sum(no_c_tensor[:, 1:, :] * torch.log(no_c_tensor[:, 1:, :] / torch.clamp(p_no_c, min=1e-9) + 1e-12), dim=-1).mean()
        
        # 2. Plain Agar Rollout (uses Theta_c and Theta_s)
        p_pl_c, p_pl_s = model.rollout_carrier(plain_c_tensor, plain_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta, tf_ratio=tf_ratio)
        loss_plain = torch.sum(plain_c_tensor[:, 1:, :] * torch.log(plain_c_tensor[:, 1:, :] / torch.clamp(p_pl_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                     torch.sum(plain_s_tensor[:, 1:, :] * torch.log(plain_s_tensor[:, 1:, :] / torch.clamp(p_pl_s, min=1e-9) + 1e-12), dim=-1).mean()
                     
        # 3. Mucin Agar Rollout (uses Theta_c and Theta_s)
        p_mu_c, p_mu_s = model.rollout_carrier(mucin_c_tensor, mucin_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta, tf_ratio=tf_ratio)
        loss_mucin = torch.sum(mucin_c_tensor[:, 1:, :] * torch.log(mucin_c_tensor[:, 1:, :] / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                     torch.sum(mucin_s_tensor[:, 1:, :] * torch.log(mucin_s_tensor[:, 1:, :] / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
                     
        l2_reg = 0.001 * (Theta_c.pow(2).sum() + Theta_s.pow(2).sum() + W.pow(2).sum())
        total_loss = loss_no_c + loss_plain + loss_mucin + l2_reg
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        
        loss_val = (loss_no_c + loss_plain + loss_mucin).item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if verbose and (epoch % 50 == 0 or epoch == epochs):
            print(f"Epoch {epoch:3d}/{epochs} | No-C: {loss_no_c.item():.3f} | Plain: {loss_plain.item():.3f} | Mucin: {loss_mucin.item():.3f} | Total: {loss_val:.3f} (TF: {tf_ratio:.2f})")
            
    model.load_state_dict(best_state)
    return model, best_loss


def evaluate_3_conditions(model, Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta):
    # 1. No Carrier (Liquid)
    p_no_c = model.rollout_no_carrier(Theta_s, W, alpha, R0[0], b0_liq, tf_ratio=0.0).cpu().numpy().flatten()
    t_no_c = no_c_tensor[:, 1:, :].cpu().numpy().flatten()
    r2_no_c = 1.0 - np.sum((t_no_c - p_no_c)**2) / np.sum((t_no_c - np.mean(t_no_c))**2)
    
    # 2. Plain Agar (Carrier + Supernatant simultaneously!)
    p_pl_c, p_pl_s = model.rollout_carrier(plain_c_tensor, plain_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta, tf_ratio=0.0)
    p_plain_simul = np.concatenate([p_pl_c.cpu().numpy().flatten(), p_pl_s.cpu().numpy().flatten()])
    t_plain_simul = np.concatenate([plain_c_tensor[:, 1:, :].cpu().numpy().flatten(), plain_s_tensor[:, 1:, :].cpu().numpy().flatten()])
    r2_plain_simul = 1.0 - np.sum((t_plain_simul - p_plain_simul)**2) / np.sum((t_plain_simul - np.mean(t_plain_simul))**2)
    
    # 3. Mucin Agar (Carrier + Supernatant simultaneously!)
    p_mu_c, p_mu_s = model.rollout_carrier(mucin_c_tensor, mucin_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta, tf_ratio=0.0)
    p_mucin_simul = np.concatenate([p_mu_c.cpu().numpy().flatten(), p_mu_s.cpu().numpy().flatten()])
    t_mucin_simul = np.concatenate([mucin_c_tensor[:, 1:, :].cpu().numpy().flatten(), mucin_s_tensor[:, 1:, :].cpu().numpy().flatten()])
    r2_mucin_simul = 1.0 - np.sum((t_mucin_simul - p_mucin_simul)**2) / np.sum((t_mucin_simul - np.mean(t_mucin_simul))**2)
    
    return np.array([r2_no_c, r2_plain_simul, r2_mucin_simul])


def main():
    print("="*70)
    print("TRAINING DUAL-THETA UNIFIED CRM (K=3) (NO CARRIER, PLAIN AGAR, MUCIN AGAR)")
    print("="*70)
    
    full_ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_DualTheta_K3_CRM.pth')
    model_full = DualThetaUnifiedCRM(N=N, K=K).to(device)
    
    if os.path.exists(full_ckpt_path):
        print(f"Loading full model from {full_ckpt_path}...")
        model_full.load_state_dict(torch.load(full_ckpt_path, map_location=device, weights_only=False)['model_state_dict'])
    else:
        print("Training full model...")
        model_full, best_loss = train_dual_theta_model(model_full, epochs=250, lr=0.015, enforce_w_zero=False)
        torch.save({'model_state_dict': model_full.state_dict(), 'best_loss': best_loss}, full_ckpt_path)
        print(f"Saved full model to {full_ckpt_path}")
        
    model_full.eval()
    
    # 2. Retrained No-W Model
    now_ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_DualTheta_K3_CRM_NoW.pth')
    model_no_w = DualThetaUnifiedCRM(N=N, K=K).to(device)
    if os.path.exists(now_ckpt_path):
        print(f"Loading No-W model from {now_ckpt_path}...")
        model_no_w.load_state_dict(torch.load(now_ckpt_path, map_location=device, weights_only=False)['model_state_dict'])
    else:
        print("Training retrained No-W model (W = 0 strictly)...")
        model_no_w, best_loss_now = train_dual_theta_model(model_no_w, epochs=250, lr=0.015, enforce_w_zero=True)
        torch.save({'model_state_dict': model_no_w.state_dict(), 'best_loss': best_loss_now}, now_ckpt_path)
        print(f"Saved No-W model to {now_ckpt_path}")
    model_no_w.eval()
    
    cond_names = ['No Carrier\n(Liquid)', 'Plain Agar\n(Carrier + Supern.)', 'Mucin Agar\n(Carrier + Supern.)']
    
    with torch.no_grad():
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model_full.get_params()
        Theta_c_np = Theta_c.detach().cpu().numpy()
        Theta_s_np = Theta_s.detach().cpu().numpy()
        W_np = W.detach().cpu().numpy()
        
        # 1. Full Model R2
        true_r2 = evaluate_3_conditions(model_full, Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta)
        print("Full Model R2 across 3 conditions:", true_r2)
        
        # 2. Retrained No-W Model R2
        Theta_c_nw, Theta_s_nw, W_nw, alpha_nw, k_shed_nw, k_attach_nw, R0_nw, b0_liq_nw, b0_c_nw, b0_s_nw, eta_nw = model_no_w.get_params(enforce_w_zero=True)
        now_r2 = evaluate_3_conditions(model_no_w, Theta_c_nw, Theta_s_nw, W_nw, alpha_nw, k_shed_nw, k_attach_nw, R0_nw, b0_liq_nw, b0_c_nw, b0_s_nw, eta_nw)
        print("Retrained No-W Model R2:", now_r2)
        
        # 3. Persistence Baseline
        p0_liq = np.repeat(no_c_tensor[:, 0:1, :].cpu().numpy(), 5, axis=1).flatten()
        t_liq = no_c_tensor[:, 1:, :].cpu().numpy().flatten()
        r2_p_liq = 1.0 - np.sum((t_liq - p0_liq)**2) / np.sum((t_liq - np.mean(t_liq))**2)
        
        p0_pl = np.concatenate([np.repeat(plain_c_tensor[:, 0:1, :].cpu().numpy(), 5, axis=1).flatten(),
                                np.repeat(plain_s_tensor[:, 0:1, :].cpu().numpy(), 5, axis=1).flatten()])
        t_pl = np.concatenate([plain_c_tensor[:, 1:, :].cpu().numpy().flatten(), plain_s_tensor[:, 1:, :].cpu().numpy().flatten()])
        r2_p_pl = 1.0 - np.sum((t_pl - p0_pl)**2) / np.sum((t_pl - np.mean(t_pl))**2)
        
        p0_mu = np.concatenate([np.repeat(mucin_c_tensor[:, 0:1, :].cpu().numpy(), 5, axis=1).flatten(),
                                np.repeat(mucin_s_tensor[:, 0:1, :].cpu().numpy(), 5, axis=1).flatten()])
        t_mu = np.concatenate([mucin_c_tensor[:, 1:, :].cpu().numpy().flatten(), mucin_s_tensor[:, 1:, :].cpu().numpy().flatten()])
        r2_p_mu = 1.0 - np.sum((t_mu - p0_mu)**2) / np.sum((t_mu - np.mean(t_mu))**2)
        persist_r2 = np.array([r2_p_liq, r2_p_pl, r2_p_mu])
        print("Persistence Baseline R2:", persist_r2)
        
        N_shuffles = 50
        # 4. Scrambled Nutrients R0 (50 runs via Dirichlet nutrient allocation preserving total nutrient pool)
        print(f"Executing {N_shuffles} random draws for Scrambled Nutrients R0 (K=3)...")
        shuff_r0_runs = []
        for _ in range(N_shuffles):
            diri = np.random.dirichlet([1.0, 1.0, 1.0], size=3)
            r0_rand = torch.tensor(diri, dtype=torch.float32, device=device) * R0.sum(dim=-1, keepdim=True)
            r2_s = evaluate_3_conditions(model_full, Theta_c, Theta_s, W, alpha, k_shed, k_attach, r0_rand, b0_liq, b0_c, b0_s, eta)
            shuff_r0_runs.append(r2_s)
        shuff_r0_runs = np.array(shuff_r0_runs)
        shuff_r0_r2 = np.mean(shuff_r0_runs, axis=0)
        shuff_r0_std = np.std(shuff_r0_runs, axis=0)
        
        # 5. Shuffled Theta (50 runs)
        print(f"Executing {N_shuffles} random permutations for Shuffled Dual-Theta...")
        shuff_th_runs = []
        for _ in range(N_shuffles):
            perm_th = torch.randperm(N)
            r2_s = evaluate_3_conditions(model_full, Theta_c[perm_th], Theta_s[perm_th], W, alpha[perm_th], k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta)
            shuff_th_runs.append(r2_s)
        shuff_th_runs = np.array(shuff_th_runs)
        shuff_th_r2 = np.mean(shuff_th_runs, axis=0)
        shuff_th_std = np.std(shuff_th_runs, axis=0)
        
        # 6. Shuffled Adhesion Traits (50 runs)
        print(f"Executing {N_shuffles} random permutations for Shuffled Adhesion Traits...")
        shuff_adh_runs = []
        for _ in range(N_shuffles):
            perm_adh = torch.randperm(N)
            r2_s = evaluate_3_conditions(model_full, Theta_c, Theta_s, W, alpha, k_shed[perm_adh], k_attach[:, perm_adh], R0, b0_liq, b0_c, b0_s, eta)
            shuff_adh_runs.append(r2_s)
        shuff_adh_runs = np.array(shuff_adh_runs)
        shuff_adh_r2 = np.mean(shuff_adh_runs, axis=0)
        shuff_adh_std = np.std(shuff_adh_runs, axis=0)
        
    print("\n--- Summary of Model and Null Performances (Mean +/- SD) ---")
    clean_names = ['No Carrier (Liquid)', 'Plain Agar (Carrier + Supern.)', 'Mucin Agar (Carrier + Supern.)']
    for i, name in enumerate(clean_names):
        print(f"{name:32s}: Full={true_r2[i]:.2f} | W=0={now_r2[i]:.2f} | Shuff_R0={shuff_r0_r2[i]:.2f} +/- {shuff_r0_std[i]:.2f} | Persist={persist_r2[i]:.2f} | Shuff_Th={shuff_th_r2[i]:.2f} +/- {shuff_th_std[i]:.2f} | Shuff_Adh={shuff_adh_r2[i]:.2f} +/- {shuff_adh_std[i]:.2f}")
        
    # Hierarchical clustering of Theta_c for order, apply to both for direct comparison!
    Z = linkage(Theta_c_np, method='ward')
    order = leaves_list(Z)
    Theta_c_clustered = Theta_c_np[order]
    Theta_s_clustered = Theta_s_np[order] # same order for row-by-row comparison
    
    # Generate Publication / Poster Figure (3 Rows)
    plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
    fig = plt.figure(figsize=(14.2, 13.5), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.55, 1.05, 1.15], hspace=0.32, wspace=0.18,
                           left=0.08, right=0.96, top=0.96, bottom=0.07)

    # -------------------------------------------------------------
    # ROW 0: Coupled Mathematical Architecture Banner (with Dual Theta)
    # -------------------------------------------------------------
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.axis('off')

    rect_bg = patches.FancyBboxPatch((0.00, 0.02), 1.00, 0.96, boxstyle='round,pad=0.02,rounding_size=0.025',
                                     facecolor='#F8FAFC', edgecolor='#00356B', lw=1.8)
    ax_eq.add_patch(rect_bg)

    ax_eq.text(0.025, 0.88, r'$\mathbf{Dual\text{-}Niche\ Consumer\text{–}Resource\ Model\ (K=3\ Latent\ Resources)}$', 
               fontsize=16.0, fontweight='bold', color='#00356B', va='top')

    ax_eq.text(0.975, 0.88, r'$\mathbf{Phenotypic\ Differentiation:\ } \mathbf{\Theta}^{(c)}\ \mathrm{(Attached\ Biofilm)}\ \mathrm{vs}\ \mathbf{\Theta}^{(s)}\ \mathrm{(Floating\ Planktonic)}$', 
               fontsize=12.0, fontweight='bold', color='#2C3E50', ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.25', facecolor='#EAEFF5', edgecolor='#8FA8C4', lw=1.0))

    eq1 = (r'$\mathbf{Carrier\ Biofilm\ (N_{i,c}):}$' + '\n' +
           r'$\frac{dN_{i,c}}{dt} = N_{i,c} \sum_{k=1}^3 \mathbf{\Theta_{ik}^{(c)}} R_k - \alpha_i N_{i,c}^2 \mathbf{- k_{\mathrm{shed},i} N_{i,c} + k_{\mathrm{attach},i}^{(e)} N_{i,s}}$' + '\n' +
           r'$\mathbf{Transfer:\ } N_{i,c}(0)^{(p+1)} = \frac{1}{5} N_{i,c}(\tau)^{(p)} + \epsilon$')

    eq2 = (r'$\mathbf{Liquid\ Supernatant\ (N_{i,s}):}$' + '\n' +
           r'$\frac{dN_{i,s}}{dt} = N_{i,s} \sum_{k=1}^3 \mathbf{\Theta_{ik}^{(s)}} R_k - \alpha_i N_{i,s}^2 \mathbf{+ k_{\mathrm{shed},i} N_{i,c} - k_{\mathrm{attach},i}^{(e)} N_{i,s}}$' + '\n' +
           r'$\mathbf{Transfer:\ } N_{i,s}(0)^{(p+1)} = \eta \cdot N_{i,s}(\tau)^{(p)} + \epsilon\ (\eta \approx 0.2\%)$')

    eq3 = (r'$\mathbf{Shared\ Media\ Resources\ (R_k):}$' + '\n' +
           r'$\frac{dR_k}{dt} = - R_k \sum_i (N_{i,c} \mathbf{\Theta_{ik}^{(c)}} + N_{i,s} \mathbf{\Theta_{ik}^{(s)}}) + \sum_{l \ne k} \mathrm{flux}_l W_{lk}$' + '\n' +
           r'$\mathbf{Replenish:\ } \mathbf{R}(0)^{(p+1)} = \mathbf{R}_0$')

    ax_eq.text(0.025, 0.38, eq1, fontsize=11.5, color='#0F2942', va='center',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='#EDF4F9', edgecolor='#1B6CA8', lw=1.2))

    ax_eq.text(0.360, 0.38, eq2, fontsize=11.5, color='#0F2942', va='center',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='#FDF3EB', edgecolor='#D95F02', lw=1.2))

    ax_eq.text(0.695, 0.38, eq3, fontsize=11.5, color='#0F2942', va='center',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='#F4F0F7', edgecolor='#7570B3', lw=1.2))

    # -------------------------------------------------------------
    # ROW 1: Two Learned Theta Matrices (Attached vs Floating)
    # -------------------------------------------------------------
    # Panel A: Attached Biofilm Traits Theta^(c)
    ax0 = fig.add_subplot(gs[1, 0])
    im0 = ax0.imshow(Theta_c_clustered, aspect='auto', cmap='YlGnBu', interpolation='nearest')
    ax0.set_box_aspect(1)
    ax0.set_anchor('W')
    badge_a = dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#00356B', alpha=0.92, lw=1.5)
    ax0.text(0.05, 0.94, r"$\mathbf{Attached\ Biofilm\ Traits\ (\Theta^{(c)})}$", transform=ax0.transAxes, fontsize=14.0, fontweight='bold', va='top', bbox=badge_a)
    ax0.set_xlabel('Resource', fontsize=16.0, fontweight='bold', color='black', labelpad=5)
    ax0.set_ylabel('Strains (N=50)', fontsize=16.0, fontweight='bold', color='black', labelpad=5)
    ax0.set_xticks(range(K))
    ax0.set_xticklabels([f'$R_{k+1}$' for k in range(K)], fontsize=14.0, fontweight='bold')
    ax0.set_yticks([])

    cax0 = inset_axes(ax0, width='4.5%', height='100%', loc='lower left',
                      bbox_to_anchor=(1.04, 0.0, 1.0, 1.0), bbox_transform=ax0.transAxes, borderpad=0)
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar0.set_label('Uptake Rate', fontsize=14.0, fontweight='bold', color='black')
    cbar0.ax.tick_params(labelsize=12.5)

    # Panel B: Floating Planktonic Traits Theta^(s)
    ax1 = fig.add_subplot(gs[1, 1])
    im1 = ax1.imshow(Theta_s_clustered, aspect='auto', cmap='YlGnBu', interpolation='nearest')
    ax1.set_box_aspect(1)
    ax1.set_anchor('W')
    badge_b = dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#00356B', alpha=0.92, lw=1.5)
    ax1.text(0.05, 0.94, r"$\mathbf{Floating\ Planktonic\ Traits\ (\Theta^{(s)})}$", transform=ax1.transAxes, fontsize=14.0, fontweight='bold', va='top', bbox=badge_b)
    ax1.set_xlabel('Resource', fontsize=16.0, fontweight='bold', color='black', labelpad=5)
    ax1.set_ylabel('Strains (N=50)', fontsize=16.0, fontweight='bold', color='black', labelpad=5)
    ax1.set_xticks(range(K))
    ax1.set_xticklabels([f'$R_{k+1}$' for k in range(K)], fontsize=14.0, fontweight='bold')
    ax1.set_yticks([])

    cax1 = inset_axes(ax1, width='4.5%', height='100%', loc='lower left',
                      bbox_to_anchor=(1.04, 0.0, 1.0, 1.0), bbox_transform=ax1.transAxes, borderpad=0)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('Uptake Rate', fontsize=14.0, fontweight='bold', color='black')
    cbar1.ax.tick_params(labelsize=12.5)

    # -------------------------------------------------------------
    # ROW 2: Benchmark Comparison across 3 Conditions (with Error Bars)
    # (Shuffled Adhesion OMITTED for No Carrier!)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[2, :])
    x = np.arange(len(cond_names))
    w = 0.135

    x_b1 = np.array([x[0] - 2.0*w, x[1] - 2.5*w, x[2] - 2.5*w])
    x_b2 = np.array([x[0] - 1.0*w, x[1] - 1.5*w, x[2] - 1.5*w])
    x_b3 = np.array([x[0] + 0.0*w, x[1] - 0.5*w, x[2] - 0.5*w])
    x_b4 = np.array([x[0] + 1.0*w, x[1] + 0.5*w, x[2] + 0.5*w])
    x_b5 = np.array([x[0] + 2.0*w, x[1] + 1.5*w, x[2] + 1.5*w])
    x_b6 = np.array([x[1] + 2.5*w, x[2] + 2.5*w])

    b1 = ax2.bar(x_b1, true_r2, width=w, label='Full Dual-Theta CRM', 
                 color='#1B6CA8', edgecolor='#0F2942', lw=1.2, zorder=3)
    b2 = ax2.bar(x_b2, now_r2, width=w, label=r'$\mathbf{W} = \mathbf{0}$', 
                 color='#E08B53', edgecolor='#803010', lw=1.2, zorder=3)
    b3 = ax2.bar(x_b3, shuff_r0_r2, width=w, yerr=shuff_r0_std, capsize=3.5,
                 label=r'Scrambled $\mathbf{R}_0$ (Dirichlet $\pm$SD)', 
                 color='#7A92A3', edgecolor='#2C3E50', lw=1.2,
                 error_kw=dict(ecolor='#2C3E50', lw=1.2, capthick=1.2), zorder=3)
    b4 = ax2.bar(x_b4, persist_r2, width=w, label='Persistence', 
                 color='#F5A623', edgecolor='#B37400', lw=1.2, zorder=3)
    b5 = ax2.bar(x_b5, shuff_th_r2, width=w, yerr=shuff_th_std, capsize=3.5,
                 label=r'Shuffled $\mathbf{\Theta}$ ($\pm$SD)', 
                 color='#D9534F', edgecolor='#701010', lw=1.2,
                 error_kw=dict(ecolor='#701010', lw=1.2, capthick=1.2), zorder=3)
    b6 = ax2.bar(x_b6, shuff_adh_r2[1:], width=w, yerr=shuff_adh_std[1:], capsize=3.5,
                 label=r'Shuffled Adhesion ($\pm$SD, Carriers only)', 
                 color='#8E44AD', edgecolor='#4A154B', lw=1.2,
                 error_kw=dict(ecolor='#4A154B', lw=1.2, capthick=1.2), zorder=3)

    ax2.axhline(0, color='black', lw=1.2, linestyle='-', zorder=4)

    # Number labels on top of bars
    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#1B6CA8')
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#803010')
    for i, bar in enumerate(b3):
        h = bar.get_height()
        err = shuff_r0_std[i]
        ax2.text(bar.get_x() + bar.get_width()/2., h + err + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#2C3E50')
    for bar in b4:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#B37400')
    for i, bar in enumerate(b5):
        h = bar.get_height()
        err = shuff_th_std[i]
        if h >= 0:
            ax2.text(bar.get_x() + bar.get_width()/2., h + err + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#701010')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., h - err - 0.04, f'{h:.2f}', ha='center', va='top', fontsize=11.0, fontweight='bold', color='#701010')
    for i, bar in enumerate(b6):
        h = bar.get_height()
        err = shuff_adh_std[i + 1]
        if h >= 0:
            ax2.text(bar.get_x() + bar.get_width()/2., h + err + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#4A154B')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., h - err - 0.04, f'{h:.2f}', ha='center', va='top', fontsize=11.0, fontweight='bold', color='#4A154B')

    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_names, rotation=0, ha='center', fontsize=15.0, fontweight='bold', color='#0F2942')
    ax2.set_ylabel(r'Coefficient of Determination ($R^2$)', fontsize=18.0, fontweight='bold', color='black', labelpad=8)
    ax2.set_ylim(-1.15, 1.45)
    ax2.tick_params(axis='y', labelsize=13.5)
    ax2.legend(loc='upper center', ncol=6, fontsize=12.0, framealpha=0.96, edgecolor='black', borderaxespad=0.5, columnspacing=0.9)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)

    out_poster_path = 'Poster/figures/Pollard_Coupled_Biofilm_Universality.png'
    os.makedirs(os.path.dirname(out_poster_path), exist_ok=True)
    fig.savefig(out_poster_path, dpi=300)
    print(f"Saved Dual-Theta (K=3) 3-row figure to {out_poster_path}")

    # Copy to artifact directory
    artifact_dir = '/home/karthik/.gemini/antigravity-cli/brain/d9158df9-e2d4-40ea-b5ee-83194049b143'
    shutil.copyfile(out_poster_path, os.path.join(artifact_dir, 'Pollard_DualTheta_K3_Universality.png'))
    print("Copied to artifact directory!")

if __name__ == '__main__':
    main()

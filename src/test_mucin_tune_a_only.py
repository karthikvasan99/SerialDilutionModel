import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location=device, weights_only=False)

no_c_tensor = data['no_c_tensor'].to(device)
plain_c_tensor = data['plain_c_tensor'].to(device)
plain_s_tensor = data['plain_s_tensor'].to(device)
mucin_c_tensor = data['mucin_c_tensor'].to(device)
mucin_s_tensor = data['mucin_s_tensor'].to(device)
N = no_c_tensor.size(-1)
strains_keep = data['strains']

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

class UnifiedCRMP0(nn.Module):
    def __init__(self, N=50, K=5, c_w=0.50):
        super().__init__()
        self.N, self.K = N, K
        self.c_w = c_w
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
        
    def get_params(self):
        Theta_c = F.softplus(self.theta_c_raw)
        Theta_s = F.softplus(self.theta_s_raw)
        alpha = F.softplus(self.alpha_raw)
        k_shed = F.softplus(self.k_shed_raw)
        k_attach = F.softplus(self.k_attach_raw)
        R0 = F.softplus(self.a)
        b0_liq = F.softplus(self.b0_liq_raw)
        b0_c = F.softplus(self.b0_c_raw)
        b0_s = F.softplus(self.b0_s_raw)
        eta = torch.tensor(1.0/30.0, device=device)
        W_sp = F.softplus(self.w_raw) * (1.0 - torch.eye(self.K, device=device))
        scale = torch.clamp(self.c_w / torch.clamp(W_sp.sum(dim=-1, keepdim=True), min=1e-6), max=1.0)
        W = W_sp * scale
        return Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta

    def rollout_carrier_from_p0(self, p0_vec, cond_idx, delta_t=0.1, tf_ratio=0.0):
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = self.get_params()
        k_att = k_attach[cond_idx - 1]
        R0_cond = R0[cond_idx]
        R0_batch = R0_cond.unsqueeze(0).expand(p0_vec.size(0), -1)
        preds_c, preds_s = [], []
        t_span = torch.linspace(0.0, 1.0, 11, device=device)
        Nc_curr = torch.zeros_like(p0_vec)
        Ns_curr = p0_vec * b0_s
        target_c = plain_c_tensor if cond_idx == 1 else mucin_c_tensor
        target_s = plain_s_tensor if cond_idx == 1 else mucin_s_tensor
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
                dNc = Nc * torch.matmul(Rt, Theta_c.t()) - alpha * (Nc**2) - k_shed * Nc + k_att * Ns
                dNs = Ns * torch.matmul(Rt, Theta_s.t()) - alpha * (Ns**2) + k_shed * Nc - k_att * Ns
                flux = Rt * (torch.matmul(Nc, Theta_c) + torch.matmul(Ns, Theta_s))
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNc, dNs, dRt], dim=-1)
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            Nc_curr = torch.clamp(S_t[-1, :, :50], min=0.0)
            Ns_curr = torch.clamp(S_t[-1, :, 50:100], min=0.0)
            preds_c.append(Nc_curr / (Nc_curr.sum(dim=-1, keepdim=True) + 1e-8))
            preds_s.append(Ns_curr / (Ns_curr.sum(dim=-1, keepdim=True) + 1e-8))
        return torch.stack(preds_c, dim=1), torch.stack(preds_s, dim=1)

    def rollout_no_carrier_from_p0(self, p0_vec, delta_t=0.1, tf_ratio=0.0):
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = self.get_params()
        R0_batch = R0[0].unsqueeze(0).expand(p0_vec.size(0), -1)
        preds = []
        t_span = torch.linspace(0.0, 1.0, 11, device=device)
        N_curr = p0_vec * b0_liq
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
                dNt = Nt * torch.matmul(Rt, Theta_s.t()) - alpha * (Nt**2)
                flux = Rt * torch.matmul(Nt, Theta_s)
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNt, dRt], dim=-1)
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :50], min=0.0)
            preds.append(N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-8))
            N_curr = N_final
        return torch.stack(preds, dim=1)

def train_base(exclude_cond='mucin', epochs=250):
    torch.manual_seed(42)
    model = UnifiedCRMP0().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        loss = 0.0
        if exclude_cond != 'no_c':
            p_no = model.rollout_no_carrier_from_p0(p0, tf_ratio=tf_ratio)
            loss += torch.sum(no_c_tensor * torch.log(no_c_tensor / torch.clamp(p_no, min=1e-9) + 1e-12), dim=-1).mean()
        if exclude_cond != 'plain':
            p_pl_c, p_pl_s = model.rollout_carrier_from_p0(p0, cond_idx=1, tf_ratio=tf_ratio)
            loss += torch.sum(plain_c_tensor * torch.log(plain_c_tensor / torch.clamp(p_pl_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                    torch.sum(plain_s_tensor * torch.log(plain_s_tensor / torch.clamp(p_pl_s, min=1e-9) + 1e-12), dim=-1).mean()
        if exclude_cond != 'mucin':
            p_mu_c, p_mu_s = model.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=tf_ratio)
            loss += torch.sum(mucin_c_tensor * torch.log(mucin_c_tensor / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                    torch.sum(mucin_s_tensor * torch.log(mucin_s_tensor / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params()
        loss += 0.001 * (Theta_c.pow(2).sum() + Theta_s.pow(2).sum() + W.pow(2).sum())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model

print("Training base model (No Carrier + Plain Agar, Mucin Agar excluded)...")
m_mu = train_base(exclude_cond='mucin')
with torch.no_grad():
    m_mu.a.data[2] = m_mu.a.data[1].clone()
    m_mu.k_attach_raw.data[1] = m_mu.k_attach_raw.data[0].clone()

t_eval = torch.cat([mucin_c_tensor[:, 1:, :], mucin_s_tensor[:, 1:, :]], dim=-1).cpu().numpy().flatten()

# Persistence from P1:
p1_c = mucin_c_tensor[:, 0:1, :].cpu().numpy()
p1_s = mucin_s_tensor[:, 0:1, :].cpu().numpy()
p_persist = np.repeat(np.concatenate([p1_c, p1_s], axis=-1), 5, axis=1).flatten()
r2_pers = compute_r2(t_eval, p_persist)
r_pers, _ = pearsonr(t_eval, p_persist)
print(f"Persistence from P1 (Predict P2..P6): R2 = {r2_pers:.4f}, r = {r_pers:.4f}")

# Pure Zero-Shot on P2..P6 (No tuning at all)
m_mu.eval()
with torch.no_grad():
    p_c, p_s = m_mu.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
    p_eval = torch.cat([p_c[:, 1:, :], p_s[:, 1:, :]], dim=-1).cpu().numpy().flatten()
    r2_zero = compute_r2(t_eval, p_eval)
    r_zero, _ = pearsonr(t_eval, p_eval)
    print(f"Pure Zero-Shot (No tuning, predict P2..P6): R2 = {r2_zero:.4f}, r = {r_zero:.4f}")

# Test 1: Tune 'a' ONLY on P0 -> P1 (freeze k_attach!)
import copy
m_a = copy.deepcopy(m_mu)
for p in m_a.parameters(): p.requires_grad = False
m_a.a.requires_grad = True
opt = torch.optim.Adam([m_a.a], lr=0.03)
for _ in range(150):
    opt.zero_grad()
    p_c, p_s = m_a.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
    loss = torch.sum(mucin_c_tensor[:, 0:1, :] * torch.log(mucin_c_tensor[:, 0:1, :] / torch.clamp(p_c[:, 0:1, :], min=1e-9) + 1e-12), dim=-1).mean() + \
           torch.sum(mucin_s_tensor[:, 0:1, :] * torch.log(mucin_s_tensor[:, 0:1, :] / torch.clamp(p_s[:, 0:1, :], min=1e-9) + 1e-12), dim=-1).mean()
    loss.backward()
    opt.step()

m_a.eval()
with torch.no_grad():
    p_c, p_s = m_a.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
    p_eval = torch.cat([p_c[:, 1:, :], p_s[:, 1:, :]], dim=-1).cpu().numpy().flatten()
    r2_tune_a = compute_r2(t_eval, p_eval)
    r_tune_a, _ = pearsonr(t_eval, p_eval)
    print(f"Tune 'a' ONLY on P0->P1 (Predict P2..P6): R2 = {r2_tune_a:.4f}, r = {r_tune_a:.4f}")


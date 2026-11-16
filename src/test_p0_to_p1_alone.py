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
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location=device, weights_only=False)

no_c_tensor = data['no_c_tensor'].to(device)       # (3, 6, 50) -> P1..P6
plain_c_tensor = data['plain_c_tensor'].to(device) # (3, 6, 50) -> P1..P6
plain_s_tensor = data['plain_s_tensor'].to(device) # (3, 6, 50) -> P1..P6
mucin_c_tensor = data['mucin_c_tensor'].to(device) # (3, 6, 50) -> P1..P6
mucin_s_tensor = data['mucin_s_tensor'].to(device) # (3, 6, 50) -> P1..P6
N = no_c_tensor.size(-1)
strains_keep = data['strains']

# Load inoculum P0
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv'))
sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
df['Relative_Abundance'] = df['Read_Fraction'] / 100.0 if sums.mean() > 50 else df['Read_Fraction']
inoc = df[df['sampleType'] == 'inoculum'].pivot_table(index='replicate', columns='Strain_Name', values='Relative_Abundance', fill_value=0.0).reindex(columns=strains_keep, fill_value=0.0)
inoc_norm = inoc.div(inoc.sum(axis=1), axis=0)
p0 = torch.tensor(inoc_norm.mean(axis=0).values, dtype=torch.float32, device=device).unsqueeze(0).expand(3, -1) # (3, 50)

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

def train_crm_base(exclude_cond='no_c', epochs=250):
    torch.manual_seed(42)
    model = UnifiedCRMP0().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        loss = 0.0
        
        if exclude_cond != 'no_c':
            p_no = model.rollout_no_carrier_from_p0(p0, tf_ratio=tf_ratio)
            loss_no = torch.sum(no_c_tensor * torch.log(no_c_tensor / torch.clamp(p_no, min=1e-9) + 1e-12), dim=-1).mean()
            loss = loss + loss_no
            
        if exclude_cond != 'plain':
            p_pl_c, p_pl_s = model.rollout_carrier_from_p0(p0, cond_idx=1, tf_ratio=tf_ratio)
            loss_pl = torch.sum(plain_c_tensor * torch.log(plain_c_tensor / torch.clamp(p_pl_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                      torch.sum(plain_s_tensor * torch.log(plain_s_tensor / torch.clamp(p_pl_s, min=1e-9) + 1e-12), dim=-1).mean()
            loss = loss + loss_pl
            
        if exclude_cond != 'mucin':
            p_mu_c, p_mu_s = model.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=tf_ratio)
            loss_mu = torch.sum(mucin_c_tensor * torch.log(mucin_c_tensor / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                      torch.sum(mucin_s_tensor * torch.log(mucin_s_tensor / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
            loss = loss + loss_mu
            
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params()
        l2 = 0.001 * (Theta_c.pow(2).sum() + Theta_s.pow(2).sum() + W.pow(2).sum())
        loss = loss + l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
    return model

print("\n" + "="*80)
print("EXPERIMENT: FINE-TUNE ON P0 -> P1 ALONE -> PREDICT P2..P6")
print("Target evaluation: Passages P2, P3, P4, P5, P6 (5 passages, 250 data points per compartment)")
print("="*80)

# =========================================================================
# 1. PLAIN AGAR
# =========================================================================
print("\n--- 1. Condition: Plain Agar ---")
m_pl = train_crm_base(exclude_cond='plain')
# Initialize plain parameters from mucin (to have good starting point)
with torch.no_grad():
    m_pl.a.data[1] = m_pl.a.data[2].clone()
    m_pl.k_attach_raw.data[0] = m_pl.k_attach_raw.data[1].clone()

for p in m_pl.parameters(): p.requires_grad = False
m_pl.a.requires_grad = True
m_pl.k_attach_raw.requires_grad = True
m_pl.b0_s_raw.requires_grad = True

opt_pl = torch.optim.Adam([m_pl.a, m_pl.k_attach_raw, m_pl.b0_s_raw], lr=0.03)
for _ in range(150):
    opt_pl.zero_grad()
    p_c, p_s = m_pl.rollout_carrier_from_p0(p0, cond_idx=1, tf_ratio=0.0)
    # Loss ON P0 -> P1 ALONE (index 0 corresponds to P1!)
    loss = torch.sum(plain_c_tensor[:, 0:1, :] * torch.log(plain_c_tensor[:, 0:1, :] / torch.clamp(p_c[:, 0:1, :], min=1e-9) + 1e-12), dim=-1).mean() + \
           torch.sum(plain_s_tensor[:, 0:1, :] * torch.log(plain_s_tensor[:, 0:1, :] / torch.clamp(p_s[:, 0:1, :], min=1e-9) + 1e-12), dim=-1).mean()
    loss.backward()
    opt_pl.step()

m_pl.eval()
with torch.no_grad():
    p_c, p_s = m_pl.rollout_carrier_from_p0(p0, cond_idx=1, tf_ratio=0.0)
    # Target: P2..P6 (indices 1..6)
    p_eval = torch.cat([p_c[:, 1:, :], p_s[:, 1:, :]], dim=-1).cpu().numpy().flatten()
    t_eval = torch.cat([plain_c_tensor[:, 1:, :], plain_s_tensor[:, 1:, :]], dim=-1).cpu().numpy().flatten()
    r2_pl_adapt = compute_r2(t_eval, p_eval)
    r_pl_adapt, _ = pearsonr(t_eval, p_eval)
    
    # Persistence baseline ON EXACT SAME FOOTING: holding P1 constant to predict P2..P6
    p1_c = plain_c_tensor[:, 0:1, :].cpu().numpy()
    p1_s = plain_s_tensor[:, 0:1, :].cpu().numpy()
    p_persist = np.repeat(np.concatenate([p1_c, p1_s], axis=-1), 5, axis=1).flatten()
    r2_pl_pers = compute_r2(t_eval, p_persist)
    r_pl_pers, _ = pearsonr(t_eval, p_persist)

print(f"Plain Agar (Predict P2..P6):")
print(f"  CRM Adapted on P0->P1: R2 = {r2_pl_adapt:.4f}, r = {r_pl_adapt:.4f}")
print(f"  Persistence from P1:   R2 = {r2_pl_pers:.4f}, r = {r_pl_pers:.4f}")
print(f"  Difference (CRM - Persistence): Delta R2 = {r2_pl_adapt - r2_pl_pers:+.4f}")

# =========================================================================
# 2. MUCIN AGAR
# =========================================================================
print("\n--- 2. Condition: Mucin Agar ---")
m_mu = train_crm_base(exclude_cond='mucin')
# Initialize mucin parameters from plain
with torch.no_grad():
    m_mu.a.data[2] = m_mu.a.data[1].clone()
    m_mu.k_attach_raw.data[1] = m_mu.k_attach_raw.data[0].clone()

for p in m_mu.parameters(): p.requires_grad = False
m_mu.a.requires_grad = True
m_mu.k_attach_raw.requires_grad = True
m_mu.b0_s_raw.requires_grad = True

opt_mu = torch.optim.Adam([m_mu.a, m_mu.k_attach_raw, m_mu.b0_s_raw], lr=0.03)
for _ in range(150):
    opt_mu.zero_grad()
    p_c, p_s = m_mu.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
    # Loss ON P0 -> P1 ALONE
    loss = torch.sum(mucin_c_tensor[:, 0:1, :] * torch.log(mucin_c_tensor[:, 0:1, :] / torch.clamp(p_c[:, 0:1, :], min=1e-9) + 1e-12), dim=-1).mean() + \
           torch.sum(mucin_s_tensor[:, 0:1, :] * torch.log(mucin_s_tensor[:, 0:1, :] / torch.clamp(p_s[:, 0:1, :], min=1e-9) + 1e-12), dim=-1).mean()
    loss.backward()
    opt_mu.step()

m_mu.eval()
with torch.no_grad():
    p_c, p_s = m_mu.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
    # Target: P2..P6
    p_eval = torch.cat([p_c[:, 1:, :], p_s[:, 1:, :]], dim=-1).cpu().numpy().flatten()
    t_eval = torch.cat([mucin_c_tensor[:, 1:, :], mucin_s_tensor[:, 1:, :]], dim=-1).cpu().numpy().flatten()
    r2_mu_adapt = compute_r2(t_eval, p_eval)
    r_mu_adapt, _ = pearsonr(t_eval, p_eval)
    
    # Persistence baseline ON EXACT SAME FOOTING: holding P1 constant to predict P2..P6
    p1_c = mucin_c_tensor[:, 0:1, :].cpu().numpy()
    p1_s = mucin_s_tensor[:, 0:1, :].cpu().numpy()
    p_persist = np.repeat(np.concatenate([p1_c, p1_s], axis=-1), 5, axis=1).flatten()
    r2_mu_pers = compute_r2(t_eval, p_persist)
    r_mu_pers, _ = pearsonr(t_eval, p_persist)

print(f"Mucin Agar (Predict P2..P6):")
print(f"  CRM Adapted on P0->P1: R2 = {r2_mu_adapt:.4f}, r = {r_mu_adapt:.4f}")
print(f"  Persistence from P1:   R2 = {r2_mu_pers:.4f}, r = {r_mu_pers:.4f}")
print(f"  Difference (CRM - Persistence): Delta R2 = {r2_mu_adapt - r2_mu_pers:+.4f}")

# =========================================================================
# 3. NO CARRIER
# =========================================================================
print("\n--- 3. Condition: No Carrier ---")
m_no = train_crm_base(exclude_cond='no_c')
with torch.no_grad():
    m_no.a.data[0] = (m_no.a.data[1] + m_no.a.data[2]) / 2.0

for p in m_no.parameters(): p.requires_grad = False
m_no.a.requires_grad = True
m_no.b0_liq_raw.requires_grad = True

opt_no = torch.optim.Adam([m_no.a, m_no.b0_liq_raw], lr=0.03)
for _ in range(150):
    opt_no.zero_grad()
    p_ro = m_no.rollout_no_carrier_from_p0(p0, tf_ratio=0.0)
    loss = torch.sum(no_c_tensor[:, 0:1, :] * torch.log(no_c_tensor[:, 0:1, :] / torch.clamp(p_ro[:, 0:1, :], min=1e-9) + 1e-12), dim=-1).mean()
    loss.backward()
    opt_no.step()

m_no.eval()
with torch.no_grad():
    p_ro = m_no.rollout_no_carrier_from_p0(p0, tf_ratio=0.0)
    p_eval = p_ro[:, 1:, :].cpu().numpy().flatten()
    t_eval = no_c_tensor[:, 1:, :].cpu().numpy().flatten()
    r2_no_adapt = compute_r2(t_eval, p_eval)
    r_no_adapt, _ = pearsonr(t_eval, p_eval)
    
    # Persistence baseline holding P1 constant
    p1_no = no_c_tensor[:, 0:1, :].cpu().numpy()
    p_persist = np.repeat(p1_no, 5, axis=1).flatten()
    r2_no_pers = compute_r2(t_eval, p_persist)
    r_no_pers, _ = pearsonr(t_eval, p_persist)

print(f"No Carrier (Predict P2..P6):")
print(f"  CRM Adapted on P0->P1: R2 = {r2_no_adapt:.4f}, r = {r_no_adapt:.4f}")
print(f"  Persistence from P1:   R2 = {r2_no_pers:.4f}, r = {r_no_pers:.4f}")
print(f"  Difference (CRM - Persistence): Delta R2 = {r2_no_adapt - r2_no_pers:+.4f}")

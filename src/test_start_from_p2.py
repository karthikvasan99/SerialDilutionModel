import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location=device, weights_only=False)

no_c_tensor = data['no_c_tensor'].to(device)
plain_c_tensor = data['plain_c_tensor'].to(device)
plain_s_tensor = data['plain_s_tensor'].to(device)
mucin_c_tensor = data['mucin_c_tensor'].to(device)
mucin_s_tensor = data['mucin_s_tensor'].to(device)
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

def compute_kl(true_p, pred_p, eps=1e-9):
    p = np.clip(true_p, eps, 1.0)
    q = np.clip(pred_p, eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    kl = np.sum(p * np.log(p / q), axis=-1)
    return np.mean(kl)

t_eval_pl = torch.cat([plain_c_tensor[:, 2:, :], plain_s_tensor[:, 2:, :]], dim=-1).cpu().numpy().flatten()
t_eval_mu = torch.cat([mucin_c_tensor[:, 2:, :], mucin_s_tensor[:, 2:, :]], dim=-1).cpu().numpy().flatten()

# Persistence baselines
p2_pl_c = plain_c_tensor[:, 1:2, :].cpu().numpy()
p2_pl_s = plain_s_tensor[:, 1:2, :].cpu().numpy()
persist_pl = np.repeat(np.concatenate([p2_pl_c, p2_pl_s], axis=-1), 4, axis=1).flatten()
r2_persist_pl = compute_r2(t_eval_pl, persist_pl)
r_persist_pl, _ = pearsonr(t_eval_pl, persist_pl)

p2_mu_c = mucin_c_tensor[:, 1:2, :].cpu().numpy()
p2_mu_s = mucin_s_tensor[:, 1:2, :].cpu().numpy()
persist_mu = np.repeat(np.concatenate([p2_mu_c, p2_mu_s], axis=-1), 4, axis=1).flatten()
r2_persist_mu = compute_r2(t_eval_mu, persist_mu)
r_persist_mu, _ = pearsonr(t_eval_mu, persist_mu)

print("="*85)
print("BENCHMARK PERSISTENCE BASELINES (Holding observed P2 constant -> P3..P6):")
print(f"  Plain Agar Persistence: R2 = {r2_persist_pl:.4f} (r = {r_persist_pl:.4f})")
print(f"  Mucin Agar Persistence: R2 = {r2_persist_mu:.4f} (r = {r_persist_mu:.4f})")
print("="*85)

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

    def rollout_carrier(self, p0_vec, cond_idx, reset_at_p2=False, delta_t=0.1, tf_ratio=0.0):
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
        
        for p in range(6): # p=0 is P0->P1, p=1 is P1->P2, p=2 is P2->P3, etc.
            if p == 0:
                Nc_0 = Nc_curr
                Ns_0 = Ns_curr
            else:
                if p == 2 and reset_at_p2:
                    # EXACT STATE INITIALIZATION FROM OBSERVED P2:
                    # Reset relative abundance to ground truth P2, preserving simulated total biomass scale
                    total_Nc = Nc_curr.sum(dim=-1, keepdim=True)
                    total_Ns = Ns_curr.sum(dim=-1, keepdim=True)
                    Nc_reset = target_c[:, 1, :] * total_Nc
                    Ns_reset = target_s[:, 1, :] * total_Ns
                    Nc_0 = torch.clamp(0.2 * Nc_reset + 1e-6, min=0.0)
                    Ns_0 = torch.clamp(eta * Ns_reset + 1e-6, min=0.0)
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

    def rollout_no_carrier(self, p0_vec, delta_t=0.1, tf_ratio=0.0):
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

def train_base(exclude_cond='plain', epochs=250):
    torch.manual_seed(42)
    model = UnifiedCRMP0().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        loss = 0.0
        if exclude_cond != 'no_c':
            p_no = model.rollout_no_carrier(p0, tf_ratio=tf_ratio)
            loss += torch.sum(no_c_tensor * torch.log(no_c_tensor / torch.clamp(p_no, min=1e-9) + 1e-12), dim=-1).mean()
        if exclude_cond != 'plain':
            p_pl_c, p_pl_s = model.rollout_carrier(p0, cond_idx=1, tf_ratio=tf_ratio)
            loss += torch.sum(plain_c_tensor * torch.log(plain_c_tensor / torch.clamp(p_pl_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                    torch.sum(plain_s_tensor * torch.log(plain_s_tensor / torch.clamp(p_pl_s, min=1e-9) + 1e-12), dim=-1).mean()
        if exclude_cond != 'mucin':
            p_mu_c, p_mu_s = model.rollout_carrier(p0, cond_idx=2, tf_ratio=tf_ratio)
            loss += torch.sum(mucin_c_tensor * torch.log(mucin_c_tensor / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                    torch.sum(mucin_s_tensor * torch.log(mucin_s_tensor / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params()
        loss += 0.001 * (Theta_c.pow(2).sum() + Theta_s.pow(2).sum() + W.pow(2).sum())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model

print("Base training ready. Training base models...")
base_plain = train_base(exclude_cond='plain', epochs=250)
base_mucin = train_base(exclude_cond='mucin', epochs=250)

print("\n" + "="*85)
print("TESTING MODEL FORECAST STARTING FROM OBSERVED P2 COMPOSITION")
print("="*85)

# 1. PLAIN AGAR
print("\n--- 1. Plain Agar ---")
m_pl = copy.deepcopy(base_plain)
with torch.no_grad():
    m_pl.a.data[1] = m_pl.a.data[2].clone()
    m_pl.k_attach_raw.data[0] = m_pl.k_attach_raw.data[1].clone()
init_k = m_pl.k_attach_raw.data[0].clone()
for p in m_pl.parameters(): p.requires_grad = False
m_pl.a.requires_grad = True
m_pl.k_attach_raw.requires_grad = True

opt = torch.optim.Adam([
    {'params': [m_pl.a], 'lr': 0.02},
    {'params': [m_pl.k_attach_raw], 'lr': 0.02}
])

# Fine-tune on P0 -> P1 -> P2
for _ in range(150):
    opt.zero_grad()
    p_c, p_s = m_pl.rollout_carrier(p0, cond_idx=1, reset_at_p2=False, tf_ratio=0.0)
    p_c_tr, p_s_tr = p_c[:, :2, :], p_s[:, :2, :]
    t_c_tr, t_s_tr = plain_c_tensor[:, :2, :], plain_s_tensor[:, :2, :]
    loss = torch.sum(t_c_tr * torch.log(t_c_tr / torch.clamp(p_c_tr, min=1e-9) + 1e-12), dim=-1).mean() + \
           torch.sum(t_s_tr * torch.log(t_s_tr / torch.clamp(p_s_tr, min=1e-9) + 1e-12), dim=-1).mean() + \
           10.0 * (F.mse_loss(p_c_tr, t_c_tr) + F.mse_loss(p_s_tr, t_s_tr)) + \
           0.001 * torch.sum((m_pl.k_attach_raw[0] - init_k)**2)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m_pl.parameters(), 5.0)
    opt.step()

m_pl.eval()
with torch.no_grad():
    # A. Autonomous from P0 (never seeing observed P2 for rollout)
    p_c_auto, p_s_auto = m_pl.rollout_carrier(p0, cond_idx=1, reset_at_p2=False, tf_ratio=0.0)
    eval_auto = torch.cat([p_c_auto[:, 2:, :], p_s_auto[:, 2:, :]], dim=-1).cpu().numpy().flatten()
    r2_pl_auto = compute_r2(t_eval_pl, eval_auto)
    r_pl_auto, _ = pearsonr(t_eval_pl, eval_auto)
    
    # B. Forecasting starting from observed P2 state (reset_at_p2=True)
    p_c_p2, p_s_p2 = m_pl.rollout_carrier(p0, cond_idx=1, reset_at_p2=True, tf_ratio=0.0)
    eval_p2 = torch.cat([p_c_p2[:, 2:, :], p_s_p2[:, 2:, :]], dim=-1).cpu().numpy().flatten()
    r2_pl_from_p2 = compute_r2(t_eval_pl, eval_p2)
    r_pl_from_p2, _ = pearsonr(t_eval_pl, eval_p2)

print(f"Plain Agar (Predict P3..P6):")
print(f"  Persistence Baseline (Holding P2):       R2 = {r2_persist_pl:.4f} (r = {r_persist_pl:.4f})")
print(f"  Autonomous CRM from P0:                  R2 = {r2_pl_auto:.4f} (r = {r_pl_auto:.4f})")
print(f"  Model Forecast starting from observed P2: R2 = {r2_pl_from_p2:.4f} (r = {r_pl_from_p2:.4f})")

# 2. MUCIN AGAR
print("\n--- 2. Mucin Agar ---")
m_mu = copy.deepcopy(base_mucin)
with torch.no_grad():
    m_mu.a.data[2] = m_mu.a.data[1].clone()
    m_mu.k_attach_raw.data[1] = m_mu.k_attach_raw.data[0].clone()
init_k_mu = m_mu.k_attach_raw.data[1].clone()
for p in m_mu.parameters(): p.requires_grad = False
m_mu.a.requires_grad = True
m_mu.k_attach_raw.requires_grad = True

opt_mu = torch.optim.Adam([
    {'params': [m_mu.a], 'lr': 0.02},
    {'params': [m_mu.k_attach_raw], 'lr': 0.02}
])

for _ in range(150):
    opt_mu.zero_grad()
    p_c, p_s = m_mu.rollout_carrier(p0, cond_idx=2, reset_at_p2=False, tf_ratio=0.0)
    p_c_tr, p_s_tr = p_c[:, :2, :], p_s[:, :2, :]
    t_c_tr, t_s_tr = mucin_c_tensor[:, :2, :], mucin_s_tensor[:, :2, :]
    loss = torch.sum(t_c_tr * torch.log(t_c_tr / torch.clamp(p_c_tr, min=1e-9) + 1e-12), dim=-1).mean() + \
           torch.sum(t_s_tr * torch.log(t_s_tr / torch.clamp(p_s_tr, min=1e-9) + 1e-12), dim=-1).mean() + \
           10.0 * (F.mse_loss(p_c_tr, t_c_tr) + F.mse_loss(p_s_tr, t_s_tr)) + \
           0.001 * torch.sum((m_mu.k_attach_raw[1] - init_k_mu)**2)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m_mu.parameters(), 5.0)
    opt_mu.step()

m_mu.eval()
with torch.no_grad():
    p_c_auto, p_s_auto = m_mu.rollout_carrier(p0, cond_idx=2, reset_at_p2=False, tf_ratio=0.0)
    eval_auto = torch.cat([p_c_auto[:, 2:, :], p_s_auto[:, 2:, :]], dim=-1).cpu().numpy().flatten()
    r2_mu_auto = compute_r2(t_eval_mu, eval_auto)
    r_mu_auto, _ = pearsonr(t_eval_mu, eval_auto)
    
    p_c_p2, p_s_p2 = m_mu.rollout_carrier(p0, cond_idx=2, reset_at_p2=True, tf_ratio=0.0)
    eval_p2 = torch.cat([p_c_p2[:, 2:, :], p_s_p2[:, 2:, :]], dim=-1).cpu().numpy().flatten()
    r2_mu_from_p2 = compute_r2(t_eval_mu, eval_p2)
    r_mu_from_p2, _ = pearsonr(t_eval_mu, eval_p2)

print(f"Mucin Agar (Predict P3..P6):")
print(f"  Persistence Baseline (Holding P2):       R2 = {r2_persist_mu:.4f} (r = {r_persist_mu:.4f})")
print(f"  Autonomous CRM from P0:                  R2 = {r2_mu_auto:.4f} (r = {r_mu_auto:.4f})")
print(f"  Model Forecast starting from observed P2: R2 = {r2_mu_from_p2:.4f} (r = {r_mu_from_p2:.4f})")

# Passage by passage for forecast from P2
print("\n" + "="*85)
print("PASSAGE-BY-PASSAGE BREAKDOWN: FORECAST FROM OBSERVED P2 vs PERSISTENCE")
print("="*85)

for name, t_c, t_s, p_c_m, p_s_m in [("Plain Agar", plain_c_tensor, plain_s_tensor, p_c_p2, p_s_p2),
                                     ("Mucin Agar", mucin_c_tensor, mucin_s_tensor, p_c_p2, p_s_p2)]:
    print(f"\nCondition: {name}")
    t_c_np = t_c.cpu().numpy()
    t_s_np = t_s.cpu().numpy()
    p_c_np = p_c_m.cpu().numpy()
    p_s_np = p_s_m.cpu().numpy()
    
    pers_c_np = np.repeat(t_c_np[:, 1:2, :], 6, axis=1)
    pers_s_np = np.repeat(t_s_np[:, 1:2, :], 6, axis=1)
    
    for p_idx, p_name in enumerate(["P3 (Day 9)", "P4 (Day 12)", "P5 (Day 15)", "P6 (Day 18)"], start=2):
        t_pass = np.concatenate([t_c_np[:, p_idx, :], t_s_np[:, p_idx, :]], axis=-1).flatten()
        m_pass = np.concatenate([p_c_np[:, p_idx, :], p_s_np[:, p_idx, :]], axis=-1).flatten()
        p_pass = np.concatenate([pers_c_np[:, p_idx, :], pers_s_np[:, p_idx, :]], axis=-1).flatten()
        
        r2_m = compute_r2(t_pass, m_pass)
        r2_p = compute_r2(t_pass, p_pass)
        r_m, _ = pearsonr(t_pass, m_pass)
        r_p, _ = pearsonr(t_pass, p_pass)
        
        diff = r2_m - r2_p
        outcome = f"CRM WINS by {diff:+.4f}" if r2_m > r2_p else f"Persist wins by {-diff:+.4f}"
        print(f"  {p_name}: CRM R2 = {r2_m:.4f} (r={r_m:.4f}) | Persistence R2 = {r2_p:.4f} (r={r_p:.4f}) --> {outcome}")


import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
from scipy.stats import pearsonr
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from search_hyperparams_beat_persistence import (
    base_for_plain, base_for_mucin, p0, 
    plain_c_tensor, plain_s_tensor, mucin_c_tensor, mucin_s_tensor,
    t_eval_pl, t_eval_mu, r2_persist_pl, r2_persist_mu, compute_r2, device
)

print("="*85)
print("TESTING ADAPTATION WITH EXTENDED ITERATIONS & ROLLOUT FROM P2")
print("="*85)

# Function to run rollout starting directly from P2 (Passage 2)
def rollout_from_p2(model, cond_idx):
    target_c = plain_c_tensor if cond_idx == 1 else mucin_c_tensor
    target_s = plain_s_tensor if cond_idx == 1 else mucin_s_tensor
    Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params()
    k_att = k_attach[cond_idx - 1]
    R0_cond = R0[cond_idx]
    R0_batch = R0_cond.unsqueeze(0).expand(target_c.size(0), -1)
    
    # Start at P2: target_c[:, 1, :], target_s[:, 1, :]
    # Need to dilute for P3: 0.2 * Nc_2, eta * Ns_2
    Nc_p2 = target_c[:, 1, :] * b0_c
    Ns_p2 = target_s[:, 1, :] * b0_s
    
    preds_c, preds_s = [], []
    t_span = torch.linspace(0.0, 1.0, 11, device=device)
    
    Nc_curr = Nc_p2
    Ns_curr = Ns_p2
    
    # Rollout for P3, P4, P5, P6 (4 passages)
    for p in range(4):
        Nc_0 = torch.clamp(0.2 * Nc_curr + 1e-6, min=0.0)
        Ns_0 = torch.clamp(eta * Ns_curr + 1e-6, min=0.0)
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
        S_t = odeint(f, S0, t_span, method='euler', options={'step_size': 0.1})
        Nc_curr = torch.clamp(S_t[-1, :, :50], min=0.0)
        Ns_curr = torch.clamp(S_t[-1, :, 50:100], min=0.0)
        preds_c.append(Nc_curr / (Nc_curr.sum(dim=-1, keepdim=True) + 1e-8))
        preds_s.append(Ns_curr / (Ns_curr.sum(dim=-1, keepdim=True) + 1e-8))
        
    return torch.stack(preds_c, dim=1), torch.stack(preds_s, dim=1)

# Fine-tune with best config: lr_a=0.02, lr_att=0.02, l2=0.001, hybrid loss
for iters in [180, 250, 350]:
    print(f"\n--- Testing iters = {iters} ---")
    
    # 1. Plain Agar
    m_pl = copy.deepcopy(base_for_plain)
    with torch.no_grad():
        m_pl.a.data[1] = m_pl.a.data[2].clone()
        m_pl.k_attach_raw.data[0] = m_pl.k_attach_raw.data[1].clone()
    init_k = m_pl.k_attach_raw.data[0].clone()
    for p in m_pl.parameters(): p.requires_grad = False
    m_pl.a.requires_grad = True
    m_pl.k_attach_raw.requires_grad = True
    m_pl.b0_s_raw.requires_grad = True
    m_pl.b0_c_raw.requires_grad = True
    
    opt = torch.optim.Adam([
        {'params': [m_pl.a], 'lr': 0.02},
        {'params': [m_pl.k_attach_raw], 'lr': 0.02},
        {'params': [m_pl.b0_s_raw, m_pl.b0_c_raw], 'lr': 0.02}
    ])
    
    for _ in range(iters):
        opt.zero_grad()
        p_c, p_s = m_pl.rollout_carrier_from_p0(p0, cond_idx=1, tf_ratio=0.0)
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
        # Mode A: Autonomous rollout from P0
        p_c_p0, p_s_p0 = m_pl.rollout_carrier_from_p0(p0, cond_idx=1, tf_ratio=0.0)
        p_eval_p0 = torch.cat([p_c_p0[:, 2:, :], p_s_p0[:, 2:, :]], dim=-1).cpu().numpy().flatten()
        r2_pl_p0 = compute_r2(t_eval_pl, p_eval_p0)
        r_pl_p0, _ = pearsonr(t_eval_pl, p_eval_p0)
        
        # Mode B: Rollout starting from P2 (same footing as persistence)
        p_c_p2, p_s_p2 = rollout_from_p2(m_pl, cond_idx=1)
        p_eval_p2 = torch.cat([p_c_p2, p_s_p2], dim=-1).cpu().numpy().flatten()
        r2_pl_p2 = compute_r2(t_eval_pl, p_eval_p2)
        r_pl_p2, _ = pearsonr(t_eval_pl, p_eval_p2)
        
    print(f"Plain Agar (iters={iters}):")
    print(f"  Autonomous from P0: R2 = {r2_pl_p0:.4f} (r = {r_pl_p0:.4f})")
    print(f"  Rollout from P2:    R2 = {r2_pl_p2:.4f} (r = {r_pl_p2:.4f})")
    print(f"  Persistence Baseline: R2 = {r2_persist_pl:.4f}")
    
    # 2. Mucin Agar
    m_mu = copy.deepcopy(base_for_mucin)
    with torch.no_grad():
        m_mu.a.data[2] = m_mu.a.data[1].clone()
        m_mu.k_attach_raw.data[1] = m_mu.k_attach_raw.data[0].clone()
    init_k = m_mu.k_attach_raw.data[1].clone()
    for p in m_mu.parameters(): p.requires_grad = False
    m_mu.a.requires_grad = True
    m_mu.k_attach_raw.requires_grad = True
    m_mu.b0_s_raw.requires_grad = True
    m_mu.b0_c_raw.requires_grad = True
    
    opt = torch.optim.Adam([
        {'params': [m_mu.a], 'lr': 0.02},
        {'params': [m_mu.k_attach_raw], 'lr': 0.02},
        {'params': [m_mu.b0_s_raw, m_mu.b0_c_raw], 'lr': 0.02}
    ])
    
    for _ in range(iters):
        opt.zero_grad()
        p_c, p_s = m_mu.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
        p_c_tr, p_s_tr = p_c[:, :2, :], p_s[:, :2, :]
        t_c_tr, t_s_tr = mucin_c_tensor[:, :2, :], mucin_s_tensor[:, :2, :]
        loss = torch.sum(t_c_tr * torch.log(t_c_tr / torch.clamp(p_c_tr, min=1e-9) + 1e-12), dim=-1).mean() + \
               torch.sum(t_s_tr * torch.log(t_s_tr / torch.clamp(p_s_tr, min=1e-9) + 1e-12), dim=-1).mean() + \
               10.0 * (F.mse_loss(p_c_tr, t_c_tr) + F.mse_loss(p_s_tr, t_s_tr)) + \
               0.001 * torch.sum((m_mu.k_attach_raw[1] - init_k)**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_mu.parameters(), 5.0)
        opt.step()
        
    m_mu.eval()
    with torch.no_grad():
        p_c_p0, p_s_p0 = m_mu.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
        p_eval_p0 = torch.cat([p_c_p0[:, 2:, :], p_s_p0[:, 2:, :]], dim=-1).cpu().numpy().flatten()
        r2_mu_p0 = compute_r2(t_eval_mu, p_eval_p0)
        r_mu_p0, _ = pearsonr(t_eval_mu, p_eval_p0)
        
        p_c_p2, p_s_p2 = rollout_from_p2(m_mu, cond_idx=2)
        p_eval_p2 = torch.cat([p_c_p2, p_s_p2], dim=-1).cpu().numpy().flatten()
        r2_mu_p2 = compute_r2(t_eval_mu, p_eval_p2)
        r_mu_p2, _ = pearsonr(t_eval_mu, p_eval_p2)
        
    print(f"Mucin Agar (iters={iters}):")
    print(f"  Autonomous from P0: R2 = {r2_mu_p0:.4f} (r = {r_mu_p0:.4f})")
    print(f"  Rollout from P2:    R2 = {r2_mu_p2:.4f} (r = {r_mu_p2:.4f})")
    print(f"  Persistence Baseline: R2 = {r2_persist_mu:.4f}")

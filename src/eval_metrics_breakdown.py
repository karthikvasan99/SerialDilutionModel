import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from search_hyperparams_beat_persistence import (
    base_for_plain, base_for_mucin, p0,
    plain_c_tensor, plain_s_tensor, mucin_c_tensor, mucin_s_tensor,
    compute_r2, device
)

def compute_kl(true_p, pred_p, eps=1e-9):
    # true_p, pred_p shape (..., N)
    p = np.clip(true_p, eps, 1.0)
    q = np.clip(pred_p, eps, 1.0)
    # Normalized
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    kl = np.sum(p * np.log(p / q), axis=-1)
    return np.mean(kl)

def compute_bray_curtis(true_p, pred_p):
    # BC = sum(|p - q|) / sum(p + q)
    num = np.sum(np.abs(true_p - pred_p), axis=-1)
    denom = np.sum(true_p + pred_p, axis=-1)
    return np.mean(num / (denom + 1e-12))

print("="*85)
print("EVALUATING PASSAGE-BY-PASSAGE BREAKDOWN, P6 ENDPOINT, AND KL DIVERGENCE")
print("Target: Plain Agar and Mucin Agar (Adapted on P0 -> P1 -> P2)")
print("="*85)

# Train best model for Plain Agar
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

for _ in range(250):
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
    p_c_pl, p_s_pl = m_pl.rollout_carrier_from_p0(p0, cond_idx=1, tf_ratio=0.0)
    p_c_pl = p_c_pl.cpu().numpy() # (3, 6, 50)
    p_s_pl = p_s_pl.cpu().numpy()

# Train best model for Mucin Agar
m_mu = copy.deepcopy(base_for_mucin)
with torch.no_grad():
    m_mu.a.data[2] = m_mu.a.data[1].clone()
    m_mu.k_attach_raw.data[1] = m_mu.k_attach_raw.data[0].clone()
init_k_mu = m_mu.k_attach_raw.data[1].clone()
for p in m_mu.parameters(): p.requires_grad = False
m_mu.a.requires_grad = True
m_mu.k_attach_raw.requires_grad = True
m_mu.b0_s_raw.requires_grad = True
m_mu.b0_c_raw.requires_grad = True

opt_mu = torch.optim.Adam([
    {'params': [m_mu.a], 'lr': 0.02},
    {'params': [m_mu.k_attach_raw], 'lr': 0.02},
    {'params': [m_mu.b0_s_raw, m_mu.b0_c_raw], 'lr': 0.02}
])

for _ in range(250):
    opt_mu.zero_grad()
    p_c, p_s = m_mu.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
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
    p_c_mu, p_s_mu = m_mu.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
    p_c_mu = p_c_mu.cpu().numpy()
    p_s_mu = p_s_mu.cpu().numpy()

# Ground truth tensors
t_c_pl = plain_c_tensor.cpu().numpy()
t_s_pl = plain_s_tensor.cpu().numpy()
t_c_mu = mucin_c_tensor.cpu().numpy()
t_s_mu = mucin_s_tensor.cpu().numpy()

# Persistence predictions (holding P2 constant for all p >= 2)
# Index 1 is P2!
pers_c_pl = np.repeat(t_c_pl[:, 1:2, :], 6, axis=1)
pers_s_pl = np.repeat(t_s_pl[:, 1:2, :], 6, axis=1)

pers_c_mu = np.repeat(t_c_mu[:, 1:2, :], 6, axis=1)
pers_s_mu = np.repeat(t_s_mu[:, 1:2, :], 6, axis=1)

print("\n" + "="*85)
print("1. PLAIN AGAR PASSAGE-BY-PASSAGE BREAKDOWN")
print("="*85)

for p_idx, p_name in enumerate(["P3 (Day 9)", "P4 (Day 12)", "P5 (Day 15)", "P6 (Day 18)"], start=2):
    # Combine compartments for this passage
    t_pass = np.concatenate([t_c_pl[:, p_idx, :], t_s_pl[:, p_idx, :]], axis=-1).flatten()
    m_pass = np.concatenate([p_c_pl[:, p_idx, :], p_s_pl[:, p_idx, :]], axis=-1).flatten()
    pers_pass = np.concatenate([pers_c_pl[:, p_idx, :], pers_s_pl[:, p_idx, :]], axis=-1).flatten()
    
    r2_m = compute_r2(t_pass, m_pass)
    r2_p = compute_r2(t_pass, pers_pass)
    r_m, _ = pearsonr(t_pass, m_pass)
    r_p, _ = pearsonr(t_pass, pers_pass)
    
    # KL divergence
    kl_m = (compute_kl(t_c_pl[:, p_idx, :], p_c_pl[:, p_idx, :]) + compute_kl(t_s_pl[:, p_idx, :], p_s_pl[:, p_idx, :])) / 2.0
    kl_p = (compute_kl(t_c_pl[:, p_idx, :], pers_c_pl[:, p_idx, :]) + compute_kl(t_s_pl[:, p_idx, :], pers_s_pl[:, p_idx, :])) / 2.0
    
    # Bray-Curtis
    bc_m = (compute_bray_curtis(t_c_pl[:, p_idx, :], p_c_pl[:, p_idx, :]) + compute_bray_curtis(t_s_pl[:, p_idx, :], p_s_pl[:, p_idx, :])) / 2.0
    bc_p = (compute_bray_curtis(t_c_pl[:, p_idx, :], pers_c_pl[:, p_idx, :]) + compute_bray_curtis(t_s_pl[:, p_idx, :], pers_s_pl[:, p_idx, :])) / 2.0
    
    beat_r2 = "CRM WINS!" if r2_m > r2_p else "Persist wins"
    beat_kl = "CRM WINS!" if kl_m < kl_p else "Persist wins"
    beat_bc = "CRM WINS!" if bc_m < bc_p else "Persist wins"
    
    print(f"\nPassage {p_name}:")
    print(f"  R2:          CRM = {r2_m:.4f} (r={r_m:.4f})  |  Persistence = {r2_p:.4f} (r={r_p:.4f})  --> {beat_r2}")
    print(f"  KL Div:      CRM = {kl_m:.4f}           |  Persistence = {kl_p:.4f}           --> {beat_kl}")
    print(f"  Bray-Curtis: CRM = {bc_m:.4f}           |  Persistence = {bc_p:.4f}           --> {beat_bc}")

print("\n" + "="*85)
print("2. MUCIN AGAR PASSAGE-BY-PASSAGE BREAKDOWN")
print("="*85)

for p_idx, p_name in enumerate(["P3 (Day 9)", "P4 (Day 12)", "P5 (Day 15)", "P6 (Day 18)"], start=2):
    t_pass = np.concatenate([t_c_mu[:, p_idx, :], t_s_mu[:, p_idx, :]], axis=-1).flatten()
    m_pass = np.concatenate([p_c_mu[:, p_idx, :], p_s_mu[:, p_idx, :]], axis=-1).flatten()
    pers_pass = np.concatenate([pers_c_mu[:, p_idx, :], pers_s_mu[:, p_idx, :]], axis=-1).flatten()
    
    r2_m = compute_r2(t_pass, m_pass)
    r2_p = compute_r2(t_pass, pers_pass)
    r_m, _ = pearsonr(t_pass, m_pass)
    r_p, _ = pearsonr(t_pass, pers_pass)
    
    kl_m = (compute_kl(t_c_mu[:, p_idx, :], p_c_mu[:, p_idx, :]) + compute_kl(t_s_mu[:, p_idx, :], p_s_mu[:, p_idx, :])) / 2.0
    kl_p = (compute_kl(t_c_mu[:, p_idx, :], pers_c_mu[:, p_idx, :]) + compute_kl(t_s_mu[:, p_idx, :], pers_s_mu[:, p_idx, :])) / 2.0
    
    bc_m = (compute_bray_curtis(t_c_mu[:, p_idx, :], p_c_mu[:, p_idx, :]) + compute_bray_curtis(t_s_mu[:, p_idx, :], p_s_mu[:, p_idx, :])) / 2.0
    bc_p = (compute_bray_curtis(t_c_mu[:, p_idx, :], pers_c_mu[:, p_idx, :]) + compute_bray_curtis(t_s_mu[:, p_idx, :], pers_s_mu[:, p_idx, :])) / 2.0
    
    beat_r2 = "CRM WINS!" if r2_m > r2_p else "Persist wins"
    beat_kl = "CRM WINS!" if kl_m < kl_p else "Persist wins"
    beat_bc = "CRM WINS!" if bc_m < bc_p else "Persist wins"
    
    print(f"\nPassage {p_name}:")
    print(f"  R2:          CRM = {r2_m:.4f} (r={r_m:.4f})  |  Persistence = {r2_p:.4f} (r={r_p:.4f})  --> {beat_r2}")
    print(f"  KL Div:      CRM = {kl_m:.4f}           |  Persistence = {kl_p:.4f}           --> {beat_kl}")
    print(f"  Bray-Curtis: CRM = {bc_m:.4f}           |  Persistence = {bc_p:.4f}           --> {beat_bc}")

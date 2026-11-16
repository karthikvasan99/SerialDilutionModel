import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location=device, weights_only=False)

no_c_tensor = data['no_c_tensor'].to(device)       # (3, 6, 50)
plain_c_tensor = data['plain_c_tensor'].to(device) # (3, 6, 50)
plain_s_tensor = data['plain_s_tensor'].to(device) # (3, 6, 50)
mucin_c_tensor = data['mucin_c_tensor'].to(device) # (3, 6, 50)
mucin_s_tensor = data['mucin_s_tensor'].to(device) # (3, 6, 50)
N = no_c_tensor.size(-1)

from search_pollard_heldout_kl import DualThetaUnifiedCRM

def compute_r2(true_vals, pred_vals):
    diff_sq = np.sum((true_vals - pred_vals)**2)
    tot_sq = np.sum((true_vals - np.mean(true_vals))**2)
    return 1.0 - diff_sq / tot_sq

def train_base(exclude_cond='no_c', K=5, c_w=0.50, epochs=200):
    torch.manual_seed(104)
    model = DualThetaUnifiedCRM(N=N, K=K, c_w=c_w, eta_mode='fixed_1_30').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params()
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        loss = 0.0
        
        if exclude_cond != 'no_c':
            p_no = model.rollout_no_carrier(no_c_tensor, Theta_s, W, alpha, R0[0], b0_liq, tf_ratio=tf_ratio)
            loss_no = torch.sum(no_c_tensor[:, 1:, :] * torch.log(no_c_tensor[:, 1:, :] / torch.clamp(p_no, min=1e-9) + 1e-12), dim=-1).mean()
            loss = loss + loss_no
            
        if exclude_cond != 'plain':
            p_pl_c, p_pl_s = model.rollout_carrier(plain_c_tensor, plain_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta, tf_ratio=tf_ratio)
            loss_pl = torch.sum(plain_c_tensor[:, 1:, :] * torch.log(plain_c_tensor[:, 1:, :] / torch.clamp(p_pl_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                      torch.sum(plain_s_tensor[:, 1:, :] * torch.log(plain_s_tensor[:, 1:, :] / torch.clamp(p_pl_s, min=1e-9) + 1e-12), dim=-1).mean()
            loss = loss + loss_pl
            
        if exclude_cond != 'mucin':
            p_mu_c, p_mu_s = model.rollout_carrier(mucin_c_tensor, mucin_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta, tf_ratio=tf_ratio)
            loss_mu = torch.sum(mucin_c_tensor[:, 1:, :] * torch.log(mucin_c_tensor[:, 1:, :] / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                      torch.sum(mucin_s_tensor[:, 1:, :] * torch.log(mucin_s_tensor[:, 1:, :] / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
            loss = loss + loss_mu
            
        l2 = 0.001 * (Theta_c.pow(2).sum() + Theta_s.pow(2).sum() + W.pow(2).sum())
        loss = loss + l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
    return model

def run_experiment(early_p_count=2):
    # early_p_count: number of early passages observed (e.g. 2 means P1, P2; predict P3..P6)
    last_early_idx = early_p_count - 1
    num_eval_passages = 6 - early_p_count
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: ADAPT ON P1..P{early_p_count} -> PREDICT P{early_p_count+1}..P6")
    print("="*80)
    
    # -------------------------------------------------------------
    # 1. NO CARRIER (Train on Plain + Mucin, Adapt on No Carrier early P)
    # -------------------------------------------------------------
    print("--- 1. Condition: No Carrier ---")
    model_no = train_base(exclude_cond='no_c')
    for p in model_no.parameters():
        p.requires_grad = False
    model_no.a.requires_grad = True
    model_no.b0_liq_raw.requires_grad = True
    
    opt_no = torch.optim.Adam([model_no.a, model_no.b0_liq_raw], lr=0.03)
    for _ in range(150):
        opt_no.zero_grad()
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model_no.get_params()
        # Rollout full 5 predictions
        p_sub = model_no.rollout_no_carrier(no_c_tensor, Theta_s, W, alpha, R0[0], b0_liq, tf_ratio=0.0)
        # Sliced loss on early passage transitions: targets are no_c_tensor[:, 1:early_p_count, :]
        loss = torch.sum(no_c_tensor[:, 1:early_p_count, :] * torch.log(no_c_tensor[:, 1:early_p_count, :] / torch.clamp(p_sub[:, :early_p_count-1, :], min=1e-9) + 1e-12), dim=-1).mean()
        loss.backward()
        opt_no.step()
        
    model_no.eval()
    with torch.no_grad():
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model_no.get_params()
        p_full = model_no.rollout_no_carrier(no_c_tensor, Theta_s, W, alpha, R0[0], b0_liq, tf_ratio=0.0)
        p_eval = p_full[:, early_p_count-1:, :].cpu().numpy().flatten()
        t_eval = no_c_tensor[:, early_p_count:, :].cpu().numpy().flatten()
        r2_no_transfer = compute_r2(t_eval, p_eval)
        r_no_transfer, _ = pearsonr(t_eval, p_eval)
        
        # Persistence baseline on SAME FOOTING: predict abundance of last early passage
        p_persist = np.repeat(no_c_tensor[:, last_early_idx:last_early_idx+1, :].cpu().numpy(), num_eval_passages, axis=1).flatten()
        r2_no_persist = compute_r2(t_eval, p_persist)
        
    print(f"No Carrier: Transfer R2 = {r2_no_transfer:.4f} (r = {r_no_transfer:.4f}) | Persistence R2 = {r2_no_persist:.4f}")

    # -------------------------------------------------------------
    # 2. PLAIN AGAR (Train on No Carrier + Mucin, Adapt on Plain early P)
    # -------------------------------------------------------------
    print("\n--- 2. Condition: Plain Agar ---")
    model_pl = train_base(exclude_cond='plain')
    for p in model_pl.parameters():
        p.requires_grad = False
    model_pl.a.requires_grad = True
    model_pl.k_attach_raw.requires_grad = True
    model_pl.b0_c_raw.requires_grad = True
    model_pl.b0_s_raw.requires_grad = True
    
    opt_pl = torch.optim.Adam([model_pl.a, model_pl.k_attach_raw, model_pl.b0_c_raw, model_pl.b0_s_raw], lr=0.03)
    for _ in range(150):
        opt_pl.zero_grad()
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model_pl.get_params()
        p_c_sub, p_s_sub = model_pl.rollout_carrier(plain_c_tensor, plain_s_tensor,
                                                   Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta, tf_ratio=0.0)
        loss = torch.sum(plain_c_tensor[:, 1:early_p_count, :] * torch.log(plain_c_tensor[:, 1:early_p_count, :] / torch.clamp(p_c_sub[:, :early_p_count-1, :], min=1e-9) + 1e-12), dim=-1).mean() + \
               torch.sum(plain_s_tensor[:, 1:early_p_count, :] * torch.log(plain_s_tensor[:, 1:early_p_count, :] / torch.clamp(p_s_sub[:, :early_p_count-1, :], min=1e-9) + 1e-12), dim=-1).mean()
        loss.backward()
        opt_pl.step()
        
    model_pl.eval()
    with torch.no_grad():
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model_pl.get_params()
        p_c_full, p_s_full = model_pl.rollout_carrier(plain_c_tensor, plain_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta, tf_ratio=0.0)
        p_eval = np.concatenate([p_c_full[:, early_p_count-1:, :].cpu().numpy().flatten(),
                                 p_s_full[:, early_p_count-1:, :].cpu().numpy().flatten()])
        t_eval = np.concatenate([plain_c_tensor[:, early_p_count:, :].cpu().numpy().flatten(),
                                 plain_s_tensor[:, early_p_count:, :].cpu().numpy().flatten()])
        r2_pl_transfer = compute_r2(t_eval, p_eval)
        r_pl_transfer, _ = pearsonr(t_eval, p_eval)
        
        # Persistence baseline
        p_persist = np.concatenate([np.repeat(plain_c_tensor[:, last_early_idx:last_early_idx+1, :].cpu().numpy(), num_eval_passages, axis=1).flatten(),
                                    np.repeat(plain_s_tensor[:, last_early_idx:last_early_idx+1, :].cpu().numpy(), num_eval_passages, axis=1).flatten()])
        r2_pl_persist = compute_r2(t_eval, p_persist)
        
    print(f"Plain Agar: Transfer R2 = {r2_pl_transfer:.4f} (r = {r_pl_transfer:.4f}) | Persistence R2 = {r2_pl_persist:.4f}")

    # -------------------------------------------------------------
    # 3. MUCIN AGAR (Train on No Carrier + Plain, Adapt on Mucin early P)
    # -------------------------------------------------------------
    print("\n--- 3. Condition: Mucin Agar ---")
    model_mu = train_base(exclude_cond='mucin')
    for p in model_mu.parameters():
        p.requires_grad = False
    model_mu.a.requires_grad = True
    model_mu.k_attach_raw.requires_grad = True
    model_mu.b0_c_raw.requires_grad = True
    model_mu.b0_s_raw.requires_grad = True
    
    opt_mu = torch.optim.Adam([model_mu.a, model_mu.k_attach_raw, model_mu.b0_c_raw, model_mu.b0_s_raw], lr=0.03)
    for _ in range(150):
        opt_mu.zero_grad()
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model_mu.get_params()
        p_c_sub, p_s_sub = model_mu.rollout_carrier(mucin_c_tensor, mucin_s_tensor,
                                                   Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta, tf_ratio=0.0)
        loss = torch.sum(mucin_c_tensor[:, 1:early_p_count, :] * torch.log(mucin_c_tensor[:, 1:early_p_count, :] / torch.clamp(p_c_sub[:, :early_p_count-1, :], min=1e-9) + 1e-12), dim=-1).mean() + \
               torch.sum(mucin_s_tensor[:, 1:early_p_count, :] * torch.log(mucin_s_tensor[:, 1:early_p_count, :] / torch.clamp(p_s_sub[:, :early_p_count-1, :], min=1e-9) + 1e-12), dim=-1).mean()
        loss.backward()
        opt_mu.step()
        
    model_mu.eval()
    with torch.no_grad():
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model_mu.get_params()
        p_c_full, p_s_full = model_mu.rollout_carrier(mucin_c_tensor, mucin_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta, tf_ratio=0.0)
        p_eval = np.concatenate([p_c_full[:, early_p_count-1:, :].cpu().numpy().flatten(),
                                 p_s_full[:, early_p_count-1:, :].cpu().numpy().flatten()])
        t_eval = np.concatenate([mucin_c_tensor[:, early_p_count:, :].cpu().numpy().flatten(),
                                 mucin_s_tensor[:, early_p_count:, :].cpu().numpy().flatten()])
        r2_mu_transfer = compute_r2(t_eval, p_eval)
        r_mu_transfer, _ = pearsonr(t_eval, p_eval)
        
        # Persistence baseline
        p_persist = np.concatenate([np.repeat(mucin_c_tensor[:, last_early_idx:last_early_idx+1, :].cpu().numpy(), num_eval_passages, axis=1).flatten(),
                                    np.repeat(mucin_s_tensor[:, last_early_idx:last_early_idx+1, :].cpu().numpy(), num_eval_passages, axis=1).flatten()])
        r2_mu_persist = compute_r2(t_eval, p_persist)
        
    print(f"Mucin Agar: Transfer R2 = {r2_mu_transfer:.4f} (r = {r_mu_transfer:.4f}) | Persistence R2 = {r2_mu_persist:.4f}")
    
    return {
        'early_p_count': early_p_count,
        'r2_transfer': [r2_no_transfer, r2_pl_transfer, r2_mu_transfer],
        'r2_persist': [r2_no_persist, r2_pl_persist, r2_mu_persist]
    }

if __name__ == '__main__':
    res2 = run_experiment(early_p_count=2)
    res3 = run_experiment(early_p_count=3)

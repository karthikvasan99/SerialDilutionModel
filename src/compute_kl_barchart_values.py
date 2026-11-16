import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

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

def compute_kl(t, p, eps=1e-12, clamp_min=1e-9):
    t_c = torch.clamp(t, min=0.0)
    p_c = torch.clamp(p, min=clamp_min)
    t_norm = t_c / (t_c.sum(dim=-1, keepdim=True) + eps)
    p_norm = p_c / (p_c.sum(dim=-1, keepdim=True) + eps)
    ratio = torch.clamp(t_norm / p_norm, min=eps)
    kl = torch.sum(torch.where(t_norm > 0, t_norm * torch.log(ratio), torch.zeros_like(t_norm)), dim=-1)
    return kl.mean().item()

# Load winning CRM
ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Winning_CRM.pth')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

from search_pollard_heldout_kl import DualThetaUnifiedCRM
model = DualThetaUnifiedCRM(N=50, K=5, c_w=0.50, eta_mode='fixed_1_30').to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

with torch.no_grad():
    Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params()
    
    # 1. Full CRM Predictions
    p_no = model.rollout_no_carrier(no_c_tensor, Theta_s, W, alpha, R0[0], b0_liq)
    p_pl_c, p_pl_s = model.rollout_carrier(plain_c_tensor, plain_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta)
    p_mu_c, p_mu_s = model.rollout_carrier(mucin_c_tensor, mucin_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta)
    
    kl_crm_no = compute_kl(no_c_tensor[:, 1:, :], p_no)
    kl_crm_pl = (compute_kl(plain_c_tensor[:, 1:, :], p_pl_c) + compute_kl(plain_s_tensor[:, 1:, :], p_pl_s)) / 2.0
    kl_crm_mu = (compute_kl(mucin_c_tensor[:, 1:, :], p_mu_c) + compute_kl(mucin_s_tensor[:, 1:, :], p_mu_s)) / 2.0
    
    # 2. Persistence Null (predicting P_{t} from P_{t-1} or P_1 held constant)
    # Holding P_1 constant for P2..P6:
    pers_no = no_c_tensor[:, 0:1, :].expand(-1, 5, -1)
    pers_pl_c = plain_c_tensor[:, 0:1, :].expand(-1, 5, -1)
    pers_pl_s = plain_s_tensor[:, 0:1, :].expand(-1, 5, -1)
    pers_mu_c = mucin_c_tensor[:, 0:1, :].expand(-1, 5, -1)
    pers_mu_s = mucin_s_tensor[:, 0:1, :].expand(-1, 5, -1)
    
    kl_pers_no = compute_kl(no_c_tensor[:, 1:, :], pers_no)
    kl_pers_pl = (compute_kl(plain_c_tensor[:, 1:, :], pers_pl_c) + compute_kl(plain_s_tensor[:, 1:, :], pers_pl_s)) / 2.0
    kl_pers_mu = (compute_kl(mucin_c_tensor[:, 1:, :], pers_mu_c) + compute_kl(mucin_s_tensor[:, 1:, :], pers_mu_s)) / 2.0
    
    # Holding P_{t-1} constant (1-step persistence):
    pers1_no = no_c_tensor[:, :-1, :]
    pers1_pl_c = plain_c_tensor[:, :-1, :]
    pers1_pl_s = plain_s_tensor[:, :-1, :]
    pers1_mu_c = mucin_c_tensor[:, :-1, :]
    pers1_mu_s = mucin_s_tensor[:, :-1, :]
    
    kl_pers1_no = compute_kl(no_c_tensor[:, 1:, :], pers1_no)
    kl_pers1_pl = (compute_kl(plain_c_tensor[:, 1:, :], pers1_pl_c) + compute_kl(plain_s_tensor[:, 1:, :], pers1_pl_s)) / 2.0
    kl_pers1_mu = (compute_kl(mucin_c_tensor[:, 1:, :], pers1_mu_c) + compute_kl(mucin_s_tensor[:, 1:, :], pers1_mu_s)) / 2.0

    # 3. W = 0 Ablation
    W_zero = torch.zeros_like(W)
    p_no_w0 = model.rollout_no_carrier(no_c_tensor, Theta_s, W_zero, alpha, R0[0], b0_liq)
    p_pl_c_w0, p_pl_s_w0 = model.rollout_carrier(plain_c_tensor, plain_s_tensor, Theta_c, Theta_s, W_zero, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta)
    p_mu_c_w0, p_mu_s_w0 = model.rollout_carrier(mucin_c_tensor, mucin_s_tensor, Theta_c, Theta_s, W_zero, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta)
    
    kl_w0_no = compute_kl(no_c_tensor[:, 1:, :], p_no_w0)
    kl_w0_pl = (compute_kl(plain_c_tensor[:, 1:, :], p_pl_c_w0) + compute_kl(plain_s_tensor[:, 1:, :], p_pl_s_w0)) / 2.0
    kl_w0_mu = (compute_kl(mucin_c_tensor[:, 1:, :], p_mu_c_w0) + compute_kl(mucin_s_tensor[:, 1:, :], p_mu_s_w0)) / 2.0

    # 4. Scrambled R0 (N=10 permutations)
    kl_r0_no_list, kl_r0_pl_list, kl_r0_mu_list = [], [], []
    for _ in range(10):
        perm = torch.randperm(5)
        R0_shuff = R0[:, perm]
        p_no_r0 = model.rollout_no_carrier(no_c_tensor, Theta_s, W, alpha, R0_shuff[0], b0_liq)
        p_pl_c_r0, p_pl_s_r0 = model.rollout_carrier(plain_c_tensor, plain_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0_shuff[1], b0_c, b0_s, eta)
        p_mu_c_r0, p_mu_s_r0 = model.rollout_carrier(mucin_c_tensor, mucin_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0_shuff[2], b0_c, b0_s, eta)
        kl_r0_no_list.append(compute_kl(no_c_tensor[:, 1:, :], p_no_r0))
        kl_r0_pl_list.append((compute_kl(plain_c_tensor[:, 1:, :], p_pl_c_r0) + compute_kl(plain_s_tensor[:, 1:, :], p_pl_s_r0)) / 2.0)
        kl_r0_mu_list.append((compute_kl(mucin_c_tensor[:, 1:, :], p_mu_c_r0) + compute_kl(mucin_s_tensor[:, 1:, :], p_mu_s_r0)) / 2.0)
        
    # 5. Shuffled Theta (N=10 permutations)
    kl_th_no_list, kl_th_pl_list, kl_th_mu_list = [], [], []
    for _ in range(10):
        perm_th = torch.randperm(50)
        Theta_c_sh = Theta_c[perm_th]
        Theta_s_sh = Theta_s[perm_th]
        p_no_th = model.rollout_no_carrier(no_c_tensor, Theta_s_sh, W, alpha, R0[0], b0_liq)
        p_pl_c_th, p_pl_s_th = model.rollout_carrier(plain_c_tensor, plain_s_tensor, Theta_c_sh, Theta_s_sh, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta)
        p_mu_c_th, p_mu_s_th = model.rollout_carrier(mucin_c_tensor, mucin_s_tensor, Theta_c_sh, Theta_s_sh, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta)
        kl_th_no_list.append(compute_kl(no_c_tensor[:, 1:, :], p_no_th))
        kl_th_pl_list.append((compute_kl(plain_c_tensor[:, 1:, :], p_pl_c_th) + compute_kl(plain_s_tensor[:, 1:, :], p_pl_s_th)) / 2.0)
        kl_th_mu_list.append((compute_kl(mucin_c_tensor[:, 1:, :], p_mu_c_th) + compute_kl(mucin_s_tensor[:, 1:, :], p_mu_s_th)) / 2.0)

    # 6. Shuffled Adhesion (N=10 permutations)
    kl_adh_pl_list, kl_adh_mu_list = [], []
    for _ in range(10):
        k_att_sh = k_attach[:, torch.randperm(50)]
        p_pl_c_ad, p_pl_s_ad = model.rollout_carrier(plain_c_tensor, plain_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_att_sh[0], R0[1], b0_c, b0_s, eta)
        p_mu_c_ad, p_mu_s_ad = model.rollout_carrier(mucin_c_tensor, mucin_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_att_sh[1], R0[2], b0_c, b0_s, eta)
        kl_adh_pl_list.append((compute_kl(plain_c_tensor[:, 1:, :], p_pl_c_ad) + compute_kl(plain_s_tensor[:, 1:, :], p_pl_s_ad)) / 2.0)
        kl_adh_mu_list.append((compute_kl(mucin_c_tensor[:, 1:, :], p_mu_c_ad) + compute_kl(mucin_s_tensor[:, 1:, :], p_mu_s_ad)) / 2.0)

print("="*85)
print("KL DIVERGENCE BENCHMARK SUMMARY (LOWER IS BETTER):")
print("="*85)
print(f"Universal CRM:      No Carrier = {kl_crm_no:.4f} | Plain = {kl_crm_pl:.4f} | Mucin = {kl_crm_mu:.4f}")
print(f"Persistence (from P1): No Carrier = {kl_pers_no:.4f} | Plain = {kl_pers_pl:.4f} | Mucin = {kl_pers_mu:.4f}")
print(f"Persistence (1-step):  No Carrier = {kl_pers1_no:.4f} | Plain = {kl_pers1_pl:.4f} | Mucin = {kl_pers1_mu:.4f}")
print(f"W = 0 (No Byproducts):No Carrier = {kl_w0_no:.4f} | Plain = {kl_w0_pl:.4f} | Mucin = {kl_w0_mu:.4f}")
print(f"Scrambled R0:       No Carrier = {np.mean(kl_r0_no_list):.4f} +/- {np.std(kl_r0_no_list):.4f} | Plain = {np.mean(kl_r0_pl_list):.4f} +/- {np.std(kl_r0_pl_list):.4f} | Mucin = {np.mean(kl_r0_mu_list):.4f} +/- {np.std(kl_r0_mu_list):.4f}")
print(f"Shuffled Theta:     No Carrier = {np.mean(kl_th_no_list):.4f} +/- {np.std(kl_th_no_list):.4f} | Plain = {np.mean(kl_th_pl_list):.4f} +/- {np.std(kl_th_pl_list):.4f} | Mucin = {np.mean(kl_th_mu_list):.4f} +/- {np.std(kl_th_mu_list):.4f}")
print(f"Shuffled Adhesion:  Plain = {np.mean(kl_adh_pl_list):.4f} +/- {np.std(kl_adh_pl_list):.4f} | Mucin = {np.mean(kl_adh_mu_list):.4f} +/- {np.std(kl_adh_mu_list):.4f}")

# Save these exact values to a dictionary
kl_dict = {
    'kl_crm': [kl_crm_no, kl_crm_pl, kl_crm_mu],
    'kl_pers': [kl_pers_no, kl_pers_pl, kl_pers_mu],
    'kl_pers1': [kl_pers1_no, kl_pers1_pl, kl_pers1_mu],
    'kl_w0': [kl_w0_no, kl_w0_pl, kl_w0_mu],
    'kl_r0_mean': [np.mean(kl_r0_no_list), np.mean(kl_r0_pl_list), np.mean(kl_r0_mu_list)],
    'kl_r0_std': [np.std(kl_r0_no_list), np.std(kl_r0_pl_list), np.std(kl_r0_mu_list)],
    'kl_th_mean': [np.mean(kl_th_no_list), np.mean(kl_th_pl_list), np.mean(kl_th_mu_list)],
    'kl_th_std': [np.std(kl_th_no_list), np.std(kl_th_pl_list), np.std(kl_th_mu_list)],
    'kl_adh_mean': [np.nan, np.mean(kl_adh_pl_list), np.mean(kl_adh_mu_list)],
    'kl_adh_std': [np.nan, np.std(kl_adh_pl_list), np.std(kl_adh_mu_list)]
}
torch.save(kl_dict, os.path.join(PROJECT_ROOT, 'Data', 'Pollard_KL_Benchmark_Values.pt'))
print("Saved KL values to Pollard_KL_Benchmark_Values.pt")

import sys; sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

from src.poster_fig2_dual_theta import (
    DualThetaUnifiedCRM,
    no_c_tensor,
    plain_c_tensor,
    plain_s_tensor,
    mucin_c_tensor,
    mucin_s_tensor,
    device
)

def average_reps(t):
    r1 = t[0:3].mean(dim=0, keepdim=True)
    r2 = t[3:6].mean(dim=0, keepdim=True)
    r3 = t[6:9].mean(dim=0, keepdim=True)
    return torch.cat([r1, r2, r3], dim=0)

no_c_avg = average_reps(no_c_tensor)
plain_c_avg = average_reps(plain_c_tensor)
plain_s_avg = average_reps(plain_s_tensor)
mucin_c_avg = average_reps(mucin_c_tensor)
mucin_s_avg = average_reps(mucin_s_tensor)

print("Averaged shapes:", no_c_avg.shape, plain_c_avg.shape)

class AveragedDualThetaCRM(DualThetaUnifiedCRM):
    def rollout_no_c(self, Theta_s, W, alpha, R0_liq, b0_liq, delta_t=0.1, tf_ratio=0.0):
        N_curr = no_c_avg[:, 0, :] * b0_liq
        R0_batch = R0_liq.unsqueeze(0).expand(no_c_avg.size(0), -1)
        preds = []
        t_span = torch.linspace(0.0, 1.0, int(1.0 / delta_t) + 1, device=device)
        for p in range(1, 6):
            if tf_ratio > 0.0 and self.training:
                N_transferred = tf_ratio * (no_c_avg[:, p-1, :] * N_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_curr
            else:
                N_transferred = N_curr
            N0 = torch.clamp(N_transferred / 100.0 + 1e-6, min=0.0)
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

    def rollout_car(self, c_avg, s_avg, Theta_c, Theta_s, W, alpha, k_shed, k_att, R0_cond, b0_c, b0_s, eta, delta_t=0.1, tf_ratio=0.0):
        N_c_curr = c_avg[:, 0, :] * b0_c
        N_s_curr = s_avg[:, 0, :] * b0_s
        R0_batch = R0_cond.unsqueeze(0).expand(c_avg.size(0), -1)
        preds_c, preds_s = [], []
        t_span = torch.linspace(0.0, 1.0, int(1.0 / delta_t) + 1, device=device)
        for p in range(1, 6):
            if tf_ratio > 0.0 and self.training:
                N_c_transferred = tf_ratio * (c_avg[:, p-1, :] * N_c_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_c_curr
                N_s_transferred = tf_ratio * (s_avg[:, p-1, :] * N_s_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_s_curr
            else:
                N_c_transferred = N_c_curr
                N_s_transferred = N_s_curr
            N_c_0 = torch.clamp(0.2 * N_c_transferred + 1e-6, min=0.0)
            N_s_0 = torch.clamp(eta * N_s_transferred + 1e-6, min=0.0)
            S0 = torch.cat([N_c_0, N_s_0, R0_batch], dim=-1)
            def f(t, S):
                Nc = torch.clamp(S[..., :50], min=0.0)
                Ns = torch.clamp(S[..., 50:100], min=0.0)
                Rt = torch.clamp(S[..., 100:], min=0.0)
                growth_c = torch.matmul(Rt, Theta_c.t())
                growth_s = torch.matmul(Rt, Theta_s.t())
                dNc = Nc * growth_c - alpha * (Nc**2) - k_shed * Nc + k_att * Ns
                dNs = Ns * growth_s - alpha * (Ns**2) + k_shed * Nc - k_att * Ns
                flux = Rt * (torch.matmul(Nc, Theta_c) + torch.matmul(Ns, Theta_s))
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNc, dNs, dRt], dim=-1)
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            Nc_fin = torch.clamp(S_t[-1, :, :50], min=0.0)
            Ns_fin = torch.clamp(S_t[-1, :, 50:100], min=0.0)
            preds_c.append(Nc_fin / (Nc_fin.sum(dim=-1, keepdim=True) + 1e-8))
            preds_s.append(Ns_fin / (Ns_fin.sum(dim=-1, keepdim=True) + 1e-8))
            N_c_curr = Nc_fin
            N_s_curr = Ns_fin
        return torch.stack(preds_c, dim=1), torch.stack(preds_s, dim=1)

model_avg = AveragedDualThetaCRM(N=50, K=3).to(device)
optimizer = torch.optim.Adam(model_avg.parameters(), lr=0.015)
for epoch in range(1, 151):
    optimizer.zero_grad()
    Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model_avg.get_params()
    tf_ratio = max(0.0, 1.0 - (epoch / (150 * 0.75)))
    
    p_no = model_avg.rollout_no_c(Theta_s, W, alpha, R0[0], b0_liq, tf_ratio=tf_ratio)
    loss_no = torch.sum(no_c_avg[:, 1:, :] * torch.log(no_c_avg[:, 1:, :] / torch.clamp(p_no, min=1e-9) + 1e-12), dim=-1).mean()
    
    p_pl_c, p_pl_s = model_avg.rollout_car(plain_c_avg, plain_s_avg, Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta, tf_ratio=tf_ratio)
    loss_pl = torch.sum(plain_c_avg[:, 1:, :] * torch.log(plain_c_avg[:, 1:, :] / torch.clamp(p_pl_c, min=1e-9) + 1e-12), dim=-1).mean() + \
              torch.sum(plain_s_avg[:, 1:, :] * torch.log(plain_s_avg[:, 1:, :] / torch.clamp(p_pl_s, min=1e-9) + 1e-12), dim=-1).mean()
              
    p_mu_c, p_mu_s = model_avg.rollout_car(mucin_c_avg, mucin_s_avg, Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta, tf_ratio=tf_ratio)
    loss_mu = torch.sum(mucin_c_avg[:, 1:, :] * torch.log(mucin_c_avg[:, 1:, :] / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
              torch.sum(mucin_s_avg[:, 1:, :] * torch.log(mucin_s_avg[:, 1:, :] / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
              
    loss = loss_no + loss_pl + loss_mu
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# Evaluation on averaged
with torch.no_grad():
    Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model_avg.get_params()
    p_no = model_avg.rollout_no_c(Theta_s, W, alpha, R0[0], b0_liq, tf_ratio=0.0).cpu().numpy().flatten()
    t_no = no_c_avg[:, 1:, :].cpu().numpy().flatten()
    r2_no = 1.0 - np.sum((t_no - p_no)**2) / np.sum((t_no - np.mean(t_no))**2)
    
    p_pl_c, p_pl_s = model_avg.rollout_car(plain_c_avg, plain_s_avg, Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta, tf_ratio=0.0)
    p_pl = np.concatenate([p_pl_c.cpu().numpy().flatten(), p_pl_s.cpu().numpy().flatten()])
    t_pl = np.concatenate([plain_c_avg[:, 1:, :].cpu().numpy().flatten(), plain_s_avg[:, 1:, :].cpu().numpy().flatten()])
    r2_pl = 1.0 - np.sum((t_pl - p_pl)**2) / np.sum((t_pl - np.mean(t_pl))**2)
    
    p_mu_c, p_mu_s = model_avg.rollout_car(mucin_c_avg, mucin_s_avg, Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta, tf_ratio=0.0)
    p_mu = np.concatenate([p_mu_c.cpu().numpy().flatten(), p_mu_s.cpu().numpy().flatten()])
    t_mu = np.concatenate([mucin_c_avg[:, 1:, :].cpu().numpy().flatten(), mucin_s_avg[:, 1:, :].cpu().numpy().flatten()])
    r2_mu = 1.0 - np.sum((t_mu - p_mu)**2) / np.sum((t_mu - np.mean(t_mu))**2)
    
    print(f"Retrained on Averaged: No-C R2 = {r2_no:.4f} | Plain R2 = {r2_pl:.4f} | Mucin R2 = {r2_mu:.4f}")
    
    # Check tradeoff correlation in this model
    r_mucin = (k_attach[1] / k_shed).cpu().numpy()
    log_rm = np.log10(r_mucin)
    th_c_norm = torch.norm(Theta_c, dim=-1).cpu().numpy()
    th_s_norm = torch.norm(Theta_s, dim=-1).cpu().numpy()
    th_ratio = th_c_norm / th_s_norm
    from scipy.stats import pearsonr
    r_to, p_to = pearsonr(log_rm, th_ratio)
    print(f"Tradeoff correlation on averaged-trained model: r = {r_to:.4f}, p = {p_to:.4e}")

import os
import sys
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from test_p0_to_p1_alone import train_crm_base, p0, mucin_c_tensor, mucin_s_tensor, compute_r2
from scipy.stats import pearsonr

print("Diagnosing Mucin Agar P0 -> P1 adaptation...")
m_mu = train_crm_base(exclude_cond='mucin')
with torch.no_grad():
    m_mu.a.data[2] = m_mu.a.data[1].clone()
    m_mu.k_attach_raw.data[1] = m_mu.k_attach_raw.data[0].clone()

print("Initial borrowed params from Plain:")
print("  a[2]:", torch.nn.functional.softplus(m_mu.a[2]).detach().cpu().numpy())
print("  k_attach[1] (mean, max):", torch.nn.functional.softplus(m_mu.k_attach_raw[1]).mean().item(), torch.nn.functional.softplus(m_mu.k_attach_raw[1]).max().item())

# Before any adaptation, what is the pure zero-shot on P2..P6?
m_mu.eval()
with torch.no_grad():
    p_c, p_s = m_mu.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
    p_eval = torch.cat([p_c[:, 1:, :], p_s[:, 1:, :]], dim=-1).cpu().numpy().flatten()
    t_eval = torch.cat([mucin_c_tensor[:, 1:, :], mucin_s_tensor[:, 1:, :]], dim=-1).cpu().numpy().flatten()
    r2_zero = compute_r2(t_eval, p_eval)
    r_zero, _ = pearsonr(t_eval, p_eval)
    print(f"Zero-shot BEFORE adaptation (Predict P2..P6): R2 = {r2_zero:.4f}, r = {r_zero:.4f}")

# Now observe what happens during optimization
for p in m_mu.parameters(): p.requires_grad = False
m_mu.a.requires_grad = True
m_mu.k_attach_raw.requires_grad = True
m_mu.b0_s_raw.requires_grad = True

opt_mu = torch.optim.Adam([m_mu.a, m_mu.k_attach_raw, m_mu.b0_s_raw], lr=0.01) # lower lr
for step in range(1, 151):
    opt_mu.zero_grad()
    p_c, p_s = m_mu.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
    loss = torch.sum(mucin_c_tensor[:, 0:1, :] * torch.log(mucin_c_tensor[:, 0:1, :] / torch.clamp(p_c[:, 0:1, :], min=1e-9) + 1e-12), dim=-1).mean() + \
           torch.sum(mucin_s_tensor[:, 0:1, :] * torch.log(mucin_s_tensor[:, 0:1, :] / torch.clamp(p_s[:, 0:1, :], min=1e-9) + 1e-12), dim=-1).mean()
    loss.backward()
    opt_mu.step()
    if step in [10, 30, 50, 100, 150]:
        with torch.no_grad():
            p_c, p_s = m_mu.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
            p_eval = torch.cat([p_c[:, 1:, :], p_s[:, 1:, :]], dim=-1).cpu().numpy().flatten()
            r2_step = compute_r2(t_eval, p_eval)
            print(f"  Step {step:3d}: Loss={loss.item():.4f}, Eval P2..P6 R2 = {r2_step:.4f}")

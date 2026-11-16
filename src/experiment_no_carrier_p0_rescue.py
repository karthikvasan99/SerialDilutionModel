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
        self.b0_s_raw = nn.Parameter(torch.tensor(2.0))
        
    def get_params(self):
        Theta_c = F.softplus(self.theta_c_raw)
        Theta_s = F.softplus(self.theta_s_raw)
        alpha = F.softplus(self.alpha_raw)
        k_shed = F.softplus(self.k_shed_raw)
        k_attach = F.softplus(self.k_attach_raw)
        R0 = F.softplus(self.a)
        b0_liq = F.softplus(self.b0_liq_raw)
        b0_s = F.softplus(self.b0_s_raw)
        eta = torch.tensor(1.0/30.0, device=device)
        W_sp = F.softplus(self.w_raw) * (1.0 - torch.eye(self.K, device=device))
        scale = torch.clamp(self.c_w / torch.clamp(W_sp.sum(dim=-1, keepdim=True), min=1e-6), max=1.0)
        W = W_sp * scale
        return Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_s, eta

    def rollout_no_carrier_from_p0(self, p0_vec, delta_t=0.1, tf_ratio=0.0):
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_s, eta = self.get_params()
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
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_s, eta = self.get_params()
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

print("Training base model on Plain and Mucin only (excluding No Carrier)...")
torch.manual_seed(42)
base_model = UnifiedCRMP0().to(device)
optimizer = torch.optim.Adam(base_model.parameters(), lr=0.015)

epochs = 250
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
    
    # Plain Agar
    p_pl_c, p_pl_s = base_model.rollout_carrier_from_p0(p0, cond_idx=1, tf_ratio=tf_ratio)
    loss_pl = torch.sum(plain_c_tensor * torch.log(plain_c_tensor / torch.clamp(p_pl_c, min=1e-9) + 1e-12), dim=-1).mean() + \
              torch.sum(plain_s_tensor * torch.log(plain_s_tensor / torch.clamp(p_pl_s, min=1e-9) + 1e-12), dim=-1).mean()
              
    # Mucin Agar
    p_mu_c, p_mu_s = base_model.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=tf_ratio)
    loss_mu = torch.sum(mucin_c_tensor * torch.log(mucin_c_tensor / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
              torch.sum(mucin_s_tensor * torch.log(mucin_s_tensor / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
              
    Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_s, eta = base_model.get_params()
    l2 = 0.001 * (Theta_c.pow(2).sum() + Theta_s.pow(2).sum() + W.pow(2).sum())
    loss = loss_pl + loss_mu + l2
    loss.backward()
    torch.nn.utils.clip_grad_norm_(base_model.parameters(), 5.0)
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Base Epoch {epoch:3d}/{epochs} | Loss = {loss.item():.4f}")

# Check base performance on Plain & Mucin
base_model.eval()
with torch.no_grad():
    p_pl_c, p_pl_s = base_model.rollout_carrier_from_p0(p0, cond_idx=1, tf_ratio=0.0)
    p_mu_c, p_mu_s = base_model.rollout_carrier_from_p0(p0, cond_idx=2, tf_ratio=0.0)
    r2_pl = compute_r2(torch.cat([plain_c_tensor, plain_s_tensor], dim=-1).cpu().numpy().flatten(),
                       torch.cat([p_pl_c, p_pl_s], dim=-1).cpu().numpy().flatten())
    r2_mu = compute_r2(torch.cat([mucin_c_tensor, mucin_s_tensor], dim=-1).cpu().numpy().flatten(),
                       torch.cat([p_mu_c, p_mu_s], dim=-1).cpu().numpy().flatten())
    print(f"Base model fit on training conditions: Plain R2 = {r2_pl:.4f}, Mucin R2 = {r2_mu:.4f}")

# Now test adaptation on No Carrier!
print("\n" + "="*80)
print("TESTING NO CARRIER OUT-OF-SAMPLE ADAPTATION PROTOCOLS")
print("Target: Predict P3..P6 (Passages 3 to 6)")
print("="*80)

# Target for evaluation: P3..P6 of No Carrier
# Note: no_c_tensor has shape (3, 6, 50), where index 0 is P1, index 1 is P2, ..., index 2..5 is P3..P6!
t_eval_p3_p6 = no_c_tensor[:, 2:, :].cpu().numpy().flatten()

# Persistence baseline on same footing (holding P2 constant for P3..P6):
p2_vals = no_c_tensor[:, 1:2, :].cpu().numpy() # shape (3, 1, 50)
persist_p3_p6 = np.repeat(p2_vals, 4, axis=1).flatten()
r2_persist_p2 = compute_r2(t_eval_p3_p6, persist_p3_p6)
r_persist_p2, _ = pearsonr(t_eval_p3_p6, persist_p3_p6)
print(f"Baseline: Holding P2 Constant -> Predict P3..P6: R2 = {r2_persist_p2:.4f}, r = {r_persist_p2:.4f}")

# Baseline 2: Holding P0 Constant -> Predict P3..P6:
p0_vals = p0.unsqueeze(1).cpu().numpy() # shape (3, 1, 50)
persist_p0 = np.repeat(p0_vals, 4, axis=1).flatten()
r2_persist_p0 = compute_r2(t_eval_p3_p6, persist_p0)
print(f"Baseline: Holding P0 Constant -> Predict P3..P6: R2 = {r2_persist_p0:.4f}")

# Protocol 1: Zero-shot (No fine-tuning at all, using average carrier resource concentration)
with torch.no_grad():
    base_model.a.data[0] = (base_model.a.data[1] + base_model.a.data[2]) / 2.0
    preds_zero = base_model.rollout_no_carrier_from_p0(p0, tf_ratio=0.0)
    p_eval_zero = preds_zero[:, 2:, :].cpu().numpy().flatten()
    r2_zero = compute_r2(t_eval_p3_p6, p_eval_zero)
    r_zero, _ = pearsonr(t_eval_p3_p6, p_eval_zero)
    print(f"\n1. Pure Zero-Shot (No adaptation, predict P3..P6): R2 = {r2_zero:.4f}, r = {r_zero:.4f}")

# Protocol 2: Fine-tune 'a' on P0 -> P1 only (observing Day 3)
def test_adaptation(learn_targets=['a'], early_passages=['P1'], lr=0.03, iters=150):
    import copy
    model = copy.deepcopy(base_model)
    for p in model.parameters():
        p.requires_grad = False
    
    params_to_opt = []
    if 'a' in learn_targets:
        model.a.requires_grad = True
        params_to_opt.append(model.a)
    if 'b0_liq' in learn_targets:
        model.b0_liq_raw.requires_grad = True
        params_to_opt.append(model.b0_liq_raw)
        
    opt = torch.optim.Adam(params_to_opt, lr=lr)
    
    # Determine target passage slice
    if early_passages == ['P1']:
        target_slice = slice(0, 1) # index 0 is P1
    elif early_passages == ['P1', 'P2']:
        target_slice = slice(0, 2) # indices 0, 1 are P1, P2
    elif early_passages == ['P2']:
        target_slice = slice(1, 2)
    
    for _ in range(iters):
        opt.zero_grad()
        p_rollout = model.rollout_no_carrier_from_p0(p0, tf_ratio=0.0)
        p_train = p_rollout[:, target_slice, :]
        t_train = no_c_tensor[:, target_slice, :]
        loss = torch.sum(t_train * torch.log(t_train / torch.clamp(p_train, min=1e-9) + 1e-12), dim=-1).mean()
        loss.backward()
        opt.step()
        
    model.eval()
    with torch.no_grad():
        p_rollout = model.rollout_no_carrier_from_p0(p0, tf_ratio=0.0)
        p_eval = p_rollout[:, 2:, :].cpu().numpy().flatten()
        r2 = compute_r2(t_eval_p3_p6, p_eval)
        r, _ = pearsonr(t_eval_p3_p6, p_eval)
        
        # Also check fit on training passages
        p_tr = p_rollout[:, target_slice, :].cpu().numpy().flatten()
        t_tr = no_c_tensor[:, target_slice, :].cpu().numpy().flatten()
        r2_tr = compute_r2(t_tr, p_tr)
    return r2, r, r2_tr

print("\n2. Fine-tuning on P0 -> P1:")
r2, r, r2_tr = test_adaptation(learn_targets=['a'], early_passages=['P1'])
print(f"   Learn 'a' only: Eval P3..P6 R2 = {r2:.4f}, r = {r:.4f} (Train P1 R2 = {r2_tr:.4f})")
r2, r, r2_tr = test_adaptation(learn_targets=['a', 'b0_liq'], early_passages=['P1'])
print(f"   Learn 'a' + b0_liq: Eval P3..P6 R2 = {r2:.4f}, r = {r:.4f} (Train P1 R2 = {r2_tr:.4f})")

print("\n3. Fine-tuning on P0 -> P1 -> P2:")
r2, r, r2_tr = test_adaptation(learn_targets=['a'], early_passages=['P1', 'P2'])
print(f"   Learn 'a' only: Eval P3..P6 R2 = {r2:.4f}, r = {r:.4f} (Train P1-P2 R2 = {r2_tr:.4f})")
r2, r, r2_tr = test_adaptation(learn_targets=['a', 'b0_liq'], early_passages=['P1', 'P2'])
print(f"   Learn 'a' + b0_liq: Eval P3..P6 R2 = {r2:.4f}, r = {r:.4f} (Train P1-P2 R2 = {r2_tr:.4f})")


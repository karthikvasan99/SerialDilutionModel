import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}", flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location=device, weights_only=False)

no_c_tensor = data['no_c_tensor'].to(device)       # (3, 6, 50)
mucin_c_tensor = data['mucin_c_tensor'].to(device) # (3, 6, 50)
mucin_s_tensor = data['mucin_s_tensor'].to(device) # (3, 6, 50)
strains = data['strains']
N = 50

# Load inoculum P0
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv'))
sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
df['Relative_Abundance'] = df['Read_Fraction'] / 100.0 if sums.mean() > 50 else df['Read_Fraction']
inoc = df[df['sampleType'] == 'inoculum'].pivot_table(index='replicate', columns='Strain_Name', values='Relative_Abundance', fill_value=0.0).reindex(columns=strains, fill_value=0.0)
inoc_norm = inoc.div(inoc.sum(axis=1), axis=0)
p0 = torch.tensor(inoc_norm.mean(axis=0).values, dtype=torch.float32, device=device).unsqueeze(0).expand(3, -1)

def compute_kl(t, p, eps=1e-12, clamp_min=1e-9):
    ratio = torch.clamp(t, min=eps) / torch.clamp(p, min=clamp_min)
    kl = torch.sum(torch.where(t > 0, t * torch.log(ratio), torch.zeros_like(t)), dim=-1)
    return kl.mean().item()

class ModularCRM_LearnableB0(nn.Module):
    def __init__(self, N=50, K=4, c_w=0.70, b0_mode='learnable_scalar', fixed_b0=1.6):
        super().__init__()
        self.N, self.K = N, K
        self.c_w = c_w
        self.b0_mode = b0_mode
        
        self.theta_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.1)
        
        self.k_attach_raw = nn.Parameter(torch.tensor([-1.0]))
        self.k_detach_raw = nn.Parameter(torch.randn(N) * 0.1 - 1.0)
        
        self.a_base = nn.Parameter(torch.randn(K) * 0.5 + 2.0)
        self.a_mucin = nn.Parameter(torch.randn(K) * 0.5 + 2.0)
        
        # Initial biomass handling
        if b0_mode == 'fixed':
            self.register_buffer('b0_fixed', torch.tensor(float(fixed_b0)))
        elif b0_mode == 'learnable_scalar':
            # Initialize such that softplus(raw) + 0.1 ~ 1.6
            # 1.6 - 0.1 = 1.5 -> softplus_inv(1.5) = log(exp(1.5) - 1) ~ 1.25
            self.b0_raw = nn.Parameter(torch.tensor(1.25))
        elif b0_mode == 'learnable_condition':
            # Separate b0 for liquid no-carrier vs carrier supernatant
            self.b0_liq_raw = nn.Parameter(torch.tensor(1.25))
            self.b0_sup_raw = nn.Parameter(torch.tensor(1.25))

    def get_b0(self):
        if self.b0_mode == 'fixed':
            return self.b0_fixed, self.b0_fixed
        elif self.b0_mode == 'learnable_scalar':
            b0 = F.softplus(self.b0_raw) + 0.1
            return b0, b0
        elif self.b0_mode == 'learnable_condition':
            b0_liq = F.softplus(self.b0_liq_raw) + 0.1
            b0_sup = F.softplus(self.b0_sup_raw) + 0.1
            return b0_liq, b0_sup

    def get_params(self):
        Theta = F.softplus(self.theta_raw)
        alpha = F.softplus(self.alpha_raw)
        k_attach = F.softplus(self.k_attach_raw)
        k_detach = F.softplus(self.k_detach_raw)
        R0_base = F.softplus(self.a_base)
        R0_mucin = F.softplus(self.a_mucin)
        W_sp = F.softplus(self.w_raw) * (1.0 - torch.eye(self.K, device=device))
        scale = torch.clamp(self.c_w / torch.clamp(W_sp.sum(dim=-1, keepdim=True), min=1e-6), max=1.0)
        W = W_sp * scale
        return Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin

    def forward_carrier(self, p0_vec, is_mucin=True, tf_targets=None, tf_ratio=0.0):
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin = self.get_params()
        _, b0_sup = self.get_b0()
        R0_cond = R0_mucin if is_mucin else R0_base
        R0_batch = R0_cond.unsqueeze(0).expand(p0_vec.size(0), -1)
        
        preds_c, preds_s = [], []
        t_span = torch.linspace(0.0, 1.0, 11, device=device)
        Nc_curr = torch.zeros_like(p0_vec)
        Ns_curr = p0_vec * b0_sup
        eta = 1.0 / 30.0
        
        for p in range(6):
            if p == 0:
                Nc_0 = Nc_curr
                Ns_0 = Ns_curr
            else:
                if tf_ratio > 0.0 and tf_targets is not None:
                    target_c, target_s = tf_targets
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
                dNc = Nc * torch.matmul(Rt, Theta.t()) - alpha * (Nc**2) - k_detach * Nc + k_attach * Ns
                dNs = Ns * torch.matmul(Rt, Theta.t()) - alpha * (Ns**2) + k_detach * Nc - k_attach * Ns
                flux = Rt * torch.matmul(Nc + Ns, Theta)
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNc, dNs, dRt], dim=-1)
                
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': 0.1})
            Nc_curr = torch.clamp(S_t[-1, :, :50], min=0.0)
            Ns_curr = torch.clamp(S_t[-1, :, 50:100], min=0.0)
            preds_c.append(Nc_curr / (Nc_curr.sum(dim=-1, keepdim=True) + 1e-8))
            preds_s.append(Ns_curr / (Ns_curr.sum(dim=-1, keepdim=True) + 1e-8))
            
        return torch.stack(preds_c, dim=1), torch.stack(preds_s, dim=1)

    def forward_no_carrier(self, p0_vec, tf_target=None, tf_ratio=0.0):
        Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin = self.get_params()
        b0_liq, _ = self.get_b0()
        R0_batch = R0_base.unsqueeze(0).expand(p0_vec.size(0), -1)
        
        preds = []
        t_span = torch.linspace(0.0, 1.0, 11, device=device)
        N_curr = p0_vec * b0_liq
        eta = 1.0 / 30.0
        
        for p in range(6):
            if p == 0:
                N_0 = N_curr
            else:
                if tf_ratio > 0.0 and tf_target is not None:
                    N_tf = tf_ratio * (tf_target[:, p-1, :] * N_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_curr
                else:
                    N_tf = N_curr
                N_0 = torch.clamp(eta * N_tf + 1e-6, min=0.0)
                
            S0 = torch.cat([N_0, R0_batch], dim=-1)
            def f(t, S):
                Nt = torch.clamp(S[..., :50], min=0.0)
                Rt = torch.clamp(S[..., 50:], min=0.0)
                dNt = Nt * torch.matmul(Rt, Theta.t()) - alpha * (Nt**2)
                flux = Rt * torch.matmul(Nt, Theta)
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNt, dRt], dim=-1)
                
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': 0.1})
            N_curr = torch.clamp(S_t[-1, :, :50], min=0.0)
            preds.append(N_curr / (N_curr.sum(dim=-1, keepdim=True) + 1e-8))
            
        return torch.stack(preds, dim=1)

def evaluate_loocv(b0_mode='learnable_scalar', fixed_b0=1.6, epochs=200, lr=0.015, K=4, c_w=0.70, reg_alpha=0.001, reg_w=0.001):
    val_kls, val_kls_no, val_kls_mu = [], [], []
    learned_b0s = []
    
    for val_vessel in range(3):
        train_vessels = [v for v in range(3) if v != val_vessel]
        
        p0_train = p0[train_vessels]
        p0_val = p0[val_vessel:val_vessel+1]
        
        no_train = no_c_tensor[train_vessels]
        no_val = no_c_tensor[val_vessel:val_vessel+1]
        
        mu_c_train = mucin_c_tensor[train_vessels]
        mu_s_train = mucin_s_tensor[train_vessels]
        mu_c_val = mucin_c_tensor[val_vessel:val_vessel+1]
        mu_s_val = mucin_s_tensor[val_vessel:val_vessel+1]
        
        torch.manual_seed(42 + val_vessel)
        model = ModularCRM_LearnableB0(N=50, K=K, c_w=c_w, b0_mode=b0_mode, fixed_b0=fixed_b0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
            
            p_no = model.forward_no_carrier(p0_train, tf_target=no_train, tf_ratio=tf_ratio)
            ratio_no = torch.clamp(no_train, min=1e-12) / torch.clamp(p_no, min=1e-9)
            loss = torch.sum(torch.where(no_train > 0, no_train * torch.log(ratio_no), torch.zeros_like(no_train)), dim=-1).mean()
            
            p_mu_c, p_mu_s = model.forward_carrier(p0_train, is_mucin=True, tf_targets=(mu_c_train, mu_s_train), tf_ratio=tf_ratio)
            ratio_mu_c = torch.clamp(mu_c_train, min=1e-12) / torch.clamp(p_mu_c, min=1e-9)
            ratio_mu_s = torch.clamp(mu_s_train, min=1e-12) / torch.clamp(p_mu_s, min=1e-9)
            loss += torch.sum(torch.where(mu_c_train > 0, mu_c_train * torch.log(ratio_mu_c), torch.zeros_like(mu_c_train)), dim=-1).mean() + \
                    torch.sum(torch.where(mu_s_train > 0, mu_s_train * torch.log(ratio_mu_s), torch.zeros_like(mu_s_train)), dim=-1).mean()
                    
            Theta, W, alpha, k_attach, k_detach, R0_base, R0_mucin = model.get_params()
            loss += reg_w * (Theta.pow(2).sum() + W.pow(2).sum()) + reg_alpha * alpha.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            b0_l, b0_s = model.get_b0()
            learned_b0s.append((b0_l.item(), b0_s.item()))
            pred_no_val = model.forward_no_carrier(p0_val, tf_ratio=0.0)
            pred_mu_c_val, pred_mu_s_val = model.forward_carrier(p0_val, is_mucin=True, tf_ratio=0.0)
            
            kl_no = compute_kl(no_val, pred_no_val)
            kl_mu = 0.5 * (compute_kl(mu_c_val, pred_mu_c_val) + compute_kl(mu_s_val, pred_mu_s_val))
            val_kls.append(0.5 * (kl_no + kl_mu))
            val_kls_no.append(kl_no)
            val_kls_mu.append(kl_mu)
            
    mean_b0_l = np.mean([b[0] for b in learned_b0s])
    mean_b0_s = np.mean([b[1] for b in learned_b0s])
    return {
        'mode': b0_mode,
        'mean_val_kl': float(np.mean(val_kls)),
        'std_val_kl': float(np.std(val_kls)),
        'val_kl_no': float(np.mean(val_kls_no)),
        'val_kl_mu': float(np.mean(val_kls_mu)),
        'mean_b0_liq': mean_b0_l,
        'mean_b0_sup': mean_b0_s
    }

if __name__ == '__main__':
    print("\n" + "="*90, flush=True)
    print("EVALUATING LEARNABLE INITIAL BIOMASS b0 (K=4, c_w=0.70, reg_alpha=0.001)")
    print("Comparing Fixed b0 vs Learnable Scalar b0 vs Learnable Condition b0")
    print("="*90, flush=True)
    
    modes = [
        ('fixed', 1.6),
        ('learnable_scalar', 1.6),
        ('learnable_condition', 1.6),
    ]
    
    results = []
    for mode, fixed_val in modes:
        res = evaluate_loocv(b0_mode=mode, fixed_b0=fixed_val, epochs=200, lr=0.015, K=4, c_w=0.70, reg_alpha=0.001, reg_w=0.001)
        results.append(res)
        print(f"Mode: {res['mode']:<20} | Comb Val KL: {res['mean_val_kl']:.4f} +/- {res['std_val_kl']:.4f} | NoCar KL: {res['val_kl_no']:.4f} | Mucin KL: {res['val_kl_mu']:.4f} | Learned b0: (liq={res['mean_b0_liq']:.2f}, sup={res['mean_b0_sup']:.2f})", flush=True)
        
    df_out = pd.DataFrame(results)
    csv_path = os.path.join(PROJECT_ROOT, 'Data', 'Learnable_b0_LOOCV_Comparison.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}", flush=True)

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}")

data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Vessel_Tensors.pt')
data = torch.load(data_path, map_location=device, weights_only=False)

no_c_tensor = data['no_c_tensor'].to(device)       # (3 vessels, 6 passages, 50 strains)
plain_c_tensor = data['plain_c_tensor'].to(device) # (3, 6, 50)
plain_s_tensor = data['plain_s_tensor'].to(device) # (3, 6, 50)
mucin_c_tensor = data['mucin_c_tensor'].to(device) # (3, 6, 50)
mucin_s_tensor = data['mucin_s_tensor'].to(device) # (3, 6, 50)

strains = data['strains']
vessels = data['vessels']
N = no_c_tensor.size(-1)

class DualThetaUnifiedCRM(nn.Module):
    def __init__(self, N, K, c_w=0.85, eta_val=1.0/30.0):
        super().__init__()
        self.N, self.K = N, K
        self.c_w = c_w
        self.eta_val = float(eta_val)
        
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

    def get_params(self, enforce_w_zero=False):
        Theta_c = F.softplus(self.theta_c_raw)
        Theta_s = F.softplus(self.theta_s_raw)
        alpha = F.softplus(self.alpha_raw)
        k_shed = F.softplus(self.k_shed_raw)
        k_attach = F.softplus(self.k_attach_raw)
        R0 = F.softplus(self.a)
        b0_liq = F.softplus(self.b0_liq_raw)
        b0_c = F.softplus(self.b0_c_raw)
        b0_s = F.softplus(self.b0_s_raw)
        eta = torch.tensor(self.eta_val, device=self.theta_c_raw.device)
        
        if enforce_w_zero:
            W = torch.zeros(self.K, self.K, device=self.w_raw.device)
        else:
            W_sp = F.softplus(self.w_raw) * (1.0 - torch.eye(self.K, device=self.w_raw.device))
            row_sums = W_sp.sum(dim=-1, keepdim=True)
            scale = torch.clamp(self.c_w / torch.clamp(row_sums, min=1e-6), max=1.0)
            W = W_sp * scale
        return Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta

    def rollout_no_carrier(self, n_c_sub, Theta_s, W, alpha, R0_liq, b0_liq, delta_t=0.1, tf_ratio=0.0):
        N_curr = n_c_sub[:, 0, :] * b0_liq
        R0_batch = R0_liq.unsqueeze(0).expand(n_c_sub.size(0), -1)
        preds = []
        t_span = torch.linspace(0.0, 1.0, int(1.0 / delta_t) + 1, device=n_c_sub.device)
        for p in range(1, 6):
            if tf_ratio > 0.0 and self.training:
                N_transferred = tf_ratio * (n_c_sub[:, p-1, :] * N_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_curr
            else:
                N_transferred = N_curr
            N0 = torch.clamp(N_transferred / 100.0 + 1e-6, min=0.0)
            S0 = torch.cat([N0, R0_batch], dim=-1)
            def f(t, S):
                Nt = torch.clamp(S[..., :self.N], min=0.0)
                Rt = torch.clamp(S[..., self.N:], min=0.0)
                dNt = Nt * torch.matmul(Rt, Theta_s.t()) - alpha * (Nt**2)
                flux = Rt * torch.matmul(Nt, Theta_s)
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNt, dRt], dim=-1)
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            N_final = torch.clamp(S_t[-1, :, :self.N], min=0.0)
            preds.append(N_final / (N_final.sum(dim=-1, keepdim=True) + 1e-8))
            N_curr = N_final
        return torch.stack(preds, dim=1)

    def rollout_carrier(self, c_sub, s_sub, Theta_c, Theta_s, W, alpha, k_shed, k_att, R0_cond, b0_c, b0_s, eta, delta_t=0.1, tf_ratio=0.0):
        N_c_curr = c_sub[:, 0, :] * b0_c
        N_s_curr = s_sub[:, 0, :] * b0_s
        R0_batch = R0_cond.unsqueeze(0).expand(c_sub.size(0), -1)
        preds_c, preds_s = [], []
        t_span = torch.linspace(0.0, 1.0, int(1.0 / delta_t) + 1, device=c_sub.device)
        for p in range(1, 6):
            if tf_ratio > 0.0 and self.training:
                N_c_transferred = tf_ratio * (c_sub[:, p-1, :] * N_c_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_c_curr
                N_s_transferred = tf_ratio * (s_sub[:, p-1, :] * N_s_curr.sum(dim=-1, keepdim=True)) + (1.0 - tf_ratio) * N_s_curr
            else:
                N_c_transferred = N_c_curr
                N_s_transferred = N_s_curr
            N_c_0 = torch.clamp(0.2 * N_c_transferred + 1e-6, min=0.0)
            N_s_0 = torch.clamp(eta * N_s_transferred + 1e-6, min=0.0)
            S0 = torch.cat([N_c_0, N_s_0, R0_batch], dim=-1)
            def f(t, S):
                Nc = torch.clamp(S[..., :self.N], min=0.0)
                Ns = torch.clamp(S[..., self.N:2*self.N], min=0.0)
                Rt = torch.clamp(S[..., 2*self.N:], min=0.0)
                growth_c = torch.matmul(Rt, Theta_c.t())
                growth_s = torch.matmul(Rt, Theta_s.t())
                dNc = Nc * growth_c - alpha * (Nc**2) - k_shed * Nc + k_att * Ns
                dNs = Ns * growth_s - alpha * (Ns**2) + k_shed * Nc - k_att * Ns
                flux = Rt * (torch.matmul(Nc, Theta_c) + torch.matmul(Ns, Theta_s))
                dRt = - flux + torch.matmul(flux, W)
                return torch.cat([dNc, dNs, dRt], dim=-1)
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            Nc_fin = torch.clamp(S_t[-1, :, :self.N], min=0.0)
            Ns_fin = torch.clamp(S_t[-1, :, self.N:2*self.N], min=0.0)
            preds_c.append(Nc_fin / (Nc_fin.sum(dim=-1, keepdim=True) + 1e-8))
            preds_s.append(Ns_fin / (Ns_fin.sum(dim=-1, keepdim=True) + 1e-8))
            N_c_curr = Nc_fin
            N_s_curr = Ns_fin
        return torch.stack(preds_c, dim=1), torch.stack(preds_s, dim=1)

def evaluate_predictions(model, Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta):
    with torch.no_grad():
        # 1. No Carrier
        p_no = model.rollout_no_carrier(no_c_tensor, Theta_s, W, alpha, R0[0], b0_liq, tf_ratio=0.0).cpu().numpy().flatten()
        t_no = no_c_tensor[:, 1:, :].cpu().numpy().flatten()
        r2_no = 1.0 - np.sum((t_no - p_no)**2) / np.sum((t_no - np.mean(t_no))**2)
        mae_no = np.mean(np.abs(t_no - p_no))
        
        # 2. Plain Agar
        p_pl_c, p_pl_s = model.rollout_carrier(plain_c_tensor, plain_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta, tf_ratio=0.0)
        p_pl_c_f = p_pl_c.cpu().numpy().flatten()
        p_pl_s_f = p_pl_s.cpu().numpy().flatten()
        t_pl_c_f = plain_c_tensor[:, 1:, :].cpu().numpy().flatten()
        t_pl_s_f = plain_s_tensor[:, 1:, :].cpu().numpy().flatten()
        
        r2_pl_c = 1.0 - np.sum((t_pl_c_f - p_pl_c_f)**2) / np.sum((t_pl_c_f - np.mean(t_pl_c_f))**2)
        r2_pl_s = 1.0 - np.sum((t_pl_s_f - p_pl_s_f)**2) / np.sum((t_pl_s_f - np.mean(t_pl_s_f))**2)
        p_plain = np.concatenate([p_pl_c_f, p_pl_s_f])
        t_plain = np.concatenate([t_pl_c_f, t_pl_s_f])
        r2_plain = 1.0 - np.sum((t_plain - p_plain)**2) / np.sum((t_plain - np.mean(t_plain))**2)
        mae_plain = np.mean(np.abs(t_plain - p_plain))
        
        # 3. Mucin Agar
        p_mu_c, p_mu_s = model.rollout_carrier(mucin_c_tensor, mucin_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta, tf_ratio=0.0)
        p_mu_c_f = p_mu_c.cpu().numpy().flatten()
        p_mu_s_f = p_mu_s.cpu().numpy().flatten()
        t_mu_c_f = mucin_c_tensor[:, 1:, :].cpu().numpy().flatten()
        t_mu_s_f = mucin_s_tensor[:, 1:, :].cpu().numpy().flatten()
        
        r2_mu_c = 1.0 - np.sum((t_mu_c_f - p_mu_c_f)**2) / np.sum((t_mu_c_f - np.mean(t_mu_c_f))**2)
        r2_mu_s = 1.0 - np.sum((t_mu_s_f - p_mu_s_f)**2) / np.sum((t_mu_s_f - np.mean(t_mu_s_f))**2)
        p_mucin = np.concatenate([p_mu_c_f, p_mu_s_f])
        t_mucin = np.concatenate([t_mu_c_f, t_mu_s_f])
        r2_mucin = 1.0 - np.sum((t_mucin - p_mucin)**2) / np.sum((t_mucin - np.mean(t_mucin))**2)
        mae_mucin = np.mean(np.abs(t_mucin - p_mucin))
        
        # Global
        all_pred = np.concatenate([p_no, p_plain, p_mucin])
        all_true = np.concatenate([t_no, t_plain, t_mucin])
        r2_global = 1.0 - np.sum((all_true - all_pred)**2) / np.sum((all_true - np.mean(all_true))**2)
        mae_global = np.mean(np.abs(all_true - all_pred))
        
        return {
            'r2_no': r2_no, 'mae_no': mae_no,
            'r2_plain': r2_plain, 'mae_plain': mae_plain,
            'r2_plain_c': r2_pl_c, 'r2_plain_s': r2_pl_s,
            'r2_mucin': r2_mucin, 'mae_mucin': mae_mucin,
            'r2_mucin_c': r2_mu_c, 'r2_mucin_s': r2_mu_s,
            'r2_global': r2_global, 'mae_global': mae_global,
            'r2_3cond': np.array([r2_no, r2_plain, r2_mucin])
        }

def train_single_seed(seed, K, c_w, lr=0.015, epochs=250, tf_decay=0.75, enforce_w_zero=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DualThetaUnifiedCRM(N=N, K=K, c_w=c_w, eta_val=1.0/30.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params(enforce_w_zero=enforce_w_zero)
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * tf_decay)))
        
        p_no = model.rollout_no_carrier(no_c_tensor, Theta_s, W, alpha, R0[0], b0_liq, tf_ratio=tf_ratio)
        loss_no = torch.sum(no_c_tensor[:, 1:, :] * torch.log(no_c_tensor[:, 1:, :] / torch.clamp(p_no, min=1e-9) + 1e-12), dim=-1).mean()
        
        p_pl_c, p_pl_s = model.rollout_carrier(plain_c_tensor, plain_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[0], R0[1], b0_c, b0_s, eta, tf_ratio=tf_ratio)
        loss_pl = torch.sum(plain_c_tensor[:, 1:, :] * torch.log(plain_c_tensor[:, 1:, :] / torch.clamp(p_pl_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                  torch.sum(plain_s_tensor[:, 1:, :] * torch.log(plain_s_tensor[:, 1:, :] / torch.clamp(p_pl_s, min=1e-9) + 1e-12), dim=-1).mean()
                  
        p_mu_c, p_mu_s = model.rollout_carrier(mucin_c_tensor, mucin_s_tensor, Theta_c, Theta_s, W, alpha, k_shed, k_attach[1], R0[2], b0_c, b0_s, eta, tf_ratio=tf_ratio)
        loss_mu = torch.sum(mucin_c_tensor[:, 1:, :] * torch.log(mucin_c_tensor[:, 1:, :] / torch.clamp(p_mu_c, min=1e-9) + 1e-12), dim=-1).mean() + \
                  torch.sum(mucin_s_tensor[:, 1:, :] * torch.log(mucin_s_tensor[:, 1:, :] / torch.clamp(p_mu_s, min=1e-9) + 1e-12), dim=-1).mean()
                  
        l2_reg = 0.001 * (Theta_c.pow(2).sum() + Theta_s.pow(2).sum() + W.pow(2).sum())
        total_loss = loss_no + loss_pl + loss_mu + l2_reg
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        
        fit_loss = (loss_no + loss_pl + loss_mu).item()
        if fit_loss < best_loss:
            best_loss = fit_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_state)
    return model, best_loss

def compute_null_models(model, Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta, N_shuffles=50):
    with torch.no_grad():
        # 1. Persistence Baseline
        p0_liq = np.repeat(no_c_tensor[:, 0:1, :].cpu().numpy(), 5, axis=1).flatten()
        t_liq = no_c_tensor[:, 1:, :].cpu().numpy().flatten()
        r2_p_liq = 1.0 - np.sum((t_liq - p0_liq)**2) / np.sum((t_liq - np.mean(t_liq))**2)
        
        p0_pl = np.concatenate([np.repeat(plain_c_tensor[:, 0:1, :].cpu().numpy(), 5, axis=1).flatten(),
                                np.repeat(plain_s_tensor[:, 0:1, :].cpu().numpy(), 5, axis=1).flatten()])
        t_pl = np.concatenate([plain_c_tensor[:, 1:, :].cpu().numpy().flatten(), plain_s_tensor[:, 1:, :].cpu().numpy().flatten()])
        r2_p_pl = 1.0 - np.sum((t_pl - p0_pl)**2) / np.sum((t_pl - np.mean(t_pl))**2)
        
        p0_mu = np.concatenate([np.repeat(mucin_c_tensor[:, 0:1, :].cpu().numpy(), 5, axis=1).flatten(),
                                np.repeat(mucin_s_tensor[:, 0:1, :].cpu().numpy(), 5, axis=1).flatten()])
        t_mu = np.concatenate([mucin_c_tensor[:, 1:, :].cpu().numpy().flatten(), mucin_s_tensor[:, 1:, :].cpu().numpy().flatten()])
        r2_p_mu = 1.0 - np.sum((t_mu - p0_mu)**2) / np.sum((t_mu - np.mean(t_mu))**2)
        persist_r2 = np.array([r2_p_liq, r2_p_pl, r2_p_mu])
        
        # 2. Scrambled Nutrients R0 (preserving total nutrient pool)
        K = R0.size(-1)
        shuff_r0_runs = []
        for _ in range(N_shuffles):
            diri = np.random.dirichlet([1.0] * K, size=3)
            r0_rand = torch.tensor(diri, dtype=torch.float32, device=device) * R0.sum(dim=-1, keepdim=True)
            res = evaluate_predictions(model, Theta_c, Theta_s, W, alpha, k_shed, k_attach, r0_rand, b0_liq, b0_c, b0_s, eta)
            shuff_r0_runs.append(res['r2_3cond'])
        shuff_r0_runs = np.array(shuff_r0_runs)
        shuff_r0_r2 = np.mean(shuff_r0_runs, axis=0)
        shuff_r0_std = np.std(shuff_r0_runs, axis=0)
        
        # 3. Shuffled Dual-Theta (permute species rows)
        shuff_th_runs = []
        for _ in range(N_shuffles):
            perm = torch.randperm(N)
            res = evaluate_predictions(model, Theta_c[perm], Theta_s[perm], W, alpha[perm], k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta)
            shuff_th_runs.append(res['r2_3cond'])
        shuff_th_runs = np.array(shuff_th_runs)
        shuff_th_r2 = np.mean(shuff_th_runs, axis=0)
        shuff_th_std = np.std(shuff_th_runs, axis=0)
        
        # 4. Shuffled Adhesion Traits (permute adhesion across species)
        shuff_adh_runs = []
        for _ in range(N_shuffles):
            perm = torch.randperm(N)
            res = evaluate_predictions(model, Theta_c, Theta_s, W, alpha, k_shed[perm], k_attach[:, perm], R0, b0_liq, b0_c, b0_s, eta)
            shuff_adh_runs.append(res['r2_3cond'])
        shuff_adh_runs = np.array(shuff_adh_runs)
        shuff_adh_r2 = np.mean(shuff_adh_runs, axis=0)
        shuff_adh_std = np.std(shuff_adh_runs, axis=0)
        
        return {
            'persist_r2': persist_r2,
            'shuff_r0_r2': shuff_r0_r2,
            'shuff_r0_std': shuff_r0_std,
            'shuff_th_r2': shuff_th_r2,
            'shuff_th_std': shuff_th_std,
            'shuff_adh_r2': shuff_adh_r2,
            'shuff_adh_std': shuff_adh_std
        }
def main():
    parser = argparse.ArgumentParser(description="Train multi-seed ensemble on Pollard averaged dataset")
    parser.add_argument('--k', type=int, default=None, help="Number of resources K")
    parser.add_argument('--c_w', type=float, default=None, help="Byproduct cross-feeding bound")
    parser.add_argument('--lr', type=float, default=None, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=None, help="Training epochs")
    parser.add_argument('--n_seeds', type=int, default=10, help="Number of seeds")
    args = parser.parse_args()

    # If parameters not explicitly provided, load from Stage 2 results if available, else Stage 1, else defaults
    stage2_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Stage2_Search_Results.csv')
    stage1_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Stage1_Search_Results.csv')
    
    if args.k is None or args.c_w is None or args.lr is None or args.epochs is None:
        if os.path.exists(stage2_path):
            df_res = pd.read_csv(stage2_path)
            top = df_res.iloc[0]
            K = int(args.k if args.k is not None else top['K'])
            c_w = float(args.c_w if args.c_w is not None else top['c_w'])
            lr = float(args.lr if args.lr is not None else top['lr'])
            epochs = int(args.epochs if args.epochs is not None else top['epochs'])
            print(f"Auto-loaded top config from Stage 2: K={K}, c_w={c_w}, lr={lr}, epochs={epochs}")
        elif os.path.exists(stage1_path):
            df_res = pd.read_csv(stage1_path)
            top = df_res.iloc[0]
            K = int(args.k if args.k is not None else top['K'])
            c_w = float(args.c_w if args.c_w is not None else top['c_w'])
            lr = float(args.lr if args.lr is not None else 0.015)
            epochs = int(args.epochs if args.epochs is not None else 200)
            print(f"Auto-loaded top config from Stage 1: K={K}, c_w={c_w}, lr={lr}, epochs={epochs}")
        else:
            K = args.k or 4
            c_w = args.c_w or 0.85
            lr = args.lr or 0.015
            epochs = args.epochs or 250
    else:
        K, c_w, lr, epochs = args.k, args.c_w, args.lr, args.epochs

    seeds = [42, 101, 102, 103, 104, 105, 106, 107, 108, 109][:args.n_seeds]
    print("="*80)
    print(f"TRAINING {len(seeds)} ENSEMBLE SEEDS ON AVERAGED DATASET")
    print(f"Configuration: K={K}, c_w={c_w}, lr={lr}, epochs={epochs}, eta=1/30 (fixed)")
    print("="*80)

    # 1. First train dedicated No-W model (W = 0 strictly) on seed 42 to get clean No-W benchmark
    print("\nTraining retrained No-W (W=0) baseline model...")
    t0_nw = time.time()
    model_no_w, loss_nw = train_single_seed(42, K=K, c_w=c_w, lr=lr, epochs=epochs, enforce_w_zero=True)
    with torch.no_grad():
        th_c_nw, th_s_nw, w_nw, a_nw, ks_nw, ka_nw, r0_nw, b0l_nw, b0c_nw, b0s_nw, eta_nw = model_no_w.get_params(enforce_w_zero=True)
        res_no_w = evaluate_predictions(model_no_w, th_c_nw, th_s_nw, w_nw, a_nw, ks_nw, ka_nw, r0_nw, b0l_nw, b0c_nw, b0s_nw, eta_nw)
    print(f"No-W Baseline trained in {time.time()-t0_nw:.1f}s | R2: No-C={res_no_w['r2_no']:.4f}, Plain={res_no_w['r2_plain']:.4f}, Mucin={res_no_w['r2_mucin']:.4f}")

    # 2. Train multi-seed ensemble
    runs_data = []
    best_global_r2 = -float('inf')
    best_model_state = None
    best_model_seed = None
    best_model_obj = None

    t_start = time.time()
    for s_idx, seed in enumerate(seeds, 1):
        t0 = time.time()
        model, fit_loss = train_single_seed(seed, K=K, c_w=c_w, lr=lr, epochs=epochs, enforce_w_zero=False)
        elapsed = time.time() - t0
        
        with torch.no_grad():
            Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta = model.get_params()
            res = evaluate_predictions(model, Theta_c, Theta_s, W, alpha, k_shed, k_attach, R0, b0_liq, b0_c, b0_s, eta)
            
            k_shed_np = k_shed.cpu().numpy()
            k_att_plain_np = k_attach[0].cpu().numpy()
            k_att_mucin_np = k_attach[1].cpu().numpy()
            Theta_c_np = Theta_c.cpu().numpy()
            Theta_s_np = Theta_s.cpu().numpy()
            W_np = W.cpu().numpy()
            
            ratio_plain = k_att_plain_np / (k_shed_np + 1e-6)
            ratio_mucin = k_att_mucin_np / (k_shed_np + 1e-6)
            th_c_norm = np.linalg.norm(Theta_c_np, axis=1)
            th_s_norm = np.linalg.norm(Theta_s_np, axis=1)
            theta_ratio = th_c_norm / (th_s_norm + 1e-6)
            
            log_rm = np.log10(ratio_mucin)
            r_val, p_val = pearsonr(log_rm, theta_ratio)
            rho_val, p_rho_val = spearmanr(ratio_mucin, theta_ratio)
            
            if res['r2_global'] > best_global_r2:
                best_global_r2 = res['r2_global']
                best_model_seed = seed
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_model_obj = model
                
            print(f"Seed {seed:3d} ({s_idx:2d}/{len(seeds)}) in {elapsed:.1f}s | "
                  f"Global R2: {res['r2_global']:.4f} (No-C: {res['r2_no']:.3f}, Plain: {res['r2_plain']:.3f}, Mucin: {res['r2_mucin']:.3f}) | "
                  f"Trade-off r: {r_val:+.3f} (p={p_val:.2e}, rho={rho_val:+.3f})", flush=True)
                  
            runs_data.append({
                'seed': seed,
                'fit_loss': fit_loss,
                'res': res,
                'k_shed': k_shed_np,
                'k_att_plain': k_att_plain_np,
                'k_att_mucin': k_att_mucin_np,
                'ratio_plain': ratio_plain,
                'ratio_mucin': ratio_mucin,
                'th_c_norm': th_c_norm,
                'th_s_norm': th_s_norm,
                'theta_ratio': theta_ratio,
                'pearson_r': r_val,
                'p_val': p_val,
                'spearman_rho': rho_val,
                'p_rho_val': p_rho_val,
                'Theta_c': Theta_c_np,
                'Theta_s': Theta_s_np,
                'W': W_np,
                'R0': R0.cpu().numpy(),
                'alpha': alpha.cpu().numpy()
            })

    total_time = time.time() - t_start
    print(f"\nCompleted {len(seeds)} ensemble runs in {total_time/60.0:.2f} min. Best Seed: {best_model_seed} (Global R2={best_global_r2:.4f})", flush=True)

    # Compute null models for the winning model
    print(f"\nEvaluating null models on best seed {best_model_seed}...", flush=True)
    t0_null = time.time()
    with torch.no_grad():
        th_c_b, th_s_b, w_b, a_b, ks_b, ka_b, r0_b, b0l_b, b0c_b, b0s_b, eta_b = best_model_obj.get_params()
        null_res = compute_null_models(best_model_obj, th_c_b, th_s_b, w_b, a_b, ks_b, ka_b, r0_b, b0l_b, b0c_b, b0s_b, eta_b, N_shuffles=50)
    print(f"Null models evaluated in {time.time()-t0_null:.1f}s:", flush=True)
    print(f"  Persistence:  {null_res['persist_r2']}", flush=True)
    print(f"  Scrambled R0: {null_res['shuff_r0_r2']} +/- {null_res['shuff_r0_std']}", flush=True)
    print(f"  Shuffled Th:  {null_res['shuff_th_r2']} +/- {null_res['shuff_th_std']}", flush=True)
    print(f"  Shuffled Adh: {null_res['shuff_adh_r2']} +/- {null_res['shuff_adh_std']}", flush=True)

    # Save best model checkpoint
    best_ckpt_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Winning_CRM.pth')
    torch.save({
        'model_state_dict': best_model_state,
        'seed': best_model_seed,
        'K': K,
        'c_w': c_w,
        'lr': lr,
        'epochs': epochs,
        'best_global_r2': best_global_r2,
        'res_no_w': res_no_w,
        'null_res': null_res
    }, best_ckpt_path)
    print(f"Saved winning checkpoint to {best_ckpt_path}", flush=True)

    # Save ensemble multi-run dataset
    out_ensemble_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_DualTheta_MultiRun_10Seeds.pt')
    results = {
        'seeds': seeds,
        'K': K,
        'c_w': c_w,
        'lr': lr,
        'epochs': epochs,
        'runs_data': runs_data,
        'res_no_w': res_no_w,
        'null_res': null_res,
        'strains': strains,
        'vessels': vessels
    }
    torch.save(results, out_ensemble_path)
    print(f"Saved ensemble results to {out_ensemble_path}", flush=True)

if __name__ == '__main__':
    main()

"""
Coupled Biofilm-Planktonic Consumer-Resource Neural ODE for Serial Dilution Microcosms.

Simultaneously models:
1. Biofilm phase on hydrogel carriers (N_c)
2. Planktonic phase in liquid supernatant (N_s)
3. Shared latent environmental resource dynamics (R)

Under the physical passaging transfer rule:
- 1 of 5 carriers transferred: N_c(0)^(p+1) = 0.2 * N_c(tau)^(p) + epsilon
- Supernatant discarded / minimal carryover: N_s(0)^(p+1) = eta * N_s(tau)^(p) + epsilon
- Fresh media replenished: R(0)^(p+1) = R_0

Predicts relative abundances of BOTH carriers and supernatants simultaneously
across all passages P1 through P6 in Pollard's 50-strain synthetic gut microbiome.
"""

import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', 'SupplementaryDataS2.csv')
SAVE_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Coupled_Biofilm_ODE_50Strains.pth')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_coupled_data(csv_path=DATA_PATH, min_prev=1e-3):
    df = pd.read_csv(csv_path)
    df = df[df['sampleType'] != 'inoculum'].copy()
    
    # Normalize read fraction to relative abundance
    sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
    if sums.mean() > 50:
        df['Relative_Abundance'] = df['Read_Fraction'] / 100.0
    else:
        df['Relative_Abundance'] = df['Read_Fraction']
        
    sample_cols = ['sampleType', 'passage', 'replicate']
    num_samples = len(df.drop_duplicates(subset=sample_cols))
    mean_abundances = df.groupby('Strain_Name')['Relative_Abundance'].sum() / num_samples
    strains_keep = sorted(list(mean_abundances[mean_abundances >= min_prev].index))
    
    conditions = ['mucin-agar', 'plain-agar']
    replicates = sorted(list(df['replicate'].unique()))
    passages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    
    carrier_data, supernatant_data, cond_labels, rep_labels = [], [], [], []
    
    for c_idx, cond in enumerate(conditions):
        c_sub = df[df['sampleType'] == f'{cond} carriers']
        s_sub = df[df['sampleType'] == f'{cond} supernatant']
        
        c_pivot = c_sub.pivot_table(index=['replicate', 'passage'], columns='Strain_Name', 
                                    values='Relative_Abundance', fill_value=0.0).reindex(columns=strains_keep, fill_value=0.0)
        s_pivot = s_sub.pivot_table(index=['replicate', 'passage'], columns='Strain_Name', 
                                    values='Relative_Abundance', fill_value=0.0).reindex(columns=strains_keep, fill_value=0.0)
        
        c_pivot = (c_pivot + 1e-6).div((c_pivot + 1e-6).sum(axis=1), axis=0)
        s_pivot = (s_pivot + 1e-6).div((s_pivot + 1e-6).sum(axis=1), axis=0)
        
        for rep in replicates:
            if rep in c_pivot.index.levels[0] and rep in s_pivot.index.levels[0]:
                carrier_data.append(c_pivot.loc[rep].reindex(passages).values)
                supernatant_data.append(s_pivot.loc[rep].reindex(passages).values)
                cond_labels.append(c_idx)
                rep_labels.append(rep)
                
    carrier_tensor = torch.tensor(np.array(carrier_data), dtype=torch.float32)
    supernatant_tensor = torch.tensor(np.array(supernatant_data), dtype=torch.float32)
    cond_tensor = torch.tensor(cond_labels, dtype=torch.long)
    
    return carrier_tensor, supernatant_tensor, cond_tensor, strains_keep, conditions, rep_labels


class CoupledBiofilmPollardODE(nn.Module):
    def __init__(self, N=50, K=5, num_conditions=2, max_crossfeed=0.5):
        super().__init__()
        self.N = N
        self.K = K
        self.num_conditions = num_conditions
        self.max_crossfeed = max_crossfeed
        
        # Universal metabolic traits Theta >= 0
        self.theta_raw = nn.Parameter(torch.randn(N, K) * 0.1)
        # Universal byproduct cross-feeding matrix W >= 0
        self.w_raw = nn.Parameter(torch.randn(K, K) * 0.1 - 1.0)
        # Universal intraspecific self-limitation alpha > 0
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.05 - 1.0)
        
        # Universal shedding rate k_shed > 0
        self.k_shed_raw = nn.Parameter(torch.randn(N) * 0.1 - 1.0)
        # Condition-dependent attachment rate k_attach > 0 (mucin vs plain)
        self.k_attach_raw = nn.Parameter(torch.randn(num_conditions, N) * 0.1 - 1.0)
        
        # Condition-specific initial media resources R0
        self.a = nn.Parameter(torch.randn(num_conditions, K) * 0.5 + 2.0)
        
        # Initial unnormalized biomass scale at P1
        self.b0_c_raw = nn.Parameter(torch.tensor(2.0))
        self.b0_s_raw = nn.Parameter(torch.tensor(2.0))
        
        # Liquid carryover fraction on tweezers/carrier (0 to 2%)
        self.eta_raw = nn.Parameter(torch.tensor(-4.0))

    def get_params(self):
        Theta = F.softplus(self.theta_raw)
        alpha = F.softplus(self.alpha_raw)
        k_shed = F.softplus(self.k_shed_raw)
        k_attach = F.softplus(self.k_attach_raw)
        R0 = F.softplus(self.a)
        b0_c = F.softplus(self.b0_c_raw)
        b0_s = F.softplus(self.b0_s_raw)
        eta = torch.sigmoid(self.eta_raw) * 0.02
        
        W_sp = F.softplus(self.w_raw) * (1.0 - torch.eye(self.K, device=self.w_raw.device))
        row_sums = W_sp.sum(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_crossfeed / torch.clamp(row_sums, min=1e-6), max=1.0)
        W = W_sp * scale
        
        return Theta, W, alpha, k_shed, k_attach, R0, b0_c, b0_s, eta

    def ode_deriv(self, t, S, Theta, W, alpha, k_shed, k_attach_batch):
        N_c = torch.clamp(S[..., :self.N], min=0.0)
        N_s = torch.clamp(S[..., self.N:2*self.N], min=0.0)
        R = torch.clamp(S[..., 2*self.N:], min=0.0)
        
        growth = torch.matmul(R, Theta.t())
        
        # Coupled Biofilm-Planktonic differential equations
        dN_c = N_c * growth - alpha * (N_c ** 2) - k_shed * N_c + k_attach_batch * N_s
        dN_s = N_s * growth - alpha * (N_s ** 2) + k_shed * N_c - k_attach_batch * N_s
        
        # Shared media resource consumption and byproduct secretion
        total_N = N_c + N_s
        flux = R * torch.matmul(total_N, Theta)
        byproduct = torch.matmul(flux, W)
        dR = - flux + byproduct
        
        return torch.cat([dN_c, dN_s, dR], dim=-1)

    def forward_rollout(self, carrier_seq, supernatant_seq, conds, delta_t=0.1, tf_ratio=0.0):
        """
        Simulates 5 serial passages starting from P1 (index 0) to predict P2..P6.
        """
        Theta, W, alpha, k_shed, k_attach, R0_table, b0_c, b0_s, eta = self.get_params()
        
        k_attach_batch = k_attach[conds]
        R0 = R0_table[conds]
        
        batch_size = carrier_seq.size(0)
        num_passages = carrier_seq.size(1)
        
        N_c_curr = carrier_seq[:, 0, :] * b0_c
        N_s_curr = supernatant_seq[:, 0, :] * b0_s
        
        pred_carriers = []
        pred_supernatants = []
        
        t_span = torch.linspace(0.0, 1.0, int(1.0 / delta_t) + 1, device=carrier_seq.device)
        
        for p in range(1, num_passages):
            # Teacher forcing mixture if training
            if tf_ratio > 0.0 and self.training:
                c_prev_obs = carrier_seq[:, p-1, :] * N_c_curr.sum(dim=-1, keepdim=True)
                s_prev_obs = supernatant_seq[:, p-1, :] * N_s_curr.sum(dim=-1, keepdim=True)
                N_c_transferred = tf_ratio * c_prev_obs + (1.0 - tf_ratio) * N_c_curr
                N_s_transferred = tf_ratio * s_prev_obs + (1.0 - tf_ratio) * N_s_curr
            else:
                N_c_transferred = N_c_curr
                N_s_transferred = N_s_curr
                
            # Inter-passage physical transfer rule:
            # 1 of 5 carriers transferred: fc = 0.2
            N_c_0 = torch.clamp(0.2 * N_c_transferred + 1e-6, min=0.0)
            # Liquid discarded: only residual carryover on carrier/tweezers: eta ~ 0.001
            N_s_0 = torch.clamp(eta * N_s_transferred + 1e-6, min=0.0)
            S0 = torch.cat([N_c_0, N_s_0, R0], dim=-1)
            
            def f(t, S):
                return self.ode_deriv(t, S, Theta, W, alpha, k_shed, k_attach_batch)
                
            S_t = odeint(f, S0, t_span, method='euler', options={'step_size': delta_t})
            N_c_final = torch.clamp(S_t[-1, :, :self.N], min=0.0)
            N_s_final = torch.clamp(S_t[-1, :, self.N:2*self.N], min=0.0)
            
            p_c = N_c_final / (N_c_final.sum(dim=-1, keepdim=True) + 1e-8)
            p_s = N_s_final / (N_s_final.sum(dim=-1, keepdim=True) + 1e-8)
            
            pred_carriers.append(p_c)
            pred_supernatants.append(p_s)
            
            N_c_curr = N_c_final
            N_s_curr = N_s_final
            
        pred_carriers = torch.stack(pred_carriers, dim=1) # [batch, 5, N]
        pred_supernatants = torch.stack(pred_supernatants, dim=1) # [batch, 5, N]
        
        return pred_carriers, pred_supernatants


def train_coupled_model(model, carrier_train, supernatant_train, conds_train,
                        epochs=300, lr=0.015, l2_lambda=0.001, delta_t=0.1, verbose=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    best_loss = float('inf')
    best_state = None
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        
        pred_c, pred_s = model.forward_rollout(carrier_train, supernatant_train, conds_train, 
                                               delta_t=delta_t, tf_ratio=tf_ratio)
        
        # Targets are passages P2..P6 (index 1 to 5)
        target_c = carrier_train[:, 1:, :]
        target_s = supernatant_train[:, 1:, :]
        
        kl_c = torch.sum(target_c * torch.log(target_c / torch.clamp(pred_c, min=1e-9) + 1e-12), dim=-1).mean()
        kl_s = torch.sum(target_s * torch.log(target_s / torch.clamp(pred_s, min=1e-9) + 1e-12), dim=-1).mean()
        
        Theta, W, _, _, _, _, _, _, _ = model.get_params()
        l2_reg = l2_lambda * (Theta.pow(2).sum() + W.pow(2).sum())
        
        total_loss = kl_c + kl_s + l2_reg
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        
        loss_val = (kl_c + kl_s).item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = copy.deepcopy(model.state_dict())
            
        if verbose and (epoch % 50 == 0 or epoch == epochs):
            print(f"Epoch {epoch:3d}/{epochs} | KL Carrier: {kl_c.item():.4f} | KL Supernatant: {kl_s.item():.4f} | Total: {loss_val:.4f} (TF: {tf_ratio:.2f})")
            
    model.load_state_dict(best_state)
    return model, best_loss


def evaluate_model(model, carrier_seq, supernatant_seq, conds, delta_t=0.1):
    model.eval()
    with torch.no_grad():
        pred_c, pred_s = model.forward_rollout(carrier_seq, supernatant_seq, conds, 
                                               delta_t=delta_t, tf_ratio=0.0) # 100% autoregressive
        
    target_c = carrier_seq[:, 1:, :].cpu().numpy()
    target_s = supernatant_seq[:, 1:, :].cpu().numpy()
    p_c = pred_c.cpu().numpy()
    p_s = pred_s.cpu().numpy()
    
    # Flatten across batch, passage, species
    y_c_true = target_c.reshape(-1)
    y_c_pred = p_c.reshape(-1)
    y_s_true = target_s.reshape(-1)
    y_s_pred = p_s.reshape(-1)
    
    r_c, _ = pearsonr(y_c_true, y_c_pred)
    r2_c = r2_score(y_c_true, y_c_pred)
    mae_c = np.mean(np.abs(y_c_true - y_c_pred))
    
    r_s, _ = pearsonr(y_s_true, y_s_pred)
    r2_s = r2_score(y_s_true, y_s_pred)
    mae_s = np.mean(np.abs(y_s_true - y_s_pred))
    
    # Combined
    y_all_true = np.concatenate([y_c_true, y_s_true])
    y_all_pred = np.concatenate([y_c_pred, y_s_pred])
    r_all, _ = pearsonr(y_all_true, y_all_pred)
    r2_all = r2_score(y_all_true, y_all_pred)
    mae_all = np.mean(np.abs(y_all_true - y_all_pred))
    
    # Bray-Curtis dissimilarity per passage sample
    bc_c_list = []
    for b in range(target_c.shape[0]):
        for p in range(target_c.shape[1]):
            bc = np.sum(np.abs(target_c[b, p] - p_c[b, p])) / (np.sum(target_c[b, p] + p_c[b, p]) + 1e-12)
            bc_c_list.append(bc)
            
    bc_s_list = []
    for b in range(target_s.shape[0]):
        for p in range(target_s.shape[1]):
            bc = np.sum(np.abs(target_s[b, p] - p_s[b, p])) / (np.sum(target_s[b, p] + p_s[b, p]) + 1e-12)
            bc_s_list.append(bc)
            
    metrics = {
        'Carrier': {'r': float(r_c), 'R2': float(r2_c), 'MAE': float(mae_c), 'BrayCurtis': float(np.mean(bc_c_list))},
        'Supernatant': {'r': float(r_s), 'R2': float(r2_s), 'MAE': float(mae_s), 'BrayCurtis': float(np.mean(bc_s_list))},
        'Combined': {'r': float(r_all), 'R2': float(r2_all), 'MAE': float(mae_all)},
        'pred_c': p_c, 'pred_s': p_s, 'target_c': target_c, 'target_s': target_s
    }
    return metrics


def run_3fold_cross_validation(carrier_tensor, supernatant_tensor, cond_tensor, rep_labels):
    print("\n" + "="*70)
    print("RUNNING 3-FOLD CROSS-VALIDATION ACROSS BIOLOGICAL BATCHES (R1, R2, R3)")
    print("="*70)
    
    batch_splits = {
        'Fold 1 (Test: R3)': {'train': ['R1a', 'R1b', 'R1c', 'R2a', 'R2b', 'R2c'], 'test': ['R3a', 'R3b', 'R3c']},
        'Fold 2 (Test: R2)': {'train': ['R1a', 'R1b', 'R1c', 'R3a', 'R3b', 'R3c'], 'test': ['R2a', 'R2b', 'R2c']},
        'Fold 3 (Test: R1)': {'train': ['R2a', 'R2b', 'R2c', 'R3a', 'R3b', 'R3c'], 'test': ['R1a', 'R1b', 'R1c']}
    }
    
    cv_results = []
    
    for fold_name, split in batch_splits.items():
        train_mask = [rep in split['train'] for rep in rep_labels]
        test_mask = [rep in split['test'] for rep in rep_labels]
        
        c_train, c_test = carrier_tensor[train_mask], carrier_tensor[test_mask]
        s_train, s_test = supernatant_tensor[train_mask], supernatant_tensor[test_mask]
        cond_train, cond_test = cond_tensor[train_mask], cond_tensor[test_mask]
        
        print(f"\n--- {fold_name} | Train N={c_train.size(0)}, Test N={c_test.size(0)} ---")
        model = CoupledBiofilmPollardODE(N=50, K=5, num_conditions=2).to(device)
        model, _ = train_coupled_model(model, c_train, s_train, cond_train, epochs=250, lr=0.015, verbose=False)
        
        test_metrics = evaluate_model(model, c_test, s_test, cond_test)
        print(f"Test Carrier:     Pearson r = {test_metrics['Carrier']['r']:.4f} | R2 = {test_metrics['Carrier']['R2']:.4f} | BC = {test_metrics['Carrier']['BrayCurtis']:.4f}")
        print(f"Test Supernatant: Pearson r = {test_metrics['Supernatant']['r']:.4f} | R2 = {test_metrics['Supernatant']['R2']:.4f} | BC = {test_metrics['Supernatant']['BrayCurtis']:.4f}")
        print(f"Test Combined:    Pearson r = {test_metrics['Combined']['r']:.4f} | R2 = {test_metrics['Combined']['R2']:.4f}")
        
        cv_results.append({
            'Fold': fold_name,
            'Carrier_r': test_metrics['Carrier']['r'],
            'Carrier_R2': test_metrics['Carrier']['R2'],
            'Carrier_BC': test_metrics['Carrier']['BrayCurtis'],
            'Supernatant_r': test_metrics['Supernatant']['r'],
            'Supernatant_R2': test_metrics['Supernatant']['R2'],
            'Supernatant_BC': test_metrics['Supernatant']['BrayCurtis'],
            'Combined_r': test_metrics['Combined']['r'],
            'Combined_R2': test_metrics['Combined']['R2']
        })
        
    cv_df = pd.DataFrame(cv_results)
    print("\n" + "="*70)
    print("3-FOLD CROSS-VALIDATION SUMMARY (OUT-OF-SAMPLE TEST PERFORMANCE):")
    print("="*70)
    print(cv_df.to_string(index=False))
    print(f"\nMean Carrier Out-of-Sample Pearson r:     {cv_df['Carrier_r'].mean():.4f} +/- {cv_df['Carrier_r'].std():.4f}")
    print(f"Mean Carrier Out-of-Sample R2:            {cv_df['Carrier_R2'].mean():.4f} +/- {cv_df['Carrier_R2'].std():.4f}")
    print(f"Mean Supernatant Out-of-Sample Pearson r: {cv_df['Supernatant_r'].mean():.4f} +/- {cv_df['Supernatant_r'].std():.4f}")
    print(f"Mean Supernatant Out-of-Sample R2:        {cv_df['Supernatant_R2'].mean():.4f} +/- {cv_df['Supernatant_R2'].std():.4f}")
    print(f"Mean Combined Out-of-Sample Pearson r:    {cv_df['Combined_r'].mean():.4f} +/- {cv_df['Combined_r'].std():.4f}")
    print(f"Mean Combined Out-of-Sample R2:           {cv_df['Combined_R2'].mean():.4f} +/- {cv_df['Combined_R2'].std():.4f}")
    return cv_df


def plot_coupled_results(model, carrier_tensor, supernatant_tensor, cond_tensor, 
                         strains_keep, conditions, rep_labels, output_prefix='Pollard_Coupled_Biofilm'):
    print("\nGenerating comprehensive publication figures...")
    metrics = evaluate_model(model, carrier_tensor, supernatant_tensor, cond_tensor)
    
    target_c = metrics['target_c']
    target_s = metrics['target_s']
    pred_c = metrics['pred_c']
    pred_s = metrics['pred_s']
    
    Theta, W, alpha, k_shed, k_attach, R0, b0_c, b0_s, eta = model.get_params()
    Theta_np = Theta.detach().cpu().numpy()
    W_np = W.detach().cpu().numpy()
    alpha_np = alpha.detach().cpu().numpy()
    k_shed_np = k_shed.detach().cpu().numpy()
    k_attach_mucin_np = k_attach[0].detach().cpu().numpy()
    k_attach_plain_np = k_attach[1].detach().cpu().numpy()
    
    sns.set_theme(style='ticks', font_scale=1.05)
    
    # -------------------------------------------------------------
    # FIGURE 1: Simultaneous Carrier & Supernatant Predictive Fit
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    
    # Panel A: Carrier
    ax = axes[0]
    ax.scatter(target_c.reshape(-1), pred_c.reshape(-1), alpha=0.35, color='#2b5c8f', edgecolors='none', s=22)
    max_val = max(target_c.max(), pred_c.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], 'k--', lw=1.5, alpha=0.7)
    ax.set_title(f"A. Biofilm Carriers (Passages P2-P6)\nPearson r = {metrics['Carrier']['r']:.3f} | R² = {metrics['Carrier']['R2']:.3f} | BC = {metrics['Carrier']['BrayCurtis']:.3f}", 
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Observed Relative Abundance", fontsize=11)
    ax.set_ylabel("Coupled ODE Predicted Abundance", fontsize=11)
    ax.set_xlim(-0.01, max_val)
    ax.set_ylim(-0.01, max_val)
    ax.grid(True, ls=':', alpha=0.4)
    
    # Panel B: Supernatant
    ax = axes[1]
    ax.scatter(target_s.reshape(-1), pred_s.reshape(-1), alpha=0.35, color='#d95f02', edgecolors='none', s=22)
    max_val_s = max(target_s.max(), pred_s.max()) * 1.05
    ax.plot([0, max_val_s], [0, max_val_s], 'k--', lw=1.5, alpha=0.7)
    ax.set_title(f"B. Liquid Supernatants (Passages P2-P6)\nPearson r = {metrics['Supernatant']['r']:.3f} | R² = {metrics['Supernatant']['R2']:.3f} | BC = {metrics['Supernatant']['BrayCurtis']:.3f}", 
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Observed Relative Abundance", fontsize=11)
    ax.set_ylabel("Coupled ODE Predicted Abundance", fontsize=11)
    ax.set_xlim(-0.01, max_val_s)
    ax.set_ylim(-0.01, max_val_s)
    ax.grid(True, ls=':', alpha=0.4)
    
    # Panel C: Combined Parity Plot with Compartment Coloring
    ax = axes[2]
    ax.scatter(target_c.reshape(-1), pred_c.reshape(-1), alpha=0.3, color='#2b5c8f', label=f"Carriers (r={metrics['Carrier']['r']:.3f})", s=18)
    ax.scatter(target_s.reshape(-1), pred_s.reshape(-1), alpha=0.3, color='#d95f02', label=f"Supernatants (r={metrics['Supernatant']['r']:.3f})", s=18)
    overall_max = max(max_val, max_val_s)
    ax.plot([0, overall_max], [0, overall_max], 'k--', lw=1.5, alpha=0.7)
    ax.set_title(f"C. Combined Simultaneous Parity\nOverall Pearson r = {metrics['Combined']['r']:.3f} | R² = {metrics['Combined']['R2']:.3f}", 
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Observed Relative Abundance", fontsize=11)
    ax.set_ylabel("Coupled ODE Predicted Abundance", fontsize=11)
    ax.set_xlim(-0.01, overall_max)
    ax.set_ylim(-0.01, overall_max)
    ax.legend(frameon=True, fontsize=10, loc='upper left')
    ax.grid(True, ls=':', alpha=0.4)
    
    plt.tight_layout()
    fig1_path = os.path.join(PROJECT_ROOT, f'{output_prefix}_Simultaneous_Fit.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # -------------------------------------------------------------
    # FIGURE 2: Biological Adhesion & Niche Trait Discovery
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Adhesion Preference Ratio (k_attach / k_shed)
    # Strains with high ratio preferentially colonize carrier; low ratio preferentially shed into supernatant
    adhesion_ratio = k_attach_mucin_np / (k_shed_np + 1e-4)
    df_traits = pd.DataFrame({
        'Strain': [s.replace('-', ' ') for s in strains_keep],
        'Ratio': adhesion_ratio,
        'k_attach_mucin': k_attach_mucin_np,
        'k_attach_plain': k_attach_plain_np,
        'k_shed': k_shed_np
    }).sort_values(by='Ratio', ascending=True)
    
    # Top 10 carrier specialists vs top 10 supernatant specialists
    extremes = pd.concat([df_traits.head(10), df_traits.tail(10)])
    colors = ['#d95f02']*10 + ['#2b5c8f']*10
    
    ax = axes[0]
    y_pos = np.arange(len(extremes))
    ax.barh(y_pos, extremes['Ratio'], color=colors, alpha=0.85, edgecolor='black', lw=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(extremes['Strain'], fontsize=8.5)
    ax.set_xlabel("Adhesion Partitioning Ratio ($k_{\mathrm{attach}} / k_{\mathrm{shed}}$)", fontsize=11)
    ax.set_title("A. Learned Adhesion Trait Divergence\n(Orange: Liquid-Preferring Shedders | Blue: Biofilm Colonizers)", 
                 fontsize=11.5, fontweight='bold', pad=10)
    ax.axvline(1.0, color='gray', linestyle='--', lw=1)
    ax.grid(True, axis='x', ls=':', alpha=0.4)
    
    # Panel B: Mucin-Specific vs Plain-Agar Attachment Rates
    ax = axes[1]
    ax.scatter(df_traits['k_attach_plain'], df_traits['k_attach_mucin'], 
               color='#7570b3', alpha=0.8, edgecolors='k', lw=0.5, s=55)
    max_k = max(df_traits['k_attach_plain'].max(), df_traits['k_attach_mucin'].max()) * 1.1
    ax.plot([0, max_k], [0, max_k], 'k--', lw=1.2, alpha=0.7, label='Equal Affinity (y = x)')
    
    # Highlight strains with prominent mucin specificity
    mucin_preferring = df_traits[df_traits['k_attach_mucin'] > 1.25 * df_traits['k_attach_plain']]
    for _, row in mucin_preferring.head(6).iterrows():
        short_name = " ".join(row['Strain'].split()[:2])
        ax.annotate(short_name, (row['k_attach_plain'], row['k_attach_mucin']),
                    xytext=(5, 2), textcoords='offset points', fontsize=7.5, fontweight='semibold', color='#1b9e77')
        
    ax.set_title("B. Substrate Affinity: Mucin vs Plain Agar Attachment\n($k_{\mathrm{attach}}^{\mathrm{mucin}}$ vs $k_{\mathrm{attach}}^{\mathrm{plain}}$)", 
                 fontsize=11.5, fontweight='bold', pad=10)
    ax.set_xlabel("Plain Agar Attachment Rate ($k_{\mathrm{attach}}^{\mathrm{plain}}$)", fontsize=11)
    ax.set_ylabel("Mucin Agar Attachment Rate ($k_{\mathrm{attach}}^{\mathrm{mucin}}$)", fontsize=11)
    ax.set_xlim(-0.01, max_k)
    ax.set_ylim(-0.01, max_k)
    ax.legend(frameon=True, fontsize=10, loc='lower right')
    ax.grid(True, ls=':', alpha=0.4)
    
    plt.tight_layout()
    fig2_path = os.path.join(PROJECT_ROOT, f'{output_prefix}_Adhesion_Traits.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # -------------------------------------------------------------
    # FIGURE 3: Longitudinal Passage Dynamics (P1..P6)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    passages_num = [1, 2, 3, 4, 5, 6]
    
    # Pick 3 representative strains: 1 carrier-dominant, 1 supernatant-dominant, 1 balanced
    # Let's inspect carrier vs supernatant ratios
    rep_idx = 0 # first replicate (mucin-agar R1a)
    
    # Let's select 3 interesting strains
    carrier_top_idx = np.argmax(adhesion_ratio)
    supernatant_top_idx = np.argmin(adhesion_ratio)
    balanced_idx = np.argsort(np.abs(adhesion_ratio - 1.0))[0]
    
    strains_to_plot = [carrier_top_idx, supernatant_top_idx, balanced_idx]
    
    for col_idx, s_idx in enumerate(strains_to_plot):
        s_name = " ".join(strains_keep[s_idx].replace('-', ' ').split()[:3])
        
        # Row 0: Carrier compartment
        ax_c = axes[0, col_idx]
        # observed P1..P6
        obs_c = carrier_tensor[rep_idx, :, s_idx].cpu().numpy()
        pred_c_traj = np.concatenate([[obs_c[0]], pred_c[rep_idx, :, s_idx]])
        
        ax_c.plot(passages_num, obs_c, 'o-', color='#2b5c8f', lw=2, ms=6, label='Observed Carrier')
        ax_c.plot(passages_num, pred_c_traj, 's--', color='#5ab4ac', lw=2, ms=5, label='Coupled ODE Carrier')
        ax_c.set_title(f"{s_name}\n[Carrier Biofilm Compartment]", fontsize=10.5, fontweight='bold')
        ax_c.set_ylabel("Relative Abundance", fontsize=10)
        ax_c.set_xticks(passages_num)
        ax_c.set_xticklabels([f'P{i}' for i in passages_num])
        ax_c.grid(True, ls=':', alpha=0.4)
        if col_idx == 0:
            ax_c.legend(frameon=True, fontsize=9, loc='best')
            
        # Row 1: Supernatant compartment
        ax_s = axes[1, col_idx]
        obs_s = supernatant_tensor[rep_idx, :, s_idx].cpu().numpy()
        pred_s_traj = np.concatenate([[obs_s[0]], pred_s[rep_idx, :, s_idx]])
        
        ax_s.plot(passages_num, obs_s, 'o-', color='#d95f02', lw=2, ms=6, label='Observed Supernatant')
        ax_s.plot(passages_num, pred_s_traj, 's--', color='#fc8d59', lw=2, ms=5, label='Coupled ODE Supernatant')
        ax_s.set_title(f"{s_name}\n[Liquid Planktonic Compartment]", fontsize=10.5, fontweight='bold')
        ax_s.set_xlabel("Passage Cycle", fontsize=10.5)
        ax_s.set_ylabel("Relative Abundance", fontsize=10)
        ax_s.set_xticks(passages_num)
        ax_s.set_xticklabels([f'P{i}' for i in passages_num])
        ax_s.grid(True, ls=':', alpha=0.4)
        if col_idx == 0:
            ax_s.legend(frameon=True, fontsize=9, loc='best')
            
    plt.tight_layout()
    fig3_path = os.path.join(PROJECT_ROOT, f'{output_prefix}_Longitudinal_Trajectories.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")
    
    return [fig1_path, fig2_path, fig3_path]


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Loading paired carrier and supernatant data from {DATA_PATH}...")
    carrier_tensor, supernatant_tensor, cond_tensor, strains_keep, conditions, rep_labels = load_coupled_data()
    print(f"Data loaded: {carrier_tensor.size(0)} matched trajectories (9 Mucin, 9 Plain Agar).")
    print(f"Number of prevalent gut strains: {len(strains_keep)}")
    
    carrier_tensor = carrier_tensor.to(device)
    supernatant_tensor = supernatant_tensor.to(device)
    cond_tensor = cond_tensor.to(device)
    
    # 1. Run 3-Fold Cross-Validation on held-out biological batches (R1, R2, R3)
    cv_df = run_3fold_cross_validation(carrier_tensor, supernatant_tensor, cond_tensor, rep_labels)
    
    # 2. Train Full Unified Model across all replicates
    print("\n" + "="*70)
    print("TRAINING UNIFIED COUPLED BIOFILM-PLANKTONIC MODEL ON ALL REPLICATES")
    print("="*70)
    full_model = CoupledBiofilmPollardODE(N=len(strains_keep), K=5, num_conditions=len(conditions)).to(device)
    full_model, best_loss = train_coupled_model(full_model, carrier_tensor, supernatant_tensor, cond_tensor,
                                                epochs=300, lr=0.015, verbose=True)
    
    # Evaluate full model
    metrics = evaluate_model(full_model, carrier_tensor, supernatant_tensor, cond_tensor)
    print("\n" + "="*70)
    print("FULL MODEL SIMULTANEOUS FIT EVALUATION (PASSAGES P2-P6):")
    print("="*70)
    print(f"Biofilm Carriers:   Pearson r = {metrics['Carrier']['r']:.4f} | R2 = {metrics['Carrier']['R2']:.4f} | MAE = {metrics['Carrier']['MAE']:.4f} | BC = {metrics['Carrier']['BrayCurtis']:.4f}")
    print(f"Liquid Supernatant: Pearson r = {metrics['Supernatant']['r']:.4f} | R2 = {metrics['Supernatant']['R2']:.4f} | MAE = {metrics['Supernatant']['MAE']:.4f} | BC = {metrics['Supernatant']['BrayCurtis']:.4f}")
    print(f"Overall Combined:   Pearson r = {metrics['Combined']['r']:.4f} | R2 = {metrics['Combined']['R2']:.4f} | MAE = {metrics['Combined']['MAE']:.4f}")
    
    # Save checkpoint
    torch.save({
        'model_state_dict': full_model.state_dict(),
        'strains_keep': strains_keep,
        'conditions': conditions,
        'metrics': metrics,
        'cv_df': cv_df.to_dict()
    }, SAVE_MODEL_PATH)
    print(f"\nSaved trained model checkpoint to {SAVE_MODEL_PATH}")
    
    # Generate Publication Figures
    fig_paths = plot_coupled_results(full_model, carrier_tensor, supernatant_tensor, cond_tensor,
                                     strains_keep, conditions, rep_labels)
    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()

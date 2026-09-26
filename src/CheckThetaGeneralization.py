import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from SerialDilutionModels import LatentAssemblyModel


def load_generalization_data(csv_path):
    """
    Parses the Jin et al. SupplementaryDataS2.csv file specifically for the 
    Mucin Generalization experiment.
    """
    print(f"Loading experimental data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Clean replicate names to group technical replicates
    df['bio_rep'] = df['replicate'].astype(str).str[:2]
    
    # 2. Average over technical replicates
    df_avg = df.groupby(['sampleType', 'passage', 'bio_rep', 'Strain_Name'])['Read_Fraction'].mean().reset_index()
    
    # 3. Pivot so each row is a sample and each column is a strain
    pivot = df_avg.pivot_table(index=['sampleType', 'bio_rep', 'passage'], 
                               columns='Strain_Name', values='Read_Fraction', fill_value=0.0)
    
    # 4. Normalize relative abundances
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    
    strains = pivot.columns.tolist()
    N = len(strains)
    
    if 'inoculum' in pivot.index.get_level_values('sampleType'):
        inoculum_data = pivot.xs('inoculum', level='sampleType')
    else:
        inoculum_data = None
        
    # Maps conditions to indices
    cond_map = {'no carriers': 0, 'plain-agar carriers': 1, 'mucin-agar carriers': 2}
    
    p1_in, p1_out, c1 = [], [], []             # Phase 1: Train Theta on Liquid & Agar
    p2_tr_in, p2_tr_out, c2_tr = [], [], []    # Phase 2 Train: Fit a_mucin on early passages
    p2_te_in, p2_te_out, c2_te = [], [], []    # Phase 2 Test: Predict final mucin passages
    
    for cond, c_idx in cond_map.items():
        if cond not in pivot.index.get_level_values('sampleType').unique():
            print(f"Warning: {cond} not found in dataset.")
            continue
            
        cond_data = pivot.xs(cond, level='sampleType')
        for rep in cond_data.index.get_level_values('bio_rep').unique():
            rep_df = cond_data.xs(rep, level='bio_rep')
            passages = sorted(rep_df.index.tolist()) 
            
            # Extract P0 -> P1
            if inoculum_data is not None and 'P1' in passages:
                p0 = inoculum_data.xs(rep, level='bio_rep').loc['P0'].values if rep in inoculum_data.index.get_level_values('bio_rep') else inoculum_data.mean(axis=0).values
                p_next_vals = rep_df.loc['P1'].values
                if cond in ['no carriers', 'plain-agar carriers']:
                    p1_in.append(p0); p1_out.append(p_next_vals); c1.append(c_idx)
                else: # Mucin
                    p2_tr_in.append(p0); p2_tr_out.append(p_next_vals); c2_tr.append(c_idx)
                    
            # Extract Inter-passage (P1->P2, P2->P3, etc)
            for i in range(len(passages) - 1):
                p_cur, p_next = passages[i], passages[i+1]
                cur_vals, next_vals = rep_df.loc[p_cur].values, rep_df.loc[p_next].values
                
                if cond in ['no carriers', 'plain-agar carriers']:
                    p1_in.append(cur_vals); p1_out.append(next_vals); c1.append(c_idx)
                else: # Mucin - Split by Early (Train) vs Late (Test)
                    if p_next in ['P1', 'P2', 'P3']:
                        p2_tr_in.append(cur_vals); p2_tr_out.append(next_vals); c2_tr.append(c_idx)
                    elif p_next in ['P4', 'P5', 'P6']:
                        p2_te_in.append(cur_vals); p2_te_out.append(next_vals); c2_te.append(c_idx)

    # Convert to Tensors
    t_p1_in, t_p1_out, t_c1 = map(lambda x: torch.tensor(np.array(x), dtype=torch.float32 if type(x[0]) is np.ndarray else torch.long), [p1_in, p1_out, c1])
    t_p2_tr_in, t_p2_tr_out, t_c2_tr = map(lambda x: torch.tensor(np.array(x), dtype=torch.float32 if type(x[0]) is np.ndarray else torch.long), [p2_tr_in, p2_tr_out, c2_tr])
    t_p2_te_in, t_p2_te_out, t_c2_te = map(lambda x: torch.tensor(np.array(x), dtype=torch.float32 if type(x[0]) is np.ndarray else torch.long), [p2_te_in, p2_te_out, c2_te])
    
    print(f"Data Split Complete:")
    print(f" - Phase 1 (Liquid & Agar): {len(t_p1_in)} transitions")
    print(f" - Phase 2 Train (Mucin Early): {len(t_p2_tr_in)} transitions")
    print(f" - Phase 2 Test  (Mucin Late): {len(t_p2_te_in)} transitions")
    
    return t_p1_in, t_p1_out, t_c1, t_p2_tr_in, t_p2_tr_out, t_c2_tr, t_p2_te_in, t_p2_te_out, t_c2_te, N, strains


def train_phase1(model, x_in, x_out, c_in, epochs=1500, lr=0.02, l1_lambda=0.0001):
    """Phase 1: Optimizes Theta, W, and 'a' (for Liquid and Agar conditions)."""
    print("\n" + "="*50)
    print("PHASE 1: Training Theta, W, and 'a' on Liquid & Plain-Agar")
    print("="*50)
    
    model.Theta.requires_grad = True
    model.Phi.requires_grad = True
    model.a.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    loss_hist = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        log_pred, _ = model(x_in, c_in)
        
        base_loss = criterion(log_pred, x_out)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = base_loss + l1_lambda * l1_norm
        
        loss.backward()
        optimizer.step()
        loss_hist.append(base_loss.item())
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Phase 1 - Epoch {epoch:4d}/{epochs} | KL Div: {base_loss.item():.6f}")
            
    return loss_hist


def train_phase2(model, x_in, x_out, c_in, epochs=1000, lr=0.02):
    """Phase 2: Freezes Theta and W. Optimizes ONLY 'a' on Early Mucin passages."""
    print("\n" + "="*50)
    print("PHASE 2: Freezing Traits. Adapting 'a' on Early Mucin Passages (P1-P3)")
    print("="*50)
    
    # Freeze the biological networks
    model.Theta.requires_grad = False
    model.Phi.requires_grad = False
    # Leave the environment vector unfrozen
    model.a.requires_grad = True
    
    # Only pass the unfrozen parameter to the optimizer
    optimizer = optim.Adam([model.a], lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    loss_hist = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        log_pred, _ = model(x_in, c_in)
        
        # No L1 penalty needed here since we are only adjusting the base environment
        loss = criterion(log_pred, x_out)
        
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Phase 2 - Epoch {epoch:4d}/{epochs} | KL Div: {loss.item():.6f}")
            
    return loss_hist


def evaluate_generalization(model, p1_in, p1_out, p1_c, p2_te_in, p2_te_out, p2_te_c, strains, p1_hist, p2_hist):
    """
    Evaluates how well the model generalized to the final passages of the Mucin environment.
    Generates a 3-panel figure comparing the Baseline fits vs the Generalization predictions.
    """
    model.eval()
    with torch.no_grad():
        _, pred_p1 = model(p1_in, p1_c)
        _, pred_p2_te = model(p2_te_in, p2_te_c)
        
    mae_p1 = torch.mean(torch.abs(pred_p1 - p1_out)).item()
    mae_test = torch.mean(torch.abs(pred_p2_te - p2_te_out)).item()
    
    print(f"\n==================================================")
    print(f"--- Generalization Evaluation ---")
    print(f"==================================================")
    print(f"Baseline MAE (Liquid/Agar)        : {mae_p1:.5f} ({mae_p1*100:.2f}%)")
    print(f"Mucin Generalization MAE (P4-P6)  : {mae_test:.5f} ({mae_test*100:.2f}%)")
    
    # --- Generate 3-Panel Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(28, 9))
    
    # Panel 1: Loss History
    axes[0].plot(p1_hist, label='Phase 1 (Fit Traits)', color='steelblue', lw=3)
    axes[0].plot(p2_hist, label='Phase 2 (Adapt Env)', color='darkorange', lw=3)
    axes[0].set_xlabel('Epoch', fontsize=24)
    axes[0].set_ylabel('KL Divergence Loss', fontsize=24)
    axes[0].set_title('Two-Phase Training History', fontsize=26)
    axes[0].tick_params(axis='both', which='major', labelsize=22)
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=20)
    axes[0].grid(True, which="both", ls="--", alpha=0.3)
    
    # Panel 2: Phase 1 Fit (Liquid & Agar)
    p1_true = p1_out.numpy().flatten()
    p1_pred = pred_p1.numpy().flatten()
    
    p1_true_log = np.log10(np.clip(p1_true, 1e-6, None))
    p1_pred_log = np.log10(np.clip(p1_pred, 1e-6, None))
    r_p1, _ = pearsonr(p1_true_log, p1_pred_log)
    r2_p1 = r_p1**2
    p1_label = f"Predictions ($R^2$={r2_p1:.2f})"
    
    axes[1].scatter(p1_true, p1_pred, alpha=0.4, edgecolors='k', c='steelblue', label=p1_label)
    
    plottable_mask = p1_true > 0
    min_x = p1_true[plottable_mask].min() if np.any(plottable_mask) else 1e-4
    max_x = p1_true.max()
    axes[1].plot([min_x, max_x], [min_x, max_x], 'r--', lw=3, label='y=x')
    
    valid_y_mask = p1_true > 0
    min_y = p1_pred[valid_y_mask].min() if np.any(valid_y_mask) else 1e-4
    max_y = p1_pred.max()
    
    plot_min_x = min_x * 0.5
    plot_max_x = min(max_x * 1.5, 1.5)
    plot_min_y = min_y * 0.5
    plot_max_y = min(max_y * 1.5, 1.5)
    
    # Decoupled Limits (Train Phase)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Actual Relative Abundance', fontsize=24)
    axes[1].set_ylabel('Predicted Relative Abundance', fontsize=24)
    axes[1].set_title('Phase 1 Fit: Liquid & Plain-Agar', fontsize=26)
    axes[1].tick_params(axis='both', which='major', labelsize=22)
    axes[1].legend(loc='upper left', fontsize=20)
    axes[1].grid(True, which="both", ls="--", alpha=0.2)
    
    # Panel 3: Mucin Generalization Test (P4, P5, P6)
    te_true = p2_te_out.numpy().flatten()
    te_pred = pred_p2_te.numpy().flatten()
    
    te_true_log = np.log10(np.clip(te_true, 1e-6, None))
    te_pred_log = np.log10(np.clip(te_pred, 1e-6, None))
    r_te, _ = pearsonr(te_true_log, te_pred_log)
    r2_te = r_te**2
    te_label = f"Generalization ($R^2$={r2_te:.2f})"
    
    axes[2].scatter(te_true, te_pred, alpha=0.6, edgecolors='k', c='darkorange', label=te_label)
    
    plottable_mask_te = te_true > 0
    min_x_te = te_true[plottable_mask_te].min() if np.any(plottable_mask_te) else 1e-4
    max_x_te = te_true.max()
    axes[2].plot([min_x_te, max_x_te], [min_x_te, max_x_te], 'r--', lw=3, label='y=x')
    
    valid_y_mask_te = te_true > 0
    min_y_te = te_pred[valid_y_mask_te].min() if np.any(valid_y_mask_te) else 1e-4
    max_y_te = te_pred.max()
    
    plot_min_x_te = min_x_te * 0.5
    plot_max_x_te = min(max_x_te * 1.5, 1.5)
    plot_min_y_te = min_y_te * 0.5
    plot_max_y_te = min(max_y_te * 1.5, 1.5)
    
    # Decoupled Limits (Test Phase) - Updated!
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('Actual Relative Abundance', fontsize=24)
    axes[2].set_ylabel('Predicted Relative Abundance', fontsize=24)
    axes[2].set_title('Generalization Test: Final Mucin', fontsize=26)
    axes[2].tick_params(axis='both', which='major', labelsize=22)
    axes[2].legend(loc='upper left', fontsize=20)
    axes[2].grid(True, which="both", ls="--", alpha=0.2)
    
    plt.tight_layout()
    
    plot_filename = 'mucin_generalization_test.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved generalization evaluation plot to '{plot_filename}'")


if __name__ == "__main__":
    csv_filename = "../Data/SupplementaryDataS2.csv"
    LATENT_DIMS = 3  
    
    if os.path.exists(csv_filename):
        t_p1_in, t_p1_out, t_c1, t_p2_tr_in, t_p2_tr_out, t_c2_tr, t_p2_te_in, t_p2_te_out, t_c2_te, NUM_SPECIES, strains = load_generalization_data(csv_filename)
    else:
        print(f"File '{csv_filename}' not found. Falling back to synthetic data.")
        t_p1_in, t_p1_out, t_c1, t_p2_tr_in, t_p2_tr_out, t_c2_tr, t_p2_te_in, t_p2_te_out, t_c2_te, NUM_SPECIES, strains = generate_mock_data(K=LATENT_DIMS)
        
    # Initialize Model (3 environments: 0=liquid, 1=agar, 2=mucin)
    model = LatentAssemblyModel(num_species=NUM_SPECIES, num_latent_dims=LATENT_DIMS, num_conditions=3)
    
    # Execute Two-Phase Training
    p1_loss_hist = train_phase1(model, t_p1_in, t_p1_out, t_c1, epochs=1500, lr=0.02, l1_lambda=0)
    p2_loss_hist = train_phase2(model, t_p2_tr_in, t_p2_tr_out, t_c2_tr, epochs=1500, lr=0.02)
    
    # Evaluate Generalization on Mucin Final Passages
    evaluate_generalization(model, t_p1_in, t_p1_out, t_c1, t_p2_te_in, t_p2_te_out, t_c2_te, strains, p1_loss_hist, p2_loss_hist)
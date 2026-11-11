import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'Figures')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results')

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(pt_path):
    """
    Loads preprocessed Gore dataset and sets up batched GPU tensors.
    """
    data = torch.load(pt_path, weights_only=False)
    rel_dict = data['relative_abundance']
    masks_dict = data['presence_masks']
    metadata_df = data['metadata']
    species_list = data['species']
    
    cond_map = {'LN': 0, 'MN': 1, 'HN': 2}
    
    # Organize into two length-matched batches for fully vectorized GPU training
    batch_11_seqs, batch_11_conds, batch_11_keys = [], [], []
    batch_4_seqs, batch_4_conds, batch_4_keys = [], [], []
    
    for (cond, s_val, c_name), tensor in rel_dict.items():
        c_idx = cond_map[cond]
        key = (cond, s_val, c_name)
        if tensor.size(0) == 11:
            batch_11_seqs.append(tensor)
            batch_11_conds.append(c_idx)
            batch_11_keys.append(key)
        else:
            batch_4_seqs.append(tensor)
            batch_4_conds.append(c_idx)
            batch_4_keys.append(key)
            
    b11_seqs = torch.stack(batch_11_seqs).to(device)
    b11_conds = torch.tensor(batch_11_conds, dtype=torch.long).to(device)
    
    b4_seqs = torch.stack(batch_4_seqs).to(device)
    b4_conds = torch.tensor(batch_4_conds, dtype=torch.long).to(device)
    
    return {
        'b11_seqs': b11_seqs,
        'b11_conds': b11_conds,
        'b11_keys': batch_11_keys,
        'b4_seqs': b4_seqs,
        'b4_conds': b4_conds,
        'b4_keys': batch_4_keys,
        'masks': masks_dict,
        'metadata': metadata_df,
        'species': species_list,
        'cond_map': cond_map,
        'rel_dict': rel_dict
    }


def train_global_model(dataset, K=5, epochs=350, lr=0.008, l2_lambda=0.01, ode_step_size=0.1):
    """
    Trains the global Neural ODE with shared Theta, W, alpha and condition-specific R(0).
    Uses Soft Scheduled Sampling (decaying teacher forcing ratio 1.0 -> 0.0).
    """
    set_seed(42)
    num_species = len(dataset['species'])
    num_conditions = len(dataset['cond_map'])
    
    model = LatentAssemblyNeuralODE(
        num_species=num_species,
        num_latent_dims=K,
        num_conditions=num_conditions,
        default_dilution_factor=30.0
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    b11_s = dataset['b11_seqs']
    b11_c = dataset['b11_conds']
    b4_s = dataset['b4_seqs']
    b4_c = dataset['b4_conds']
    
    print(f"\nTraining Global Neural ODE on {device}...")
    print(f"  Configuration: K={K}, Epochs={epochs}, LR={lr}, L2={l2_lambda}, Dilution=30x")
    print(f"  Batches: 11-step batch ({b11_s.size(0)} seqs) | 4-step batch ({b4_s.size(0)} seqs)\n")
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Scheduled sampling decay
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.75)))
        total_loss = 0.0
        
        # --- BATCH 1: Length 11 (Days 0 to 10) ---
        x_curr = b11_s[:, 0, :]
        batch11_loss = 0.0
        for t in range(1, b11_s.size(1)):
            log_pred, x_pred, _, _ = model(x_curr, b11_c, ode_method='euler', ode_step_size=ode_step_size)
            target = b11_s[:, t, :]
            batch11_loss += criterion(log_pred, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred
            
        # --- BATCH 2: Length 4 (Days 7 to 10) ---
        x_curr_4 = b4_s[:, 0, :]
        batch4_loss = 0.0
        for t in range(1, b4_s.size(1)):
            log_pred_4, x_pred_4, _, _ = model(x_curr_4, b4_c, ode_method='euler', ode_step_size=ode_step_size)
            target_4 = b4_s[:, t, :]
            batch4_loss += criterion(log_pred_4, target_4)
            x_curr_4 = tf_ratio * target_4 + (1.0 - tf_ratio) * x_pred_4
            
        # L2 Regularization on biological interaction matrices
        l2_reg = sum(p.pow(2.0).sum() for name, p in model.named_parameters() if 'Theta' in name or 'W' in name)
        
        epoch_loss = (batch11_loss + batch4_loss) / (b11_s.size(1) - 1 + b4_s.size(1) - 1) + l2_lambda * l2_reg
        epoch_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        loss_val = epoch_loss.item()
        loss_history.append(loss_val)
        
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {loss_val:.6f} | TF Ratio: {tf_ratio:.2f} | Time: {time.time()-start_time:.1f}s")
            
    total_time = time.time() - start_time
    print(f"\n[DONE] Training complete in {total_time:.1f} seconds.")
    return model, loss_history


def evaluate_autoregressive_rollouts(model, dataset, ode_step_size=0.1):
    """
    Performs pure multi-step autoregressive rollouts for all 99 trajectories:
    Feeds only the initial observed timepoint, and unrolls predictions across all subsequent days.
    Computes log-scale Pearson R2 and JSD for each community.
    """
    model.eval()
    results = []
    
    rel_dict = dataset['rel_dict']
    masks_dict = dataset['masks']
    cond_map = dataset['cond_map']
    
    with torch.no_grad():
        for (cond, s_val, c_name), true_traj in rel_dict.items():
            c_idx = torch.tensor([cond_map[cond]], dtype=torch.long, device=device)
            mask = masks_dict[(s_val, c_name)].numpy()
            
            # Initial state
            curr_state = true_traj[0:1, :].to(device)
            pred_traj = [curr_state.cpu().squeeze(0).numpy()]
            
            # Full forward rollout
            for t in range(1, true_traj.size(0)):
                _, next_state, _, _ = model(curr_state, c_idx, ode_method='euler', ode_step_size=ode_step_size)
                pred_traj.append(next_state.squeeze(0).cpu().numpy())
                curr_state = next_state
                
            pred_matrix = np.array(pred_traj) # (timepoints, 48)
            true_matrix = true_traj.numpy()   # (timepoints, 48)
            
            # Filter to member species across unrolled steps (t >= 1)
            true_unrolled = true_matrix[1:, mask].flatten()
            pred_unrolled = pred_matrix[1:, mask].flatten()
            
            # Log scale Pearson R2
            true_log = np.log10(np.clip(true_unrolled, 1e-6, None))
            pred_log = np.log10(np.clip(pred_unrolled, 1e-6, None))
            
            r_val, _ = pearsonr(true_log, pred_log)
            r2_val = float(r_val**2) if not np.isnan(r_val) else 0.0
            
            # Final Day 10 R2
            final_true_log = np.log10(np.clip(true_matrix[-1, mask], 1e-6, None))
            final_pred_log = np.log10(np.clip(pred_matrix[-1, mask], 1e-6, None))
            r_final, _ = pearsonr(final_true_log, final_pred_log)
            r2_final = float(r_final**2) if not np.isnan(r_final) else 0.0
            
            # Calculate predicted CV on Days 7..10
            late_pred = pred_matrix[-4:, mask]
            mean_p = late_pred.mean(axis=0)
            std_p = late_pred.std(axis=0)
            surv_p = mean_p >= 1e-3
            pred_cv = float((std_p[surv_p] / mean_p[surv_p]).mean()) if np.any(surv_p) else 0.0
            
            results.append({
                'condition': cond,
                'S': s_val,
                'community': c_name,
                'key': (cond, s_val, c_name),
                'r2_trajectory': r2_val,
                'r2_endpoint': r2_final,
                'pred_cv': pred_cv,
                'true_matrix': true_matrix,
                'pred_matrix': pred_matrix,
                'mask': mask
            })
            
    df_eval = pd.DataFrame(results)
    return df_eval


def generate_evaluation_figures(model, df_eval, dataset, loss_history):
    """
    Generates a 4-panel publication-grade evaluation figure:
      Panel A: True vs Predicted Relative Abundance Parity Plot (Log Scale).
      Panel B: Multi-step Predictability (R2) by Condition & Community Size S.
      Panel C: Time Course Rollouts for Stable (LN) vs Fluctuating (HN) Communities.
      Panel D: PCA of Learned Environmental Resources R(0) along the Nutrient Axis.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    cond_colors = {'LN': '#2ecc71', 'MN': '#f39c12', 'HN': '#e74c3c'}
    
    # -------------------------------------------------------------------------
    # PANEL A: Parity Scatter Plot across All 99 Communities
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    all_true, all_pred, all_c = [], [], []
    for _, row in df_eval.iterrows():
        mask = row['mask']
        t_vals = row['true_matrix'][1:, mask].flatten()
        p_vals = row['pred_matrix'][1:, mask].flatten()
        all_true.extend(t_vals)
        all_pred.extend(p_vals)
        all_c.extend([cond_colors[row['condition']]] * len(t_vals))
        
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    
    t_log = np.log10(np.clip(all_true, 1e-6, None))
    p_log = np.log10(np.clip(all_pred, 1e-6, None))
    overall_r, _ = pearsonr(t_log, p_log)
    overall_r2 = overall_r**2
    
    for c_name, c_hex in cond_colors.items():
        idx_c = [i for i, row in df_eval.iterrows() if row['condition'] == c_name]
        t_sub = np.concatenate([df_eval.loc[i, 'true_matrix'][1:, df_eval.loc[i, 'mask']].flatten() for i in idx_c])
        p_sub = np.concatenate([df_eval.loc[i, 'pred_matrix'][1:, df_eval.loc[i, 'mask']].flatten() for i in idx_c])
        ax_a.scatter(t_sub, p_sub, alpha=0.35, s=20, c=c_hex, label=f'{c_name} Regime')
        
    ax_a.plot([1e-6, 1.0], [1e-6, 1.0], 'k--', lw=2.5, label='1:1 Parity')
    ax_a.set_xscale('log')
    ax_a.set_yscale('log')
    ax_a.set_xlim([5e-7, 1.5])
    ax_a.set_ylim([5e-7, 1.5])
    ax_a.set_xlabel('Empirical Relative Abundance', fontsize=14, fontweight='bold')
    ax_a.set_ylabel('Model Autoregressive Prediction', fontsize=14, fontweight='bold')
    ax_a.set_title(f'A) Global Multi-Passage Fit ($R^2 = {overall_r2:.2f}$)', fontsize=16, fontweight='bold')
    ax_a.legend(fontsize=12, loc='upper left')
    ax_a.grid(True, linestyle=':', alpha=0.6)
    
    # -------------------------------------------------------------------------
    # PANEL B: Predictability (R2) Grouped by S and Nutrient Regime
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    s_groups = [3, 6, 12, 24, 48]
    x = np.arange(len(s_groups))
    width = 0.25
    
    for i, cond in enumerate(['LN', 'MN', 'HN']):
        mean_r2 = []
        std_r2 = []
        for s in s_groups:
            sub = df_eval[(df_eval['condition'] == cond) & (df_eval['S'] == s)]
            mean_r2.append(sub['r2_trajectory'].mean())
            std_r2.append(sub['r2_trajectory'].std() if len(sub) > 1 else 0.0)
            
        ax_b.bar(x + (i - 1) * width, mean_r2, width, yerr=std_r2, capsize=4,
                 label=f'{cond}', color=cond_colors[cond], edgecolor='black', linewidth=1.2, zorder=3)
                 
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f'S={s}' for s in s_groups], fontsize=13, fontweight='bold')
    ax_b.set_ylim([0, 1.05])
    ax_b.set_ylabel('Multi-Step Trajectory $R^2$ (Log Scale)', fontsize=14, fontweight='bold')
    ax_b.set_title('B) Predictability Across Community Pool Sizes', fontsize=16, fontweight='bold')
    ax_b.legend(fontsize=12, loc='upper right')
    ax_b.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)
    
    # -------------------------------------------------------------------------
    # PANEL C: Dynamics Contrast (LN Damping vs. HN Oscillations)
    # -------------------------------------------------------------------------
    ax_c = axes[1, 0]
    
    # Pick S=12 Community 1 for direct LN vs HN contrast
    row_ln = df_eval[(df_eval['condition'] == 'LN') & (df_eval['S'] == 12) & (df_eval['community'] == 'Community1')].iloc[0]
    row_hn = df_eval[(df_eval['condition'] == 'HN') & (df_eval['S'] == 12) & (df_eval['community'] == 'Community1')].iloc[0]
    
    # Find most abundant species in HN
    top_sp = np.argsort(row_hn['true_matrix'][-1, :])[-3:] # top 3 species
    sp_names = [dataset['species'][idx] for idx in top_sp]
    
    days_11 = np.arange(11)
    
    # Plot HN (solid=true, dashed=pred)
    ax_c.plot(days_11, row_ln['true_matrix'][:, top_sp[-1]], 'g-', lw=2.5, label=f"{sp_names[-1]} (LN True)")
    ax_c.plot(days_11, row_ln['pred_matrix'][:, top_sp[-1]], 'g--', lw=2.0, label=f"{sp_names[-1]} (LN Model)")
    
    ax_c.plot(days_11, row_hn['true_matrix'][:, top_sp[-1]], 'r-', lw=2.5, label=f"{sp_names[-1]} (HN True)")
    ax_c.plot(days_11, row_hn['pred_matrix'][:, top_sp[-1]], 'r--', lw=2.0, label=f"{sp_names[-1]} (HN Model)")
    
    ax_c.set_xlabel('Passage Day (0 to 10)', fontsize=14, fontweight='bold')
    ax_c.set_ylabel('Relative Abundance', fontsize=14, fontweight='bold')
    ax_c.set_title('C) Trajectory Rollout (S=12 Comm 1: LN Damping vs HN Bloom)', fontsize=16, fontweight='bold')
    ax_c.legend(fontsize=11, loc='best')
    ax_c.grid(True, linestyle=':', alpha=0.6)
    
    # -------------------------------------------------------------------------
    # PANEL D: Environmental Resource Space R(0) along Glucose:Urea Axis
    # -------------------------------------------------------------------------
    ax_d = axes[1, 1]
    
    # Extract R(0) = softplus(a) from model
    with torch.no_grad():
        R_0 = torch.nn.functional.softplus(model.a).cpu().numpy() # shape (3, K)
        
    pca = PCA(n_components=2)
    R_pca = pca.fit_transform(R_0)
    var_exp = pca.explained_variance_ratio_ * 100
    
    cond_labels = ['LN (0g/L Gluc, 0g/L Urea)', 'MN (5g/L Gluc, 4g/L Urea)', 'HN (20g/L Gluc, 16g/L Urea)']
    
    # Plot trajectory line connecting LN -> MN -> HN
    ax_d.plot(R_pca[:, 0], R_pca[:, 1], 'k--', lw=2.0, zorder=2)
    
    for i, c_name in enumerate(['LN', 'MN', 'HN']):
        ax_d.scatter(R_pca[i, 0], R_pca[i, 1], s=400, c=cond_colors[c_name],
                     edgecolors='black', linewidth=2.0, zorder=3, label=cond_labels[i])
        ax_d.text(R_pca[i, 0], R_pca[i, 1] + 0.08, c_name, fontsize=14, fontweight='bold', ha='center')
        
    ax_d.set_xlabel(f'Resource PC1 ({var_exp[0]:.1f}% Variance)', fontsize=14, fontweight='bold')
    ax_d.set_ylabel(f'Resource PC2 ({var_exp[1]:.1f}% Variance)', fontsize=14, fontweight='bold')
    ax_d.set_title('D) Learned Latent Resource Coordinates $R(0)$', fontsize=16, fontweight='bold')
    ax_d.legend(fontsize=11, loc='best')
    ax_d.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'Gore_Global_Fit_Evaluation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[FIGURE] Saved comprehensive evaluation figure to:\n  {save_path}")


def main():
    pt_path = os.path.join(DATA_DIR, 'Gore_processed_data.pt')
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Missing {pt_path}. Please run Gore_Data_Processing.py first.")
        
    dataset = load_dataset(pt_path)
    
    # Train Global Neural ODE Model
    # K=5 latent resources, 350 epochs, lr=0.008, l2=0.01
    model, loss_history = train_global_model(
        dataset, K=5, epochs=350, lr=0.008, l2_lambda=0.01, ode_step_size=0.1
    )
    
    # Save Model Weights
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_save_path = os.path.join(DATA_DIR, 'Gore_Global_ODE_Model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"\n[MODEL] Saved trained model weights to:\n  {model_save_path}")
    
    # Autoregressive Rollout Evaluation
    print("\nRunning multi-passage autoregressive evaluations across all 99 trajectories...")
    df_eval = evaluate_autoregressive_rollouts(model, dataset)
    
    # Summary Table
    print("\n" + "="*70)
    print("GLOBAL MODEL EVALUATION SUMMARY (R2 by Regime & Pool Size)")
    print("="*70)
    summary = df_eval.groupby(['condition', 'S'])['r2_trajectory'].agg(['mean', 'std', 'count']).reset_index()
    for _, row in summary.iterrows():
        print(f"  Regime: {row['condition']:<3} | S={int(row['S']):2d} | Trajectory R2: {row['mean']:.3f} ± {row['std']:.3f} (n={int(row['count'])})")
        
    overall_mean = df_eval['r2_trajectory'].mean()
    print("-" * 70)
    print(f"  Overall Average Autoregressive Trajectory R2: {overall_mean:.3f}")
    print("="*70)
    
    # Save Evaluation CSV
    eval_csv_path = os.path.join(RESULTS_DIR, 'Gore_Global_Evaluation_Results.csv')
    df_eval[['condition', 'S', 'community', 'r2_trajectory', 'r2_endpoint', 'pred_cv']].to_csv(eval_csv_path, index=False)
    print(f"\n[RESULTS] Detailed evaluation metrics saved to:\n  {eval_csv_path}")
    
    # Generate Figures
    generate_evaluation_figures(model, df_eval, dataset, loss_history)


if __name__ == '__main__':
    main()

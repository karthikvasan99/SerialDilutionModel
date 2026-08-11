import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import pearsonr

class LatentAssemblyModel(nn.Module):
    """
    Discrete-Time Latent Microbiome Assembly Model.
    Predicts x_{T+1} given x_T using latent environmental variables.
    """
    def __init__(self, num_species, num_latent_dims, num_conditions=1):
        super(LatentAssemblyModel, self).__init__()
        self.N = num_species
        self.K = num_latent_dims
        
        # --- Model Parameters ---
        # Theta (N x K): The latent metabolic trait matrix for each species
        self.Theta = nn.Parameter(torch.randn(self.N, self.K) * 0.1)
        
        # W (K x K): The metabolic cross-feeding/pleiotropy matrix
        self.W = nn.Parameter(torch.randn(self.K, self.K) * 0.1)
        
        # a (num_conditions x K): The baseline environmental states 
        self.a = nn.Parameter(torch.randn(num_conditions, self.K) * 0.1)

    def forward(self, x_T, condition_idx):
        # 1. Community Aggregate Trait Expression
        community_traits = torch.matmul(x_T, self.Theta)
        
        # 2. Metabolic Modification (f)
        f = torch.matmul(community_traits, self.W.t())
        
        # 3. Final Environmental State (z_final)
        a_batch = self.a[condition_idx]
        z_final = a_batch - f
        
        # 4. Per-species fitness
        base_logits = -torch.matmul(z_final, self.Theta.t())
        
        # 5. Scores: Removed the clamp. If x_T is 0, score is exactly 0.
        x_T = torch.clamp(x_T, min=0)
        scores = x_T * torch.exp(base_logits)
        
        # 6. Normalize to get relative abundances
        x_T_plus_1 = scores / (scores.sum(dim=-1, keepdim=True))
        
        # 7. Safe log for KLDivLoss
        # Clamp ONLY for the log calculation, leaving the returned x_T_plus_1 with exact zeros
        safe_x_T_plus_1 = torch.clamp(x_T_plus_1, min=1e-9)
        log_x_T_plus_1 = torch.log(safe_x_T_plus_1)
        
        return log_x_T_plus_1, x_T_plus_1

def load_all_trajectories(csv_path, max_grouping_length=None):
    """
    Parses the Jin et al. SupplementaryDataS2.csv file and extracts all possible
    causal sub-sequences across all conditions for Scheduled Sampling training.
    No train/test splits are performed. All data is used for training.
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
    
    train_trajectories = []
    
    for cond, c_idx in cond_map.items():
        if cond not in pivot.index.get_level_values('sampleType').unique():
            print(f"Warning: {cond} not found in dataset.")
            continue
            
        cond_data = pivot.xs(cond, level='sampleType')
        for rep in cond_data.index.get_level_values('bio_rep').unique():
            rep_df = cond_data.xs(rep, level='bio_rep')
            passages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
            
            # Fetch P0 (Inoculum)
            if inoculum_data is not None and rep in inoculum_data.index.get_level_values('bio_rep'):
                p0 = inoculum_data.xs(rep, level='bio_rep').loc['P0'].values
            elif inoculum_data is not None:
                p0 = inoculum_data.mean(axis=0).values
            else:
                continue
                
            valid_timepoints = [p0]
            
            for p in passages:
                if p in rep_df.index:
                    valid_timepoints.append(rep_df.loc[p].values)
                else:
                    break 
            
            if len(valid_timepoints) < 2:
                continue
                
            # Extract ALL causal sub-sequences for scheduled sampling training
            n_train = len(valid_timepoints)
            for i in range(n_train - 1):
                for j in range(i + 1, n_train):
                    sub_seq = valid_timepoints[i:j+1]
                    if max_grouping_length is None or len(sub_seq) <= max_grouping_length:
                        train_trajectories.append((torch.tensor(np.array(sub_seq), dtype=torch.float32), c_idx))

    print(f"Extraction Complete: {len(train_trajectories)} Total Training Trajectories generated.")
    return train_trajectories, N, strains, cond_map

def train_model(model, train_trajectories, epochs=1500, lr=0.02, l1_lambda=0.0, l2_lambda=0.4, verbose=True):
    """
    Trains the model autoregressively across multi-passage trajectories using Soft Scheduled Sampling.
    Uses Ophelia-style length grouping and proximal gradient descent for regularizations.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    # Group by sequence length to allow vectorized multi-step unrolling
    train_grouped = defaultdict(list)
    for seq, c in train_trajectories:
        train_grouped[len(seq)].append((seq, c))
        
    train_batches = []
    for length, items in train_grouped.items():
        seqs = torch.stack([item[0] for item in items])
        conds = torch.tensor([item[1] for item in items], dtype=torch.long)
        train_batches.append((seqs, conds))
        
    train_loss_hist = []
    
    if verbose:
        print("\nStarting Multi-Passage Scheduled Sampling Training...")
        
    for epoch in range(epochs):
        # Teacher forcing decays from 1.0 down to 0.0 at 80% of training
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.8)))

        model.train()
        optimizer.zero_grad()
        epoch_train_loss = 0.0
        total_train_steps = 0
        
        for seqs, conds in train_batches:
            seq_len = seqs.size(1)
            x_curr = seqs[:, 0, :] 
            
            batch_loss = 0.0
            for t in range(1, seq_len):
                log_pred, x_pred = model(x_curr, conds)
                target = seqs[:, t, :]
                
                batch_loss = batch_loss + criterion(log_pred, target)
                
                # SCHEDULED SAMPLING: Soft Mixing of True Target and Model Prediction
                x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred 
                total_train_steps += 1
                
            (batch_loss / (seq_len - 1)).backward()
            epoch_train_loss += batch_loss.item()
            
        optimizer.step()
        
        # Proximal Gradient Descent for Regularizations
        with torch.no_grad():
            if l2_lambda > 0.0:
                l2_factor = 1.0 / (1.0 + 2.0 * lr * l2_lambda)
                model.Theta.data *= l2_factor
                model.W.data *= l2_factor

            if l1_lambda > 0.0:
                for param in model.parameters():
                    alpha_l1 = lr * l1_lambda
                    param.data = torch.sign(param.data) * torch.clamp(torch.abs(param.data) - alpha_l1, min=0.0)
                    
        avg_train_loss = epoch_train_loss / total_train_steps if total_train_steps > 0 else 0
        train_loss_hist.append(avg_train_loss)
        
        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d}/{epochs} | KL Loss (Avg/Step): {avg_train_loss:.6f} | TF-Ratio: {tf_ratio:.2f}")
            
    return model, train_loss_hist

def evaluate_entropy_retention(model, csv_path):
    """
    Plots the true vs. simulated trajectories of Shannon entropy.
    Simulated trajectories are generated via pure autoregressive rollout from P0.
    """
    print("\nEvaluating Entropy Retention via Autoregressive Rollout...")
    
    df = pd.read_csv(csv_path)
    df['bio_rep'] = df['replicate'].astype(str).str[:2]
    df_avg = df.groupby(['sampleType', 'passage', 'bio_rep', 'Strain_Name'])['Read_Fraction'].mean().reset_index()
    pivot = df_avg.pivot_table(index=['sampleType', 'bio_rep', 'passage'], 
                               columns='Strain_Name', values='Read_Fraction', fill_value=0.0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    
    if 'inoculum' not in pivot.index.get_level_values('sampleType'):
        print("Error: Inoculum data required for simulation.")
        return
    inoc_data = pivot.xs('inoculum', level='sampleType')
    
    cond_map = {'no carriers': 0, 'plain-agar carriers': 1, 'mucin-agar carriers': 2}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    def calc_shannon_entropy(traj_matrix):
        t = np.array(traj_matrix)
        t = np.clip(t, 1e-9, 1.0) 
        return -np.sum(t * np.log(t), axis=1)

    model.eval()
    
    for i, (cond, c_idx) in enumerate(cond_map.items()):
        if cond not in pivot.index.get_level_values('sampleType').unique():
            continue
            
        cond_data = pivot.xs(cond, level='sampleType')
        reps = cond_data.index.get_level_values('bio_rep').unique()
        
        for rep in reps:
            rep_df = cond_data.xs(rep, level='bio_rep')
            
            # Identify P0 (Inoculum)
            if rep in inoc_data.index.get_level_values('bio_rep'):
                p0 = inoc_data.xs(rep, level='bio_rep').loc['P0'].values
            else:
                p0 = inoc_data.mean(axis=0).values
                
            # Grab True Trajectory
            passages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
            true_traj = [p0]
            for p in passages:
                if p in rep_df.index:
                    true_traj.append(rep_df.loc[p].values)
            
            true_ent = calc_shannon_entropy(true_traj)
            
            # Generate Simulated Trajectory (Autoregressive Rollout)
            sim_traj = [p0]
            curr_state = torch.tensor(p0, dtype=torch.float32).unsqueeze(0)
            c_tensor = torch.tensor([c_idx], dtype=torch.long)
            
            with torch.no_grad():
                for _ in range(len(true_traj) - 1):
                    # Feed raw probabilities back into the model
                    _, next_state = model(curr_state, c_tensor)
                    sim_traj.append(next_state.squeeze(0).numpy())
                    curr_state = next_state 
                    
            sim_ent = calc_shannon_entropy(sim_traj)
            
            # Plotting
            label_true = 'True Experimental' if rep == reps[0] else ""
            label_sim = 'Model Rollout' if rep == reps[0] else ""
            
            axes[i].plot(range(len(true_ent)), true_ent, color='royalblue', alpha=0.6, linewidth=2, label=label_true)
            axes[i].plot(range(len(sim_ent)), sim_ent, color='crimson', linestyle='--', alpha=0.8, linewidth=2, label=label_sim)
            
        axes[i].set_title(f"Diversity: {cond.title()}", fontsize=14, fontweight='bold')
        axes[i].set_xlabel("Passage Number (P0 - P6)", fontsize=12)
        axes[i].set_xticks(range(7))
        axes[i].grid(True, linestyle=':', alpha=0.6)
        axes[i].legend(loc='upper right')
        
    axes[0].set_ylabel("Shannon Entropy", fontsize=14)
    plt.tight_layout()
    
    # Ensure Images directory exists and save
    output_dir = "../Images"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, 'entropy_retention.png')
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved evaluation plot to '{plot_filename}'")
    plt.close()


def evaluate_autoregressive_scatter(model, csv_path):
    """
    Plots a scatter plot of true vs predicted relative abundances across all 
    passages, generated via purely autoregressive rollout starting from the inoculum.
    Results are split into a 2x4 grid (Overall + Individual Passages P1-P6).
    """
    print("\nEvaluating Autoregressive Scatter (Split by Passage)...")
    
    df = pd.read_csv(csv_path)
    df['bio_rep'] = df['replicate'].astype(str).str[:2]
    df_avg = df.groupby(['sampleType', 'passage', 'bio_rep', 'Strain_Name'])['Read_Fraction'].mean().reset_index()
    pivot = df_avg.pivot_table(index=['sampleType', 'bio_rep', 'passage'], 
                               columns='Strain_Name', values='Read_Fraction', fill_value=0.0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    
    if 'inoculum' not in pivot.index.get_level_values('sampleType'):
        print("Error: Inoculum data required for simulation.")
        return
    inoc_data = pivot.xs('inoculum', level='sampleType')
    
    cond_map = {'no carriers': 0, 'plain-agar carriers': 1, 'mucin-agar carriers': 2}
    
    model.eval()
    
    passages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    true_dict = {p: [] for p in passages}
    pred_dict = {p: [] for p in passages}
    true_all = []
    pred_all = []
    
    for cond, c_idx in cond_map.items():
        if cond not in pivot.index.get_level_values('sampleType').unique():
            continue
            
        cond_data = pivot.xs(cond, level='sampleType')
        reps = cond_data.index.get_level_values('bio_rep').unique()
        
        for rep in reps:
            rep_df = cond_data.xs(rep, level='bio_rep')
            
            # Identify P0 (Inoculum)
            if rep in inoc_data.index.get_level_values('bio_rep'):
                p0 = inoc_data.xs(rep, level='bio_rep').loc['P0'].values
            else:
                p0 = inoc_data.mean(axis=0).values
                
            # Grab True Trajectory (P1 through P6)
            true_traj = []
            true_traj_passages = []
            for p in passages:
                if p in rep_df.index:
                    true_traj.append(rep_df.loc[p].values)
                    true_traj_passages.append(p)
            
            if len(true_traj) == 0:
                continue
                
            # Generate Simulated Trajectory (Autoregressive Rollout)
            curr_state = torch.tensor(p0, dtype=torch.float32).unsqueeze(0)
            c_tensor = torch.tensor([c_idx], dtype=torch.long)
            
            with torch.no_grad():
                for i in range(len(true_traj)):
                    # Feed predictions into the model
                    _, next_state = model(curr_state, c_tensor)
                    pred_np = next_state.squeeze(0).numpy()
                    
                    p_name = true_traj_passages[i]
                    true_dict[p_name].append(true_traj[i])
                    pred_dict[p_name].append(pred_np)
                    
                    true_all.append(true_traj[i])
                    pred_all.append(pred_np)
                    
                    # PURE AUTOREGRESSION: set the output as the next step's input
                    curr_state = next_state 
                    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes_flat = axes.flatten()
    
    def plot_panel(ax, t, p, title):
        t_arr = np.array(t).flatten()
        p_arr = np.array(p).flatten()
        
        if len(t_arr) == 0:
            ax.axis('off')
            return
            
        # Calculate R^2 on log scale
        t_log = np.log10(np.clip(t_arr, 1e-6, None))
        p_log = np.log10(np.clip(p_arr, 1e-6, None))
        r, _ = pearsonr(t_log, p_log)
        r2 = r**2
        
        ax.scatter(t_arr, p_arr, alpha=0.4, edgecolors='k', c='steelblue', label=f'Predicted ($R^2$={r2:.2f})')
        
        plottable_mask = t_arr > 0
        min_x = t_arr[plottable_mask].min() if np.any(plottable_mask) else 1e-4
        max_x = t_arr.max() if len(t_arr) > 0 else 1.0
        
        valid_y_mask = p_arr > 0
        min_y = p_arr[valid_y_mask].min() if np.any(valid_y_mask) else 1e-4
        max_y = p_arr.max() if len(p_arr) > 0 else 1.0
        
        ax.plot([min_x, max_x], [min_x, max_x], 'r--', lw=3, label='y=x')
        
        plot_min = min(min_x, min_y) * 0.5
        plot_max = max(max_x, max_y) * 1.5
        
        # ax.set_xlim([plot_min, plot_max])
        # ax.set_ylim([plot_min, plot_max])
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc='upper left', fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.2)

    # Plot All Passages
    plot_panel(axes_flat[0], true_all, pred_all, 'All Passages (P1 - P6)')
    
    # Plot Individual Passages
    for i, p_name in enumerate(passages):
        plot_panel(axes_flat[i+1], true_dict[p_name], pred_dict[p_name], f'Passage {p_name}')
        
    # Hide the 8th unused panel
    axes_flat[7].axis('off')
    
    fig.supxlabel('True Relative Abundance', fontsize=22, y=0.02)
    fig.supylabel('Predicted Relative Abundance', fontsize=22, x=0.01)
    
    plt.tight_layout(rect=[0.03, 0.04, 1, 1])
    
    # Ensure Images directory exists and save
    output_dir = "../Images"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, 'autoregressive_scatter_split.png')
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved evaluation plot to '{plot_filename}'")
    plt.close()


if __name__ == "__main__":
    csv_filename = "../Data/SupplementaryDataS2.csv"
    LATENT_DIMS = 3  
    MAX_TRAJ_LEN = 7 # P0 -> P6
    
    if os.path.exists(csv_filename):
        train_traj, NUM_SPECIES, strains, cond_map = load_all_trajectories(
            csv_filename, 
            max_grouping_length=MAX_TRAJ_LEN
        )
    else:
        print(f"File '{csv_filename}' not found. Ensure the path is correct.")
        exit(1)
        
    # Initialize Model (3 environments: 0=liquid, 1=agar, 2=mucin)
    model = LatentAssemblyModel(num_species=NUM_SPECIES, num_latent_dims=LATENT_DIMS, num_conditions=len(cond_map))
    
    # Train using dynamic Soft Scheduled Sampling (decaying teacher forcing) on ALL data
    trained_model, train_loss_hist = train_model(
        model, train_traj, 
        epochs=1500, lr=0.02, 
        l1_lambda=0.0, l2_lambda=0.15, verbose=True
    )

    # Save the trained model parameters
    save_dir = "../Data"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "latent_assembly_model.pth")
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"\nSaved trained model parameters to '{model_save_path}'")

    # Evaluate Model's ability to retain diversity during full autoregressive simulation
    evaluate_entropy_retention(trained_model, csv_filename)
    
    # Evaluate scatter plot of full autoregressive rollouts
    evaluate_autoregressive_scatter(trained_model, csv_filename)
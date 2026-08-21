import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def load_and_filter_data(data_path, abundance_threshold=1e-3):
    """
    Loads German processed dataset and applies T1 mean relative abundance filter (mean >= threshold & present in all 12 T1 samples).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path '{data_path}' not found.")
        
    print(f"Loading German microbiome dataset from: {data_path}")
    data = torch.load(data_path, weights_only=False)
    strains = data['strains']
    trajectories = data['trajectories']
    num_before = len(strains)
    
    # Compute mean relative abundance across all T1 samples
    t1_passages = torch.stack([seq[0] for seq in trajectories.values()], dim=0) # (12, N)
    mean_t1_abundance = t1_passages.mean(dim=0)
    all_t1_present = (t1_passages > 0).all(dim=0)
    
    keep_mask = (mean_t1_abundance >= abundance_threshold) & all_t1_present
    num_after = int(keep_mask.sum().item())
    
    print(f"\nSpecies T1 Abundance Filtering (Threshold >= {abundance_threshold} & Present in All 12 T1 Samples):")
    print(f" - Total species BEFORE filter: {num_before}")
    print(f" - Total species AFTER filter:  {num_after}")
    print(f" - Filtered out: {num_before - num_after} low-abundance / non-ubiquitous noise species")
    
    filtered_trajectories = {}
    for k, seq in trajectories.items():
        filt_seq = seq[:, keep_mask]
        # Re-normalize relative abundances
        filtered_trajectories[k] = filt_seq / filt_seq.sum(dim=-1, keepdim=True)
        
    filtered_strains = [s for i, s in enumerate(strains) if keep_mask[i]]
    
    return filtered_trajectories, filtered_strains, keep_mask


def format_key_short(k):
    """Formats trajectory tuple key into short readable label: Origin|Substrate|Pressure."""
    origin, substrate, pressure = k[0], k[1], k[2]
    sub_short = substrate.replace("H2:CO2", "H2").replace(" + ", "+")
    return f"{origin} {sub_short} ({pressure}b)"


def compute_pairwise_alignment(trajectories, passage_idx=-1):
    """
    Computes pairwise alignment matrices across all trajectories at a given passage (default final passage T6).
    """
    keys = list(trajectories.keys())
    n = len(keys)
    
    matrix_r2_nz = np.zeros((n, n))
    matrix_r2_all = np.zeros((n, n))
    matrix_spearman = np.zeros((n, n))
    matrix_bray_curtis = np.zeros((n, n))
    
    records = []
    
    for i in range(n):
        for j in range(n):
            k1, k2 = keys[i], keys[j]
            t1 = trajectories[k1][passage_idx].numpy()
            t2 = trajectories[k2][passage_idx].numpy()
            
            # Determine relationship category
            same_host = (k1[0] == k2[0])
            same_env = (k1[1] == k2[1]) and (k1[2] == k2[2])
            
            if i == j:
                category = 'Self'
            elif same_env and not same_host:
                category = 'Same Env, Diff Host'
            elif same_host and not same_env:
                category = 'Same Host, Diff Env'
            else:
                category = 'Diff Host, Diff Env'
                
            # Metric 1: Log R^2 on non-zero species where both > 0
            nz_mask = (t1 > 0) & (t2 > 0)
            if np.sum(nz_mask) >= 2:
                r_nz, _ = pearsonr(np.log10(t1[nz_mask]), np.log10(t2[nz_mask]))
                r2_nz = float(r_nz ** 2) if not np.isnan(r_nz) else 0.0
            else:
                r2_nz = 0.0
                
            # Metric 2: Log R^2 with 1e-6 clip on all species
            r_all, _ = pearsonr(np.log10(np.clip(t1, 1e-6, None)), np.log10(np.clip(t2, 1e-6, None)))
            r2_all = float(r_all ** 2) if not np.isnan(r_all) else 0.0
            
            # Metric 3: Spearman rank correlation
            spear_r, _ = spearmanr(t1, t2)
            spear_r = float(spear_r) if not np.isnan(spear_r) else 0.0
            
            # Metric 4: Bray-Curtis similarity
            bc_sim = float(1.0 - (np.abs(t1 - t2).sum() / (t1 + t2).sum()))
            
            matrix_r2_nz[i, j] = r2_nz
            matrix_r2_all[i, j] = r2_all
            matrix_spearman[i, j] = spear_r
            matrix_bray_curtis[i, j] = bc_sim
            
            records.append({
                'i': i, 'j': j,
                'k1': format_key_short(k1),
                'k2': format_key_short(k2),
                'host1': k1[0], 'host2': k2[0],
                'env1': f"{k1[1]} ({k1[2]}b)", 'env2': f"{k2[1]} ({k2[2]}b)",
                'category': category,
                'r2_nonzero': r2_nz,
                'r2_all_clip': r2_all,
                'spearman_r': spear_r,
                'bray_curtis_sim': bc_sim
            })
            
    df_pairs = pd.DataFrame(records)
    labels = [format_key_short(k) for k in keys]
    
    matrices = {
        'r2_nonzero': pd.DataFrame(matrix_r2_nz, index=labels, columns=labels),
        'r2_all_clip': pd.DataFrame(matrix_r2_all, index=labels, columns=labels),
        'spearman_r': pd.DataFrame(matrix_spearman, index=labels, columns=labels),
        'bray_curtis_sim': pd.DataFrame(matrix_bray_curtis, index=labels, columns=labels)
    }
    
    return df_pairs, matrices, keys


def plot_alignment_summary(df_pairs, matrices, output_dir="../Images"):
    """
    Generates publication-quality visualizations for ground-truth microbiome alignment at T6.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # -----------------------------------------------------------------
    # Plot 1: 12x12 Heatmap of Pairwise Log R^2 (Non-Zero Species)
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))
    mat = matrices['r2_nonzero'].values
    labels = list(matrices['r2_nonzero'].index)
    
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1.0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Log-Scale Pearson $R^2$ (Non-Zero Species)', fontsize=12, fontweight='bold')
    
    # Annotate matrix values
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)
            
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Ground-Truth Final Passage (T6) Alignment Matrix\nPearson $R^2$ (Log Scale, Non-Zero Species)", fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "German_final_passage_alignment_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Saved alignment heatmap to: '{heatmap_path}'")
    plt.close()
    
    # -----------------------------------------------------------------
    # Plot 2: Category Breakdown Boxplot & Summary Bar Chart
    # -----------------------------------------------------------------
    df_no_self = df_pairs[df_pairs['category'] != 'Self'].copy()
    categories = ['Same Env, Diff Host', 'Same Host, Diff Env', 'Diff Host, Diff Env']
    
    data_by_cat_nz = [df_no_self[df_no_self['category'] == c]['r2_nonzero'].values for c in categories]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Boxplot of Non-Zero Log Pearson R^2 Distribution by Category
    bplot = axes[0].boxplot(data_by_cat_nz, labels=categories, patch_artist=True, medianprops=dict(color='black', linewidth=2))
    colors = ['#1f77b4', '#ff7f0e', '#7f7f7f']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    # Overlay jittered data points
    for idx, data_pts in enumerate(data_by_cat_nz):
        jitter = np.random.normal(0, 0.04, size=len(data_pts))
        axes[0].scatter(np.ones_like(data_pts) * (idx + 1) + jitter, data_pts, color='black', alpha=0.6, s=25)
        
    axes[0].set_title("A. Non-Zero Log Pearson $R^2$ Distribution", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Pearson $R^2$ (Log Scale, Non-Zero Species)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Comparison Category", fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Panel B: Grouped Bar Chart of Multi-Metric Ground-Truth Alignment
    metrics = ['r2_nonzero', 'r2_all_clip', 'bray_curtis_sim']
    metric_labels = ['Log R^2 (Non-Zero)', 'Log R^2 (All 1e-6 Clip)', 'Bray-Curtis Similarity']
    bar_colors = ['#2b5c8f', '#4682b4', '#e74c3c']
    
    x = np.arange(len(categories))
    width = 0.25
    
    for m_idx, (m_col, m_label, col_color) in enumerate(zip(metrics, metric_labels, bar_colors)):
        means = [df_no_self[df_no_self['category'] == c][m_col].mean() for c in categories]
        stds = [df_no_self[df_no_self['category'] == c][m_col].std() for c in categories]
        rects = axes[1].bar(x + (m_idx - 1)*width, means, width, yerr=stds, capsize=4, label=m_label, color=col_color, alpha=0.85, edgecolor='black')
        
        for rect in rects:
            height = rect.get_height()
            axes[1].text(rect.get_x() + rect.get_width()/2.0, height + 0.02, f"{height:.2f}", ha='center', va='bottom', fontsize=9)
            
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, fontsize=10)
    axes[1].set_title("B. Multi-Metric Ground-Truth Alignment (Mean ± SD)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Alignment Score", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Comparison Category", fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.suptitle("Experimental Final Passage (T6) Microbiome Alignment Across Conditions & Hosts", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, "German_final_passage_alignment_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Saved alignment category summary plot to: '{summary_path}'")
    plt.close()


def print_alignment_report(df_pairs):
    """
    Prints detailed alignment breakdown to console.
    """
    df_no_self = df_pairs[df_pairs['category'] != 'Self']
    
    print("\n==================================================")
    print("EXPERIMENTAL FINAL PASSAGE (T6) ALIGNMENT REPORT")
    print("==================================================")
    
    summary_stats = df_no_self.groupby('category')[['r2_nonzero', 'r2_all_clip', 'spearman_r', 'bray_curtis_sim']].agg(
        ['mean', 'std', 'min', 'max']
    )
    print("\nSummary Statistics Across Comparison Categories:")
    print(summary_stats.to_string())
    
    print("\n--------------------------------------------------")
    print("Same Environment across Different Host Origins (BRRS vs OSU):")
    print("--------------------------------------------------")
    same_env_df = df_no_self[df_no_self['category'] == 'Same Env, Diff Host']
    for _, row in same_env_df.iterrows():
        print(f"  {row['env1']:<25}: Log R^2 (NonZero) = {row['r2_nonzero']:.4f} | Log R^2 (All Clip) = {row['r2_all_clip']:.4f} | Bray-Curtis = {row['bray_curtis_sim']:.4f}")
        
    print("\nKey Takeaway for Model Evaluation:")
    mean_same_env = same_env_df['r2_nonzero'].mean()
    print(f" -> Real experimental microbiomes tested in identical environments across different host origins")
    print(f"    have a mean ground-truth alignment of R^2 = {mean_same_env:.4f} (on non-zero species).")
    print(f" -> This indicates that host inoculum differences define an intrinsic upper bound on cross-host predictive alignment!")


if __name__ == "__main__":
    DATA_PATH = '../Data/German_processed_data.pt'
    
    trajectories, strains, keep_mask = load_and_filter_data(DATA_PATH, abundance_threshold=1e-3)
    df_pairs, matrices, keys = compute_pairwise_alignment(trajectories, passage_idx=-1)
    
    print_alignment_report(df_pairs)
    plot_alignment_summary(df_pairs, matrices, output_dir="../Images")

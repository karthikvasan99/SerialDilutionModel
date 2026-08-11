import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_format_timeline(csv_path):
    """
    Loads the data and builds a clean timeline (P0 -> P_final) of average 
    relative abundances for each media condition.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Average across replicates if bio_rep is present
    if 'replicate' in df.columns:
        df['bio_rep'] = df['replicate'].astype(str).str[:2]
        df_avg = df.groupby(['sampleType', 'passage', 'Strain_Name'])['Read_Fraction'].mean().reset_index()
    else:
        df_avg = df.groupby(['sampleType', 'passage', 'Strain_Name'])['Read_Fraction'].mean().reset_index()

    # Pivot the data
    pivot = df_avg.pivot_table(index=['sampleType', 'passage'], 
                               columns='Strain_Name', values='Read_Fraction', fill_value=0.0)
    
    # Normalize to ensure everything sums to 1.0
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    
    timelines = {}
    target_conditions = ['no carriers', 'plain-agar carriers', 'mucin-agar carriers']
    
    # Extract the Inoculum (P0) which is shared across conditions
    if 'inoculum' in pivot.index.get_level_values('sampleType'):
        inoculum_data = pivot.xs('inoculum', level='sampleType').mean(axis=0)
    else:
        inoculum_data = None
        print("Warning: 'inoculum' sampleType not found. Starting plots from P1.")

    # Build the sequence for each condition
    for cond in target_conditions:
        if cond in pivot.index.get_level_values('sampleType'):
            cond_data = pivot.xs(cond, level='sampleType')
            # Sort passages alphabetically (P1, P2, P3, etc.)
            cond_data = cond_data.sort_index() 
            
            if inoculum_data is not None:
                # Prepend the inoculum and label it 'P0'
                cond_data.loc['P0'] = inoculum_data
                cond_data = cond_data.sort_index()
                
            timelines[cond] = cond_data

    return timelines

def bray_curtis_dissimilarity(x, y):
    """Calculates the Bray-Curtis dissimilarity between two community vectors."""
    # BC = sum(|x_i - y_i|) / sum(x_i + y_i)
    # Since sum(x) = 1 and sum(y) = 1, sum(x+y) = 2
    return np.sum(np.abs(x - y)) / 2.0

def plot_passage_dynamics(timelines):
    """
    Plots stacked area charts of relative abundance and bar charts of 
    step-wise Bray-Curtis dissimilarity for each condition.
    """
    conditions = list(timelines.keys())
    num_conds = len(conditions)
    
    fig, axes = plt.subplots(2, num_conds, figsize=(6 * num_conds, 10))
    fig.suptitle('Microbiome Assembly Dynamics Across Passages', fontsize=20, y=0.98)
    
    # Standardize colors across subplots based on the overall top strains
    all_strains = set()
    for cond in conditions:
        all_strains.update(timelines[cond].columns.tolist())
    
    # Find the top 15 most abundant strains globally for the legend color-mapping
    global_mean = pd.concat(timelines.values()).mean(axis=0)
    top_strains = global_mean.sort_values(ascending=False).head(15).index.tolist()
    
    cmap = plt.get_cmap('tab20')
    color_map = {strain: cmap(i % 20) for i, strain in enumerate(top_strains)}
    color_map['Other'] = '#dddddd' # Light grey for all other strains combined

    for i, cond in enumerate(conditions):
        df = timelines[cond]
        passages = df.index.tolist()
        
        # --- ROW 1: Community Composition (Stacked Area) ---
        ax_top = axes[0, i] if num_conds > 1 else axes[0]
        
        # Isolate top strains and group the rest into "Other"
        plot_df = df[top_strains].copy()
        plot_df['Other'] = df.drop(columns=top_strains, errors='ignore').sum(axis=1)
        
        # Plot stacked area
        x = np.arange(len(passages))
        y = plot_df.values.T
        labels = plot_df.columns.tolist()
        colors = [color_map.get(l, '#000000') for l in labels]
        
        ax_top.stackplot(x, y, labels=labels, colors=colors, alpha=0.8)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(passages)
        ax_top.set_xlim(0, len(passages) - 1)
        ax_top.set_ylim(0, 1.0)
        ax_top.set_title(f"{cond.title()}\nRelative Abundance", fontsize=14)
        ax_top.set_ylabel("Relative Abundance" if i == 0 else "")
        ax_top.margins(0, 0)
        
        if i == num_conds - 1:
            ax_top.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Top Strains")

        # --- ROW 2: Step-Wise Shift (Bray-Curtis Dissimilarity) ---
        ax_bottom = axes[1, i] if num_conds > 1 else axes[1]
        
        shifts = []
        labels_shift = []
        for j in range(len(passages) - 1):
            p_curr = passages[j]
            p_next = passages[j+1]
            x_curr = df.loc[p_curr].values
            x_next = df.loc[p_next].values
            
            bc_dist = bray_curtis_dissimilarity(x_curr, x_next)
            shifts.append(bc_dist)
            labels_shift.append(f"{p_curr} \u2192 {p_next}")
            
        x_shift = np.arange(len(shifts))
        bars = ax_bottom.bar(x_shift, shifts, color='coral', edgecolor='black', alpha=0.8)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax_bottom.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
        ax_bottom.set_xticks(x_shift)
        ax_bottom.set_xticklabels(labels_shift, rotation=45, ha='right')
        ax_bottom.set_ylim(0, max(shifts + [0.5]) * 1.2) # Give headroom for text
        ax_bottom.set_title("Passage-to-Passage Shift\n(Bray-Curtis Dissimilarity)", fontsize=14)
        ax_bottom.set_ylabel("Dissimilarity (0=Same, 1=Different)" if i == 0 else "")
        ax_bottom.grid(axis='y', linestyle='--', alpha=0.5)

    # Use the rect parameter to prevent tight_layout from overlapping the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plot_filename = "passage_dynamics_visualization.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved successfully to '{plot_filename}'")
    plt.show()

if __name__ == "__main__":
    CSV_FILENAME = "SupplementaryDataS2.csv"
    
    # Load and format the data timelines
    media_timelines = load_and_format_timeline(CSV_FILENAME)
    
    # Generate the visualization
    if media_timelines:
        plot_passage_dynamics(media_timelines)
    else:
        print("Failed to process timelines. Check dataset structure.")
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

def main():
    # Load Data
    pca_csv = os.path.join(PROJECT_ROOT, 'Results', 'Grant_PCA_Values.csv')
    transfer_csv = os.path.join(PROJECT_ROOT, 'Results', 'Grant_Transferability_Values.csv')
    
    df_pca = pd.read_csv(pca_csv)
    df_trans = pd.read_csv(transfer_csv)
    
    # Set up Figure
    plt.rcParams['font.family'] = 'sans-serif'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    substrate_colors = {'H2:CO2': '#e74c3c', 'H2:CO2 + MeOH': '#3498db', 'H2:CO2 + TMA': '#2ecc71'}
    pressure_markers = {5.0: 'o', 30.0: '^'}
    
    # ==========================================
    # PANEL A: PCA Plot
    # ==========================================
    for _, row in df_pca.iterrows():
        sub = row['Substrate']
        press = row['Pressure_bar']
        color = substrate_colors.get(sub, 'k')
        marker = pressure_markers.get(float(press), 'X')
        
        ax1.scatter(row['PC1'], row['PC2'], s=350, c=color, marker=marker, edgecolors='black', linewidth=2.0, zorder=3)
        # Low verbage labels
        ax1.text(row['PC1'], row['PC2'] + 0.25, f"{sub}\n{press} bar", fontsize=14, fontweight='bold', ha='center', va='bottom', zorder=4)

    # Simplified legend
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', label='5 bar', markerfacecolor='gray', markersize=14, markeredgecolor='black'),
        mlines.Line2D([0], [0], marker='^', color='w', label='30 bar', markerfacecolor='gray', markersize=14, markeredgecolor='black')
    ]
    ax1.legend(handles=legend_elements, loc='best', fontsize=14, frameon=True)
    
    ax1.set_xlabel("Principal Component 1 (55%)", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Principal Component 2 (28%)", fontsize=16, fontweight='bold')
    ax1.set_title("Learned Environmental Spaces", fontsize=18, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    ax1.set_xlim(df_pca['PC1'].min() - 1.5, df_pca['PC1'].max() + 1.5)
    ax1.set_ylim(df_pca['PC2'].min() - 1.0, df_pca['PC2'].max() + 1.5)
    
    ax1.text(-0.05, 1.05, 'A', transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')

    # ==========================================
    # PANEL B: Transferability Plot
    # ==========================================
    x_labels = [f"{row['Substrate'].replace(' + ', '\n+ ')}\n{row['Pressure_bar']} bar" for _, row in df_trans.iterrows()]
    y_vals = df_trans['LOO_R2'].values
    bar_colors = [substrate_colors.get(row['Substrate'], 'gray') for _, row in df_trans.iterrows()]
    
    bars = ax2.bar(x_labels, y_vals, color=bar_colors, edgecolor='black', linewidth=2.0, zorder=3)
    ax2.set_ylim([0, 1.0])
    ax2.set_ylabel("Predictability ($R^2$)", fontsize=16, fontweight='bold')
    ax2.set_title("Cross-Community Transferability", fontsize=18, fontweight='bold', pad=15)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    # X-axis ticks
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), 
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=14, fontweight='bold')
                     
    ax2.text(-0.05, 1.05, 'B', transform=ax2.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')

    # Save
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, 'Images'), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, 'Images', 'Grant_Combined_Figure.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\nSaved Combined Figure to {save_path}")

if __name__ == '__main__':
    main()

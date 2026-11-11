import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

def main():
    # Load the saved PCA coordinates (No retraining required!)
    csv_path = os.path.join(PROJECT_ROOT, 'Results', 'Grant_PCA_Values.csv')
    df = pd.read_csv(csv_path)
    
    fig, ax1 = plt.subplots(figsize=(8, 7))
    plt.rcParams['font.family'] = 'sans-serif'
    
    substrate_colors = {'H2:CO2': '#e74c3c', 'H2:CO2 + MeOH': '#3498db', 'H2:CO2 + TMA': '#2ecc71'}
    pressure_markers = {5.0: 'o', 30.0: '^'}
    
    for _, row in df.iterrows():
        substrate = row['Substrate']
        pressure = row['Pressure_bar']
        pc1 = row['PC1']
        pc2 = row['PC2']
        
        color = substrate_colors.get(substrate, 'k')
        marker = pressure_markers.get(float(pressure), 'X')
        
        ax1.scatter(pc1, pc2, s=250, c=color, marker=marker, edgecolors='black', linewidth=1.5, zorder=3)
        
        # Increased fontsize from 10 to 12
        ax1.text(pc1, pc2 + 0.25, f"{substrate}\n{pressure} bar", fontsize=12, ha='center', va='bottom', zorder=4)

    # Custom legend for PCA
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', label='H2:CO2', markerfacecolor='#e74c3c', markersize=10, markeredgecolor='black'),
        mlines.Line2D([0], [0], marker='o', color='w', label='H2:CO2 + MeOH', markerfacecolor='#3498db', markersize=10, markeredgecolor='black'),
        mlines.Line2D([0], [0], marker='o', color='w', label='H2:CO2 + TMA', markerfacecolor='#2ecc71', markersize=10, markeredgecolor='black'),
        mlines.Line2D([0], [0], marker='o', color='w', label='5.0 bar', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        mlines.Line2D([0], [0], marker='^', color='w', label='30.0 bar', markerfacecolor='gray', markersize=10, markeredgecolor='black')
    ]
    ax1.legend(handles=legend_elements, loc='best', fontsize=12, frameon=True)
    
    ax1.set_xlabel("Principal Component 1 (54.9%)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Principal Component 2 (27.5%)", fontsize=14, fontweight='bold')
    ax1.set_title("PCA of Learned Environmental Resources ($R_0$)", fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    # Increase Y and X limits slightly so labels don't get cut off
    ax1.set_xlim(df['PC1'].min() - 1.5, df['PC1'].max() + 1.5)
    ax1.set_ylim(df['PC2'].min() - 1.0, df['PC2'].max() + 1.5)

    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, 'Images'), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, 'Images', 'German_Grant_PCA_CV.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\nSaved Grant Plot to {save_path}")

if __name__ == '__main__':
    main()

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

def main():
    # Hardcoded optimal LOO R^2 averages from the extensive grid sweeps 
    # (Results/MeOH_Finetune_Results.csv and Results/TMA_Finetune_Results.csv)
    # H2:CO2 conditions use the global baseline K=4, LR=0.01, L2=0.05 results.
    
    unique_envs = [
        ('H2:CO2', 5.0),
        ('H2:CO2', 30.0),
        ('H2:CO2 + MeOH', 5.0),
        ('H2:CO2 + MeOH', 30.0),
        ('H2:CO2 + TMA', 5.0),
        ('H2:CO2 + TMA', 30.0),
    ]
    
    env_r2 = {
        ('H2:CO2', 5.0): 0.513,       # Average of BRRS and OSU from baseline K=4
        ('H2:CO2', 30.0): 0.698,      # Average of BRRS and OSU from baseline K=4
        ('H2:CO2 + MeOH', 5.0): 0.454, # Optimal K=5, LR=0.001, L2=0.05
        ('H2:CO2 + MeOH', 30.0): 0.479,# Optimal K=4, LR=0.001, L2=0.05
        ('H2:CO2 + TMA', 5.0): 0.371,  # Optimal K=2, LR=0.01, L2=0.05
        ('H2:CO2 + TMA', 30.0): 0.532, # Optimal K=4, LR=0.01, L2=0.05
    }
    
    # PLOTTING
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.family'] = 'sans-serif'
    
    substrate_colors = {'H2:CO2': '#e74c3c', 'H2:CO2 + MeOH': '#3498db', 'H2:CO2 + TMA': '#2ecc71'}
        
    x_labels = [f"{env[0]}\n{env[1]} bar" for env in unique_envs]
    y_vals = [env_r2[env] for env in unique_envs]
    
    bar_colors = [substrate_colors.get(env[0], 'gray') for env in unique_envs]
    
    bars = ax.bar(x_labels, y_vals, color=bar_colors, edgecolor='black', linewidth=1.5, zorder=3)
    ax.set_ylim([0, 1.0])
    ax.set_ylabel('Leave-One-Out $R^2$ at $T_6$', fontsize=14, fontweight='bold')
    ax.set_title('Cross-Community Transferability (Condition Fine-Tuned)', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, 'Images'), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, 'Images', 'Grant_Cross_Community_Transferability_Finetuned.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved Grant Plot to {save_path}")

if __name__ == '__main__':
    main()

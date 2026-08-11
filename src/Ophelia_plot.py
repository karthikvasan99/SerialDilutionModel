import os
import numpy as np
import matplotlib.pyplot as plt

def regenerate_plot(data_file='venturelli_plot_data.npz'):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Could not find '{data_file}'. Run the main model script first to generate it.")
        
    data = np.load(data_file, allow_pickle=True)
    train_true = data['train_true']
    train_pred = data['train_pred']
    r2_train = float(data['r2_train'])
    timepoints_to_plot = data['timepoints_to_plot']

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes = axes.flatten()
    
    # --- PANEL 0: Train Scatter ---
    axes[0].scatter(train_true, train_pred, alpha=0.3, edgecolors='k', c='steelblue', label=f'Predicted ($R^2$={r2_train:.2f})')
    
    plottable_mask = train_true > 0
    min_x = train_true[plottable_mask].min() if np.any(plottable_mask) else 1e-4
    max_x = train_true.max()
    axes[0].plot([min_x, max_x], [min_x, max_x], 'r--', lw=3, label='y=x')
    
    valid_y_mask = train_true > 0
    min_y = train_pred[valid_y_mask].min() if np.any(valid_y_mask) else 1e-4
    max_y = train_pred.max()
    
    plot_min_x = min_x * 0.5
    plot_max_x = min(max_x * 1.5, 1.5)
    plot_min_y = min_y * 0.5
    plot_max_y = min(max_y * 1.5, 1.5)
    
    axes[0].set_xlim([plot_min_x, plot_max_x])
    axes[0].set_ylim([plot_min_y, plot_max_y])
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title('Train Set (All Timepoints)', fontsize=26)
    axes[0].tick_params(axis='both', which='major', labelsize=22)
    axes[0].legend(loc='upper left', fontsize=20)
    axes[0].grid(True, which="both", ls="--", alpha=0.2)
    
    # --- PANELS 1-5: Test Scatters ---
    for idx, p in enumerate(timepoints_to_plot):
        ax = axes[idx + 1]
        
        # Check if this timepoint was saved
        if f'true_{p}' in data:
            true_p = data[f'true_{p}']
            pred_p = data[f'pred_{p}']
            r2_p = float(data[f'r2_{p}'])
            
            ax.scatter(true_p, pred_p, alpha=0.6, edgecolors='k', c='darkorange', label=f'Predicted ($R^2$={r2_p:.2f})')
            
            plottable_mask_p = true_p > 0
            min_x_test = true_p[plottable_mask_p].min() if np.any(plottable_mask_p) else 1e-4
            max_x_test = true_p.max()
            ax.plot([min_x_test, max_x_test], [min_x_test, max_x_test], 'r--', lw=3)
            
            valid_y_mask_test = true_p > 0
            min_y_test = pred_p[valid_y_mask_test].min() if np.any(valid_y_mask_test) else 1e-4
            max_y_test = pred_p.max()
            
            plot_min_x_test = min_x_test * 0.5
            plot_max_x_test = min(max_x_test * 1.5, 1.5)
            plot_min_y_test = min_y_test * 0.5
            plot_max_y_test = min(max_y_test * 1.5, 1.5)
            
            ax.set_xlim([min(plot_min_x_test, plot_min_y_test), max(plot_max_x_test, plot_max_y_test)])
            ax.set_ylim([min(plot_min_x_test, plot_min_y_test), max(plot_max_x_test, plot_max_y_test)])
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f"Test Evaluation ({p})", fontsize=26)
            ax.tick_params(axis='both', which='major', labelsize=22)
            ax.legend(loc='upper left', fontsize=20)
            ax.grid(True, which="both", ls="--", alpha=0.2)
            
    fig.supxlabel('Actual Relative Abundance', fontsize=30, y=0.02)
    fig.supylabel('Predicted Relative Abundance', fontsize=30, x=0.01)
            
    plt.tight_layout(rect=[0.03, 0.04, 1, 1])
    
    output_dir = "Images"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, 'venturelli_evaluation_regenerated.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot regenerated and saved to '{plot_filename}'")

if __name__ == "__main__":
    regenerate_plot()
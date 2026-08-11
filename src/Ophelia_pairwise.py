import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
import sys
from SerialDilutionModels import LatentAssemblyModel
from Ophelia_evaluation import load_venturelli_arrays, train_model, evaluate_predictions
import copy


data_path = '../Data'

train_path = os.path.join(data_path, 'train_data.mat')
test_path = os.path.join(data_path, 'test_data.mat')

if os.path.exists(train_path) and os.path.exists(test_path):   

    train_data_red = loadmat(train_path)
    X_train_ophelia = train_data_red['train_data']

    test_data_red  = loadmat(test_path)
    X_test_ophelia  = test_data_red['test_data']

X_all = np.concat((X_train_ophelia, X_test_ophelia))
X_all_inoculum = X_all[:, 0, :]

MAX_TRAJECTORY_LENGTH = 7  

# Find all inoculum where bacteria 'i' and 'j' are present
for i in range(X_all_inoculum.shape[1]):
    for j in range(i+1, X_all_inoculum.shape[1]):

        if os.path.isfile(f'../Pairwise_results/Ophelia_{i}_{j}_plot_data.npz'):
            print(f'Skipping {i}_{j}')
            continue

        mask_ij = (X_all_inoculum[:, i] != 0) & (X_all_inoculum[:, j] != 0)
        
        # 2. Slice X_all directly using the boolean mask
        X_test_ij = X_all[mask_ij]        # Rows where both are present
        X_train_ij = X_all[~mask_ij]  

        # print(X_train_ij.shape, X_test_ij.shape)

        # --- 10% Validation Split ---
        np.random.seed(42) # Ensure reproducible splits
        num_train_samples = X_train_ij.shape[0]
        indices = np.random.permutation(num_train_samples)
        val_size = max(1, int(num_train_samples * 0.10))
        
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        X_train_split_ij = X_train_ij[train_indices]
        X_val_ij = X_train_ij[val_indices]
        
        print(f"\nSplit complete: {X_train_split_ij.shape[0]} Train arrays, {X_val_ij.shape[0]} Val arrays.")
        
        # Process into causal trajectories
        train_trajectories, val_trajectories, test_trajectories, NUM_SPECIES, strains, valid_conds = load_venturelli_arrays(
            train_array=X_train_split_ij,
            val_array=X_val_ij,
            test_array=X_test_ij, 
            max_grouping_length=MAX_TRAJECTORY_LENGTH
        )
        num_conds = len(valid_conds)
        
        # --- Grid Search ---
        latent_dims_grid = [4, 5, 6]
        epochs_grid = [800, 1000, 1500]
        l2_lambda_grid = [0.15, 0.2, 0.25]
        
        best_val_loss = float('inf')
        best_params = {}
        best_model = None
        best_train_hist = None
        best_val_hist = None
        
        total_iterations = len(latent_dims_grid) * len(epochs_grid) * len(l2_lambda_grid)
        current_iter = 0
        
        print(f"\nStarting Grid Search ({total_iterations} combinations)...")
        
        for k_val in latent_dims_grid:
            for epochs_val in epochs_grid:
                for l2_val in l2_lambda_grid:
                    current_iter += 1
                    
                    # --- Update Progress Bar ---
                    progress = current_iter / total_iterations
                    bar_length = 40
                    block = int(round(bar_length * progress))
                    text = f"\rProgress: [{'=' * block + ' ' * (bar_length - block)}] {current_iter}/{total_iterations} "
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    
                    model = LatentAssemblyModel(num_species=NUM_SPECIES, num_latent_dims=k_val, num_conditions=num_conds)
                    
                    trained_model, train_loss_hist, val_loss_hist = train_model(
                        model, train_trajectories, val_trajectories,
                        epochs=epochs_val, lr=0.02, l1_lambda=0.0, l2_lambda=l2_val, verbose=False
                    )
                    
                    final_val_loss = val_loss_hist[-1]
                    
                    if final_val_loss < best_val_loss:
                        best_val_loss = final_val_loss
                        best_params = {'latent_dims': k_val, 'epochs': epochs_val, 'l2_lambda': l2_val}
                        best_model = copy.deepcopy(trained_model)
                        best_train_hist = train_loss_hist
                        best_val_hist = val_loss_hist
                        
        print() # New line after progress bar finishes
                    
        print("\n==================================================")
        print("Grid Search Complete!")
        print(f"Best Parameters: {best_params}")
        print(f"Best Val KL Loss: {best_val_loss:.6f}")
        print("==================================================")
        
        print("\nRunning final evaluation and plotting on the best model...")
        evaluate_predictions(best_model, train_trajectories, test_trajectories, strains, 
            valid_conds, best_train_hist, best_val_hist, plot_eval=False, save_data=True,
            name=f'Ophelia_{i}_{j}')

    #     break
    # break

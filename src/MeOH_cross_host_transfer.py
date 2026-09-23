import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
import itertools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_data_filtered(data_path, test_keys, threshold=1e-3):
    print(f"Loading data from {data_path} with cutoff > {threshold}")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    strains = data['strains']
    trajectories = data['trajectories']

    train_seqs = []
    for k, seq in trajectories.items():
        if k not in test_keys:
            train_seqs.append(seq)
    train_seqs = torch.stack(train_seqs)
    
    mean_abundance = train_seqs.mean(dim=(0, 1))
    
    keep_mask = mean_abundance > threshold
    num_kept = int(keep_mask.sum().item())
    print(f"Number of species retained: {num_kept}")
    
    filtered_strains = [s for idx, s in enumerate(strains) if keep_mask[idx]]
    filtered_traj = {}
    for k in trajectories.keys():
        filt_seq = trajectories[k][:, keep_mask]
        filt_seq = filt_seq + 1e-6
        filtered_traj[k] = filt_seq / filt_seq.sum(dim=-1, keepdim=True)
        
    return filtered_traj, filtered_strains

def train_neural_ode(model, train_seqs, train_conds, epochs=400, lr=0.01, l2_lambda=0.05):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    seq_len = train_seqs.size(1)
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        tf_ratio = max(0.0, 1.0 - (epoch / (epochs * 0.8)))
        
        model.train()
        optimizer.zero_grad()
        
        x_curr = train_seqs[:, 0, :]
        batch_loss = 0.0
        
        for t in range(1, seq_len):
            log_pred, x_pred, _, _ = model(x_curr, train_conds)
            target = train_seqs[:, t, :]
            batch_loss = batch_loss + criterion(log_pred, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred
            
        (batch_loss / (seq_len - 1)).backward()
        optimizer.step()
        
        with torch.no_grad():
            if l2_lambda > 0.0:
                l2_factor = 1.0 / (1.0 + 2.0 * lr * l2_lambda)
                model.ode_func.Theta.data *= l2_factor
                model.ode_func.W.data *= l2_factor
                
        current_loss = (batch_loss / (seq_len - 1)).item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_state)
    return best_loss

def main():
    set_seed(42)
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
    
    test_keys = [('BRRS', 'H2:CO2 + MeOH', 5.0), ('BRRS', 'H2:CO2 + MeOH', 30.0)]
    trajectories, strains = load_data_filtered(data_path, test_keys, threshold=1e-3)
    
    methanogen_indices = [idx for idx, s in enumerate(strains) if 'Methano' in s or 'Thermoplasmata' in s or 'archaeon' in s]
    print(f"Found {len(methanogen_indices)} Methanogens.")
    
    env_map = {}
    unique_envs = []
    for cond_key in trajectories.keys():
        env_label = cond_key[1:] if isinstance(cond_key, tuple) and len(cond_key) >= 2 else cond_key
        if env_label not in unique_envs:
            unique_envs.append(env_label)
        env_map[cond_key] = unique_envs.index(env_label)
        
    num_species = len(strains)
    num_conditions = len(unique_envs)
    
    cv_val_keys = [k for k in trajectories.keys() if 'BRRS' in k[0] and k not in test_keys]
    print(f"\nCV Validation Folds (BRRS non-MeOH): {cv_val_keys}")
    
    lrs = [0.01, 0.005, 0.001]
    l2s = [0.001, 0.01, 0.05, 0.1]
    dims = [2, 4, 8, 16]
    
    best_avg_val_loss = float('inf')
    best_params = {}
    
    print("\n=== Running LOOCV Hyperparameter Search ===")
    import io
    from contextlib import redirect_stdout
    
    for lr, l2, dim in itertools.product(lrs, l2s, dims):
        fold_losses = []
        for val_key in cv_val_keys:
            train_keys = [k for k in trajectories.keys() if k not in test_keys and k != val_key]
            train_seqs = torch.stack([trajectories[k] for k in train_keys])
            train_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)
            
            val_seq = trajectories[val_key].unsqueeze(0)
            val_cond = torch.tensor([env_map[val_key]], dtype=torch.long)
            
            model = LatentAssemblyNeuralODE(num_species, dim, num_conditions, 100.0)
            with redirect_stdout(io.StringIO()):
                train_neural_ode(model, train_seqs, train_conds, epochs=400, lr=lr, l2_lambda=l2)
            
            model.eval()
            with torch.no_grad():
                x_curr = val_seq[:, 0, :]
                pred_rel_all = [x_curr]
                for _ in range(1, val_seq.size(1)):
                    _, x_pred, _, _ = model(x_curr, val_cond)
                    pred_rel_all.append(x_pred)
                    x_curr = x_pred
                pred_rel_all = torch.cat(pred_rel_all, dim=0)
                safe_pred = torch.clamp(pred_rel_all, min=1e-9)
                v_loss = nn.KLDivLoss(reduction='batchmean')(torch.log(safe_pred[1:]), val_seq[0, 1:]).item()
                fold_losses.append(v_loss)
                
        avg_loss = sum(fold_losses) / len(fold_losses)
        print(f"lr={lr}, l2={l2}, dim={dim} | Avg CV Loss: {avg_loss:.4f}")
        if avg_loss < best_avg_val_loss:
            best_avg_val_loss = avg_loss
            best_params = {'lr': lr, 'l2': l2, 'dim': dim}
            
    print(f"\nBest Hyperparameters Found: {best_params} (CV Loss: {best_avg_val_loss:.4f})")
    
    print("\n=== Training Final Model on ALL Non-Test Data ===")
    final_train_keys = [k for k in trajectories.keys() if k not in test_keys]
    final_train_seqs = torch.stack([trajectories[k] for k in final_train_keys])
    final_train_conds = torch.tensor([env_map[k] for k in final_train_keys], dtype=torch.long)
    
    final_model = LatentAssemblyNeuralODE(num_species, best_params['dim'], num_conditions, 100.0)
    train_neural_ode(final_model, final_train_seqs, final_train_conds, epochs=400, lr=best_params['lr'], l2_lambda=best_params['l2'])
    
    print("\n=== Model Prediction for Test Sets (BRRS MeOH) ===")
    final_model.eval()
    all_pred_log = []
    all_true_log = []
    with torch.no_grad():
        for t_key in test_keys:
            seq = trajectories[t_key]
            c = torch.tensor([env_map[t_key]], dtype=torch.long)
            
            x_curr = seq[0].unsqueeze(0)
            pred_rel_all = [x_curr]
            for _ in range(1, seq.size(0)):
                _, x_pred, _, _ = final_model(x_curr, c)
                pred_rel_all.append(x_pred)
                x_curr = x_pred
            pred_rel_all = torch.cat(pred_rel_all, dim=0)
            
            pred_flat = pred_rel_all[1:].flatten().numpy()
            true_flat = seq[1:].flatten().numpy()
            pred_log = np.log10(np.clip(pred_flat, 1e-6, None))
            true_log = np.log10(np.clip(true_flat, 1e-6, None))
            
            r_val, _ = pearsonr(true_log, pred_log)
            all_pred_log.extend(pred_log)
            all_true_log.extend(true_log)
            
            pred_t6_methano = pred_rel_all[-1, methanogen_indices].sum().item()
            true_t6_methano = seq[-1, methanogen_indices].sum().item()
            
            print(f"Prediction for {t_key}:")
            print(f"  Pearson R (log-scale):             {r_val:.4f}")
            print(f"  Predicted T6 Methanogen Abundance: {pred_t6_methano:.4f}")
            print(f"  True T6 Methanogen Abundance:      {true_t6_methano:.4f}\n")
            
        overall_r, _ = pearsonr(np.array(all_true_log), np.array(all_pred_log))
        print(f"Overall Pearson R for Test Sets: {overall_r:.4f}")

    # Save model weights
    save_path = os.path.join(PROJECT_ROOT, 'Results', 'MeOH_Transfer_Model.pth')
    torch.save(final_model.state_dict(), save_path)
    print(f"\nSaved final model weights to: {save_path}")

if __name__ == "__main__":
    main()

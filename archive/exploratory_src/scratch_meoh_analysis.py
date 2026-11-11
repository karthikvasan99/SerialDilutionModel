import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SerialDilutionModels import LatentAssemblyNeuralODE

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_data_114_strains(data_path):
    print(f"Loading data from {data_path}")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    strains = data['strains']
    trajectories = data['trajectories']

    # Check across ALL timepoints, since some bacteria are zero at T1 and grow later
    brrs_all = torch.stack([seq for k, seq in trajectories.items() if 'BRRS' in k[0]])
    osu_all = torch.stack([seq for k, seq in trajectories.items() if 'OSU' in k[0]])
    
    brrs_present = (brrs_all > 0).any(dim=0).any(dim=0)
    osu_present = (osu_all > 0).any(dim=0).any(dim=0)
    keep_mask = brrs_present & osu_present
    
    num_kept = int(keep_mask.sum().item())
    print(f"Number of species present in at least one BRRS and OSU sample across all passages (Intersection): {num_kept}")
    
    filtered_strains = [s for idx, s in enumerate(strains) if keep_mask[idx]]
    filtered_traj = {}
    for k in trajectories.keys():
        filt_seq = trajectories[k][:, keep_mask]
        
        # Add pseudo-count to eliminate exact zeros
        filt_seq = filt_seq + 1e-6
        
        # Re-normalize to relative abundance
        filtered_traj[k] = filt_seq / filt_seq.sum(dim=-1, keepdim=True)
        
    return filtered_traj, filtered_strains

def train_neural_ode(model, train_seqs, train_conds, epochs=200, lr=0.01, l2_lambda=0.05):
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
                
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {(batch_loss / (seq_len - 1)).item():.6f}")
            
        current_loss = (batch_loss / (seq_len - 1)).item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with loss {best_loss:.6f}")

def main():
    set_seed(42)
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'German_processed_data.pt')
    trajectories, strains = load_data_114_strains(data_path)
    
    # Identify Methanogens
    methanogen_indices = [idx for idx, s in enumerate(strains) if 'Methano' in s or 'Thermoplasmata' in s or 'archaeon' in s]
    print(f"Found {len(methanogen_indices)} Methanogens.")
    
    def get_methanogen_abundance(seq):
        return seq[:, methanogen_indices].sum(dim=-1)
    
    print("\n=== Empirical Results for MeOH Conditions ===")
    
    brrs_5 = get_methanogen_abundance(trajectories[('BRRS', 'H2:CO2 + MeOH', 5.0)])
    brrs_30 = get_methanogen_abundance(trajectories[('BRRS', 'H2:CO2 + MeOH', 30.0)])
    print(f"BRRS MeOH 5.0 bar  | T1: {brrs_5[0]:.4f} -> T6: {brrs_5[-1]:.4f} (Change: {brrs_5[-1]-brrs_5[0]:.4f})")
    print(f"BRRS MeOH 30.0 bar | T1: {brrs_30[0]:.4f} -> T6: {brrs_30[-1]:.4f} (Change: {brrs_30[-1]-brrs_30[0]:.4f})")
    print(f" Empirically, BRRS 30.0 bar produces {'MORE' if brrs_30[-1] > brrs_5[-1] else 'LESS'} methanogens than 5.0 bar")
    
    osu_5 = get_methanogen_abundance(trajectories[('OSU', 'H2:CO2 + MeOH', 5.0)])
    osu_30 = get_methanogen_abundance(trajectories[('OSU', 'H2:CO2 + MeOH', 30.0)])
    print(f"\nOSU MeOH 5.0 bar   | T1: {osu_5[0]:.4f} -> T6: {osu_5[-1]:.4f} (Change: {osu_5[-1]-osu_5[0]:.4f})")
    print(f"OSU MeOH 30.0 bar  | T1: {osu_30[0]:.4f} -> T6: {osu_30[-1]:.4f} (Change: {osu_30[-1]-osu_30[0]:.4f})")
    print(f" Empirically, OSU 30.0 bar produces {'MORE' if osu_30[-1] > osu_5[-1] else 'LESS'} methanogens than 5.0 bar")
    print(" (Notice the empirical trend difference between BRRS and OSU!)")
    
    # Train model on OSU
    print("\n=== Training Model on OSU (Zero-Shot Setup) ===")
    
    env_map = {}
    unique_envs = []
    for cond_key in trajectories.keys():
        env_label = cond_key[1:] if isinstance(cond_key, tuple) and len(cond_key) >= 2 else cond_key
        if env_label not in unique_envs:
            unique_envs.append(env_label)
        env_map[cond_key] = unique_envs.index(env_label)
        
    num_species = len(strains)
    num_conditions = len(unique_envs)
    
    val_key = ('OSU', 'H2:CO2 + TMA', 30.0)
    test_keys = [('BRRS', 'H2:CO2 + MeOH', 5.0), ('BRRS', 'H2:CO2 + MeOH', 30.0)]
    
    # Train on all OSU except val_key
    train_keys = [k for k in trajectories.keys() if 'OSU' in k[0] and k != val_key]
    
    train_seqs = torch.stack([trajectories[k] for k in train_keys])
    train_conds = torch.tensor([env_map[k] for k in train_keys], dtype=torch.long)
    
    val_seq = trajectories[val_key].unsqueeze(0)
    val_cond = torch.tensor([env_map[val_key]], dtype=torch.long)
    
    print(f"\n=== Fine-tuning Hyperparameters on {val_key} ===")
    
    best_val_loss = float('inf')
    best_params = {}
    best_model_state = None
    best_model_obj = None
    
    lrs = [0.01, 0.005]
    l2s = [0.01, 0.05]
    dims = [4, 8]
    
    import itertools
    for lr, l2, dim in itertools.product(lrs, l2s, dims):
        print(f"Training with lr={lr}, l2={l2}, dim={dim}...")
        model = LatentAssemblyNeuralODE(
            num_species=num_species,
            num_latent_dims=dim,
            num_conditions=num_conditions,
            default_dilution_factor=100.0
        )
        # Suppress prints from train loop by capturing stdout or just letting it run quickly
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            train_neural_ode(model, train_seqs, train_conds, epochs=400, lr=lr, l2_lambda=l2)
            
        # Eval on val set
        model.eval()
        with torch.no_grad():
            x_curr = val_seq[:, 0, :]
            seq_len = val_seq.size(1)
            pred_rel_all = [x_curr]
            for t in range(1, seq_len):
                _, x_pred, _, _ = model(x_curr, val_cond)
                pred_rel_all.append(x_pred)
                x_curr = x_pred
            pred_rel_all = torch.cat(pred_rel_all, dim=0) # shape: (6, N)
            
            # Use KLDiv loss as val metric
            safe_pred = torch.clamp(pred_rel_all, min=1e-9)
            val_loss = nn.KLDivLoss(reduction='batchmean')(torch.log(safe_pred[1:]), val_seq[0, 1:]).item()
            
            print(f"  Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {'lr': lr, 'l2': l2, 'dim': dim}
                best_model_state = copy.deepcopy(model.state_dict())
                best_model_obj = model

    print(f"\nBest Hyperparameters: {best_params} with Val Loss: {best_val_loss:.4f}")
    
    # Load best model and predict test set
    best_model_obj.load_state_dict(best_model_state)
    best_model_obj.eval()
    
    print(f"\n=== Model Prediction for Test Sets ===")
    all_pred_log = []
    all_true_log = []
    
    with torch.no_grad():
        for t_key in test_keys:
            seq = trajectories[t_key]
            c = torch.tensor([env_map[t_key]], dtype=torch.long)
            
            x_curr = seq[0].unsqueeze(0)
            seq_len = seq.size(0)
            
            pred_rel_all = [x_curr]
            for t in range(1, seq_len):
                _, x_pred, _, _ = best_model_obj(x_curr, c)
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
            print(f"  True T6 Methanogen Abundance:      {true_t6_methano:.4f}")
            print()
            
        overall_r, _ = pearsonr(np.array(all_true_log), np.array(all_pred_log))
        print(f"Overall Pearson R (log-scale) for BRRS MeOH Test predictions: {overall_r:.4f}")
        
    # Also evaluate on Val set to show the user what it predicted for the validation set
    with torch.no_grad():
        seq = trajectories[val_key]
        c = torch.tensor([env_map[val_key]], dtype=torch.long)
        
        x_curr = seq[0].unsqueeze(0)
        pred_rel_all = [x_curr]
        for t in range(1, seq_len):
            _, x_pred, _, _ = best_model_obj(x_curr, c)
            pred_rel_all.append(x_pred)
            x_curr = x_pred
        pred_rel_all = torch.cat(pred_rel_all, dim=0)
        
        pred_t6_methano = pred_rel_all[-1, methanogen_indices].sum().item()
        true_t6_methano = seq[-1, methanogen_indices].sum().item()
        print(f"\nFor reference, prediction on validation set {val_key}:")
        print(f"  Predicted T6 Methanogen Abundance: {pred_t6_methano:.4f}")
        print(f"  True T6 Methanogen Abundance:      {true_t6_methano:.4f}")

if __name__ == "__main__":
    main()

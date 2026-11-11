import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = 'Data/Gore_processed_data.pt'
data = torch.load(data_path, weights_only=False)
cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

b11_N, b11_od, b11_rel, b11_c, b11_m, b11_keys = [], [], [], [], [], []
b4_N, b4_od, b4_rel, b4_c, b4_m, b4_keys = [], [], [], [], [], []

for (nutrient, S, comm), traj_rel in data['relative_abundance'].items():
    c_idx = cond_to_idx[nutrient]
    mask = data['presence_masks'][(S, comm)]
    od_mean = data['biomass_od'][(nutrient, S, comm)].mean(dim=0)
    
    if traj_rel.shape[0] == 11:
        b11_N.append(traj_rel * od_mean.unsqueeze(-1))
        b11_od.append(od_mean)
        b11_rel.append(traj_rel)
        b11_c.append(c_idx)
        b11_m.append(mask)
        b11_keys.append((nutrient, S, comm))
    elif traj_rel.shape[0] == 4:
        od_4 = od_mean[[0, 7, 8, 10]]
        b4_N.append(traj_rel * od_4.unsqueeze(-1))
        b4_od.append(od_mean)
        b4_rel.append(traj_rel)
        b4_c.append(c_idx)
        b4_m.append(mask)
        b4_keys.append((nutrient, S, comm))

b11_N = torch.stack(b11_N).to(device)
b11_c = torch.tensor(b11_c, device=device)
b11_m = torch.stack(b11_m).to(device)

b4_N = torch.stack(b4_N).to(device)
b4_c = torch.tensor(b4_c, device=device)
b4_m = torch.stack(b4_m).to(device)

all_N0 = torch.cat([b11_N[:, 0, :], b4_N[:, 0, :]], dim=0).to(device)
all_c = torch.cat([b11_c, b4_c], dim=0).to(device)
all_m = torch.cat([b11_m, b4_m], dim=0).to(device)
all_keys = b11_keys + b4_keys

# Ground truth binary chaotic labels per Gore et al.
gt_labels = []
gt_exp_cvs = []
for (nutrient, S, comm) in all_keys:
    mask = data['presence_masks'][(S, comm)].numpy().astype(bool)
    rel = data['relative_abundance'][(nutrient, S, comm)].numpy()
    od = data['biomass_od'][(nutrient, S, comm)].mean(dim=0).numpy()
    if rel.shape[0] == 11:
        N_abs = rel * od[:, None]
        late_N = N_abs[7:11, mask]
    else:
        od_4 = od[[0, 7, 8, 10]]
        N_abs = rel * od_4[:, None]
        late_N = N_abs[1:, mask]
    mu_N = late_N.mean(axis=0)
    sig_N = late_N.std(axis=0)
    cv_N = np.mean(sig_N / (0.01 + mu_N))
    od_7_10 = data['biomass_od'][(nutrient, S, comm)].numpy()[:, 7:11]
    od_cv = np.mean(np.std(od_7_10, axis=1) / np.mean(od_7_10, axis=1))
    gt_exp_cvs.append(cv_N)
    gt_labels.append(bool(cv_N >= 0.25 or od_cv >= 0.14))

gt_labels = np.array(gt_labels)
gt_exp_cvs = np.array(gt_exp_cvs)

from src.Gore_PhysODE_LowKSweep import AdaptivePureConsumerResourceODE, HierarchicalPhysicalNeuralODE

def compute_lyapunov_spectrum(model, eps=1e-5, num_passages=10, burn_in=5):
    """
    Computes the maximal finite-time Lyapunov exponent (lambda_max) using Benettin's algorithm
    across serial dilution passages.
    """
    model.eval()
    B = all_N0.shape[0]
    
    with torch.no_grad():
        # First, run burn_in passages to get onto the attractor
        N_curr = all_N0.clone()
        for _ in range(burn_in):
            N_curr = model.step_forward(N_curr, all_c, all_m, delta_t=0.10)
            
        # Initialize random orthogonal perturbation on present species
        torch.manual_seed(42)
        delta_0 = torch.randn_like(N_curr) * all_m
        # Normalize delta_0 to norm eps per community
        delta_norm = torch.norm(delta_0, dim=-1, keepdim=True) + 1e-12
        delta = (delta_0 / delta_norm) * eps
        
        lyap_sum = torch.zeros(B, device=device)
        
        for p in range(num_passages):
            # Base forward step
            N_next = model.step_forward(N_curr, all_c, all_m, delta_t=0.10)
            
            # Perturbed forward step
            N_pert_curr = torch.clamp(N_curr + delta, min=0.0)
            N_pert_next = model.step_forward(N_pert_curr, all_c, all_m, delta_t=0.10)
            
            # Distance after 1 passage
            diff = (N_pert_next - N_next) * all_m
            d1 = torch.norm(diff, dim=-1, keepdim=True) + 1e-12
            
            # Expansion ratio
            expansion = d1 / eps
            lyap_sum += torch.log(expansion).squeeze(-1)
            
            # Renormalize perturbation back to eps
            delta = (diff / d1) * eps
            N_curr = N_next
            
        lyap_max = (lyap_sum / float(num_passages)).cpu().numpy()
        return lyap_max

if __name__ == '__main__':
    models = {
        'PhysODE_K2 (800 ep)': ('Data/Gore_Global_ODE_Model_PhysODE_K2_800ep.pth', 2),
        'PhysODE_K7 (400 ep)': ('Data/Gore_Global_ODE_Model_PhysODE_K7_400ep.pth', 7),
        'PhysODE_K2 (400 ep)': ('Data/Gore_Global_ODE_Model_PhysODE_BestLowK.pth', 2),
    }
    
    results = {}
    for name, (path, K) in models.items():
        if not os.path.exists(path):
            continue
        print(f"\n=======================================================")
        print(f"ANALYZING LYAPUNOV SPECTRUM FOR: {name}")
        print(f"=======================================================")
        ode_func = AdaptivePureConsumerResourceODE(N=48, K=K)
        model = HierarchicalPhysicalNeuralODE(ode_func, num_conditions=3, K=K).to(device)
        model.load_state_dict(torch.load(path, weights_only=False))
        
        # Test 1: Finite-time Lyapunov during experimental passages (Day 1 to 10, burn_in=0)
        lyap_exp_window = compute_lyapunov_spectrum(model, eps=1e-5, num_passages=10, burn_in=0)
        
        # Test 2: Attractor Lyapunov after burn-in (burn_in=10, num_passages=10)
        lyap_attractor = compute_lyapunov_spectrum(model, eps=1e-5, num_passages=10, burn_in=10)
        
        for mode, lyap in [("Experimental Window (Days 1-10)", lyap_exp_window), 
                           ("Asymptotic Attractor (Post-Burn-in)", lyap_attractor)]:
            auc = roc_auc_score(gt_labels, lyap)
            r_val, _ = pearsonr(lyap, gt_exp_cvs)
            rho_val, _ = spearmanr(lyap, gt_exp_cvs)
            
            print(f"\n--- Mode: {mode} ---")
            print(f"Lyapunov Range: min={lyap.min():.4f}, mean={lyap.mean():.4f}, median={np.median(lyap):.4f}, max={lyap.max():.4f}")
            print(f"Correlation with Gore Exp CV: Pearson r = {r_val:.4f}, Spearman rho = {rho_val:.4f}")
            print(f"Chaos ROC AUC vs Gore Ground Truth: {auc:.4f}")
            
            # Check sign-based classification: lambda > 0 (Chaos) vs lambda <= 0 (Stable)
            preds_sign = (lyap > 0)
            acc_sign = accuracy_score(gt_labels, preds_sign)
            bacc_sign = balanced_accuracy_score(gt_labels, preds_sign)
            f1_sign = f1_score(gt_labels, preds_sign, zero_division=0)
            cm_sign = confusion_matrix(gt_labels, preds_sign, labels=[False, True])
            print(f"Zero-Threshold Classification (lambda > 0 => Chaos):")
            print(f"  Overall Acc: {acc_sign*100:.1f}% | Balanced Acc: {bacc_sign*100:.1f}% | F1: {f1_sign:.3f}")
            print(f"  TN: {cm_sign[0,0]:2d} | FP: {cm_sign[0,1]:2d} | FN: {cm_sign[1,0]:2d} | TP: {cm_sign[1,1]:2d}")
            
            # Condition breakdown
            for nut in ['LN', 'MN', 'HN']:
                idxs = [i for i, key in enumerate(all_keys) if key[0] == nut]
                sub_lyap = lyap[idxs]
                sub_gt = gt_labels[idxs]
                sub_preds = (sub_lyap > 0)
                sub_acc = accuracy_score(sub_gt, sub_preds)
                print(f"    [{nut}] Mean lambda = {sub_lyap.mean():.4f} | Pred Chaos: {sub_preds.sum()}/33 | GT Chaos: {sub_gt.sum()}/33 | Acc: {sub_acc*100:.1f}%")
                
            # Optimal threshold on Lyapunov
            sorted_lyap = np.sort(np.unique(lyap))
            candidates = np.concatenate([[sorted_lyap[0] - 1e-4], (sorted_lyap[:-1] + sorted_lyap[1:]) / 2.0, [sorted_lyap[-1] + 1e-4]])
            best_bacc, best_t = 0, 0
            for t in candidates:
                preds = lyap >= t
                bacc = balanced_accuracy_score(gt_labels, preds)
                if bacc > best_bacc:
                    best_bacc = bacc
                    best_t = t
            preds_opt = lyap >= best_t
            cm_opt = confusion_matrix(gt_labels, preds_opt, labels=[False, True])
            print(f"Optimal Threshold Classification (T = {best_t:.4f}):")
            print(f"  Overall Acc: {accuracy_score(gt_labels, preds_opt)*100:.1f}% | Balanced Acc: {best_bacc*100:.1f}% | F1: {f1_score(gt_labels, preds_opt):.3f}")
            print(f"  TN: {cm_opt[0,0]:2d} | FP: {cm_opt[0,1]:2d} | FN: {cm_opt[1,0]:2d} | TP: {cm_opt[1,1]:2d}")

import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.mixture import GaussianMixture

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def assign_phylum(strain_name):
    s = strain_name.lower()
    if any(k in s for k in ['bacteroides', 'parabacteroides', 'prevotella', 'aliistipes', 'odoribacter', 'barnesiella']):
        return 'Bacteroidota'
    elif any(k in s for k in ['clostridium', 'ruminococcus', 'faecalibacterium', 'coprococcus', 'eubacterium', 
                             'roseburia', 'anaerostipes', 'dorea', 'blautia', 'lachnospira', 'streptococcus', 
                             'enterococcus', 'holdemanella']):
        return 'Bacillota (Firmicutes)'
    elif any(k in s for k in ['veillonella', 'dialister', 'megasphaera', 'phascolarctobacterium']):
        return 'Negativicutes'
    elif any(k in s for k in ['bifidobacterium', 'collinsella', 'eggerthella', 'actinomyces', 'adlercreutzia']):
        return 'Actinomycetota'
    elif any(k in s for k in ['escherichia', 'klebsiella', 'desulfovibrio', 'sutterella', 'bilophila', 'parasutterella']):
        return 'Pseudomonadota'
    else:
        return 'Other'

def analyze():
    data_path = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_DualTheta_MultiRun_10Seeds.pt')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please train ensemble first.")
        return

    data = torch.load(data_path, map_location='cpu', weights_only=False)
    runs = data['runs_data']
    strains = data['strains']
    clean_strains = [s.replace('-', ' ') for s in strains]
    phyla = [assign_phylum(s) for s in clean_strains]
    N_seeds = len(runs)
    N_strains = len(strains)

    print("="*80)
    print(f"BIMODALITY & TRADE-OFF STATISTICAL ANALYSIS (N={N_seeds} ENSEMBLE RUNS)")
    print("="*80)

    # 1. Gather distributions across seeds
    ratios_plain = np.array([r['ratio_plain'] for r in runs])   # (N_seeds, N_strains)
    ratios_mucin = np.array([r['ratio_mucin'] for r in runs])   # (N_seeds, N_strains)
    theta_ratios = np.array([r['theta_ratio'] for r in runs])   # (N_seeds, N_strains)

    mean_plain = np.mean(ratios_plain, axis=0)
    mean_mucin = np.mean(ratios_mucin, axis=0)
    mean_theta = np.mean(theta_ratios, axis=0)

    sem_plain = np.std(ratios_plain, axis=0, ddof=1) / np.sqrt(N_seeds)
    sem_mucin = np.std(ratios_mucin, axis=0, ddof=1) / np.sqrt(N_seeds)
    sem_theta = np.std(theta_ratios, axis=0, ddof=1) / np.sqrt(N_seeds)

    log_plain = np.log10(mean_plain)
    log_mucin = np.log10(mean_mucin)
    log_ratios_mucin_all = np.log10(ratios_mucin)
    mean_log_mucin = np.mean(log_ratios_mucin_all, axis=0)
    sem_log_mucin = np.std(log_ratios_mucin_all, axis=0, ddof=1) / np.sqrt(N_seeds)

    # 2. GMM on Mucin Agar
    X_mu = log_mucin.reshape(-1, 1)
    gmm_mu_1 = GaussianMixture(n_components=1, random_state=42).fit(X_mu)
    gmm_mu_2 = GaussianMixture(n_components=2, random_state=42).fit(X_mu)
    bic_mu_1 = gmm_mu_1.bic(X_mu)
    bic_mu_2 = gmm_mu_2.bic(X_mu)
    delta_bic_mu = bic_mu_2 - bic_mu_1

    # Sort components by mean
    idx_mu = np.argsort(gmm_mu_2.means_.flatten())
    weights_mu = gmm_mu_2.weights_[idx_mu]
    means_mu = gmm_mu_2.means_.flatten()[idx_mu]
    vars_mu = gmm_mu_2.covariances_.flatten()[idx_mu]

    print("\n--- 1. MUCIN AGAR ADHERENCE BIMODALITY ---")
    print(f"BIC (1 component): {bic_mu_1:.2f}")
    print(f"BIC (2 components): {bic_mu_2:.2f}")
    print(f"Delta BIC (2 vs 1): {delta_bic_mu:.2f} {'(Strong support for bimodality)' if delta_bic_mu < -10 else ''}")
    print(f"Component 1 (Planktonic): Weight = {weights_mu[0]*100:.1f}%, Mean = {means_mu[0]:.3f}, SD = {np.sqrt(vars_mu[0]):.3f}")
    print(f"Component 2 (Biofilm):    Weight = {weights_mu[1]*100:.1f}%, Mean = {means_mu[1]:.3f}, SD = {np.sqrt(vars_mu[1]):.3f}")

    # 3. GMM on Plain Agar
    X_pl = log_plain.reshape(-1, 1)
    gmm_pl_1 = GaussianMixture(n_components=1, random_state=42).fit(X_pl)
    gmm_pl_2 = GaussianMixture(n_components=2, random_state=42).fit(X_pl)
    bic_pl_1 = gmm_pl_1.bic(X_pl)
    bic_pl_2 = gmm_pl_2.bic(X_pl)
    delta_bic_pl = bic_pl_2 - bic_pl_1

    idx_pl = np.argsort(gmm_pl_2.means_.flatten())
    weights_pl = gmm_pl_2.weights_[idx_pl]
    means_pl = gmm_pl_2.means_.flatten()[idx_pl]
    vars_pl = gmm_pl_2.covariances_.flatten()[idx_pl]

    print("\n--- 2. PLAIN AGAR ADHERENCE BIMODALITY ---")
    print(f"BIC (1 component): {bic_pl_1:.2f}")
    print(f"BIC (2 components): {bic_pl_2:.2f}")
    print(f"Delta BIC (2 vs 1): {delta_bic_pl:.2f}")
    print(f"Component 1 (Planktonic): Weight = {weights_pl[0]*100:.1f}%, Mean = {means_pl[0]:.3f}, SD = {np.sqrt(vars_pl[0]):.3f}")
    print(f"Component 2 (Biofilm):    Weight = {weights_pl[1]*100:.1f}%, Mean = {means_pl[1]:.3f}, SD = {np.sqrt(vars_pl[1]):.3f}")

    # 4. Physical-Metabolic Trade-off Regression
    slope, intercept, r_val, p_val, std_err = linregress(mean_log_mucin, mean_theta)
    rho_val, p_rho_val = spearmanr(mean_log_mucin, mean_theta)

    print("\n--- 3. PHYSICAL-METABOLIC TRADE-OFF ('STICK OR SPECIALIZE') ---")
    print(f"Pearson r:  {r_val:+.4f} (p = {p_val:.3e})")
    print(f"Spearman rho: {rho_val:+.4f} (p = {p_rho_val:.3e})")
    print(f"Linear Fit: Metabolic_Shift = {slope:.4f} * log10(k_att/k_shed) + {intercept:.4f} (SE = {std_err:.4f})")
    print(f"R-squared:  {r_val**2:.4f}")

    # 5. Phylum Breakdown
    df = pd.DataFrame({
        'Strain': clean_strains,
        'Phylum': phyla,
        'Mean_Log_Adhesion_Mucin': mean_log_mucin,
        'SEM_Log_Adhesion_Mucin': sem_log_mucin,
        'Mean_Log_Adhesion_Plain': log_plain,
        'Mean_Theta_Ratio': mean_theta,
        'SEM_Theta_Ratio': sem_theta
    })

    print("\n--- 4. PHYLUM SUMMARY ---")
    phy_summary = df.groupby('Phylum').agg(
        Count=('Strain', 'count'),
        Mean_Adhesion_Mucin=('Mean_Log_Adhesion_Mucin', 'mean'),
        Mean_Metabolic_Shift=('Mean_Theta_Ratio', 'mean')
    ).sort_values(by='Count', ascending=False)
    print(phy_summary.to_string())

    out_csv = os.path.join(PROJECT_ROOT, 'Data', 'Pollard_Averaged_Tradeoff_Summary.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved trade-off table to {out_csv}")

if __name__ == '__main__':
    analyze()

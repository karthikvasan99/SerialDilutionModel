import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress

data_path = 'Data/Pollard_DualTheta_MultiRun_10Seeds.pt'
results = torch.load(data_path, map_location='cpu', weights_only=False)
runs_data = results['runs_data']
strains = results['strains']

ratios_mucin = np.array([r['ratio_mucin'] for r in runs_data])
theta_ratios = np.array([r['theta_ratio'] for r in runs_data])

log_ratios_mucin = np.log10(ratios_mucin)

mean_log_ratio = np.mean(log_ratios_mucin, axis=0)
sem_log_ratio = np.std(log_ratios_mucin, axis=0, ddof=1) / np.sqrt(len(runs_data))

mean_theta_ratio = np.mean(theta_ratios, axis=0)
sem_theta_ratio = np.std(theta_ratios, axis=0, ddof=1) / np.sqrt(len(runs_data))

r_mean, p_mean = pearsonr(mean_log_ratio, mean_theta_ratio)
rho_mean, p_rho_mean = spearmanr(mean_log_ratio, mean_theta_ratio)
slope, intercept, r_val, p_val, std_err = linregress(mean_log_ratio, mean_theta_ratio)

print(f"Tradeoff Pearson r = {r_mean:.4f}, p = {p_mean:.4e}")
print(f"Tradeoff Spearman rho = {rho_mean:.4f}, p = {p_rho_mean:.4e}")
print(f"Linear regression: slope = {slope:.4f}, intercept = {intercept:.4f}")

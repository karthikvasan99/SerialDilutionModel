import torch
import numpy as np
from sklearn.mixture import GaussianMixture

data_path = 'Data/Pollard_DualTheta_MultiRun_10Seeds.pt'
results = torch.load(data_path, map_location='cpu', weights_only=False)
runs = results['runs_data']
strains = results['strains']

ratios_mucin = np.array([r['ratio_mucin'] for r in runs])
ratios_plain = np.array([r['ratio_plain'] for r in runs])

mean_mucin = np.mean(ratios_mucin, axis=0)
mean_plain = np.mean(ratios_plain, axis=0)

log_mucin = np.log10(mean_mucin)
log_plain = np.log10(mean_plain)

gmm_m1 = GaussianMixture(n_components=1, random_state=42).fit(log_mucin.reshape(-1, 1))
gmm_m2 = GaussianMixture(n_components=2, random_state=42).fit(log_mucin.reshape(-1, 1))
gmm_p1 = GaussianMixture(n_components=1, random_state=42).fit(log_plain.reshape(-1, 1))
gmm_p2 = GaussianMixture(n_components=2, random_state=42).fit(log_plain.reshape(-1, 1))

print("Mucin GMM 1-comp BIC:", gmm_m1.bic(log_mucin.reshape(-1, 1)))
print("Mucin GMM 2-comp BIC:", gmm_m2.bic(log_mucin.reshape(-1, 1)))
print("Delta BIC Mucin:", gmm_m2.bic(log_mucin.reshape(-1, 1)) - gmm_m1.bic(log_mucin.reshape(-1, 1)))
print("Mucin Means:", 10**gmm_m2.means_.flatten(), "Weights:", gmm_m2.weights_)

print("Plain GMM 1-comp BIC:", gmm_p1.bic(log_plain.reshape(-1, 1)))
print("Plain GMM 2-comp BIC:", gmm_p2.bic(log_plain.reshape(-1, 1)))
print("Plain Means:", 10**gmm_p2.means_.flatten(), "Weights:", gmm_p2.weights_)

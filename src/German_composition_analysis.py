import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist, squareform
from matplotlib.lines import Line2D

# Set global matplotlib styles
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# 1. Load Data
DATA_PATH = 'Data/German_processed_data.pt'
data = torch.load(DATA_PATH, weights_only=False)
trajectories = data['trajectories']
strains = data['strains']
num_species = len(strains)

print(f"Loaded {len(trajectories)} trajectories with {num_species} species.")

# Helper to clean strain names
def clean_taxon_name(s):
    parts = s.split('|')
    species = parts[-1].strip() if len(parts) > 0 else 'Unknown'
    genus = parts[5].strip() if len(parts) > 5 and parts[5] != '' else ''
    fam = parts[4].strip() if len(parts) > 4 and parts[4] != '' else ''
    
    # Specific clean labels
    if 'Methanomicrobium' in s:
        return 'Methanomicrobium sp.'
    if 'Pyramidobacter sp. C12-8' in s:
        return 'Pyramidobacter sp. C12-8'
    if 'Pyramidobacter' in s:
        return 'Pyramidobacter sp. CG50-2'
    if 'Thermoplasmata' in s:
        return 'Thermoplasmata archaeon'
    if 'methanogenic archaeon' in s:
        return 'Methanogenic archaeon'
    if 'UBA7028' in s or ('Prevotellaceae' in s and 'UBA' in s):
        return 'Prevotellaceae UBA7028'
    if 'WCE2008' in s:
        return 'Bacteroidales WCE2008'
    if 'Streptococcus henryi' in s:
        return 'Streptococcus henryi'
    if 'Streptococcus vicugnae' in s:
        return 'Streptococcus vicugnae'
    if 'Streptococcus' in s:
        return 'Streptococcus sp.'
    if 'Anaerovibrio' in s:
        return 'Anaerovibrio spp.'
    if 'Kandleria' in s:
        return 'Kandleria spp.'
    if 'Bacteroides' in s:
        return 'Bacteroides spp.'
    if 'Denitrobacterium' in s:
        return 'Denitrobacterium detoxificans'
    if 'aminophilum' in s or 'Lachnoclostridium' in s:
        return '[Clostridium] aminophilum'
    if species != '' and species != 'Unknown':
        return species
    if genus != '':
        return f'{genus} sp.'
    if fam != '':
        return f'{fam} (fam)'
    return 'Other taxa'

cleaned_taxa = [clean_taxon_name(s) for s in strains]

# Identify top overall taxa
all_abund = []
for k, v in trajectories.items():
    all_abund.append(v.numpy())
all_abund = np.concatenate(all_abund, axis=0) # (72, 751)

# Group strains by cleaned taxon name
unique_clean = sorted(list(set(cleaned_taxa)))
taxon_to_idx = {t: [] for t in unique_clean}
for i, t in enumerate(cleaned_taxa):
    taxon_to_idx[t].append(i)

# Aggregate abundance matrix by clean taxa
agg_mat = np.zeros((72, len(unique_clean)))
for j, t in enumerate(unique_clean):
    agg_mat[:, j] = all_abund[:, taxon_to_idx[t]].sum(axis=1)

# Find top taxa by mean abundance
mean_taxa = agg_mat.mean(axis=0)
top_k = 10
top_indices = np.argsort(mean_taxa)[::-1][:top_k]
top_taxa_names = [unique_clean[idx] for idx in top_indices]
print("Top taxa identified:", top_taxa_names)


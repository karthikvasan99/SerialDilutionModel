import torch
import numpy as np
import os
import sys

# load data
data_path = 'Data/German_processed_data.pt'
data = torch.load(data_path, map_location='cpu', weights_only=False)
strains = data['strains']
trajectories = data['trajectories']

# The keys are tuples like ('BRRS', 'H2:CO2', 5.0), ('OSU', 'H2:CO2 + MeOH', 30.0), etc.
# Find intersection strains
t1_passages = torch.stack([seq[0] for seq in trajectories.values()], dim=0)

# The user mentioned "114 or how many every bacteria are present in the intersection without the mean abundance cutoff".
# Let's just find bacteria that are present in AT LEAST ONE BRRS T1 sample AND AT LEAST ONE OSU T1 sample?
# Or in ALL T1 samples?
all_t1_present = (t1_passages > 0).all(dim=0)
print(f"Number of strains present in ALL T1 samples: {all_t1_present.sum().item()}")

brrs_t1 = []
osu_t1 = []
for k, seq in trajectories.items():
    if 'BRRS' in k[0]:
        brrs_t1.append(seq[0])
    elif 'OSU' in k[0]:
        osu_t1.append(seq[0])

brrs_t1 = torch.stack(brrs_t1, dim=0)
osu_t1 = torch.stack(osu_t1, dim=0)

brrs_present = (brrs_t1 > 0).any(dim=0)
osu_present = (osu_t1 > 0).any(dim=0)
intersection = brrs_present & osu_present

print(f"Number of strains present in at least one BRRS and at least one OSU T1 sample: {intersection.sum().item()}")

brrs_all = (brrs_t1 > 0).all(dim=0)
osu_all = (osu_t1 > 0).all(dim=0)
intersection_all = brrs_all & osu_all
print(f"Number of strains present in ALL BRRS and ALL OSU T1 samples: {intersection_all.sum().item()}")

# Identify methanogens
for idx, s in enumerate(strains):
    if intersection_all[idx]:
        if 'Methano' in s or 'Thermoplasmata' in s or 'archaeon' in s:
            print(f"Methanogen found: {s}")


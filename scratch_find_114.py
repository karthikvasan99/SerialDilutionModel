import torch
import numpy as np

data_path = 'Data/German_processed_data.pt'
data = torch.load(data_path, map_location='cpu', weights_only=False)
strains = data['strains']
trajectories = data['trajectories']

# Let's count how many strains are >0 in BRRS T0 (or T1?)
# In BRRS, is it T0? The trajectories are 6 passages.
for k, seq in trajectories.items():
    print(f"{k}: shape {seq.shape}")

# Let's just find exactly 114
for thresh in [0.0, 1e-5, 1e-4, 1e-3]:
    # What if it's the intersection of the *entire* dataset for BRRS vs OSU?
    brrs_strains = set()
    osu_strains = set()
    for k, seq in trajectories.items():
        if 'BRRS' in k[0]:
            present = (seq > thresh).any(dim=0)
            for idx, p in enumerate(present):
                if p: brrs_strains.add(idx)
        elif 'OSU' in k[0]:
            present = (seq > thresh).any(dim=0)
            for idx, p in enumerate(present):
                if p: osu_strains.add(idx)
    
    inter = brrs_strains.intersection(osu_strains)
    print(f"Thresh {thresh}: BRRS {len(brrs_strains)}, OSU {len(osu_strains)}, Inter {len(inter)}")
    
    methanogens = 0
    for idx in inter:
        s = strains[idx]
        if 'Methano' in s or 'Thermoplasmata' in s or 'archaeon' in s:
            methanogens += 1
    print(f"  Methanogens in inter: {methanogens}")

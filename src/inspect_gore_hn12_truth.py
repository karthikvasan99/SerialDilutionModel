import sys, os
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

# Load experimental Gore data
df_exp = pd.read_excel('Data/gore_data.xlsx', sheet_name=None)
print("Sheet names:", df_exp.keys())

# Look at Community Metadata / Phase Classification in gore_data
for name in df_exp.keys():
    if 'comm' in name.lower() or 'phase' in name.lower() or 'metadata' in name.lower() or 'summary' in name.lower():
        print(f"Sheet {name}:")
        print(df_exp[name].head())
        
# Let's inspect the raw OD data for HN S=12
if 'OD' in df_exp:
    df_od = df_exp['OD']
    print("\nOD Sheet columns:", df_od.columns[:10])
    # Filter for HN and S=12
    hn12_od = df_od[(df_od['Nutrient'] == 'HN') & (df_od['Richness'] == 12)]
    print("\nHN S=12 OD:")
    print(hn12_od.head(16))


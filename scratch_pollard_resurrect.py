import pandas as pd
import numpy as np

df = pd.read_csv('/home/karthik/SerialDilutionModel/Data/SupplementaryDataS2.csv')
df = df[df['sampleType'] != 'inoculum']

# Convert to fractions
sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
if sums.mean() > 50:
    df['Relative_Abundance'] = df['Read_Fraction'] / 100.0
else:
    df['Relative_Abundance'] = df['Read_Fraction']

# Calculate mean abundance across all samples
sample_cols = ['sampleType', 'passage', 'replicate']
num_samples = len(df.drop_duplicates(subset=sample_cols))
mean_abundances = df.groupby('Strain_Name')['Relative_Abundance'].sum() / num_samples

# Filter to >= 1e-3
strains_keep = mean_abundances[mean_abundances >= 1e-3].index
print(f"Number of strains >= 1e-3 mean abundance: {len(strains_keep)}")

# Pivot to shape (sampleType+replicate, Strain, passage)
pivot_df = df.pivot_table(index=['sampleType', 'replicate'], 
                          columns=['Strain_Name', 'passage'], 
                          values='Relative_Abundance', 
                          fill_value=0.0)

# Reorder passages explicitly P1 to P6
passages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

resurrected_strains = set()

for strain in strains_keep:
    # Get the time series for this strain across all 45 trajectories
    if strain not in pivot_df.columns.get_level_values('Strain_Name'):
        continue
        
    ts_data = pivot_df[strain].reindex(columns=passages).values # shape (45, 6)
    
    for row in ts_data:
        # Check if there is a 0 followed by a >0 later
        # Specifically: >0 then 0 then >0 (disappear and reappear)
        # Or just 0 then >0
        
        has_zero = False
        has_resurrected = False
        
        for p in range(6):
            val = row[p]
            if val == 0:
                has_zero = True
            elif val > 0 and has_zero:
                # It was 0 at some point earlier, and now it's >0!
                has_resurrected = True
                break
                
        if has_resurrected:
            resurrected_strains.add(strain)
            break

print(f"\nNumber of bacteria that go to zero and reappear: {len(resurrected_strains)}")
for s in resurrected_strains:
    print(s)

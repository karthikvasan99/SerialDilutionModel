import pandas as pd
import numpy as np

df = pd.read_csv('/home/karthik/SerialDilutionModel/Data/SupplementaryDataS2.csv')

# Drop inoculum
df = df[df['sampleType'] != 'inoculum']

# Convert percentage to fraction if needed
sums = df.groupby(['sampleType', 'passage', 'replicate'])['Read_Fraction'].sum()
if sums.mean() > 50:
    df['Relative_Abundance'] = df['Read_Fraction'] / 100.0
else:
    df['Relative_Abundance'] = df['Read_Fraction']

# Pivot to get shape (samples, strains)
pivot_df = df.pivot_table(index=['sampleType', 'replicate', 'passage'], 
                          columns='Strain_Name', 
                          values='Relative_Abundance', 
                          fill_value=0.0)

print("Pivoted shape:", pivot_df.shape)
print("Index levels:", pivot_df.index.names)

# Check passage counts per replicate
passage_counts = pivot_df.groupby(['sampleType', 'replicate']).size()
print("\nPassage counts per replicate:")
print(passage_counts)

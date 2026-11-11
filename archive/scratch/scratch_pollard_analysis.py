import pandas as pd
import numpy as np

df = pd.read_csv('/home/karthik/SerialDilutionModel/Data/SupplementaryDataS2.csv')

# Unique samples
sample_cols = ['sampleType', 'passage', 'replicate']
num_samples = len(df.drop_duplicates(subset=sample_cols))

# Check if Read_Fraction sums to 100 or 1
sums = df.groupby(sample_cols)['Read_Fraction'].sum()
print("Average sum of Read_Fraction per sample:", sums.mean())

if sums.mean() > 50:
    # It's a percentage
    df['Relative_Abundance'] = df['Read_Fraction'] / 100.0
else:
    # It's a fraction
    df['Relative_Abundance'] = df['Read_Fraction']

# Environments
environments = df['sampleType'].unique()
print(f"\nNumber of environments present: {len(environments)}")
print(f"Environments: {list(environments)}")

# Mean relative abundance across all samples
# Sum of relative abundance for each strain divided by total number of samples
mean_abundances = df.groupby('Strain_Name')['Relative_Abundance'].sum() / num_samples

thresholds = [1e-5, 1e-4, 1e-3]
print("\nBacteria present at different mean relative abundance thresholds:")
for t in thresholds:
    count = (mean_abundances >= t).sum()
    print(f"Threshold {t}: {count} bacteria")

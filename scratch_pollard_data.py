import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('/home/karthik/SerialDilutionModel/Data/SupplementaryDataS2.csv')

print("Columns:", df.columns.tolist())
print("First few rows:")
print(df.head())

# Look for environment column
# It might be named something like 'Environment', 'Condition', 'Media', etc.
# We will just print columns to see

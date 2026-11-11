import pandas as pd
import numpy as np

df_hn_od = pd.read_excel('Data/gore_data.xlsx', sheet_name='HN Biomass (OD)')
print("HN Biomass (OD) columns:", df_hn_od.columns)
print("HN Biomass (OD) shape:", df_hn_od.shape)
print(df_hn_od.head(10))

# Compute experimental CV for passages 10 to 18 (or whatever late passages Gore used)
# Let's inspect the column names
for col in df_hn_od.columns:
    print(col, df_hn_od[col].dtype)


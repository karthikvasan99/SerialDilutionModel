import os
import pandas as pd

os.makedirs('Results', exist_ok=True)

# 1. Save Transferability Values
transfer_data = [
    {'Substrate': 'H2:CO2', 'Pressure_bar': 5.0, 'LOO_R2': 0.513},
    {'Substrate': 'H2:CO2', 'Pressure_bar': 30.0, 'LOO_R2': 0.698},
    {'Substrate': 'H2:CO2 + MeOH', 'Pressure_bar': 5.0, 'LOO_R2': 0.454},
    {'Substrate': 'H2:CO2 + MeOH', 'Pressure_bar': 30.0, 'LOO_R2': 0.479},
    {'Substrate': 'H2:CO2 + TMA', 'Pressure_bar': 5.0, 'LOO_R2': 0.371},
    {'Substrate': 'H2:CO2 + TMA', 'Pressure_bar': 30.0, 'LOO_R2': 0.532},
]
df_trans = pd.DataFrame(transfer_data)
df_trans.to_csv('Results/Grant_Transferability_Values.csv', index=False)

# 2. Save PCA Values
pca_data = [
    {'Substrate': 'H2:CO2', 'Pressure_bar': 5.0, 'PC1': 0.28, 'PC2': -1.08},
    {'Substrate': 'H2:CO2', 'Pressure_bar': 30.0, 'PC1': -0.82, 'PC2': 1.43},
    {'Substrate': 'H2:CO2 + MeOH', 'Pressure_bar': 5.0, 'PC1': 3.13, 'PC2': -1.92},
    {'Substrate': 'H2:CO2 + MeOH', 'Pressure_bar': 30.0, 'PC1': 2.81, 'PC2': 2.50},
    {'Substrate': 'H2:CO2 + TMA', 'Pressure_bar': 5.0, 'PC1': -3.48, 'PC2': 0.87},
    {'Substrate': 'H2:CO2 + TMA', 'Pressure_bar': 30.0, 'PC1': -1.92, 'PC2': -1.81},
]
df_pca = pd.DataFrame(pca_data)
df_pca.to_csv('Results/Grant_PCA_Values.csv', index=False)

print("Saved CSVs successfully.")

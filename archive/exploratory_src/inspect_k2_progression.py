import sys, os
import pandas as pd
for f in ['Results/Gore_PhysODE_K2_Progression.csv', 'Results/Gore_PhysODE_K2_ExtendedProgression.csv']:
    if os.path.exists(f):
        print(f"=== {f} ===")
        df = pd.read_csv(f)
        print(df.tail(10).to_string())

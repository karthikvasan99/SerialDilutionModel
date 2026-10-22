import sys, os
sys.path.insert(0, '.')
import pandas as pd
import glob

# Check any CSV in Results/
results_files = glob.glob('Results/*.csv')
for f in results_files:
    print(f"--- File: {f} ---")
    try:
        df = pd.read_csv(f)
        print("Columns:", df.columns.tolist())
        print(df.tail(5))
    except Exception as e:
        print("Error:", e)


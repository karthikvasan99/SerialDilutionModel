import os
import sys
import torch
import numpy as np
import pandas as pd
from test_learnable_b0_loocv import evaluate_loocv, PROJECT_ROOT

print("Testing K in [3, 4, 5] with learnable condition-specific b0 (c_w=0.70, reg_alpha=0.001)...", flush=True)

results = []
for k in [3, 4, 5]:
    res = evaluate_loocv(b0_mode='learnable_condition', epochs=200, lr=0.015, K=k, c_w=0.70, reg_alpha=0.001, reg_w=0.001)
    results.append(res)
    print(f"K={k:d} | Comb Val KL: {res['mean_val_kl']:.4f} +/- {res['std_val_kl']:.4f} | NoCar: {res['val_kl_no']:.4f} | Mucin: {res['val_kl_mu']:.4f} | Learned b0: (liq={res['mean_b0_liq']:.2f}, sup={res['mean_b0_sup']:.2f})", flush=True)

df_out = pd.DataFrame(results)
csv_path = os.path.join(PROJECT_ROOT, 'Data', 'Learnable_b0_K_Sweep.csv')
df_out.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}", flush=True)

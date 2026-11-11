import sys, os
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

from src.scratch_k2_ultimate_f1 import X, y, feature_names

# 8 Core Features
feats = ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'S', 'c_ln', 'c_hn']
X_sub = X[:, [feature_names.index(f) for f in feats]]

scaler = StandardScaler()
X_s = scaler.fit_transform(X_sub)

X_t = torch.tensor(X_s, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

# Model 1: Trained with BCE Loss (Log-Loss)
torch.manual_seed(42)
linear_bce = nn.Linear(len(feats), 1)
opt_bce = torch.optim.Adam(linear_bce.parameters(), lr=0.03)
crit_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([77.0 / 22.0])) # balanced BCE

for ep in range(300):
    opt_bce.zero_grad()
    logits = linear_bce(X_t).squeeze(-1)
    loss = crit_bce(logits, y_t)
    loss.backward()
    opt_bce.step()

# Model 2: Trained with Soft-F1 Loss (Fixed T=0.5, no extra parameter)
class SoftF1Loss(nn.Module):
    def forward(self, probs, y_true):
        tp = (probs * y_true).sum()
        fp = (probs * (1 - y_true)).sum()
        fn = ((1 - probs) * y_true).sum()
        return 1.0 - (2 * tp / (2 * tp + fp + fn + 1e-7))

torch.manual_seed(42)
linear_f1 = nn.Linear(len(feats), 1)
opt_f1 = torch.optim.Adam(linear_f1.parameters(), lr=0.03)
crit_f1 = SoftF1Loss()

for ep in range(300):
    opt_f1.zero_grad()
    logits = linear_f1(X_t).squeeze(-1)
    probs = torch.sigmoid(logits)
    loss = crit_f1(probs, y_t)
    loss.backward()
    opt_f1.step()

# Compare Weights & Intercepts
w_bce = linear_bce.weight.detach().numpy()[0]
b_bce = linear_bce.bias.detach().numpy()[0]

w_f1 = linear_f1.weight.detach().numpy()[0]
b_f1 = linear_f1.bias.detach().numpy()[0]

df_comp = pd.DataFrame({
    'Feature': feats,
    'Weight_BCE': w_bce,
    'Weight_SoftF1': w_f1,
    'Ratio (F1/BCE)': w_f1 / (w_bce + 1e-6)
})

print("="*75)
print("WEIGHT COMPARISON: BCE (LOG-LOSS) VS SOFT-F1 LOSS (SAME T=0.5)")
print("="*75)
print(f"Intercept b (BCE):     {b_bce:+.4f}")
print(f"Intercept b (Soft-F1): {b_f1:+.4f}")
print(df_comp.to_string(index=False))

# Evaluate both models strictly at T=0.50
with torch.no_grad():
    p_bce = torch.sigmoid(linear_bce(X_t).squeeze(-1)).numpy()
    p_f1 = torch.sigmoid(linear_f1(X_t).squeeze(-1)).numpy()

pred_bce = (p_bce >= 0.5).astype(int)
pred_f1 = (p_f1 >= 0.5).astype(int)

cm_bce = confusion_matrix(y, pred_bce)
cm_f1 = confusion_matrix(y, pred_f1)

print("\n" + "="*75)
print("PERFORMANCE BENCHMARK AT FIXED DEFAULT T = 0.50")
print("="*75)
print(f"BCE Model (T=0.50):     F1 = {f1_score(y, pred_bce):.4f} | Rec = {recall_score(y, pred_bce)*100:.1f}% | Prec = {precision_score(y, pred_bce)*100:.1f}% | TP={cm_bce[1,1]} FP={cm_bce[0,1]}")
print(f"Soft-F1 Model (T=0.50): F1 = {f1_score(y, pred_f1):.4f} | Rec = {recall_score(y, pred_f1)*100:.1f}% | Prec = {precision_score(y, pred_f1)*100:.1f}% | TP={cm_f1[1,1]} FP={cm_f1[0,1]}")


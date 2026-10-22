import sys, os
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix

from src.scratch_k2_ultimate_f1 import X, y, feature_names

# Features: 10 core features
feats_10 = ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S', 'c_ln', 'c_hn']
X_10 = X[:, [feature_names.index(f) for f in feats_10]]

s = StandardScaler()
X_s = s.fit_transform(X_10)

X_t = torch.tensor(X_s, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

# Architecture where Threshold T is an EXPLICIT trainable nn.Parameter!
class DifferentiableThresholdModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        # Learnable threshold T initialized at 0.5
        self.T_raw = nn.Parameter(torch.tensor([0.0])) # sigmoid(T_raw) in (0, 1)
        
    def forward(self, x, tau=0.1):
        logits = self.linear(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        T = torch.sigmoid(self.T_raw)
        # Smooth surrogate step function with temperature tau
        soft_binary = torch.sigmoid((probs - T) / tau)
        return probs, T, soft_binary

def soft_f1_loss(soft_binary, y_true):
    tp = (soft_binary * y_true).sum()
    fp = (soft_binary * (1 - y_true)).sum()
    fn = ((1 - soft_binary) * y_true).sum()
    return 1.0 - (2 * tp / (2 * tp + fp + fn + 1e-7))

model = DifferentiableThresholdModel(in_dim=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

print("--- TRAINING LOGISTIC REGRESSION WITH GRADIENT ON THRESHOLD T ---")
for epoch in range(200):
    optimizer.zero_grad()
    probs, T, soft_binary = model(X_t, tau=0.08)
    loss = soft_f1_loss(soft_binary, y_t)
    loss.backward()
    
    # Check gradient on T_raw
    if epoch % 40 == 0 or epoch == 199:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | T value: {T.item():.4f} | Grad on T: {model.T_raw.grad.item():+.6f}")
    optimizer.step()

# Evaluate
model.eval()
with torch.no_grad():
    probs, learned_T, _ = model(X_t)
    final_preds = (probs >= learned_T).int().numpy()
    
f1 = f1_score(y, final_preds)
cm = confusion_matrix(y, final_preds)
print("\nFinal Learned Results:")
print(f"Learned Optimal T: {learned_T.item():.4f}")
print(f"Final F1-Score: {f1:.4f}")
print(f"Confusion Matrix: TP={cm[1,1]} FP={cm[0,1]} FN={cm[1,0]} TN={cm[0,0]}")


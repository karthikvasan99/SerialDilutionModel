import sys, os
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, precision_recall_curve

from src.scratch_k2_ultimate_f1 import X, y, feature_names, all_keys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SoftF1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        # y_pred is sigmoid probabilities in [0, 1]
        # y_true is binary 0 or 1
        tp = (y_pred * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()
        fn = ((1 - y_pred) * y_true).sum()
        
        soft_f1 = 2 * tp / (2 * tp + fp + fn + self.epsilon)
        return 1.0 - soft_f1

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(y))

# Train across 5 folds
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    
    y_train_t = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y[val_idx], dtype=torch.float32, device=device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    
    torch.manual_seed(42 + fold)
    model = SimpleClassifier(in_dim=X.shape[1], hidden=32, dropout=0.2).to(device)
    criterion = SoftF1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    
    best_val_f1 = 0.0
    best_val_preds = None
    
    for epoch in range(250):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            v_preds = model(X_val_t).cpu().numpy()
            # check f1 at standard 0.5
            f1 = f1_score(y[val_idx], (v_preds >= 0.5).astype(int), zero_division=0)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_val_preds = v_preds
                
    oof_probs[val_idx] = best_val_preds if best_val_preds is not None else v_preds

precisions, recalls, thresholds = precision_recall_curve(y, oof_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
best_idx = np.argmax(f1_scores)
best_f1 = f1_scores[best_idx]
best_t = thresholds[best_idx]
y_pred = (oof_probs >= best_t).astype(int)
cm = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)

print(f"\n=======================================================")
print(f"PYTORCH SOFT-F1 NEURAL NETWORK (STRICTLY K=2)")
print(f"=======================================================")
print(f"Best Out-Of-Fold F1: {best_f1:.4f}")
print(f"Precision: {precisions[best_idx]:.3f} | Recall: {recalls[best_idx]:.3f} | Accuracy: {acc*100:.1f}%")
print(f"Confusion Matrix:")
print(f"  TN: {cm[0,0]:2d} | FP: {cm[0,1]:2d}")
print(f"  FN: {cm[1,0]:2d} | TP: {cm[1,1]:2d}")

for nut in ['LN', 'MN', 'HN']:
    idxs = [i for i, k in enumerate(all_keys) if k[0] == nut]
    cm_n = confusion_matrix(y[idxs], y_pred[idxs], labels=[0, 1])
    acc_n = accuracy_score(y[idxs], y_pred[idxs])
    print(f"  [{nut}] Acc: {acc_n*100:.1f}% | TN: {cm_n[0,0]:2d}, FP: {cm_n[0,1]:2d}, FN: {cm_n[1,0]:2d}, TP: {cm_n[1,1]:2d}")

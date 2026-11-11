import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix, precision_recall_curve

from src.scratch_k2_ultimate_f1 import X, y, feature_names, all_keys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

feats_10 = [
    'mean_tv_path', 'mean_poincare_shift', 'mean_overlap',
    'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S',
    'surv_count', 'extinct_ratio'
]
f_idx = [feature_names.index(f) for f in feats_10]
X_sub = X[:, f_idx]

class SmallMLP(nn.Module):
    def __init__(self, in_dim=10, hidden=3, act='tanh', dropout=0.0):
        super().__init__()
        if act == 'tanh':
            activation = nn.Tanh()
        elif act == 'silu':
            activation = nn.SiLU()
        else:
            activation = nn.ReLU()
            
        layers = [nn.Linear(in_dim, hidden), activation]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

loo = LeaveOneOut()

def evaluate_loocv(hidden=3, act='tanh', dropout=0.0, lr=0.01, wd=1e-2, epochs=200):
    oof_probs = np.zeros(len(y))
    pos_weight = torch.tensor([77.0 / 22.0], device=device)
    
    for tr, te in loo.split(X_sub):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(X_sub[tr])
        x_te = scaler.transform(X_sub[te])
        
        torch.manual_seed(42)
        model = SmallMLP(in_dim=10, hidden=hidden, act=act, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
        x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y[tr], dtype=torch.float32, device=device)
        x_te_t = torch.tensor(x_te, dtype=torch.float32, device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        for ep in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model.net(x_tr_t).squeeze(-1)
            loss = bce(logits, y_tr_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            oof_probs[te[0]] = model(x_te_t).cpu().item()
            
    preds = (oof_probs >= 0.5).astype(int)
    f1 = f1_score(y, preds)
    acc = accuracy_score(y, preds)
    rec = recall_score(y, preds)
    prec = precision_score(y, preds)
    auc = roc_auc_score(y, oof_probs)
    cm = confusion_matrix(y, preds)
    
    p_c, r_c, th_c = precision_recall_curve(y, oof_probs)
    f1_c = 2 * (p_c * r_c) / (p_c + r_c + 1e-12)
    b_idx = np.argmax(f1_c)
    
    params = count_parameters(SmallMLP(10, hidden, act, dropout))
    
    return {
        'hidden': hidden, 'act': act, 'dropout': dropout, 'lr': lr, 'wd': wd, 'params': params,
        'f1_05': f1, 'acc_05': acc, 'rec_05': rec, 'prec_05': prec, 'auc': auc, 'cm': cm,
        'opt_f1': f1_c[b_idx], 'opt_t': th_c[b_idx] if b_idx < len(th_c) else 0.5,
        'oof_probs': oof_probs, 'preds': preds
    }

print("="*85)
print("GRID SEARCH: 2-LAYER NEURAL NETWORKS ON 10 PURE PHYSICAL FEATURES (ZERO MEDIA)")
print("="*85)

results = []
for h in [2, 3, 4, 5]:
    for act in ['tanh', 'silu']:
        for wd in [1e-3, 1e-2, 5e-2]:
            for lr in [0.005, 0.01]:
                res = evaluate_loocv(hidden=h, act=act, dropout=0.0, lr=lr, wd=wd, epochs=200)
                results.append(res)

results.sort(key=lambda x: (x['f1_05'], x['auc']), reverse=True)

print(f"{'Hidden':<6} {'Act':<5} {'WD':<7} {'LR':<6} {'Params':<6} | {'F1_05':<7} {'Acc':<6} {'Rec':<6} {'Prec':<6} {'AUC':<7} | {'TP':<3} {'FP':<3} {'FN':<3} {'TN':<3} (Opt F1 at T*)")
print("-" * 90)
for r in results[:10]:
    cm = r['cm']
    print(f"{r['hidden']:<6} {r['act']:<5} {r['wd']:<7.1e} {r['lr']:<6.3f} {r['params']:<6} | {r['f1_05']:<7.4f} {r['acc_05']*100:<5.1f}% {r['rec_05']*100:<5.1f}% {r['prec_05']*100:<5.1f}% {r['auc']:<7.4f} | {cm[1,1]:<3} {cm[0,1]:<3} {cm[1,0]:<3} {cm[0,0]:<3} ({r['opt_f1']:.4f} at T={r['opt_t']:.3f})")

best = results[0]
print("\n" + "="*85)
print(f"CHAMPION NEURAL NET: Hidden={best['hidden']}, Act={best['act']}, WD={best['wd']}, LR={best['lr']} ({best['params']} params)")
print("="*85)

df_eval = pd.DataFrame({
    'Nutrient': [k[0] for k in all_keys],
    'S': [k[1] for k in all_keys],
    'Comm': [k[2] for k in all_keys],
    'y_true': y,
    'oof_prob': best['oof_probs'],
    'oof_pred': best['preds']
})

print("\n--- MEDIA BY MEDIA CONFUSION MATRIX (LOOCV at T=0.50) ---")
for nut in ['LN', 'MN', 'HN']:
    sub = df_eval[df_eval.Nutrient == nut]
    cm_sub = confusion_matrix(sub.y_true, sub.oof_pred, labels=[0, 1])
    print(f"{nut} Media ({len(sub)} comms): TP={cm_sub[1,1]:2d} | TN={cm_sub[0,0]:2d} | FP={cm_sub[0,1]:2d} | FN={cm_sub[1,0]:2d} | Acc={accuracy_score(sub.y_true, sub.oof_pred)*100:.1f}% | Recall={recall_score(sub.y_true, sub.oof_pred, zero_division=0)*100:.1f}%")


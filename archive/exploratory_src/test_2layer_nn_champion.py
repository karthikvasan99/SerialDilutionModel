import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)

from src.scratch_k2_ultimate_f1 import X, y, feature_names, all_keys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 10 core features
selected_features = [
    'mean_tv_path', 'mean_poincare_shift', 'mean_overlap',
    'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S',
    'c_ln', 'c_hn'
]
f_idx = [feature_names.index(f) for f in selected_features]
X_sub = X[:, f_idx]

class SoftF1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        tp = (y_pred * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()
        fn = ((1 - y_pred) * y_true).sum()
        return 1.0 - (2 * tp / (2 * tp + fp + fn + self.epsilon))

class TinyMLP(nn.Module):
    def __init__(self, in_dim=10, hidden=5, act='tanh'):
        super().__init__()
        activation = nn.Tanh() if act == 'tanh' else nn.SiLU()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            activation,
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

loo = LeaveOneOut()

def evaluate_nn_loocv(hidden=5, act='tanh', loss_type='bce', epochs=250, lr=0.02, wd=1e-3):
    oof_probs = np.zeros(len(y))
    criterion = SoftF1Loss() if loss_type == 'softf1' else None
    pos_weight = torch.tensor([77.0 / 22.0], device=device)
    
    for tr, te in loo.split(X_sub):
        s = StandardScaler()
        x_tr = s.fit_transform(X_sub[tr])
        x_te = s.transform(X_sub[te])
        
        torch.manual_seed(42)
        model = TinyMLP(in_dim=10, hidden=hidden, act=act).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
        x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y[tr], dtype=torch.float32, device=device)
        x_te_t = torch.tensor(x_te, dtype=torch.float32, device=device)
        
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        for ep in range(epochs):
            model.train()
            optimizer.zero_grad()
            if loss_type == 'softf1':
                preds = model(x_tr_t)
                loss = criterion(preds, y_tr_t)
            else:
                logits = model.net(x_tr_t).squeeze(-1)
                loss = bce_loss_fn(logits, y_tr_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            oof_probs[te[0]] = model(x_te_t).cpu().item()
            
    preds_05 = (oof_probs >= 0.5).astype(int)
    cm = confusion_matrix(y, preds_05)
    f1 = f1_score(y, preds_05)
    acc = accuracy_score(y, preds_05)
    rec = recall_score(y, preds_05)
    prec = precision_score(y, preds_05)
    auc = roc_auc_score(y, oof_probs)
    
    p_c, r_c, th_c = precision_recall_curve(y, oof_probs)
    f1_c = 2 * (p_c * r_c) / (p_c + r_c + 1e-12)
    b_idx = np.argmax(f1_c)
    
    return {
        'hidden': hidden,
        'act': act,
        'loss_type': loss_type,
        'params': count_parameters(TinyMLP(10, hidden, act)),
        'f1_05': f1,
        'acc_05': acc,
        'rec_05': rec,
        'prec_05': prec,
        'auc': auc,
        'cm': cm,
        'opt_f1': f1_c[b_idx],
        'opt_t': th_c[b_idx] if b_idx < len(th_c) else 0.5,
        'oof_probs': oof_probs,
        'preds_05': preds_05
    }

print("="*85)
print("COMPARING 2-LAYER NEURAL NETWORKS (LOOCV BENCHMARK ON 10 FEATURES)")
print("="*85)

results = []
for h in [3, 4, 5, 6]:
    for loss in ['bce', 'softf1']:
        for act in ['tanh', 'silu']:
            res = evaluate_nn_loocv(hidden=h, act=act, loss_type=loss, epochs=250, lr=0.02)
            cm = res['cm']
            results.append(res)
            print(f"MLP (h={h:d}, {act:4s}, {loss:6s}) [{res['params']:2d} params] | F1_05={res['f1_05']:.4f} | Acc={res['acc_05']*100:.1f}% | Rec={res['rec_05']*100:.1f}% | Prec={res['prec_05']*100:.1f}% | AUC={res['auc']:.4f} | TP:{cm[1,1]:2d} FP:{cm[0,1]:2d} FN:{cm[1,0]:2d} (Opt F1={res['opt_f1']:.4f} at T={res['opt_t']:.3f})")

# Pick the best model and inspect its media-by-media breakdown
df_res = pd.DataFrame(results)
best_res = sorted(results, key=lambda x: (x['f1_05'], x['auc']), reverse=True)[0]

print("\n" + "="*85)
print(f"CHAMPION NEURAL NET: Hidden={best_res['hidden']}, Act={best_res['act']}, Loss={best_res['loss_type']} ({best_res['params']} params)")
print("="*85)

df_eval = pd.DataFrame({
    'Nutrient': [k[0] for k in all_keys],
    'S': [k[1] for k in all_keys],
    'Comm': [k[2] for k in all_keys],
    'y_true': y,
    'oof_prob': best_res['oof_probs'],
    'oof_pred': best_res['preds_05']
})

print("\n--- MEDIA BY MEDIA CONFUSION MATRIX (LOOCV at T=0.50) ---")
for nut in ['LN', 'MN', 'HN']:
    sub = df_eval[df_eval.Nutrient == nut]
    cm_sub = confusion_matrix(sub.y_true, sub.oof_pred, labels=[0, 1])
    print(f"{nut} Media ({len(sub)} comms): TP={cm_sub[1,1]:2d} | TN={cm_sub[0,0]:2d} | FP={cm_sub[0,1]:2d} | FN={cm_sub[1,0]:2d} | Acc={accuracy_score(sub.y_true, sub.oof_pred)*100:.1f}% | Recall={recall_score(sub.y_true, sub.oof_pred, zero_division=0)*100:.1f}%")


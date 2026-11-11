import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, confusion_matrix

from src.scratch_k2_ultimate_f1 import X, y, feature_names

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

feature_idx = {name: i for i, name in enumerate(feature_names)}

candidate_subsets = {
    "Dynamics (4 feat)": ['mean_path_diff', 'bio_cv', 'end_cv', 'max_sp_cv'],
    "Dynamics + HN (5 feat)": ['mean_path_diff', 'bio_cv', 'end_cv', 'max_sp_cv', 'c_hn'],
    "Full Minimal (6 feat)": ['mean_path_diff', 'bio_cv', 'end_cv', 'max_sp_cv', 'c_ln', 'c_hn'],
    "Path + Bio + Media (5 feat)": ['mean_path_diff', 'bio_cv', 'bio_final', 'c_ln', 'c_hn'],
    "TV Path + Bio CV + Media (4 feat)": ['mean_tv_path', 'bio_cv', 'c_ln', 'c_hn'],
    "Core 7 (7 feat)": ['bio_cv', 'bio_final', 'max_sp_cv', 'mean_path_diff', 'end_cv', 'c_ln', 'c_hn']
}

class SoftF1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        tp = (y_pred * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()
        fn = ((1 - y_pred) * y_true).sum()
        return 1.0 - (2 * tp / (2 * tp + fp + fn + self.epsilon))

class TinyMLP2Layer(nn.Module):
    """2-layer MLP: in_dim -> hidden -> 1"""
    def __init__(self, in_dim, hidden=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

class TinyMLP3Layer(nn.Module):
    """3-layer MLP: in_dim -> h1 -> h2 -> 1"""
    def __init__(self, in_dim, h1=8, h2=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_eval_nn(model_cls, kwargs, X_sub, y, loss_type='softf1', epochs=200, lr=0.02, wd=1e-3):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    oof_probs = np.zeros(len(y))
    oof_preds_nested = np.zeros(len(y))
    
    criterion = SoftF1Loss() if loss_type == 'softf1' else nn.BCELoss(weight=None)
    
    for train_idx, val_idx in outer_cv.split(X_sub, y):
        X_tr, y_tr = X_sub[train_idx], y[train_idx]
        X_va, y_va = X_sub[val_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        
        # Inner CV for threshold
        inner_probs = np.zeros(len(y_tr))
        for in_tr, in_va in inner_cv.split(X_tr, y_tr):
            in_s = StandardScaler()
            X_in_tr = in_s.fit_transform(X_tr[in_tr])
            X_in_va = in_s.transform(X_tr[in_va])
            
            torch.manual_seed(42)
            m_in = model_cls(in_dim=X_sub.shape[1], **kwargs).to(device)
            opt_in = torch.optim.AdamW(m_in.parameters(), lr=lr, weight_decay=wd)
            
            x_in_tr_t = torch.tensor(X_in_tr, dtype=torch.float32, device=device)
            y_in_tr_t = torch.tensor(y_tr[in_tr], dtype=torch.float32, device=device)
            x_in_va_t = torch.tensor(X_in_va, dtype=torch.float32, device=device)
            
            pos_weight = (len(y_tr[in_tr]) - y_tr[in_tr].sum()) / (y_tr[in_tr].sum() + 1e-6)
            bce_inner = nn.BCELoss()
            
            for ep in range(epochs):
                m_in.train()
                opt_in.zero_grad()
                pred = m_in(x_in_tr_t)
                loss = criterion(pred, y_in_tr_t) if loss_type == 'softf1' else bce_inner(pred, y_in_tr_t)
                loss.backward()
                opt_in.step()
                
            m_in.eval()
            with torch.no_grad():
                inner_probs[in_va] = m_in(x_in_va_t).cpu().numpy()
                
        p, r, th = precision_recall_curve(y_tr, inner_probs)
        f1_inner = 2 * (p * r) / (p + r + 1e-12)
        best_t = th[np.argmax(f1_inner)] if len(th) > 0 and np.argmax(f1_inner) < len(th) else 0.5
        
        # Outer train
        torch.manual_seed(42)
        m_out = model_cls(in_dim=X_sub.shape[1], **kwargs).to(device)
        opt_out = torch.optim.AdamW(m_out.parameters(), lr=lr, weight_decay=wd)
        x_tr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
        x_va_t = torch.tensor(X_va_s, dtype=torch.float32, device=device)
        
        for ep in range(epochs):
            m_out.train()
            opt_out.zero_grad()
            pred = m_out(x_tr_t)
            loss = criterion(pred, y_tr_t) if loss_type == 'softf1' else bce_inner(pred, y_tr_t)
            loss.backward()
            opt_out.step()
            
        m_out.eval()
        with torch.no_grad():
            pr_va = m_out(x_va_t).cpu().numpy()
            oof_probs[val_idx] = pr_va
            oof_preds_nested[val_idx] = (pr_va >= best_t).astype(int)
            
    p_oof, r_oof, th_oof = precision_recall_curve(y, oof_probs)
    f1_curve = 2 * (p_oof * r_oof) / (p_oof + r_oof + 1e-12)
    
    return {
        'nested_f1': f1_score(y, oof_preds_nested),
        'nested_acc': accuracy_score(y, oof_preds_nested),
        'nested_rec': recall_score(y, oof_preds_nested),
        'nested_prec': precision_score(y, oof_preds_nested),
        'f1_t05': f1_score(y, (oof_probs >= 0.5).astype(int)),
        'oracle_f1': np.max(f1_curve),
        'oracle_rec': r_oof[np.argmax(f1_curve)],
        'oracle_prec': p_oof[np.argmax(f1_curve)],
        'cm': confusion_matrix(y, oof_preds_nested)
    }

print("="*80)
print("TINY NEURAL NETWORKS (2-LAYER & 3-LAYER) BENCHMARK")
print("="*80)

results = []
for name, feats in candidate_subsets.items():
    idxs = [feature_idx[f] for f in feats]
    X_sub = X[:, idxs]
    
    # 2-layer MLP (hidden=6)
    m2 = TinyMLP2Layer(in_dim=len(feats), hidden=6)
    p2 = count_parameters(m2)
    res2 = train_and_eval_nn(TinyMLP2Layer, {'hidden': 6}, X_sub, y, loss_type='softf1', epochs=250, lr=0.015)
    results.append({
        'Model': f"2-Layer MLP (h=6) [{name}]",
        'Params': p2,
        'Nested F1': res2['nested_f1'],
        'Nested Acc': res2['nested_acc'],
        'Nested Rec': res2['nested_rec'],
        'Nested Prec': res2['nested_prec'],
        'T=0.5 F1': res2['f1_t05'],
        'Oracle F1': res2['oracle_f1'],
        'TP': res2['cm'][1,1], 'FP': res2['cm'][0,1], 'FN': res2['cm'][1,0], 'TN': res2['cm'][0,0]
    })
    
    # 3-layer MLP (h1=6, h2=3)
    m3 = TinyMLP3Layer(in_dim=len(feats), h1=6, h2=3)
    p3 = count_parameters(m3)
    res3 = train_and_eval_nn(TinyMLP3Layer, {'h1': 6, 'h2': 3}, X_sub, y, loss_type='softf1', epochs=250, lr=0.015)
    results.append({
        'Model': f"3-Layer MLP (6-3) [{name}]",
        'Params': p3,
        'Nested F1': res3['nested_f1'],
        'Nested Acc': res3['nested_acc'],
        'Nested Rec': res3['nested_rec'],
        'Nested Prec': res3['nested_prec'],
        'T=0.5 F1': res3['f1_t05'],
        'Oracle F1': res3['oracle_f1'],
        'TP': res3['cm'][1,1], 'FP': res3['cm'][0,1], 'FN': res3['cm'][1,0], 'TN': res3['cm'][0,0]
    })

df_nn = pd.DataFrame(results)
print(df_nn.to_string(index=False))


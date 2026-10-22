import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, confusion_matrix, roc_auc_score

from src.scratch_k2_ultimate_f1 import X, y, feature_names

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Let's test the 7 features selected by L1_C0.2 and L1_C0.5:
sets = {
    "L1_C0.2 (7 feats)": ['mean_tv_path', 'S', 'bio_cv', 'c_hn', 'max_sp_cv', 'mean_poincare_shift', 'mean_overlap'],
    "L1_C0.5 (7 feats)": ['S', 'mean_tv_path', 'c_ln', 'mean_overlap', 'c_hn', 'mean_poincare_shift', 'max_sp_cv'],
    "Top Physical (6 feats)": ['mean_tv_path', 'mean_path_diff', 'bio_cv', 'bio_final', 'c_ln', 'c_hn'],
    "All 26 feats": feature_names
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

class SoftF1Logistic(nn.Module):
    """Logistic model trained directly with Soft-F1 loss!"""
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)

class TinySoftF1MLP(nn.Module):
    """2-layer MLP (in -> 6 -> 1) trained with Soft-F1 loss"""
    def __init__(self, in_dim, hidden=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

def evaluate_torch_model(model_cls, feats, epochs=300, lr=0.03, wd=1e-4):
    f_idx = [feature_names.index(f) for f in feats]
    X_sub = X[:, f_idx]
    
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    oof_probs = np.zeros(len(y))
    oof_preds_nested = np.zeros(len(y))
    
    criterion = SoftF1Loss()
    
    for train_idx, val_idx in outer_cv.split(X_sub, y):
        X_tr, y_tr = X_sub[train_idx], y[train_idx]
        X_va, y_va = X_sub[val_idx], y[val_idx]
        
        s = StandardScaler()
        X_tr_s = s.fit_transform(X_tr)
        X_va_s = s.transform(X_va)
        
        inner_probs = np.zeros(len(y_tr))
        for in_tr, in_va in inner_cv.split(X_tr, y_tr):
            in_s = StandardScaler()
            X_in_tr = in_s.fit_transform(X_tr[in_tr])
            X_in_va = in_s.transform(X_tr[in_va])
            
            torch.manual_seed(42)
            m_in = model_cls(in_dim=len(feats)).to(device)
            opt_in = torch.optim.AdamW(m_in.parameters(), lr=lr, weight_decay=wd)
            
            x_in_tr_t = torch.tensor(X_in_tr, dtype=torch.float32, device=device)
            y_in_tr_t = torch.tensor(y_tr[in_tr], dtype=torch.float32, device=device)
            x_in_va_t = torch.tensor(X_in_va, dtype=torch.float32, device=device)
            
            for ep in range(epochs):
                m_in.train()
                opt_in.zero_grad()
                pred = m_in(x_in_tr_t)
                loss = criterion(pred, y_in_tr_t)
                loss.backward()
                opt_in.step()
                
            m_in.eval()
            with torch.no_grad():
                inner_probs[in_va] = m_in(x_in_va_t).cpu().numpy()
                
        p, r, th = precision_recall_curve(y_tr, inner_probs)
        f1_inner = 2 * (p * r) / (p + r + 1e-12)
        best_t = th[np.argmax(f1_inner)] if len(th) > 0 and np.argmax(f1_inner) < len(th) else 0.5
        
        torch.manual_seed(42)
        m_out = model_cls(in_dim=len(feats)).to(device)
        opt_out = torch.optim.AdamW(m_out.parameters(), lr=lr, weight_decay=wd)
        x_tr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
        x_va_t = torch.tensor(X_va_s, dtype=torch.float32, device=device)
        
        for ep in range(epochs):
            m_out.train()
            opt_out.zero_grad()
            pred = m_out(x_tr_t)
            loss = criterion(pred, y_tr_t)
            loss.backward()
            opt_out.step()
            
        m_out.eval()
        with torch.no_grad():
            pr_va = m_out(x_va_t).cpu().numpy()
            oof_probs[val_idx] = pr_va
            oof_preds_nested[val_idx] = (pr_va >= best_t).astype(int)
            
    p_oof, r_oof, th_oof = precision_recall_curve(y, oof_probs)
    f1_curve = 2 * (p_oof * r_oof) / (p_oof + r_oof + 1e-12)
    best_idx = np.argmax(f1_curve)
    
    return {
        'nested_f1': f1_score(y, oof_preds_nested),
        'nested_acc': accuracy_score(y, oof_preds_nested),
        'nested_rec': recall_score(y, oof_preds_nested),
        'nested_prec': precision_score(y, oof_preds_nested),
        'cm': confusion_matrix(y, oof_preds_nested),
        't05_f1': f1_score(y, (oof_probs >= 0.5).astype(int)),
        'auc': roc_auc_score(y, oof_probs),
        'oracle_f1': f1_curve[best_idx],
        'oracle_rec': r_oof[best_idx],
        'oracle_prec': p_oof[best_idx],
        'oracle_acc': accuracy_score(y, (oof_probs >= th_oof[best_idx]).astype(int)) if best_idx < len(th_oof) else 0.0,
        'oracle_t': th_oof[best_idx] if best_idx < len(th_oof) else 0.5
    }

print("="*85)
print("SOFT-F1 OPTIMIZATION BENCHMARK (STRICT NESTED CV)")
print("="*85)

for s_name, feats in sets.items():
    # 1. Soft-F1 Logistic
    res_log = evaluate_torch_model(SoftF1Logistic, feats, epochs=300, lr=0.03)
    p_count = len(feats) + 1
    print(f"Soft-F1 Logistic ({p_count} params) [{s_name}]:")
    print(f"  Nested F1 = {res_log['nested_f1']:.4f} | Acc = {res_log['nested_acc']*100:.1f}% | Rec = {res_log['nested_rec']*100:.1f}% ({res_log['cm'][1,1]}/{res_log['cm'][1,1]+res_log['cm'][1,0]}) | Prec = {res_log['nested_prec']*100:.1f}% | AUC = {res_log['auc']:.4f}")
    print(f"  Default T=0.5 F1: {res_log['t05_f1']:.4f} | Oracle F1: {res_log['oracle_f1']:.4f} (Rec={res_log['oracle_rec']*100:.1f}%, Prec={res_log['oracle_prec']*100:.1f}%, Acc={res_log['oracle_acc']*100:.1f}% at T={res_log['oracle_t']:.3f})\n")
    
    # 2. Soft-F1 Tiny MLP
    res_mlp = evaluate_torch_model(TinySoftF1MLP, feats, epochs=300, lr=0.02)
    p_mlp = len(feats)*6 + 6 + 6*1 + 1
    print(f"Soft-F1 2-Layer MLP ({p_mlp} params) [{s_name}]:")
    print(f"  Nested F1 = {res_mlp['nested_f1']:.4f} | Acc = {res_mlp['nested_acc']*100:.1f}% | Rec = {res_mlp['nested_rec']*100:.1f}% ({res_mlp['cm'][1,1]}/{res_mlp['cm'][1,1]+res_mlp['cm'][1,0]}) | Prec = {res_mlp['nested_prec']*100:.1f}% | AUC = {res_mlp['auc']:.4f}")
    print(f"  Default T=0.5 F1: {res_mlp['t05_f1']:.4f} | Oracle F1: {res_mlp['oracle_f1']:.4f} (Rec={res_mlp['oracle_rec']*100:.1f}%, Prec={res_mlp['oracle_prec']*100:.1f}%, Acc={res_mlp['oracle_acc']*100:.1f}% at T={res_mlp['oracle_t']:.3f})\n")


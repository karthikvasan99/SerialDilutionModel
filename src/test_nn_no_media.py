import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix

from src.scratch_k2_ultimate_f1 import X, y, feature_names, all_keys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

loo = LeaveOneOut()

def evaluate_nn(feats, hidden=5, act='tanh', epochs=250, lr=0.02, wd=1e-3):
    f_idx = [feature_names.index(f) for f in feats]
    X_sub = X[:, f_idx]
    
    oof_probs = np.zeros(len(y))
    pos_weight = torch.tensor([77.0 / 22.0], device=device)
    
    for tr, te in loo.split(X_sub):
        s = StandardScaler()
        x_tr = s.fit_transform(X_sub[tr])
        x_te = s.transform(X_sub[te])
        
        torch.manual_seed(42)
        model = TinyMLP(in_dim=len(feats), hidden=hidden, act=act).to(device)
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
    cm = confusion_matrix(y, preds)
    f1 = f1_score(y, preds)
    acc = accuracy_score(y, preds)
    rec = recall_score(y, preds)
    prec = precision_score(y, preds)
    auc = roc_auc_score(y, oof_probs)
    
    return {
        'features': feats,
        'f1': f1, 'acc': acc, 'rec': rec, 'prec': prec, 'auc': auc,
        'cm': cm, 'oof_probs': oof_probs, 'preds': preds
    }

candidate_sets = [
    # Candidate 1: Base 8 + mean_sp_cv + r1_halflife_std
    ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S', 'mean_sp_cv', 'r1_halflife_std'],
    # Candidate 2: Base 8 + surv_count + extinct_ratio
    ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S', 'surv_count', 'extinct_ratio'],
    # Candidate 3: Base 8 + mean_path_diff + end_cv
    ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S', 'mean_path_diff', 'end_cv'],
    # Candidate 4: Base 8 + bio_diff_consec + max_path_diff
    ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S', 'bio_diff_consec', 'max_path_diff'],
    # Candidate 5: Base 8 + r1_halflife_std + r2_halflife_std
    ['mean_tv_path', 'mean_poincare_shift', 'mean_overlap', 'bio_cv', 'max_sp_cv', 'bio_final', 'path_len', 'S', 'r1_halflife_std', 'r2_halflife_std'],
    # Candidate 6: Pure Top 10 by individual correlation with chaos (all non-media)
    ['mean_tv_path', 'mean_path_diff', 'end_cv', 'mean_end_diff', 'std_tv_path', 'bio_final', 'max_path_diff', 'bio_diff_consec', 'bio_cv', 'max_sp_cv']
]

print("="*85)
print("EVALUATING 10 PHYSICAL FEATURES (ZERO MEDIA CONDITIONS) WITH 2-LAYER MLP (LOOCV)")
print("="*85)

for i, feats in enumerate(candidate_sets):
    res = evaluate_nn(feats, hidden=5, act='tanh')
    cm = res['cm']
    print(f"Set {i+1} | F1 = {res['f1']:.4f} | Acc = {res['acc']*100:.1f}% | Rec = {res['rec']*100:.1f}% | Prec = {res['prec']*100:.1f}% | AUC = {res['auc']:.4f} | TP:{cm[1,1]:2d} FP:{cm[0,1]:2d} FN:{cm[1,0]:2d} TN:{cm[0,0]:2d}")


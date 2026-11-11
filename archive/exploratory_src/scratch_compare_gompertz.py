import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
data_path = 'Data/Gore_processed_data.pt'
data = torch.load(data_path, weights_only=False)

N_species = 48
K = 7
dt = 0.10
dilution = 30.0
dispersal = 1e-6
t_span = torch.tensor([0.0, 1.0], device=device)

# Prepare batches
b11_trajs, b11_conds, b11_masks = [], [], []
b4_trajs, b4_conds, b4_masks = [], [], []

cond_to_idx = {'LN': 0, 'MN': 1, 'HN': 2}

for (nutrient, S, comm), traj in data['relative_abundance'].items():
    cond_idx = cond_to_idx[nutrient]
    mask = data['presence_masks'][(S, comm)]
    if traj.shape[0] == 11:
        b11_trajs.append(traj)
        b11_conds.append(cond_idx)
        b11_masks.append(mask)
    elif traj.shape[0] == 4:
        b4_trajs.append(traj)
        b4_conds.append(cond_idx)
        b4_masks.append(mask)

b11_s = torch.stack(b11_trajs).to(device) # (N11, 11, 48)
b11_c = torch.tensor(b11_conds, device=device)
b11_m = torch.stack(b11_masks).to(device) # (N11, 48)

b4_s = torch.stack(b4_trajs).to(device) # (N4, 4, 48)
b4_c = torch.tensor(b4_conds, device=device)
b4_m = torch.stack(b4_masks).to(device) # (N4, 48)

class ModelContainer(nn.Module):
    def __init__(self, ode_func, num_conditions=3, K=7):
        super().__init__()
        self.ode_func = ode_func
        self.a = nn.Parameter(torch.randn(num_conditions, K) * 0.1)
        
    def step_forward(self, x_curr, cond_idx, mask):
        R0 = F.softplus(self.a[cond_idx])
        N0 = x_curr / dilution + dispersal * mask
        N0_safe = torch.clamp(N0, min=0.0)
        S0 = torch.cat([N0_safe, R0], dim=-1)
        
        S_t = odeint(self.ode_func, S0, t_span, method='euler', options={'step_size': dt})
        S_final = S_t[-1]
        N_final = S_final[..., :N_species] * mask
        N_safe = torch.clamp(N_final, min=0.0)
        tot = N_safe.sum(dim=-1, keepdim=True)
        tot = torch.where(tot > 0, tot, torch.ones_like(tot))
        x_pred = N_safe / tot
        log_pred = torch.log(torch.clamp(x_pred, min=1e-9))
        return log_pred, x_pred

class LogisticODE(nn.Module):
    def __init__(self, N, K):
        super().__init__()
        self.N, self.K = N, K
        self.Theta = nn.Parameter(torch.randn(N, K) * 0.1)
        self.W = nn.Parameter(torch.randn(K, K) * 0.1)
        self.alpha_raw = nn.Parameter(torch.randn(N) * 0.01)
    def forward(self, t, S):
        N_t, R_t = S[..., :self.N], S[..., self.N:]
        alpha = F.softplus(self.alpha_raw)
        N_safe, R_safe = torch.clamp(N_t, min=0.0), torch.clamp(R_t, min=0.0)
        growth = torch.matmul(R_safe, self.Theta.t())
        dN = (N_safe * growth) - (alpha * (N_safe ** 2))
        sink = torch.matmul(N_safe, torch.matmul(self.Theta, self.W))
        dR = -R_safe * sink
        return torch.cat([dN, dR], dim=-1)

class GompertzODE(nn.Module):
    def __init__(self, N, K, eps=1e-4):
        super().__init__()
        self.N, self.K, self.eps = N, K, eps
        self.Theta = nn.Parameter(torch.randn(N, K) * 0.1)
        self.W = nn.Parameter(torch.randn(K, K) * 0.1)
        self.beta_raw = nn.Parameter(torch.randn(N) * 0.01)
    def forward(self, t, S):
        N_t, R_t = S[..., :self.N], S[..., self.N:]
        beta = F.softplus(self.beta_raw)
        N_safe, R_safe = torch.clamp(N_t, min=0.0), torch.clamp(R_t, min=0.0)
        growth = torch.matmul(R_safe, self.Theta.t())
        # Gompertz density-dependent suppression: - beta * N * log(1 + N/eps)
        crowding = beta * N_safe * torch.log(1.0 + N_safe / self.eps)
        dN = (N_safe * growth) - crowding
        sink = torch.matmul(N_safe, torch.matmul(self.Theta, self.W))
        dR = -R_safe * sink
        return torch.cat([dN, dR], dim=-1)

def evaluate_model(model):
    model.eval()
    with torch.no_grad():
        # Evaluate multi-step autoregressive rollout on b11
        x_curr = b11_s[:, 0, :]
        preds = [x_curr.cpu().numpy()]
        for t in range(1, 11):
            _, x_curr = model.step_forward(x_curr, b11_c, b11_m)
            preds.append(x_curr.cpu().numpy())
        preds = np.stack(preds, axis=1) # (N11, 11, 48)
        actual = b11_s.cpu().numpy()
        
        # Linear R2 on active species across all steps
        r2_list = []
        for i in range(b11_s.size(0)):
            m = b11_masks[i].numpy().astype(bool)
            y_t = actual[i, :, m].flatten()
            y_p = preds[i, :, m].flatten()
            ss_res = np.sum((y_t - y_p)**2)
            ss_tot = np.sum((y_t - np.mean(y_t))**2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-9)
            r2_list.append(r2)
            
        mean_r2 = float(np.mean(r2_list))
        median_r2 = float(np.median(r2_list))
    return mean_r2, median_r2

def train_and_eval(name, ode_func, epochs=200):
    torch.manual_seed(42)
    model = ModelContainer(ode_func, num_conditions=3, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    print(f"\n--- Training {name} for {epochs} epochs ---")
    for epoch in range(1, epochs + 1):
        model.train()
        tf_ratio = max(0.10, 1.0 - 0.90 * (epoch / epochs))
        
        # Batch 11
        x_curr = b11_s[:, 0, :]
        loss11 = 0.0
        for t in range(1, 11):
            log_p, x_p = model.step_forward(x_curr, b11_c, b11_m)
            target = b11_s[:, t, :]
            loss11 += criterion(log_p, target)
            x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_p
            
        # Batch 4
        x_curr_4 = b4_s[:, 0, :]
        loss4 = 0.0
        for t in range(1, 4):
            log_p, x_p = model.step_forward(x_curr_4, b4_c, b4_m)
            target = b4_s[:, t, :]
            loss4 += criterion(log_p, target)
            x_curr_4 = tf_ratio * target + (1.0 - tf_ratio) * x_p
            
        loss = (loss11 / 10.0) + (loss4 / 3.0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}/{epochs}: Loss = {loss.item():.4f} (TF={tf_ratio:.2f})")
            
    mean_r2, med_r2 = evaluate_model(model)
    print(f"Result for {name}: Mean Linear R^2 = {mean_r2:.4f}, Median Linear R^2 = {med_r2:.4f}")
    return mean_r2, med_r2

res_log = train_and_eval("Logistic / Quadratic Baseline", LogisticODE(N_species, K), epochs=200)
res_gomp = train_and_eval("Gompertz Logarithmic Self-Limitation", GompertzODE(N_species, K, eps=1e-3), epochs=200)


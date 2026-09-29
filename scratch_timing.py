import time
import os
import torch
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/karthik/SerialDilutionModel/src/')
from SerialDilutionModels import LatentAssemblyNeuralODE

# Just mock data to see how fast it runs on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 50
K = 5
num_conditions = 5

model = LatentAssemblyNeuralODE(N, K, num_conditions).to(device)

train_seqs = torch.rand(36, 6, 50).to(device)
train_conds = torch.randint(0, 5, (36,)).to(device)

criterion = torch.nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"Timing 500 epochs on {device}...")
start = time.time()
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    
    batch_loss = 0.0
    tf_ratio = max(0.0, 1.0 - (epoch / (500 * 0.75)))
    
    x_curr = train_seqs[:, 0, :]
    for t in range(1, train_seqs.size(1)):
        log_pred, x_pred, _, _ = model(x_curr, train_conds)
        target = train_seqs[:, t, :]
        batch_loss += criterion(log_pred, target)
        x_curr = tf_ratio * target + (1.0 - tf_ratio) * x_pred
            
    loss = batch_loss
    loss.backward()
    optimizer.step()
    
    if epoch == 9:
        print(f"Time for 10 epochs: {time.time() - start:.2f}s")
end = time.time()
print(f"Time for 500 epochs: {end - start:.2f}s")

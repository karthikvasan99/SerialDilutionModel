import torch
data = torch.load('Data/Pollard_model.pth', weights_only=True)
print(data.keys())

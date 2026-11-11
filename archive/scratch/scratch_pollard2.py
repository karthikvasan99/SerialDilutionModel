import torch
import sys
sys.path.append('/home/karthik/SerialDilutionModel/src/')
from SerialDilutionModels import LatentAssemblyModel

print("Loading Pollard model...")
model_dict = torch.load('Data/Pollard_model.pth', map_location='cpu', weights_only=True)
for k, v in model_dict.items():
    print(f"{k}: {v.shape}")

print("Checking for any Pollard data files...")
import glob
files = glob.glob('Data/*Pollard*')
print(files)

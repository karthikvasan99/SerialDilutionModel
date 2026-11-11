import torch
data = torch.load('Data/German_processed_data.pt', weights_only=False)
trajectories = data['trajectories']

# concat all passages
all_passages = torch.cat(list(trajectories.values()), dim=0)
mean_abundance = all_passages.mean(dim=0)
keep_mask = mean_abundance >= 1e-3
print("Mean across all passages & all conditions >= 1e-3:", keep_mask.sum().item())


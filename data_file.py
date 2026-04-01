from torch.utils.data import Dataset
import torch
from dataclasses import dataclass


class TensorIndexDataset(Dataset):
    def __init__(self, x_in, x_target, colors):
        self.c = x_in.float().contiguous()
        self.x_in = x_in.float().contiguous()

        self.x_target = x_target.float().contiguous()
        self.colors = torch.tensor(colors, dtype=torch.float32)

    def __len__(self):
        return self.x_in.shape[0]

    def __getitem__(self, idx):
        return self.x_in[idx], self.x_target[idx], self.colors[idx]
    

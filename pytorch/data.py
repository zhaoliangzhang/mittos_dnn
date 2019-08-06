import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class trainSet(Dataset):
    def __init__(self, root):
        raw_data = np.loadtxt(root, dtype=np.float32)
        self.data = torch.from_numpy(raw_data[:,0:40])
        self.label = torch.from_numpy(raw_data[:,40:41])

    def __getitem__(self, index):
        return self.data[index,:], self.label[index,:]

    def __len__(self):
        return self.data.shape[0]
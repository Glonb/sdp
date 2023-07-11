import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, numpy_file, label_file):
        self.data = torch.from_numpy(np.load(numpy_file))
        self.labels = pd.read_csv(label_file, usecols=['bug']).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx].item()

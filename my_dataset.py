import torch
from torch.utils.data import Dataset
import pandas as pd


class MyDataset(Dataset):

    def __init__(self, embed_file, label_file):
        self.data = torch.load(embed_file)
        labels = pd.read_csv(label_file, usecols=['bug']).values
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].float(), self.labels[idx]

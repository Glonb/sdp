import torch
from torch.utils.data import Dataset
import pandas as pd


class MyDataset(Dataset):

    def __init__(self, embed_file, csv_file):
        self.emb_data = torch.load(embed_file)
        self.csv_data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        tr_features = torch.tensor(self.csv_data.iloc[idx, 1:-1], dtype=torch.float32)
        label = torch.tensor(self.csv_data.iloc[idx, -1], dtype=torch.long)

        return self.emb_data[idx].float(), tr_features, label


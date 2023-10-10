import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, file_path):
        self.data = torch.load(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        token_ids = item['token_ids']
        label = item['label']
        return token_ids, label


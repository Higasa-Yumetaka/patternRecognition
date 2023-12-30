import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd


class CSVDataset(Dataset):
    def __init__(self, path, skip_header=True):
        self.X = pd.read_csv(path, skip_header=skip_header)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index].values
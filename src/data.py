import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
import os

class OptimizedDataset(Dataset):
    def __init__(self, tensor_folder):
        self.files = sorted(glob.glob(os.path.join(tensor_folder, "./data/optimized_tensors/*.pt")))
        self.demographics = pd.read_csv('./data/demographic.csv',)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Loading the pre-processed tensor
        
        data = torch.load(self.files[idx])
        label = torch.tensor(float(self.demographics[' group'][idx]), dtype=torch.long)
        return data , label
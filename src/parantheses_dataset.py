import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch
import numpy as np
from constants import MAX_CUT, MAPPING, MAX_STRING_LENGTH

class ToTensorParentheses:
    def __call__(self, sample):
        features = sample[0]
        label = sample[1]
        return {
            'features': torch.tensor([MAPPING[c] for c in features], dtype=torch.long),
            'label': torch.tensor([MAPPING[c] for c in label], dtype=torch.long),
        }
        
        
class NormalizeParantheses:
    def __call__(self, sample):
        features = sample[0]
        label = sample[1]
        normalized_features = (features - np.mean(features)) / np.std(features)
        return (normalized_features, label)


class CustomParenthesesDataset(Dataset):
    def __init__(self,
                 data_file_path: str,
                 transform = ToTensorParentheses()):
        self.data_file_path = data_file_path
        self.transform = transform
        with open(data_file_path, "r", encoding="utf-8") as f:
            self.data = f.readlines()
        line_count = 0
        with open(data_file_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        self.data_size = line_count


    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = self.data[idx].strip().split(',')
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
    

if __name__ == "__main__":
    dataset = CustomParenthesesDataset("belief-state-geometry-in-the-residual-stream/src/data/trainingdata/training_?_paranthesees.txt")
    print(dataset[0])
    print(len(dataset))

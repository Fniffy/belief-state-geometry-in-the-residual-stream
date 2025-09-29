import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Lambda

class CustomImageDataset(Dataset):
    def __init__(self,
                 data_file_path: str,
                 transform = None, target_transform = None):
        self.data_file_path = data_file_path
        self.transform = transform
        self.target_transform = target_transform
        with open(data_file_path, "r", encoding="utf-8") as f:
            self.data = f.readlines()
        line_count = 0
        with open(data_file_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        self.data_size = line_count


    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.data[idx].strip().split(',')
    
    

if __name__ == "__main__":
    dataset = CustomImageDataset("belief-state-geometry-in-the-residual-stream/src/data/trainingdata/training_?_paranthesees.txt")
    print(dataset[2])
    print(len(dataset))

import sys
import torchvision
import torch
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from parantheses_dataset import CustomParenthesesDataset as Dataset
import torchvision.transforms as transforms
from constants import *
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd

if SHOW_VERSIONS == True and __name__ == "__main__":
    print("System version:", sys.version)                       #Version used: System version: 3.13.3 (main, Apr 10 2025, 21:38:51) [GCC 14.2.0]
    print("PyTorch version:", torch.__version__)                #Version used: PyTorch version: 2.8.0+cu128
    print("Pytorch version:", torchvision.__version__)          #Version used: Pytorch version: 0.23.0+cu128
    print("Numpy version:", np.__version__)                     #Version used: Numpy version: 2.3.2
    print("Pandas version:", pd.__version__)                    #Version used: Pandas version: 2.3.1
    print("CUDA available:", torch.cuda.is_available())         #              CUDA available: False
    
# -----------------------------
# Collate function for padding sequences
# -----------------------------
def collate_fn(batch):
    features = [item['features'].long() for item in batch]  # long for embedding
    labels = [item['label'] for item in batch]
    features = pad_sequence(features, batch_first=True, padding_value=0)
    labels = torch.stack(labels).squeeze()
    return features, labels

class SimpleParenthesesTransformer(nn.Module):
    """"
        Heart of my project <3
    """
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=LAYERS, seq_len=MAX_STRING_LENGTH*2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, 1)  # predicting integer label
        self.seq_len = seq_len
        
    def forward(self, x):
        src_key_padding_mask = (x == MAPPING['_'])
        residuals_per_layer = []
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        for layer in self.layers:
            x_residual = x.clone()
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
            residuals_per_layer.append(x.clone())
        mask = (~src_key_padding_mask).unsqueeze(-1)  # [batch, seq_len, 1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)   # masked average
        logits = self.fc_out(x)  # simple pooling
        return logits, residuals_per_layer
    

    
    
if __name__ == "__main__":
    # load dataset
    dataset = Dataset("belief-state-geometry-in-the-residual-stream/src/data/trainingdata/training_?_paranthesees.txt")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print(f"loaded Dataset of size: {len(dataset)}")


    model = SimpleParenthesesTransformer(vocab_size=len(MAPPING))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(model)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if TRAINING == True:
        print("starting Training...")
        model 
        if Path(CUT_PARENTHESES_MODEL).is_file():
            model.load_state_dict(torch.load(CUT_PARENTHESES_MODEL))
            print("loaded existing model weights")
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0

            for batch in dataloader:
                features = batch[0].to(device)   # shape: [batch_size, seq_len] features
                labels = batch[1].to(device)        # shape: [batch_size] labels

                optimizer.zero_grad()
                outputs = model(features)                 # shape: [batch_size, 1]
                outputs = outputs[0]            # make shape: [batch_size]

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * features.size(0)

            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), CUT_PARENTHESES_MODEL)
        
    if TRAINING == False:
        if not Path(CUT_PARENTHESES_MODEL).is_file():
            print("No trained model found, cannot run inference.")
            exit(1)
        model.load_state_dict(torch.load(CUT_PARENTHESES_MODEL))
        model.eval()
        
        while True:
            try:
                user_input = input("Enter parentheses everything else exits the loop: ")
                parentheses_string = user_input.strip()+ '_'*(MAX_STRING_LENGTH*2 - len(user_input.strip()))
                parentheses_vector = torch.tensor([MAPPING[c] for c in parentheses_string], dtype=torch.float)
                output = model(parentheses_vector.unsqueeze(0).long())
                prediction = output[0].item()
                print(f"Predicted depth: {prediction}")
            except ValueError:
                print("eval phase stopped")
                break
            
    
    
        
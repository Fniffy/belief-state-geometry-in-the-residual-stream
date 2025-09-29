"""
    Just a simple file to play around with pytorch
"""

#if __name__ =="__main__":
import torch
import numpy as np


tensor = torch.ones(3,2,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
print(tensor.mT)




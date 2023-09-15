import torch
import torch.optim as optim

def adam(parameters):
    optimizer = optim.Adam(parameters(), lr=0.01, weight_decay=1e-5)
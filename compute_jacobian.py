import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd.functional import jacobian

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.V = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

    def forward(self, s):
        Vs = self.V(s)

        return torch.norm(Vs, dim=1).reshape(-1,1)



h = NeuralNetwork().to(device)
s0 = torch.tensor([-1.8545, 0.2772, 23.45, 0.32], dtype=torch.float).reshape((2,2)).to(device)

h_s0 = h(s0)
print(h_s0)

print(jacobian(h, h_s0))
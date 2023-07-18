
import numpy as np
import itertools
from typing import Tuple, List, Optional
import json
import time
import copy
import itertools

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.autograd import grad

class LinearBlock(nn.Module):
    def __init__(self, num_inputs, num_outputs) -> None:
        super().__init__()
        
        self.Z_layer = nn.Linear(num_inputs, num_outputs)
        self.activation_layer = nn.ReLU()

    def forward(self, input):

        z = self.Z_layer(input)
        z = self.activation_layer(z)

        return z


class ResBlock(nn.Module):
    def __init__(self, num_input, expansion_factor=2) -> None:
        super().__init__()
        
        self.norm_layer = nn.LayerNorm(num_input) 

        self.feed_forward = nn.Sequential(
                          nn.Linear(num_input, expansion_factor * num_input),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*num_input, num_input),
                          nn.ReLU()
        )

    def forward(self, H, U, V):
        
        Z = self.feed_forward(H)

        out = torch.mul( 1 -  Z, U ) + torch.mul(Z, V)
        out = self.norm_layer(out)
        return out



class Transformer(nn.Module):
    def __init__(self, num_inputs, num_neurals):
        super().__init__()

        self.V_layer = LinearBlock(num_inputs, num_neurals)
        self.U_layer = LinearBlock(num_inputs, num_neurals)
        self.H_layer = LinearBlock(num_inputs, num_neurals)

        self.res_block_1 = ResBlock(num_neurals)
        self.res_block_2 = ResBlock(num_neurals)

        
        self.output_layer = nn.Linear(num_neurals, 1)

    def forward(self, input):
        
        U = self.U_layer(input)
        V = self.V_layer(input)
        H_1 = self.H_layer(input)
        H_2 = self.res_block_1(H_1, U, V)
        H_3 = self.res_block_2(H_2, U, V)
        
        f = self.output_layer(H_3)
        

        return f

def jacobian(y, x):
        ''' jacobian of y wrt x '''
        meta_batch_size, num_observations = y.shape[:2]
        jac = torch.zeros(meta_batch_size, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
        for i in range(y.shape[-1]):
            # calculate dydx over batches for each feature value of y
            y_flat = y[...,i].view(-1, 1)
            jac[:, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

        status = 0
        if torch.any(torch.isnan(jac)):
            status = -1

        return jac, status

if __name__ == "__main__":
    print("hello world")
    transformer = Transformer(2, 48)

    x = torch.rand(3, 2, requires_grad=True)

    f = transformer(x)
    print(f)

    jacob, _ = jacobian(f, x)

    print(jacob)

    dx = torch.ones(3, 2, requires_grad=True)* 0.05

    df = torch.bmm(jacob, dx.unsqueeze(dim=-1)).squeeze(dim=-1)

    f_bar = f + df
    print(f"f_bar = {f_bar}")

    x_2 = x + dx
    f_2 = transformer(x_2)
    print(f"f_2 = {f_2}")
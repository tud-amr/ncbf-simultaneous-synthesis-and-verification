import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd.functional import jacobian

from NeuralCBF.MyNeuralNetwork import *
from CARs import car1
from Dataset.DataModule import DataModule


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

NN = torch.load("NN.pt")
NN.to(device)

car1.set_barrier_function(NN)
car1.dt = 0.005

s0 = torch.tensor([-0.0232, -0.1550]).reshape((1,2)).float().to(device)
print(f" the input is: \n {s0} \n")

h_s0, _ = NN(s0)
print(f"the output of NN is \n {h_s0} \n")

_, JV = NN.V_with_jacobian(s0)
# print(f"JV from V_with_jacobian is \n {JV} \n")

Lf_V , Lg_V = car1.V_lie_derivatives(s0)
print(f"Lf_V is {Lf_V} \n Lg_V is {Lg_V} \n")

batch_idx = 0
Lg_V_np = Lg_V[batch_idx, 0].detach().cpu().numpy()
Lf_V_np = Lf_V[batch_idx, 0].detach().cpu().numpy()

dh = (Lf_V_np + Lg_V_np * 1 )* car1.dt
print(f" dh from hand compute is {dh} ")

JV_torch = jacobian(NN.h, s0).squeeze(dim=0)
print(JV_torch.shape)


h2 = h_s0 + dh
s2 = car1.step(s0, 1)
ds = s2 - s0

dh_torch = torch.bmm(JV_torch, torch.unsqueeze(ds.T, dim=0 ))
print(f" dh_torch is {dh_torch}")

print(f"ds = {ds} , s2 = {s2}")
h2_start = NN(s2)

print(f"true h2 is: \n {h2_start} \n while h2 from jacobian is \n {h2} \n")


# print(f"the jacobian from torch library is: \n {jacobian(NN.h, s0)}")



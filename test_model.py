import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from MyNeuralNetwork import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

h = torch.load("h.pt")

loss_record = []
sample_count = 0
for i in np.arange(-2, 2, 0.1):
    for j in np.arange(-1,1,0.1):
        sample_count += 1
        X_test = torch.tensor([[i, j]], dtype=torch.float, requires_grad=True, device=device)
        
        dhdx = get_dhdx(X_test, h)
        C_bar_s = C_bar(X_test, h, u_v1, u_v2, dhdx)
        h_bar_s = h_bar(X_test)

        #calculate the loss
        loss = myCustomLoss(h_bar_s, C_bar_s)
        loss_record.append(loss.item())
        print(f"test sample: {i}, {j}. loss is {loss.item()}")


average_loss = sum(loss_record)/sample_count
print(f"average loss is : {average_loss}")

# X_test = torch.tensor([[1.9, 0.9]], dtype=torch.float, requires_grad=True, device=device)
        
# dhdx = get_dhdx(X_test, h)
# C_bar_s = C_bar(X_test, h, u_v1, u_v2, dhdx)
# h_bar_s = h_bar(X_test)

# #calculate the loss
# loss = myCustomLoss(h_bar_s, C_bar_s)
# print(loss.item())
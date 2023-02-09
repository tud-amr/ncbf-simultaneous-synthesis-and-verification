import torch
from CARs import car1
from MyNeuralNetwork import *
from DataModule import DataModule

t = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])

print(t.shape)
a = torch.flatten(t)
print(a.shape)
b = torch.flatten(t, start_dim=1)
print(b.shape)

# x0 = torch.rand(3, 2,dtype=torch.float)*2 - 1

# data_module = DataModule(system=car1, training_sample_num=50)

# h = NeuralNetwork(dynamic_system=car1, data_module=data_module)

# h.prepare_data()

# h.descent_loss(h.data_module.s_training, h.data_module.safe_mask_training, h.data_module.unsafe_mask_training)

# h.V_loss(h.data_module.s_training, h.data_module.safe_mask_training, h.data_module.unsafe_mask_training)

# del h
import torch
from CARs import car1
from MyNeuralNetwork import *
from DataModule import DataModule






x0 = torch.rand(3, 2,dtype=torch.float)*2 - 1

data_module = DataModule(system=car1, train_grid_gap=0.1, test_grid_gap=0.3)

h = NeuralNetwork(dynamic_system=car1, data_module=data_module)

h.prepare_data()



# h.boundary_loss(h.data_module.s_training, h.data_module.safe_mask_training, h.data_module.unsafe_mask_training, accuracy=True)
h.descent_loss(h.data_module.s_training, h.data_module.safe_mask_training, h.data_module.unsafe_mask_training)
# h.V_loss(h.data_module.s_training, h.data_module.safe_mask_training, h.data_module.unsafe_mask_training)

del h
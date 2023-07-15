import torch
from dynamic_system_instances import car1, inverted_pendulum_1
from MyNeuralNetwork import NeuralNetwork
# from ValueFunctionNeuralNetwork import ValueFunctionNeuralNetwork
from DataModule import DataModule
from treelib import Tree, Node
from dynamic_system_instances import car1, inverted_pendulum_1
from control_affine_system import ControlAffineSystem
from itertools import product
import matplotlib.pyplot as plt
# from auto_LiRPA import BoundedModule, BoundedTensor
# from auto_LiRPA.perturbations import *
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


data_module = DataModule(system=inverted_pendulum_1, val_split=0.1, train_batch_size=6, test_batch_size=128, train_grid_gap=0.1, test_grid_gap=0.01)
data_module.prepare_data()
NN0 = NeuralNetwork.load_from_checkpoint("CBF_logs/robust_training_maximum_without_nominal_controller/lightning_logs/version_4/checkpoints/epoch=499-step=1500.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module)

NN = NeuralNetwork(dynamic_system=inverted_pendulum_1, data_module=data_module, require_grad_descent_loss=True, fine_tune=False)
NN.set_previous_cbf(NN0.h)

batch = next(iter(NN.data_module.train_dataloader()))

NN.training_stage = 1

# s = torch.Tensor([-3, -3]).float().reshape((-1, NN.dynamic_system.ns)).to(device)

# print(NN.nominal_controller(s))
NN.training_step(batch, 0)

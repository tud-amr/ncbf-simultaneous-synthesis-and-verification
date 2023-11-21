import torch
from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1
from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import NeuralNetwork
# from ValueFunctionNeuralNetwork import ValueFunctionNeuralNetwork
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule
from treelib import Tree, Node
from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1, dubins_car_rotate, dubins_car_acc
from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem
from itertools import product
import matplotlib.pyplot as plt
# from auto_LiRPA import BoundedModule, BoundedTensor
# from auto_LiRPA.perturbations import *
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

########################### hyperparameters #############################

train_mode = 1
system = inverted_pendulum_1
default_root_dir = "./logs/CBF_logs/inverted_pendulum_1"
checkpoint_dir = "logs/CBF_logs/inverted_pendulum_1/lightning_logs/version_8/checkpoints/epoch=199-step=2436.ckpt"
grid_gap = torch.Tensor([0.2, 0.2])  


########################################################

data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=1024, training_points_num=int(1e6), train_mode=train_mode)
data_module.prepare_data()
data_module.set_dataset()

# NN = NeuralNetwork.load_from_checkpoint(checkpoint_dir, dynamic_system=system, data_module=data_module, train_mode=train_mode)
# NN0 = NeuralNetwork.load_from_checkpoint(checkpoint_dir, dynamic_system=system, data_module=data_module, train_mode=train_mode)
NN = NeuralNetwork(dynamic_system=inverted_pendulum_1, data_module=data_module, train_mode=train_mode)
# NN.set_previous_cbf(NN0.h)

batch = next(iter(NN.data_module.train_dataloader()))


batch[0] = torch.Tensor([2, 3.6]).to(NN.device).expand(3, system.ns)
# s = torch.Tensor([-3, -3]).float().reshape((-1, NN.dynamic_system.ns)).to(device)

# print(NN.nominal_controller(s))
NN.training_step(batch, 0)

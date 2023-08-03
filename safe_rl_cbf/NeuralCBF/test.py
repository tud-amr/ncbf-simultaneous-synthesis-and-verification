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

system = dubins_car_acc

data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=512, training_points_num=int(1e6))
data_module.prepare_data()
NN = NeuralNetwork.load_from_checkpoint("logs/CBF_logs/dubins_car_acc/lightning_logs/version_1/checkpoints/epoch=86-step=14181.ckpt", dynamic_system=system, data_module=data_module)

# NN = NeuralNetwork(dynamic_system=inverted_pendulum_1, data_module=data_module, require_grad_descent_loss=True, fine_tune=False)
# NN.set_previous_cbf(NN0.h)

batch = next(iter(NN.data_module.train_dataloader()))

NN.training_stage = 0
NN.use_h0 = False

batch[0] = torch.Tensor([2, 2, 0, 0.4, 0]).to(NN.device).reshape(-1,system.ns)
# s = torch.Tensor([-3, -3]).float().reshape((-1, NN.dynamic_system.ns)).to(device)

# print(NN.nominal_controller(s))
NN.training_step(batch, 0)

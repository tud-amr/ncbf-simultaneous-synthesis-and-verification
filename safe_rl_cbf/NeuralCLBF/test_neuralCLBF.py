import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import lightning.pytorch as pl

from safe_rl_cbf.NeuralCLBF.NeuralCLBF import *
from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1, dubins_car, dubins_car_rotate ,dubins_car_acc, point_robot, point_robots_dis, robot_arm_2d, two_vehicle_avoidance
from safe_rl_cbf.Dataset.TestingDataModule import TestingDataModule
from safe_rl_cbf.Analysis.draw_cbf import draw_cbf



############################# hyperparameters #############################

system = inverted_pendulum_1
checkpoint_path = "logs/CBF_logs/neural_clbf/inverted_pendulum_1/lightning_logs/version_26/checkpoints/epoch=169-step=1530.ckpt"
data_module = TestingDataModule(system=system, test_batch_size=1024, test_points_num=int(1e3), test_index={0: None, 1: None})



############################ load model ###################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

NN = NeuralCLBF.load_from_checkpoint(checkpoint_path, dynamic_system=system, data_module=data_module)
NN.to(device)

trainer = pl.Trainer(accelerator = "gpu",
    devices = 1,
    max_epochs=1,)
trainer.test(NN)

draw_cbf(system=system)
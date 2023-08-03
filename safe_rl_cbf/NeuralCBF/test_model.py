import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import lightning.pytorch as pl

from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import *
from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1, dubins_car, dubins_car_rotate ,dubins_car_acc, point_robot, point_robots_dis, robot_arm_2d
from safe_rl_cbf.Dataset.TestingDataModule import TestingDataModule
from safe_rl_cbf.Analysis.draw_cbf import draw_cbf



############################# hyperparameters #############################

system = dubins_car_rotate
checkpoint_path = "logs/CBF_logs/dubins_car_rotate/lightning_logs/version_1/checkpoints/epoch=139-step=22820.ckpt"
data_module = TestingDataModule(system=system, test_batch_size=512, test_points_num=int(1e2), test_index={0: None, 1: None, 2: 0})



############################ load model ###################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

NN = NeuralNetwork.load_from_checkpoint(checkpoint_path, dynamic_system=system, data_module=data_module)
NN.to(device)

trainer = pl.Trainer(accelerator = "gpu",
    devices = 1,
    max_epochs=1,)
trainer.test(NN)

draw_cbf(system=system)
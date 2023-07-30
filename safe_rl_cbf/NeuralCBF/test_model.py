import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import lightning.pytorch as pl

from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import *
from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1, cart_pole_1, dubins_car, dubins_car_acc, point_robot
from safe_rl_cbf.Dataset.TestingDataModule import TestingDataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

system = point_robot

data_module = TestingDataModule(system=system, test_batch_size=512, test_points_num=int(1e2), test_index={0: None, 1: None, 2: -0.6, 3: -0.4})

NN = NeuralNetwork.load_from_checkpoint("logs/CBF_logs/point_robot/lightning_logs/version_1/checkpoints/epoch=68-step=11247.ckpt", dynamic_system=system, data_module=data_module)
NN.to(device)

trainer = pl.Trainer(accelerator = "gpu",
    devices = 1,
    max_epochs=1,)
trainer.test(NN)

with open("safe_rl_cbf/Analysis/draw_cbf.py") as f:
    exec(f.read())
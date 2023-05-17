import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pytorch_lightning as pl

from MyNeuralNetwork import *
from dynamic_system_instances import car1, inverted_pendulum_1
from DataModule import DataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


data_module = DataModule(system=inverted_pendulum_1, val_split=0, train_batch_size=64, test_batch_size=128, train_grid_gap=0.3, test_grid_gap=0.01)

G = ValueFunctionNeuralNetwork.load_from_checkpoint("/home/wangxinyu/.mujoco/mujoco210/sunny_test/masterthesis_test/CBF_logs/train_g/lightning_logs/version_0/checkpoints/epoch=399-step=4800.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module)
G.to(device)

NN = NeuralNetwork.load_from_checkpoint("/home/wangxinyu/.mujoco/mujoco210/sunny_test/masterthesis_test/CBF_logs/robust_training_maximum/lightning_logs/version_14/checkpoints/epoch=167-step=16632.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module, value_function=G)
NN.to(device)

trainer = pl.Trainer(accelerator = "gpu",
    devices = 1,
    max_epochs=1)
trainer.test(NN)

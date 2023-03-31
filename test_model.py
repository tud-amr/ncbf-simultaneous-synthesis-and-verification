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

NN = NeuralNetwork.load_from_checkpoint("masterthesis_test/CBF_logs/run0_without_condition_1/lightning_logs/version_0/checkpoints/epoch=454-step=4095.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module, learn_shape_epochs=2 )
NN.to(device)

trainer = pl.Trainer(accelerator = "gpu",
    devices = 1,
    max_epochs=1)
trainer.test(NN)

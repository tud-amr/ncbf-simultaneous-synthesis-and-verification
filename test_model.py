import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import lightning.pytorch as pl

from MyNeuralNetwork import *
from dynamic_system_instances import car1, inverted_pendulum_1
from DataModule import DataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


data_module = DataModule(system=inverted_pendulum_1, val_split=0, train_batch_size=64, test_batch_size=1024, train_grid_gap=0.3, test_grid_gap=0.01)


NN = NeuralNetwork.load_from_checkpoint("CBF_logs/robust_training_maximum_without_nominal_controller/lightning_logs/version_14/checkpoints/epoch=68-step=759.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module)
NN.to(device)

trainer = pl.Trainer(accelerator = "gpu",
    devices = 1,
    max_epochs=1,
    inference_mode=False)
trainer.test(NN)

with open("draw_CBF_shape.py") as f:
    exec(f.read())
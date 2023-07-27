import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import lightning.pytorch as pl

from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import *
from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1
from safe_rl_cbf.Dataset.DataModule import DataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


data_module = DataModule(system=inverted_pendulum_1, val_split=0, train_batch_size=64, test_batch_size=1024, train_grid_gap=0.3, test_grid_gap=0.01)


NN = NeuralNetwork.load_from_checkpoint("logs/CBF_logs/robust_training_maximum_with_value_function_4/lightning_logs/version_1/checkpoints/epoch=74-step=3225.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module)
NN.to(device)

trainer = pl.Trainer(accelerator = "gpu",
    devices = 1,
    max_epochs=1,
    inference_mode=False)
trainer.test(NN)

with open("safe_rl_cbf/Analysis/draw_CBF_shape.py") as f:
    exec(f.read())
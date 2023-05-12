import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from matplotlib import cm
from matplotlib.ticker import LinearLocator

from MyNeuralNetwork import *
from dynamic_system_instances import car1, inverted_pendulum_1
from DataModule import DataModule


data_module = DataModule(system=inverted_pendulum_1, val_split=0, train_batch_size=64, test_batch_size=128, train_grid_gap=0.3, test_grid_gap=0.01)

NN = NeuralNetwork(dynamic_system=inverted_pendulum_1, data_module=data_module, require_grad_descent_loss=True)

default_root_dir = "./masterthesis_test/CBF_logs/robust_training"

trainer = pl.Trainer(
    accelerator = "gpu",
    devices = 1,
    max_epochs=50,
    callbacks=[ EarlyStopping(monitor="Safety_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=50) ], 
    # default_root_dir=default_root_dir,
    )

torch.autograd.set_detect_anomaly(True)
# trainer.fit(NN, ckpt_path="./masterthesis_test/lightning_logs/version_4/checkpoints/epoch=399-step=26400.ckpt")
trainer.fit(NN)




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

for i in range(5, 15):
    data_module = DataModule(system=inverted_pendulum_1, val_split=0.1, train_batch_size=64, test_batch_size=128, train_grid_gap=0.1, test_grid_gap=0.01)

    NN = NeuralNetwork(dynamic_system=inverted_pendulum_1, data_module=data_module, learn_shape_epochs=2, require_grad_descent_loss=True)

    default_root_dir = "./masterthesis_test/CBF_logs/run" + str(i) + "_with_OptNet"

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=60,
        # callbacks=[ EarlyStopping(monitor="Total loss / val", mode="min", stopping_threshold=0.05) ], 
        default_root_dir=default_root_dir,
        )

    torch.autograd.set_detect_anomaly(True)
    # trainer.fit(NN, ckpt_path="./masterthesis_test/lightning_logs/version_4/checkpoints/epoch=399-step=26400.ckpt")
    trainer.fit(NN)
    
    
    data_module = DataModule(system=inverted_pendulum_1, val_split=0.1, train_batch_size=64, test_batch_size=128, train_grid_gap=0.1, test_grid_gap=0.01)

    NN = NeuralNetwork(dynamic_system=inverted_pendulum_1, data_module=data_module, learn_shape_epochs=2, require_grad_descent_loss=False)

    default_root_dir = "./masterthesis_test/CBF_logs/run" + str(i) + "_without_OptNet"

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=60,
        # callbacks=[ EarlyStopping(monitor="Total loss / val", mode="min", stopping_threshold=0.05) ], 
        default_root_dir=default_root_dir,
        )

    torch.autograd.set_detect_anomaly(True)
    # trainer.fit(NN, ckpt_path="./masterthesis_test/lightning_logs/version_4/checkpoints/epoch=399-step=26400.ckpt")
    trainer.fit(NN)



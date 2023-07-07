import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from matplotlib import cm
from matplotlib.ticker import LinearLocator

from MyNeuralNetwork import *
# from ValueFunctionNeuralNetwork import *
from dynamic_system_instances import car1, inverted_pendulum_1
from DataModule import DataModule


train_mode = 'train_h'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

data_module = DataModule(system=inverted_pendulum_1, val_split=0, train_batch_size=64, test_batch_size=128, train_grid_gap=0.3, test_grid_gap=0.01)


if train_mode == 'train_h':


    NN = NeuralNetwork(dynamic_system=inverted_pendulum_1, data_module=data_module, require_grad_descent_loss=True)

    default_root_dir = "./CBF_logs/robust_training_maximum"

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=200,
        callbacks=[ EarlyStopping(monitor="Safety_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=50, stopping_threshold=1e-3) ], 
        default_root_dir=default_root_dir,
        reload_dataloaders_every_n_epochs=20,
        )

    torch.autograd.set_detect_anomaly(True)
    # trainer.fit(NN, ckpt_path="/home/wangxinyu/.mujoco/mujoco210/sunny_test/masterthesis_test/CBF_logs/robust_training_maximum/lightning_logs/version_4/checkpoints/epoch=92-step=1116.ckpt")
    trainer.fit(NN)

    torch.save(NN.data_module.s_training, "s_training.pt")




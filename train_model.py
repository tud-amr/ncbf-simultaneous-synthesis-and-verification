import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from matplotlib import cm
from matplotlib.ticker import LinearLocator

from MyNeuralNetwork import *
# from ValueFunctionNeuralNetwork import *
from dynamic_system_instances import car1, inverted_pendulum_1
from DataModule import DataModule


first_train = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

data_module = DataModule(system=inverted_pendulum_1, val_split=0, train_batch_size=128, test_batch_size=128, train_grid_gap=0.1, test_grid_gap=0.01)

default_root_dir = "./CBF_logs/robust_training_maximum_without_nominal_controller"

checkpoint_callback = ModelCheckpoint(dirpath=default_root_dir, save_top_k=1, monitor="Total_loss/train")

if first_train:


    NN = NeuralNetwork(dynamic_system=inverted_pendulum_1, data_module=data_module, require_grad_descent_loss=True, first_train=first_train)


    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=2000,
        # callbacks=[ EarlyStopping(monitor="Safety_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=50, stopping_threshold=1e-3) ], 
        callbacks = [checkpoint_callback],
        default_root_dir=default_root_dir,
        # reload_dataloaders_every_n_epochs=3,
        )

    torch.autograd.set_detect_anomaly(True)
    # trainer.fit(NN, ckpt_path="/home/wangxinyu/.mujoco/mujoco210/sunny_test/masterthesis_test/CBF_logs/robust_training_maximum/lightning_logs/version_4/checkpoints/epoch=92-step=1116.ckpt")
    trainer.fit(NN)

    torch.save(NN.data_module.s_training, "s_training.pt")

else:
    

    log_dir = default_root_dir + "/lightning_logs"

    version_list = os.listdir(log_dir)
    
    version_dir =  log_dir + "/" + version_list[-1]
    

    checkpoint_folder_dir = version_dir + "/checkpoints"
    

    checkpoint_name = os.listdir(checkpoint_folder_dir)[0]

    checkpoint_path = checkpoint_folder_dir + "/" + checkpoint_name 
    
   
    NN0 = NeuralNetwork.load_from_checkpoint(checkpoint_path,dynamic_system=inverted_pendulum_1, data_module=data_module, require_grad_descent_loss=True, first_train=first_train)


    NN = NeuralNetwork(dynamic_system=inverted_pendulum_1, data_module=data_module, require_grad_descent_loss=True, first_train=first_train)
    NN.set_previous_cbf(NN0.h)

    
    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=2000,
        # callbacks=[ EarlyStopping(monitor="Safety_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=50, stopping_threshold=1e-3) ], 
        default_root_dir=default_root_dir,
        # reload_dataloaders_every_n_epochs=3,
        accumulate_grad_batches=4
        )

    torch.autograd.set_detect_anomaly(True)
    # trainer.fit(NN, ckpt_path="/home/wangxinyu/.mujoco/mujoco210/sunny_test/masterthesis_test/CBF_logs/robust_training_maximum/lightning_logs/version_4/checkpoints/epoch=92-step=1116.ckpt")
    trainer.fit(NN)

    torch.save(NN.data_module.s_training, "s_training.pt")

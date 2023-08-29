import numpy as np
import os
import re
import matplotlib.pyplot as plt
import torch
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging

from matplotlib import cm
from matplotlib.ticker import LinearLocator

from safe_rl_cbf.NeuralCBF.NeuralCLBF import *
from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1, dubins_car, dubins_car_rotate ,dubins_car_acc, point_robots_dis, robot_arm_2d, two_vehicle_avoidance
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule


def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

########################### hyperparameters #############################

train_mode = 0
system = inverted_pendulum_1
default_root_dir = "./logs/CBF_logs/neural_clbf/inverted_pendulum_1"
checkpoint_dir = "logs/CBF_logs/inverted_pendulum_2/lightning_logs/version_1/checkpoints/epoch=93-step=7708.ckpt"
grid_gap = torch.Tensor([0.2, 0.2])  

########################## start training ###############################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# checkpoint_callback = ModelCheckpoint(dirpath=default_root_dir, save_top_k=1, monitor="Total_loss/train")
if train_mode==0:

    data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=1024, training_points_num=int(4e4), train_mode=train_mode)

    NN = NeuralCLBF(dynamic_system=system, data_module=data_module, train_mode=train_mode)
    

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=50,
        # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
        # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        default_root_dir=default_root_dir,
        # reload_dataloaders_every_n_epochs=15,
        accumulate_grad_batches=12,
        # gradient_clip_val=0.5
        )

    torch.autograd.set_detect_anomaly(True)
    trainer.fit(NN)

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

from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import *
# from ValueFunctionNeuralNetwork import *
from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1
from safe_rl_cbf.Dataset.DataModule import DataModule


def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)


fine_tune = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

data_module = DataModule(system=inverted_pendulum_1, val_split=0, train_batch_size=128, test_batch_size=1024, train_grid_gap=0.025, test_grid_gap=0.01)

default_root_dir = "./logs/CBF_logs/robust_training_maximum_with_value_function_4"

# checkpoint_callback = ModelCheckpoint(dirpath=default_root_dir, save_top_k=1, monitor="Total_loss/train")

if not fine_tune:


    # NN = NeuralNetwork(dynamic_system=inverted_pendulum_1, data_module=data_module, require_grad_descent_loss=True)
    NN = NeuralNetwork.load_from_checkpoint("logs/CBF_logs/robust_training_maximum_with_value_function_4/lightning_logs/version_0/checkpoints/epoch=117-step=5074.ckpt",dynamic_system=inverted_pendulum_1, data_module=data_module, require_grad_descent_loss=True, primal_learning_rate=8e-4, fine_tune=fine_tune)
   
    NN.training_stage = 1
    NN.set_previous_cbf(NN.h)

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=400,
        # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
        # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        default_root_dir=default_root_dir,
        reload_dataloaders_every_n_epochs=15,
        accumulate_grad_batches=12,
        # gradient_clip_val=0.5
        )

    torch.autograd.set_detect_anomaly(True)
    # trainer.fit(NN, ckpt_path="CBF_logs/robust_training_maximum_without_nominal_controller/lightning_logs/version_3/checkpoints/epoch=44-step=1935.ckpt")
    trainer.fit(NN)

    torch.save(NN.data_module.s_training, "s_training.pt")

else:
    
    
    log_dir = default_root_dir + "/lightning_logs"

    version_list = os.listdir(log_dir)

    version_dir =  log_dir + "/" +  max(version_list,key=extract_number)
    

    checkpoint_folder_dir = version_dir + "/checkpoints"
    

    checkpoint_name = os.listdir(checkpoint_folder_dir)[0]

    checkpoint_path = checkpoint_folder_dir + "/" + checkpoint_name 
    

    NN = NeuralNetwork.load_from_checkpoint(checkpoint_path,dynamic_system=inverted_pendulum_1, data_module=data_module, require_grad_descent_loss=True, fine_tune=fine_tune)

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=200,
        # callbacks=[ EarlyStopping(monitor="Safety_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=50, stopping_threshold=1e-3) ], 
        # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        default_root_dir=default_root_dir,
        reload_dataloaders_every_n_epochs=15,
        accumulate_grad_batches=12,
        )

    torch.autograd.set_detect_anomaly(True)
    # trainer.fit(NN, ckpt_path="/home/wangxinyu/.mujoco/mujoco210/sunny_test/masterthesis_test/CBF_logs/robust_training_maximum/lightning_logs/version_4/checkpoints/epoch=92-step=1116.ckpt")
    trainer.fit(NN)

    torch.save(NN.data_module.s_training, "s_training.pt")
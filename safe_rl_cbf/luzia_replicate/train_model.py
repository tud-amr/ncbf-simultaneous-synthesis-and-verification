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
from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1, dubins_car, dubins_car_rotate ,dubins_car_acc, point_robot ,point_robots_dis, robot_arm_2d, two_vehicle_avoidance
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule
from safe_rl_cbf.Dataset.TestingDataModule import TestingDataModule
from safe_rl_cbf.Analysis.draw_cbf import draw_cbf

def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

########################### hyperparameters #############################

train_mode = 2
system = inverted_pendulum_1
default_root_dir = "logs/cbf_logs"
checkpoint_dir = "saved_models/inverted_pendulum_stage_1/checkpoints/epoch=293-step=2646.ckpt"
grid_gap = torch.Tensor([0.2, 0.2])  

########################## start training ###############################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

if train_mode==2:
     
    data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=1024, training_points_num=int(1e5), train_mode=1, training_grid_gap=None)

    NN0 =  NeuralNetwork.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module, train_mode=1)
    NN = NeuralNetwork.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module, train_mode=1)
   
    NN.set_previous_cbf(NN0.h)

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=1,
        # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
        # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        default_root_dir=default_root_dir,
        reload_dataloaders_every_n_epochs=1000,
        accumulate_grad_batches=12,
        # gradient_clip_val=0.5
        )

    torch.autograd.set_detect_anomaly(True)
    trainer.fit(NN)
    
    del NN0, NN, trainer
    
    verification_time = 0
    training_start_time = time.time()

    for i in range(20):

        verification_start_time = time.time()
        
        data_module.train_mode = 2
        data_module.training_grid_gap = grid_gap
        data_module.prepare_data()

        log_dir = default_root_dir + "/lightning_logs"
        version_list = os.listdir(log_dir)
        version_dir =  log_dir + "/" +  max(version_list,key=extract_number)
        checkpoint_folder_dir = version_dir + "/checkpoints"
        checkpoint_name = os.listdir(checkpoint_folder_dir)[0]
        latest_checkpoint = checkpoint_folder_dir + "/" + checkpoint_name 
        print("latest_checkpoint: ", latest_checkpoint)

        NN0 =  NeuralNetwork.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module, train_mode=train_mode)
        NN = NeuralNetwork.load_from_checkpoint(latest_checkpoint,dynamic_system=system, data_module=data_module, train_mode=train_mode)
    
        NN.set_previous_cbf(NN0.h)

        trainer = pl.Trainer(
            accelerator = "gpu",
            devices = 1,
            max_epochs=1000,
            # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
            # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
            default_root_dir=default_root_dir,
            reload_dataloaders_every_n_epochs=1,
            accumulate_grad_batches=12,
            # gradient_clip_val=0.5
            )

        torch.autograd.set_detect_anomaly(True)
        trainer.fit(NN)

        print(f"data_module.verified = {data_module.verified}")
        print(f"augment_data.shape = {data_module.augment_data.shape}")
        torch.save(data_module.augment_data, "s_training.pt")
       
        verification_time += time.time() - verification_start_time
        counter_examples_num = data_module.augment_data.shape[0]

        if data_module.verified == 1:
            break
        
        del NN0, NN, trainer

        data_module.train_mode = 1
        data_module.training_grid_gap = None
        data_module.augment_dataset()

        log_dir = default_root_dir + "/lightning_logs"
        version_list = os.listdir(log_dir)
        version_dir =  log_dir + "/" +  max(version_list,key=extract_number)
        checkpoint_folder_dir = version_dir + "/checkpoints"
        checkpoint_name = os.listdir(checkpoint_folder_dir)[0]
        latest_checkpoint = checkpoint_folder_dir + "/" + checkpoint_name 

        NN0 =  NeuralNetwork.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module, train_mode=1)
        NN = NeuralNetwork.load_from_checkpoint(latest_checkpoint,dynamic_system=system, data_module=data_module, train_mode=1)
    
        NN.set_previous_cbf(NN0.h)

        trainer = pl.Trainer(
            accelerator = "gpu",
            devices = 1,
            max_epochs=20,
            # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
            # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
            default_root_dir=default_root_dir,
            reload_dataloaders_every_n_epochs=1000,
            accumulate_grad_batches=12,
            # gradient_clip_val=0.5
            )

        torch.autograd.set_detect_anomaly(True)
        trainer.fit(NN)

log_dir = default_root_dir + "/lightning_logs"
version_list = os.listdir(log_dir)
version_dir =  log_dir + "/" +  max(version_list,key=extract_number)
checkpoint_folder_dir = version_dir + "/checkpoints"
checkpoint_name = os.listdir(checkpoint_folder_dir)[0]
latest_checkpoint = checkpoint_folder_dir + "/" + checkpoint_name 

data_module = TestingDataModule(system=system, test_batch_size=1024, test_points_num=int(1e3), test_index={0: None, 1: None})

############################ plot figures ###################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

NN = NeuralNetwork.load_from_checkpoint(latest_checkpoint, dynamic_system=system, data_module=data_module)
NN.to(device)

trainer = pl.Trainer(accelerator = "gpu",
    devices = 1,
    max_epochs=1,)
trainer.test(NN)

draw_cbf(system=system)
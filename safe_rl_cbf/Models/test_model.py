import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import lightning.pytorch as pl


from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Models.custom_header import *
from safe_rl_cbf.Analysis.draw_cbf import draw_cbf



############################# hyperparameters #############################

system = select_dynamic_system("InvertedPendulum", "constraints_inverted_pendulum")
checkpoint_path = "saved_models/inverted_pendulum_umax_12/checkpoints/epoch=4-step=5.ckpt"
log_dir = "logs/CBF_logs/test"
data_module = TestingDataModule(system=system, test_index= {"0": "None", "1": "None"} , test_batch_size=1024, prefix= "test", log_dir=log_dir)   


############################ load model ###################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

NN = NeuralCBF.load_from_checkpoint(checkpoint_path, dynamic_system=system, data_module=data_module)
NN.to(device)
# NN.synchronize_cbf()
trainer = pl.Trainer(accelerator = "gpu",
    devices = 1,
    max_epochs=1,)
trainer.test(NN, datamodule=data_module.dataloader())

draw_cbf(system=system, log_dir=log_dir)
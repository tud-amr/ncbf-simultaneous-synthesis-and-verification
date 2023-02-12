import os
from os import path

import torch
import gymnasium as gym
from MyPendulum import MyPendulumEnv
from inverted_pendulum_callback import CustomCallback

# import stable_baseline3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

# import CBF
from MyNeuralNetwork import NeuralNetwork
from dynamic_system_instances import inverted_pendulum_1
from DataModule import DataModule


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

data_module = DataModule(system=inverted_pendulum_1, val_split=0.1, train_batch_size=64, test_batch_size=128, train_grid_gap=0.1, test_grid_gap=0.01)

NN = NeuralNetwork.load_from_checkpoint("masterthesis_test/lightning_logs/version_4/checkpoints/epoch=399-step=26400.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module, learn_shape_epochs=2 )
NN = NN.to(device)

log_dir = path.join(path.dirname(__file__), "stable_baseline_logs")
os.makedirs(log_dir, exist_ok=True)

my_ip_env = MyPendulumEnv("rgb_array", g=9.81, with_CBF= True)
my_ip_env.set_barrier_function(NN)

my_ip_env = Monitor(my_ip_env, filename=log_dir, info_keywords=("username",))

custom_cb = CustomCallback(log_dir=log_dir)

model = PPO("MlpPolicy", my_ip_env, verbose=1, tensorboard_log=log_dir )

model.learn(total_timesteps=50000, callback=custom_cb, tb_log_name= my_ip_env.prefix+"run1")
model.save("ppo_car")
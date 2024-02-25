import os
from os import path

import torch
import gymnasium as gym
from safe_rl_cbf.RL.InvertedPendulum.MyPendulum import MyPendulumEnv
from safe_rl_cbf.RL.InvertedPendulum.inverted_pendulum_callback import CustomCallback

# import stable_baseline3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

# import CBF
from safe_rl_cbf.Models.NeuralCBF import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


for i in range(0, 1):
    data_module = TrainingDataModule(system=inverted_pendulum_1, val_split=0, train_batch_size=1024, training_points_num=int(1e5), train_mode=0)

    NN = NeuralNetwork.load_from_checkpoint("saved_models/inverted_pendulum_umax_12/checkpoints/epoch=4-step=5.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module )
    

    log_dir = "logs/stable_baseline_logs_IP_23_Oct/IP_without/" + "run" + str(i) + "/"
    os.makedirs(log_dir, exist_ok=True)

    my_ip_env = MyPendulumEnv("rgb_array", g=9.81, with_CBF=False)
    my_ip_env.set_barrier_function(NN)

    custom_cb = CustomCallback(log_dir=log_dir)

    model = PPO("MlpPolicy", my_ip_env, verbose=1, tensorboard_log=log_dir )

    model.learn(total_timesteps=50000, callback=custom_cb, tb_log_name= my_ip_env.prefix)
    model.save(log_dir + "ip_without_CBF")

for i in range(0, 1):

    data_module = TrainingDataModule(system=inverted_pendulum_1, val_split=0, train_batch_size=1024, training_points_num=int(1e5), train_mode=0)

    NN = NeuralNetwork.load_from_checkpoint("saved_models/inverted_pendulum_umax_12/checkpoints/epoch=4-step=5.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module )
    

    log_dir = "logs/stable_baseline_logs_IP_23_Oct/IP_with/" + "run" + str(i) + "/"
    os.makedirs(log_dir, exist_ok=True)

    my_ip_env = MyPendulumEnv("rgb_array", g=9.81, with_CBF=True)
    my_ip_env.set_barrier_function(NN)

    my_ip_env = Monitor(my_ip_env, filename=log_dir, info_keywords=("username",))

    custom_cb = CustomCallback(log_dir=log_dir)

    model = PPO("MlpPolicy", my_ip_env, verbose=1, tensorboard_log=log_dir )

    model.learn(total_timesteps=50000, callback=custom_cb, tb_log_name= my_ip_env.prefix)
    model.save(log_dir + "ip_with_CBF")



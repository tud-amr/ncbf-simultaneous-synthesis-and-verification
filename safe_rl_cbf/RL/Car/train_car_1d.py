import os
from os import path

import torch
import gymnasium as gym
import numpy as np
from safe_rl_cbf.RL.Car.Car_1D import CAR1D
from safe_rl_cbf.RL.Car.Car_1D_callback import CustomCallback

# import stable_baseline3
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

# import CBF
from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise



log_dir = "logs/stable_baseline_logs_car_10_Nov/" + "run" + str(0) + "/"
os.makedirs(log_dir, exist_ok=True)

my_car_env = CAR1D("rgb_array", g=9.81, with_CBF=False)

custom_cb = CustomCallback()

model = PPO("MlpPolicy", my_car_env, verbose=1, tensorboard_log=log_dir )


model.learn(total_timesteps=100000, callback=custom_cb)
model.save(log_dir + "car_1d")



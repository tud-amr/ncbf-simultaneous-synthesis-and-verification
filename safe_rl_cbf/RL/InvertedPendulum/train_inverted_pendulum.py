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
from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1
from safe_rl_cbf.Dataset.DataModule import DataModule


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


for i in range(0, 1):

    # data_module = DataModule(system=inverted_pendulum_1, val_split=0.1, train_batch_size=64, test_batch_size=128, train_grid_gap=0.1, test_grid_gap=0.01)

    # NN = NeuralNetwork.load_from_checkpoint("saved_models/inverted_pendulum_umax_12/checkpoints/epoch=74-step=3225.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module, learn_shape_epochs=2 )
    # NN = NN.to(device)

    # # log_dir = path.join(path.dirname(__file__), "logs/stable_baseline_logs/" + "run" + str(i) + "/")
    # log_dir = "logs/stable_baseline_logs/" + "run" + str(i) + "/"
    # os.makedirs(log_dir, exist_ok=True)

    # my_ip_env = MyPendulumEnv("rgb_array", g=9.81, with_CBF=False)
    # my_ip_env.set_barrier_function(NN)

    # my_ip_env = Monitor(my_ip_env, filename=log_dir, info_keywords=("username",))

    # custom_cb = CustomCallback(log_dir=log_dir)

    # model = PPO("MlpPolicy", my_ip_env, verbose=1, tensorboard_log=log_dir )

    # model.learn(total_timesteps=100000, callback=custom_cb, tb_log_name= my_ip_env.prefix)
    # model.save(log_dir + "ip_without_CBF")


    data_module = DataModule(system=inverted_pendulum_1, val_split=0.1, train_batch_size=64, test_batch_size=128, train_grid_gap=0.1, test_grid_gap=0.01)

    NN = NeuralNetwork.load_from_checkpoint("saved_models/inverted_pendulum_umax_12/checkpoints/epoch=74-step=3225.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module, learn_shape_epochs=2 )
    NN = NN.to(device)

    log_dir = log_dir = "logs/stable_baseline_logs/" + "run" + str(i) + "/"
    os.makedirs(log_dir, exist_ok=True)

    my_ip_env = MyPendulumEnv("human", g=9.81, with_CBF=True)
    my_ip_env.set_barrier_function(NN)

    my_ip_env = Monitor(my_ip_env, filename=log_dir, info_keywords=("username",))

    custom_cb = CustomCallback(log_dir=log_dir)

    model = PPO("MlpPolicy", my_ip_env, verbose=1, tensorboard_log=log_dir )

    model.learn(total_timesteps=150000, callback=custom_cb, tb_log_name= my_ip_env.prefix)
    model.save(log_dir + "ip_with_CBF")



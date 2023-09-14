from stable_baselines3 import PPO
import gym
import os 
from safe_rl_cbf.RL.PointRobot.point_robot_callback import CustomCallback
from safe_rl_cbf.RL.PointRobot.point_robot_env import PointRobotEnv
import torch
from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import dubins_car, point_robot
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule


device = "cuda" if torch.cuda.is_available() else "cpu"
system = point_robot

for i in range(6):
    render_sim = False
    log_dir = "logs/stable_baseline_logs/point_robot_without/" + "run" + str(i) + "/"
    os.makedirs(log_dir, exist_ok=True)

    # data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=512, training_points_num=int(1e6))
    # NN = NeuralNetwork.load_from_checkpoint("saved_models/point_robot/checkpoints/epoch=68-step=11247.ckpt", dynamic_system=system, data_module=data_module )
    # NN.to(device)


    env = PointRobotEnv(render_sim=render_sim)
    # env.set_barrier_function(NN)

    custom_cb = CustomCallback(log_dir=log_dir)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    model.learn(total_timesteps=700000, callback=custom_cb, tb_log_name= env.prefix)
    model.save('point_robot_without_cbf')

for i in range(6):
    render_sim = False
    log_dir = "logs/stable_baseline_logs/point_robot_with/" + "run" + str(i) + "/"
    os.makedirs(log_dir, exist_ok=True)

    data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=512, training_points_num=int(1e6))
    NN = NeuralNetwork.load_from_checkpoint("saved_models/point_robot/checkpoints/epoch=68-step=11247.ckpt", dynamic_system=system, data_module=data_module )
    NN.to(device)


    env = PointRobotEnv(render_sim=render_sim)
    env.set_barrier_function(NN)

    custom_cb = CustomCallback(log_dir=log_dir)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    model.learn(total_timesteps=700000, callback=custom_cb, tb_log_name= env.prefix)
    model.save('point_robot_with_cbf')
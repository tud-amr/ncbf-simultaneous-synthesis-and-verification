from stable_baselines3 import PPO
import gym
import os 
from safe_rl_cbf.RL.PointRobot.point_robot_callback import CustomCallback
from safe_rl_cbf.RL.PointRobot.PointRobotEnv import PointRobotEnv
import torch
from safe_rl_cbf.Models.NeuralCBF import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import dubins_car, point_robot
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


device = "cuda" if torch.cuda.is_available() else "cpu"
system = point_robot

# for i in range(1, 2):
#     render_sim = False
#     log_dir = "logs/stable_baseline_logs/point_robot_without/" + "run" + str(i) + "/"
#     os.makedirs(log_dir, exist_ok=True)

#     # data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=512, training_points_num=int(1e6))
#     # NN = NeuralNetwork.load_from_checkpoint("saved_models/point_robot/checkpoints/epoch=68-step=11247.ckpt", dynamic_system=system, data_module=data_module )
#     # NN.to(device)


#     env = PointRobotEnv(render_sim=render_sim)
#     # env.set_barrier_function(NN)

#     custom_cb = CustomCallback(log_dir=log_dir)

#     model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

#     model.learn(total_timesteps=1500000, callback=custom_cb, tb_log_name= env.prefix)
#     model.save('point_robot_without')

for i in range(0, 2):
    render_sim = False
    log_dir = "logs/stable_baseline_logs_action_penalty/point_robot_with/" + "run" + str(i) + "/"
    os.makedirs(log_dir, exist_ok=True)

    data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=512, training_points_num=int(1e6))
    NN = NeuralNetwork.load_from_checkpoint("saved_models/point_robot/checkpoints/epoch=4-step=20.ckpt", dynamic_system=system, data_module=data_module )
    # NN.to(device)


    env = PointRobotEnv(render_sim=render_sim)
    env.set_barrier_function(NN)

    custom_cb = CustomCallback(log_dir=log_dir)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    model.learn(total_timesteps=1500000, callback=custom_cb, tb_log_name= env.prefix)
    model.save('point_robot_with_cbf')
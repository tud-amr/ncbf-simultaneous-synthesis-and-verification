import numpy as np
from stable_baselines3 import PPO
import torch

from safe_rl_cbf.RL.PointRobot.point_robot_env import PointRobotEnv
from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import dubins_car, point_robot
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
system = point_robot
render_sim = True #if True, a graphic is generated

env = PointRobotEnv(render_sim=render_sim)

# model = PPO.load("new_agent")
# model.set_env(env)

data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=512, training_points_num=int(1e6))
NN = NeuralNetwork.load_from_checkpoint("saved_models/point_robot/checkpoints/epoch=68-step=11247.ckpt", dynamic_system=system, data_module=data_module )
NN.to(device)

env.set_barrier_function(NN)

obs = env.reset()


try:
    while True:
        # action, _ = model.predict(obs) 
        action = np.array([0.1, 0.1])
        obs, reward, done, info = env.step(action)
        if render_sim is True:
            env.render()
        if done:
            obs = env.reset()
        
finally:
    env.close()
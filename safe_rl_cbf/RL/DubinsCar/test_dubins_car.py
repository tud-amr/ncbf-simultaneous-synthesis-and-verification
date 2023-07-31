import numpy as np
from stable_baselines3 import PPO
import torch

from safe_rl_cbf.RL.DubinsCar.dubins_car_env import DubinsCarEnv
from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import dubins_car
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
system = dubins_car
render_sim = True #if True, a graphic is generated

env = DubinsCarEnv(render_sim=render_sim)

# model = PPO.load("new_agent")
# model.set_env(env)

# data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=512, training_points_num=int(1e6))
# NN = NeuralNetwork.load_from_checkpoint("saved_models/dubins_car/checkpoints/epoch=64-step=10595.ckpt", dynamic_system=system, data_module=data_module )
# NN.to(device)

# env.set_barrier_function(NN)

obs = env.reset()


try:
    while True:
        # action, _ = model.predict(obs) 
        action = np.array([0.5, 1])
        obs, reward, done, info = env.step(action)
        if render_sim is True:
            env.render()
        if done:
            obs = env.reset()
        
finally:
    env.close()
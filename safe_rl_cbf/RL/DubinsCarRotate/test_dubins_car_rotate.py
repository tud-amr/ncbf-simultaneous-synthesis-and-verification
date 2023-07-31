import numpy as np
from stable_baselines3 import PPO
import torch

from safe_rl_cbf.RL.DubinsCarRotate.dubins_car_rotate_env import DubinsCarRotateEnv
from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import dubins_car_rotate
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
system = dubins_car_rotate
render_sim = True #if True, a graphic is generated

env = DubinsCarRotateEnv(render_sim=render_sim, v=0.5)

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
        action = np.array([-0.2])
        obs, reward, done, info = env.step(action)
        if render_sim is True:
            env.render()
        if done:
            obs = env.reset()
        
finally:
    env.close()
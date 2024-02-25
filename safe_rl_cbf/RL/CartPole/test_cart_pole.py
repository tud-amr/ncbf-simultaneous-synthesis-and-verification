import numpy as np
import torch
from safe_rl_cbf.RL.CartPole.MyCartPole import CartPoleEnv
from stable_baselines3 import PPO


from safe_rl_cbf.Models.NeuralCBF import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1
from safe_rl_cbf.Dataset.DataModule import DataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

data_module = DataModule(system=inverted_pendulum_1, val_split=0.1, train_batch_size=64, test_batch_size=128, train_grid_gap=0.1, test_grid_gap=0.01)

# NN = NeuralNetwork.load_from_checkpoint("saved_models/inverted_pendulum_umax_12/checkpoints/epoch=74-step=3225.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module, learn_shape_epochs=2 )
# NN = NN.to(device)


env = CartPoleEnv("human")
# env.set_barrier_function(NN)
model = PPO.load("logs/stable_baseline_logs/cartpole/run0/cartpole_without_CBF")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    action[0] = 0
    obs, rewards, dones, info = env.step(action)
    env.render()
    if  dones is True:     
        obs = env.reset()
    

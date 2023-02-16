import numpy as np
import torch
from MyPendulum import MyPendulumEnv
from stable_baselines3 import PPO


from MyNeuralNetwork import NeuralNetwork
from dynamic_system_instances import inverted_pendulum_1
from DataModule import DataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

data_module = DataModule(system=inverted_pendulum_1, val_split=0.1, train_batch_size=64, test_batch_size=128, train_grid_gap=0.1, test_grid_gap=0.01)

NN = NeuralNetwork.load_from_checkpoint("./masterthesis_test/CBF_logs/run6_with_OptNet/lightning_logs/version_0/checkpoints/epoch=59-step=4260.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module, learn_shape_epochs=2 )
NN = NN.to(device)


env = MyPendulumEnv("human", g=9.81, with_CBF=False)
env.set_barrier_function(NN)
model = PPO.load("./masterthesis_test/stable_baseline_logs/run1/ip_without_CBF.zip")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if np.linalg.norm(env.state) < 0.2:    # dones is True or 
        obs = env.reset()
    

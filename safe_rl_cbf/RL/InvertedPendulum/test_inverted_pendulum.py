import numpy as np
import torch
from safe_rl_cbf.RL.InvertedPendulum.MyPendulum import MyPendulumEnv
from stable_baselines3 import PPO


from safe_rl_cbf.Models.NeuralCBF import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

data_module = TrainingDataModule(system=inverted_pendulum_1, val_split=0, train_batch_size=1024, training_points_num=int(1e5), train_mode=0)

NN = NeuralNetwork.load_from_checkpoint("saved_models/inverted_pendulum_umax_12/checkpoints/epoch=4-step=5.ckpt", dynamic_system=inverted_pendulum_1, data_module=data_module )
NN = NN.to(device)

env = MyPendulumEnv("human", g=9.81, with_CBF=False)
env.set_barrier_function(NN)
model = PPO.load("logs/stable_baseline_logs_IP_23_Oct/IP_with/run2/ip_with_CBF.zip")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if  dones is True or np.linalg.norm(env.state) < 0.2:     
        obs = env.reset()
    

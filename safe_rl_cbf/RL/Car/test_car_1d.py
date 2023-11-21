import numpy as np
import torch
from safe_rl_cbf.RL.Car.Car_1D import CAR1D
from stable_baselines3 import PPO
import copy
import matplotlib.pyplot as plt

from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule
from safe_rl_cbf.RL.Car.car1d_neural_controller import ActionNet

env = CAR1D("human", g=9.81, with_CBF=False)

model = PPO.load("logs/stable_baseline_logs_car_10_Nov/run0/car_1d.zip")

neural_controller = ActionNet()
neural_controller.MlpExtractor.load_state_dict(model.policy.mlp_extractor.policy_net.state_dict()) 
neural_controller.action_net = copy.deepcopy(model.policy.action_net).cpu()
torch.save(neural_controller, "car1d_neural_controller.pt")

env.set_state(np.array([1.0, 0.0]))
obs = env._get_obs()

while True:
    action, _states = model.predict(obs)
    
    action2 = neural_controller(torch.tensor(obs).float()).detach().numpy()
    print(action, action2)
    exit()
    obs, rewards, dones, info = env.step(action2)
    env.render()
    if  dones is True or np.linalg.norm(env.state) < 0.2:     
        obs = env.reset()


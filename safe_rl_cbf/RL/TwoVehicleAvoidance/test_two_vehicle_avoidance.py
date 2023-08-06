import numpy as np
from stable_baselines3 import PPO
import torch

from safe_rl_cbf.RL.TwoVehicleAvoidance.two_vehicle_avoidance_env import TwoVehicleAvoidanceEnv
from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import dubins_car, point_robot, point_robots_dis, two_vehicle_avoidance
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
system = two_vehicle_avoidance
render_sim = True #if True, a graphic is generated

env = TwoVehicleAvoidanceEnv(render_sim=render_sim)

# model = PPO.load("new_agent")
# model.set_env(env)

data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=512, training_points_num=int(1e6))
NN = NeuralNetwork.load_from_checkpoint("saved_models/two_vehicle_avoidance/checkpoints/epoch=655-step=106928.ckpt", dynamic_system=system, data_module=data_module )
NN.to(device)

env.set_barrier_function(NN)

obs = env.reset()
random_action = np.random.rand(1) - 0.5

try:
    while True:
        # action, _ = model.predict(obs) 
        action = np.zeros(1)
        obs, reward, done, info = env.step(action)
        if render_sim is True:
            env.render()
        if done:
            random_action = np.random.rand(1) - 0.5
            print("random_action", random_action)
            print("Done")
            obs = env.reset()

        if env.current_time_step > 700:

            random_action = np.random.rand(1) - 0.5
            print("random_action", random_action)
            print("Time out")
            obs = env.reset()
finally:
    env.close()
import numpy as np
from stable_baselines3 import PPO
import torch

from safe_rl_cbf.RL.RobotArm.robot_arm_env import RobotArmEnv
from safe_rl_cbf.Models.NeuralCBF import NeuralNetwork
from safe_rl_cbf.Dynamics.dynamic_system_instances import dubins_car, point_robot, robot_arm_2d
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
system = robot_arm_2d
render_sim = True #if True, a graphic is generated

env = RobotArmEnv(render_sim=render_sim)

# model = PPO.load("new_agent")
# model.set_env(env)

data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=512, training_points_num=int(1e6))
NN = NeuralNetwork.load_from_checkpoint("saved_models/robot_arm_2d/checkpoints/epoch=208-step=34067.ckpt", dynamic_system=system, data_module=data_module )
NN.to(device)

env.set_barrier_function(NN)

obs = env.reset()

random_action = np.random.rand(2) - 0.5 
print("random_action", random_action)
try:
    while True:
        # action, _ = model.predict(obs) 
        action = random_action
        obs, reward, done, info = env.step(action)
        if render_sim is True:
            env.render()
        if done:
            obs = env.reset()
        if env.current_time_step > 700:

            random_action = np.random.rand(2) - 0.5
            print("random_action", random_action)
            print("Time out")
            obs = env.reset()
finally:
    env.close()
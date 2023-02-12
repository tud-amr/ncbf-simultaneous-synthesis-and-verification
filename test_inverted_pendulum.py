import numpy as np
from MyPendulum import MyPendulumEnv
from stable_baselines3 import PPO
env = MyPendulumEnv("human", g=9.81)
model = PPO.load("ppo_car")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones is True:   
        obs = env.reset()
    

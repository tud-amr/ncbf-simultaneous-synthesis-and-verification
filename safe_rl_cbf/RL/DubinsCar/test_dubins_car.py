import numpy as np
from stable_baselines3 import PPO

from safe_rl_cbf.RL.DubinsCar.dubins_car_env import DubinsCarEnv


render_sim = True #if True, a graphic is generated

env = DubinsCarEnv(render_sim=render_sim)

#model = PPO.load("new_agent")

# model.set_env(env)

obs = env.reset()

try:
    while True:
        # action, _ = model.predict(obs) 
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if render_sim is True:
            env.render()
        if done:
            obs - env.reset()
finally:
    env.close()
from stable_baselines3 import PPO
import gym
from safe_rl_cbf.RL.DubinsCarAcc.dubins_car_acc_env import DubinsCarEnv

render_sim = False
env = DubinsCarEnv(render_sim=render_sim)


model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=1000000)
model.save('dubins_car_acc')

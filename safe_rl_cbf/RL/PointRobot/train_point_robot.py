from stable_baselines3 import PPO
import gym
from safe_rl_cbf.RL.PointRobot.point_robot_env import PointRobotEnv

render_sim = False
env = PointRobotEnv(render_sim=render_sim)


model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=1000000)
model.save('new_agent')

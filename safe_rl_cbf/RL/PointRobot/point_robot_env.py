
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
import random
import os
import pygame
import numpy as np
import pymunk
from pymunk import Vec2d
import pymunk.pygame_util
import time
from safe_rl_cbf.RL.PointRobot.PointRobot import PointRobot

class PointRobotEnv(gym.Env):
    def __init__(self, render_sim=False):
        """
        Initialize the PointRobot environment
        the state is [x, y, v_x, v_y]
        the action is [a_x, a_y]
        """
       
        
        # state property
        self.x_max = 4.0
        self.y_max = 4.0    
        self.v_max = 0.4
       
        
        min_observation = np.array([0, 0, -1, -1], dtype=np.float32)
        max_observation = np.array([1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=min_observation, high=max_observation, dtype=np.float32)
        
        self.radius = 0.15

        self.x_init = 0.5
        self.y_init = 1.0
        self.v_init = 0.0
      
        self.state = np.array([self.x_init, self.y_init, self.v_init, self.v_init])

        # action property
        self.max_a_x = 1.0
        self.max_a_y = 1.0
        self.action_space = spaces.Box(low=np.array([-self.max_a_x, -self.max_a_y]), high=np.array([self.max_a_x, self.max_a_y]), dtype=np.float32)
        
        # simulation property
        self.current_time_step = 0
        self.max_time_steps = 1500
        self.scale = 100.0
        self.dt = 1/50
        self.render_sim = render_sim

        # target property
        self.x_target_max = self.x_max * 0.7 
        self.x_target_min = self.x_max * 0.4
        self.y_target_max = self.y_max * 0.7
        self.y_target_min = self.y_max * 0.4
        self.x_target = 3 # random.uniform(self.x_target_min, self.x_target_max)
        self.y_target = 3 # random.uniform(self.y_target_min, self.y_target_max)
        self.target = np.array([self.x_target, self.y_target])
        self.target_radius = 0.2

        # pygame property
       
        if self.render_sim is True:
            self.init_pygame()

        self.init_pymunk()
        self.init_agent()
        self.init_obstacles()

        self.break_safety = 0
        self.epoch_trajectory = []
        self.epoch_reward = []
        self.training_trajectories = []
        self.training_rewards = []
        self.h = None
        self.use_cbf = False
        self.prefix = "without_CBF_"

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.x_max * self.scale, self.y_max * self.scale))
        pygame.display.set_caption("Dubins Car 2d Environment")
        self.clock = pygame.time.Clock()

    def init_pymunk(self):
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, 0)
        
        if self.render_sim is True:
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
            pymunk.pygame_util.positive_y_is_up = True
    
    def init_agent(self):
        self.point_robot = PointRobot(1, self.radius, self.state, self.space, self.scale, self.dt)

    def init_obstacles(self):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        x = 2
        y = 1
        body.position = (x * self.scale, y * self.scale)
        width = 1 - self.radius * 2
        height = 2 - self.radius * 2
        shape = pymunk.Poly.create_box(body, size=( width * self.scale, height * self.scale ))
        self.space.add(body, shape)
    
    def set_barrier_function(self, h):
        self.h = h
        self.use_cbf = True
        self.prefix = "with_CBF_"

    def step(self, action):
        start_time = time.time()
        reward = 0
        action = action 

        if self.use_cbf:
            if self.h is None:
                raise Exception(f"Please use self.set_barrier_function to set barrier function")
            
            device = self.h.device
            s = torch.from_numpy(self.state).float().reshape((-1, self.h.dynamic_system.ns)).to(device)
            
            hs = self.h(s)

            if hs < 0.1:
                gradh = self.h.jacobian(hs, s)
                
                
                u_ref = torch.from_numpy(action).float().reshape((-1,self.h.dynamic_system.nu)).to(device)            
                u_result, r_result = self.h.solve_CLF_QP(s, gradh, u_ref, epsilon=0.0)

            
                u = u_result.cpu().numpy().flatten()

                if hs < 0:
                    # print(f"Current state [{self.state}] is unsafe, h(s)={hs}, u={u}")
                    # raise Exception(f"Current state [{self.state[0]}, {self.state[1]}] is unsafe, h(s)={hs}")
                    pass

                if r_result > 0.0:
                    self.break_safety += 1
                #     raise Exception(f"The QP is infeasible, slack variable is {r_result}")

                if torch.abs( torch.norm(u_result - u_ref) ) > 0.1:
                    # pass
                    # print(f"u_ref={u_ref}, u_result={u_result}, hs={hs}")
                    reward += -0.02
                    # done = True
            else:
                u = action
        else:
            u = action


        obs = self.get_observation()
        x, y, v_x, v_y = self.state

        if obs[2] == 1:
            u[0] = np.clip(u[0], -self.max_a_x, 0)
        if obs[3] == 1:
            u[1] = np.clip(u[1], -self.max_a_x, 0)
        if obs[2] == -1:
            u[0] = np.clip(u[0], 0, self.max_a_x)
        if obs[3] == -1:
            u[1] = np.clip(u[1], 0, self.max_a_x)
           
        F1 = u[0] * self.point_robot.mass * self.scale
        F2 = u[1] * self.point_robot.mass * self.scale

        self.point_robot.shape.body.apply_force_at_local_point((F1, 0), (0, 0))
        self.point_robot.shape.body.apply_force_at_local_point((0, F2), (0, 0))
        # self.car.shape.body.apply_force_at_local_point((0, -F2), (-self.car.radius, 0))

        self.space.step(self.dt)
        # self.car.step(u)
        self.current_time_step += 1


        

        distance_to_target_x = self.x_target - x
        distance_to_target_y = self.y_target - y
        distance_to_target = np.sqrt(distance_to_target_x ** 2 + distance_to_target_y ** 2 )
        heading_to_target = self.normalize_angle(np.arctan2(distance_to_target_y, distance_to_target_x))
        heading = self.normalize_angle(np.arctan2(v_y, v_x))
        heading_error = heading_to_target - heading


        # reward += (1.0 / (  np.abs(distance_to_target_x) + 0.1 )) + (1.0 / (  np.abs(distance_to_target_y) + 0.1 )) - 0.01 * np.abs(heading_error)
        reward += - 0.01 * distance_to_target -  0.01 * np.abs(heading_error) 
        # reward = -0.01
        
        done = False
        info = {}
        self.epoch_trajectory.append( np.array([[x], [y]]) )

        if np.abs(obs[0]) == 0 or np.abs(obs[1]) == 0 or np.abs(obs[0])==1 or np.abs(obs[1])==1:
            # reach the boundary
            print("collision with wall!!!!!!!!!!!!")
            reward = -5
            self.break_safety += 1
            done = True
            self.current_time_step = 0
            self.training_trajectories.append( {"traj": np.hstack(self.epoch_trajectory), "collision": True})
            self.epoch_trajectory.clear()
            
            # print(f"state is {self.state}")
            

        elif np.abs(x - 2) < 0.5  and np.abs(y - 1) < 1 :
            print("collision with obstacle!!!!!!!!!!!!")
            reward = -5
            self.break_safety += 1
            done = True
            self.current_time_step = 0
            self.training_trajectories.append( {"traj": np.hstack(self.epoch_trajectory), "collision": True})
            self.epoch_trajectory.clear()
            
            

        elif np.linalg.norm(np.array([x, y]) - np.array([self.x_target, self.y_target])) < 0.2:
            print("reach goal!!!!!!!!!!!!")
            reward = 10
            done = True
            self.current_time_step = 0
            self.training_trajectories.append( {"traj": np.hstack(self.epoch_trajectory), "collision": False})
            self.epoch_trajectory.clear()

           
             
        
        elif self.current_time_step > self.max_time_steps:
            print("reach max time steps!!!!!!!!!!!!")
            done = True
            self.current_time_step = 0
            self.training_trajectories.append( {"traj": np.hstack(self.epoch_trajectory), "collision": False})
            self.epoch_trajectory.clear()
            
            
        else:
            pass

        

        self.reward = reward

        end_time = time.time()
        self.step_executing_time = end_time - start_time

        return obs, reward, done, info
    
    def get_observation(self):
        x, y = self.point_robot.shape.body.position / self.scale
        v_x, v_y = self.point_robot.shape.body.velocity / self.scale

        self.state = np.array([x, y, v_x, v_y])


        x_obs = np.clip(x/self.x_max, 0, 1)
        y_obs = np.clip(y/self.y_max, 0, 1)
        v_x_obs = np.clip(v_x/self.v_max, -1, 1)
        v_y_obs = np.clip(v_y/self.v_max, -1, 1)
        
        return np.array([x_obs, y_obs, v_x_obs, v_y_obs], dtype=np.float32)

    def reset(self):
        
        self.x_init = random.uniform(0.3, 1)
        self.y_init = random.uniform(0.5, 3)
        self.point_robot.set_states(self.x_init, self.y_init)
        return self.get_observation()

    def render(self, mode='human'): 
        if self.render_sim is False:
            return None
        
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)

        target_pos_in_screen = (self.x_target * self.scale, self.y_max * self.scale - self.y_target * self.scale)
        pygame.draw.circle(self.screen, (0, 255, 0), target_pos_in_screen, self.target_radius * self.scale)
        self.space.step(self.dt)
        pygame.display.flip()
        self.clock.tick(1/self.dt)

    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    env = PointRobotEnv(render_sim=True)
    
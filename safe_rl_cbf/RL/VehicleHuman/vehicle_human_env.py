
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
import random
import os

import pygame
from pygame.locals import *

import pymunk
from pymunk.pygame_util import *
from pymunk.vec2d import Vec2d

from safe_rl_cbf.RL.VehicleHuman.DubinsCarAcc import DubinsCar
from safe_rl_cbf.RL.VehicleHuman.PointRobot import PointRobot

class VehicleHumanEnv(gym.Env):
    def __init__(self, render_sim=False):
        """
        Initialize the PointRobot environment
        the state is [x, y, v_x, v_y]
        the action is [a_x, a_y]
        """
       

        # state property
        self.x_e_max = 3.0
        self.y_e_max = 3.0    
        self.theta_e_max = np.pi
        self.v_e_max = 1.0
        self.w_e_max = 1.0
        
        min_observation = np.array([0, 0, -1, -1, -1, 0, 0, -1, -1], dtype=np.float32)
        max_observation = np.array([1, 1,  1,  1,  1, 1, 1,  1,  1], dtype=np.float32)
        self.observation_space = spaces.Box(low=min_observation, high=max_observation, dtype=np.float32)
        
        self.radius = 0.2

        self.x_e_init = 1.0
        self.y_e_init = 1.0
        self.theta_e_init = 0.0
        self.v_e_init = 0.3
        self.w_e_init = 0.0
      
        self.ego_state = np.array([self.x_e_init, self.y_e_init, self.theta_e_init, self.v_e_init, self.w_e_init])
        self.human_state = np.array([self.x_e_init, self.y_e_init, self.v_e_init, self.v_e_init])
        # action property
        self.max_a_e = 1.0
        self.max_alpha_e = 1.0
        self.action_space = spaces.Box(low=np.array([-self.max_a_e, -self.max_alpha_e]), high=np.array([self.max_a_e, self.max_alpha_e]), dtype=np.float32)
        
        # simulation property
        self.current_time_step = 0
        self.max_time_steps = 1000
        self.scale = 150.0
        self.dt = 1/50
        self.render_sim = render_sim

        # target property
        self.x_target_max = self.x_e_max * 0.7 
        self.x_target_min = self.x_e_max * 0.4
        self.y_target_max = self.y_e_max * 0.7
        self.y_target_min = self.y_e_max * 0.4
        self.x_target = 1.5 # random.uniform(self.x_target_min, self.x_target_max)
        self.y_target = 1.5 # random.uniform(self.y_target_min, self.y_target_max)
        self.target = np.array([self.x_target, self.y_target])
        self.target_radius = 0.2

        # pygame property
       
        if self.render_sim is True:
            self.init_pygame()

        self.init_pymunk()
        self.init_agent()
        # self.init_obstacles()

        self.h = None
        self.use_cbf = False

        self.keys = {K_LEFT: (-0.2, 0), K_RIGHT: (0.2, 0),
                K_UP: (0, 0.2), K_DOWN: (0, -0.2)}

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.x_e_max * self.scale, self.y_e_max * self.scale))
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
        self.ego_vehicle = DubinsCar(1, self.radius, self.ego_state, self.space, self.scale, self.dt)
        self.human = PointRobot(1, self.radius, self.human_state, self.space, self.scale, self.dt)
        
    def init_obstacles(self):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        x = 5
        y = 5
        body.position = (x * self.scale, y * self.scale)
        width = 2
        height = 2
        shape = pymunk.Poly.create_box(body, size=( width * self.scale, height * self.scale ))
        self.space.add(body, shape)
    
    def set_barrier_function(self, h):
        self.h = h
        self.use_cbf = True

    def step(self, action):
        
        reward = 0
        action = action 

        if self.use_cbf:
            if self.h is None:
                raise Exception(f"Please use self.set_barrier_function to set barrier function")
            
            device = self.h.device
            s = np.hstack([self.ego_state, self.human_state])
            s = torch.from_numpy(s).float().reshape((-1, self.h.dynamic_system.ns)).to(device)
            
            hs = self.h(s)
            gradh = self.h.jacobian(hs, s)
            if hs < 0:
                # raise Exception(f"Current state [{self.state[0]}, {self.state[1]}] is unsafe, h(s)={hs}")
                print(f"Current state [{self.ego_state}, {self.human_state}] is unsafe, h(s)={hs}")
            
            u_ref = torch.from_numpy(action).float().reshape((-1,self.h.dynamic_system.nu)).to(device)            
            u_result, r_result = self.h.solve_CLF_QP(s, gradh, u_ref, epsilon=0.1)

            if r_result > 0.0:
                self.break_safety += 1
                # raise Exception(f"The QP is infeasible, slack variable is {r_result}")
                print(f"The QP is infeasible, slack variable is {r_result}")

            if torch.abs( torch.norm(u_result - u_ref) ) > 0.1:
                reward += -15

            u = u_result.cpu().numpy().flatten()
        else:
            u = action

        
       
        
        # get command from keyboard
        pygame.event.get()
        self.do_event()
        
        self.ego_vehicle.step(u)
        # self.space.step(self.dt)
       
        self.current_time_step += 1


        obs = self.get_observation()
        x, y, theta, v, w= self.ego_state

        distance_to_target_x = self.x_target - x
        distance_to_target_y = self.y_target - y

        reward = (1.0 / (  np.abs(distance_to_target_x) + 0.1 )) + (1.0 / (  np.abs(distance_to_target_y) + 0.1 ))
        done = False

        if np.abs(obs[0]) == 0 or np.abs(obs[1]) == 0 or np.abs(obs[0])==1 or np.abs(obs[1])==1:
            # reach the boundary
            reward = -10
            done = True
        elif np.abs(obs[3]) == 1 or np.abs(obs[4]) == 1 or np.abs(obs[3]) == -1 or np.abs(obs[4]) == -1:
            # reach the maximum velocity or angular velocity
            print("reach the maximum velocity or angular velocity")
            reward = -10
            done = True
      
        if self.current_time_step == self.max_time_steps:
            self.done = True
        
        # if np.abs(x - 5) < 1 + self.radius and np.abs(y - 5) < 1 + self.radius:
        #     reward = -10
        #     done = True

        
        info = {}
        return obs, reward, done, info
    
    def get_observation(self):
        x, y = self.human.shape.body.position / self.scale
        v_x, v_y = self.human.shape.body.velocity / self.scale
        self.human_state = np.array([x, y, v_x, v_y])

        x, y = self.ego_vehicle.shape.body.position / self.scale
        v = self.ego_vehicle.shape.body.velocity.length / self.scale
        w = self.ego_vehicle.shape.body.angular_velocity
        theta = self.normalize_angle(self.ego_vehicle.shape.body.angle)

        self.ego_state = np.array([x, y, theta, v, w])

        x_obs = np.clip(x/self.x_e_max, 0, 1)
        y_obs = np.clip(y/self.y_e_max, 0, 1)
        theta_obs = np.clip(v_x/np.pi, -1, 1)
        v_obs = np.clip(v/self.v_e_max, -1, 1)
        w_obs = np.clip(w/self.w_e_max, -1, 1)
        
        return np.array([x_obs, y_obs, theta_obs, v_obs, w_obs ], dtype=np.float32)

    def reset(self):
        self.current_time_step = 0
        self.x_init = random.uniform(0, self.x_e_max)
        self.y_init = random.uniform(0, self.y_e_max)
        theta_to_target = np.arctan2(self.y_target - self.y_init, self.x_target - self.x_init)
        self.theta_init = theta_to_target  # random.uniform(-self.theta_max, self.theta_max)
        self.ego_vehicle.set_states(self.x_init, self.y_init, self.theta_init)
        
        self.x_init = 2 * self.x_target - self.x_init
        self.y_init = 2 * self.y_target - self.y_init
        
        self.human.set_states(self.x_init, self.y_init)
        return self.get_observation()


    def render(self, mode='human'): 
        if self.render_sim is False:
            return None
        
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)

        target_pos_in_screen = (self.x_target * self.scale, self.y_e_max * self.scale - self.y_target * self.scale)
        pygame.draw.circle(self.screen, (0, 255, 0), target_pos_in_screen, self.target_radius * self.scale)
        self.space.step(self.dt)
        pygame.display.flip()
        self.clock.tick(1/self.dt)

    def do_event(self):
        keys = pygame.key.get_pressed()
        keys_pressed = [k for k, v in self.keys.items() if keys[k]]
        x, y , v_x, v_y = self.human_state
        if abs(v_x) < 1 and abs(v_y) < 1:
            for key in keys_pressed:
                F = Vec2d(self.keys[key][0], self.keys[key][1] ) * self.human.mass * self.scale
                self.human.shape.body.apply_force_at_local_point(F, (0, 0))
        # if event.type == KEYDOWN:
        #     if event.key in self.keys:
        #         F = Vec2d(self.keys[event.key][0], self.keys[event.key][1] ) * self.ego_point_robot.mass * self.scale
        #         self.component_point_robot.shape.body.apply_force_at_local_point(F, (0, 0)) 

    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


if __name__ == "__main__":
    env = PointRobotsEnv(render_sim=True)
    
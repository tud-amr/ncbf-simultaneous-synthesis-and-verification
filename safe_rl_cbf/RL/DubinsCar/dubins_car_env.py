
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
from safe_rl_cbf.RL.DubinsCar.DubinsCar import DubinsCar

class DubinsCarEnv(gym.Env):
    def __init__(self, render_sim=False):
        """
        Initialize the DubinsCar environment
        the state is [x, y, theta]
        the action is [velocity, omega]
        """
       

        # state property
        self.x_max = 8.0
        self.y_max = 8.0
        self.theta_max = 4.0
        self.v_max = 1.0
        self.omega_max = 1.0
       
        
        min_observation = np.array([0, 0, -1, -1, -1], dtype=np.float32)
        max_observation = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=min_observation, high=max_observation, dtype=np.float32)
        
        self.radius = 0.3

        self.x_init = 1.0
        self.y_init = 1.0
        self.theta_init = 0.0
        self.v_init = 0.0
        self.omega_init = 0.0
        self.state = np.array([self.x_init, self.y_init, self.theta_init, self.v_init, self.omega_init])

        # action property
        self.max_acc = 1.0
        self.max_alpha = 1.0
        self.action_space = spaces.Box(low=np.array([-self.max_acc, -self.max_alpha]), high=np.array([self.max_acc, self.max_alpha]), dtype=np.float32)
        
        # simulation property
        self.current_time_step = 0
        self.max_time_steps = 1000
        self.scale = 50.0
        self.dt = 1/50
        self.render_sim = render_sim

        # target property
        self.x_target_max = self.x_max * 0.7 
        self.x_target_min = self.x_max * 0.4
        self.y_target_max = self.y_max * 0.7
        self.y_target_min = self.y_max * 0.4
        self.x_target = 5 # random.uniform(self.x_target_min, self.x_target_max)
        self.y_target = 2 # random.uniform(self.y_target_min, self.y_target_max)
        self.target = np.array([self.x_target, self.y_target])
        self.target_radius = 0.2

        # pygame property
       
        if self.render_sim is True:
            self.init_pygame()

        self.init_pymunk()
        self.init_agent()
        self.init_obstacles()

        self.h = None
        self.use_cbf = False

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
        self.car = DubinsCar(1, self.radius, self.state, self.space, self.scale, self.dt)

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
        action = action / 5

        if self.use_cbf:
            if self.h is None:
                raise Exception(f"Please use self.set_barrier_function to set barrier function")
            
            device = self.h.device
            s = torch.from_numpy(self.state).float().reshape((-1, self.h.dynamic_system.ns)).to(device)
            
            hs = self.h(s)
            gradh = self.h.jacobian(hs, s)
            # if hs < 0:
            #     raise Exception(f"Current state [{self.state[0]}, {self.state[1]}] is unsafe, h(s)={hs}")
            
            u_ref = torch.from_numpy(action).float().reshape((-1,self.h.dynamic_system.nu)).to(device)            
            u_result, r_result = self.h.solve_CLF_QP(s, gradh, u_ref, epsilon=0.0)

            if r_result > 0.0:
                self.break_safety += 1
            #     raise Exception(f"The QP is infeasible, slack variable is {r_result}")

            if torch.abs( torch.norm(u_result - u_ref) ) > 0.1:
                reward += -15

            u = u_result.cpu().numpy().flatten()
        else:
            u = action

        
        # F1 = u[0] * self.car.mass * self.scale
        # F2 = u[1] * self.car.mass * self.car.radius / 4 

        # self.car.shape.body.apply_force_at_local_point((F1, 0), (0, 0))
        # self.car.shape.body.apply_force_at_local_point((0, F2), (self.car.radius, 0))
        # self.car.shape.body.apply_force_at_local_point((0, -F2), (-self.car.radius, 0))

        # self.space.step(self.dt)
        self.car.step(u)
        self.current_time_step += 1


        obs = self.get_observation()
        x, y, theta, v, w = self.state

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
            reward = -10
            done = True

        if self.current_time_step == self.max_time_steps:
            self.done = True
        
        if np.abs(x - 5) < 1 + self.radius and np.abs(y - 5) < 1 + self.radius:
            reward = -10
            done = True

        
        info = {}
        return obs, reward, done, info
    
    def get_observation(self):
        x, y = self.car.shape.body.position / self.scale
        theta = self.car.shape.body.angle
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        v = self.car.shape.body.velocity.length / self.scale
        w = self.car.shape.body.angular_velocity
        self.state = np.array([x, y, theta, v, w])


        x_obs = np.clip(x/self.x_max, 0, 1)
        y_obs = np.clip(y/self.y_max, 0, 1)
        theta_obs = np.clip(theta/(4), -1, 1)
        v_obs = np.clip(v/self.v_max, -1, 1)
        w_obs = np.clip(w/self.omega_max, -1, 1)
        
        return np.array([x_obs, y_obs, theta_obs, v_obs, w_obs], dtype=np.float32)

    def reset(self):
        self.current_time_step = 0
        self.x_init = random.uniform(0, self.x_max)
        self.y_init = random.uniform(0, self.y_max)
        self.theta_init = random.uniform(-self.theta_max, self.theta_max)
        self.car.set_states(self.x_init, self.y_init, self.theta_init)
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


if __name__ == "__main__":
    env = DubinsCarEnv(render_sim=True)
    
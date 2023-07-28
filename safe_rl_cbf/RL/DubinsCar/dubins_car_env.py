
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
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
        self.theta_max = np.pi
        self.state = np.array([0.0, 0.0, 0.0])
        self.observation_space = spaces.Box(low=np.array([0, 0, -self.theta_max]), high=np.array([self.x_max, self.y_max, self.theta_max]), dtype=np.float32)
        self.radius = 0.3

        self.x_init = 1.0
        self.y_init = 1.0
        self.theta_init = 0.0

        # action property
        self.max_velocity = 1.0
        self.max_omega = 1.0
        self.action_space = spaces.Box(low=np.array([-self.max_velocity, -self.max_omega]), high=np.array([self.max_velocity, self.max_omega]), dtype=np.float32)
        
        # simulation property
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
        self.init_obstacles()

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
        
        self.car = DubinsCar(1, self.radius, self.x_init, self.y_init, self.theta_init, self.space, self.scale, self.dt)

    def init_obstacles(self):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        x = 5
        y = 5
        body.position = (x * self.scale, y * self.scale)
        width = 2
        height = 2
        shape = pymunk.Poly.create_box(body, size=( width * self.scale, height * self.scale ))
        self.space.add(body, shape)

    def step(self, action):

        self.car.step(action)
        obs = self.get_observation()
        x, y, theta = obs

        distance_to_target_x = self.x_target - x
        distance_to_target_y = self.y_target - y

        reward = (1.0 / ( np.abs(distance_to_target_x) + 0.1 )) + (1.0 / ( np.abs(distance_to_target_y) + 0.1 ))
        done = False

        if np.abs(x) == 0 or np.abs(y) == 0 or np.abs(x)==self.x_max or np.abs(y)==self.y_max:
            reward -= 10
            done = True
        
        if np.abs(x - 5) < 1 + self.radius and np.abs(y - 5) < 1 + self.radius:
            reward -= 10
            done = True

        
        info = {}
        return obs, reward, done, info
    
    def get_observation(self):
        x, y = self.car.shape.body.position / self.scale
        theta = self.car.shape.body.angle

        x = np.clip(x, 0, self.x_max)
        y = np.clip(y, 0, self.y_max)
        theta = np.clip(theta, -self.theta_max, self.theta_max)
        self.state = np.array([x, y, theta])
        return self.state

    def reset(self):
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

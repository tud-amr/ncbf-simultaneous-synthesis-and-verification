
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
from safe_rl_cbf.RL.RobotArm.RobotArm import RobotArm

class RobotArmEnv(gym.Env):
    def __init__(self, render_sim=False):
        """
        Initialize the PointRobot environment
        the state is [x, y, v_x, v_y]
        the action is [a_x, a_y]
        """
       

        # state property
        self.x_max = 8.0
        self.y_max = 8.0    
        self.theta_max = np.pi
       
        
        min_observation = np.array([0, 0, -1], dtype=np.float32)
        max_observation = np.array([1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=min_observation, high=max_observation, dtype=np.float32)
        
        

        # action property
        self.max_w = 1.0
        self.action_space = spaces.Box(low=np.array([-self.max_w, -self.max_w]), high=np.array([self.max_w, self.max_w]), dtype=np.float32)
        
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
        # self.init_obstacles()

        self.h = None
        self.use_cbf = False

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.x_max * self.scale, self.y_max * self.scale))
        pygame.display.set_caption("Dubins Car 2d Environment")
        self.clock = pygame.time.Clock()

    def init_pymunk(self):
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, -9.8 * self.scale)
        
        if self.render_sim is True:
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
            pymunk.pygame_util.positive_y_is_up = True
    
    def init_agent(self):
        self.robot_arm = RobotArm(self.space, self.scale, self.dt)

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
            s = np.array([self.l1_state[2], self.l2_state[2]])
            s = torch.from_numpy(s).float().reshape((-1, self.h.dynamic_system.ns)).to(device)
            
            hs = self.h(s)
            gradh = self.h.jacobian(hs, s)
            # if hs < 0:
            #     raise Exception(f"Current state [{self.state[0]}, {self.state[1]}] is unsafe, h(s)={hs}")
            
            u_ref = torch.from_numpy(action).float().reshape((-1,self.h.dynamic_system.nu)).to(device)            
            u_result, r_result = self.h.solve_CLF_QP(s, gradh, u_ref, epsilon=0.1)

            if r_result > 0.0:
                self.break_safety += 1
            #     raise Exception(f"The QP is infeasible, slack variable is {r_result}")

            if torch.abs( torch.norm(u_result - u_ref) ) > 0.1:
                reward += -15

            u = u_result.cpu().numpy().flatten()
        else:
            u = action

        
        # F1 = u[0] * self.point_robot.mass * self.scale
        # F2 = u[1] * self.point_robot.mass * self.scale
        
        self.robot_arm.motor_1.rate = u[0]
        self.robot_arm.motor_2.rate = u[1]
        # self.point_robot.shape.body.apply_force_at_local_point((F1, 0), (0, 0))
        # self.point_robot.shape.body.apply_force_at_local_point((0, F2), (0, 0))
        # self.car.shape.body.apply_force_at_local_point((0, -F2), (-self.car.radius, 0))

        self.space.step(self.dt)
        # self.car.step(u)
        self.current_time_step += 1


        obs = self.get_observation()
        x, y, theta2 = self.l2_state

        distance_to_target_x = self.x_target - x
        distance_to_target_y = self.y_target - y

        reward = (1.0 / (  np.abs(distance_to_target_x) + 0.1 )) + (1.0 / (  np.abs(distance_to_target_y) + 0.1 ))
        done = False

        # if np.abs(obs[0]) == 0 or np.abs(obs[1]) == 0 or np.abs(obs[0])==1 or np.abs(obs[1])==1:
        #     # reach the boundary
        #     reward = -10
        #     done = True
        # elif np.abs(obs[2]) == 1 or np.abs(obs[3]) == 1:
        #     # reach the maximum velocity or angular velocity
        #     print("reach the maximum velocity or angular velocity")
        #     reward = -10
        #     done = True

        if self.current_time_step == self.max_time_steps:
            self.done = True
        
        # if np.abs(x - 5) < 1 + self.radius and np.abs(y - 5) < 1 + self.radius:
        #     reward = -10
        #     done = True

        
        info = {}
        return obs, reward, done, info
    
    def get_observation(self):


        x2, y2 = self.robot_arm.l2.body.position / self.scale
        theta2 = self.normalize_angle(self.robot_arm.l2.body.angle)
        

        x1, y1 = self.robot_arm.l1.body.position / self.scale
        theta1 = self.normalize_angle(self.robot_arm.l1.body.angle)

        self.l1_state = np.array([x1, y1, theta1])
        self.l2_state = np.array([x2, y2, theta2 - theta1])

        x_obs = np.clip(x1/self.x_max, 0, 1)
        y_obs = np.clip(y1/self.y_max, 0, 1)
        theta_obs = np.clip(theta1/self.theta_max, -1, 1)
        
        return np.array([x_obs, y_obs, theta_obs], dtype=np.float32)

    def reset(self):
        self.current_time_step = 0
        self.robot_arm.reset()
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
    
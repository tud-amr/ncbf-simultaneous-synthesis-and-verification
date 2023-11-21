from os import path
from typing import Optional
import time

import numpy as np
import copy

import torch
import gym
from gym import spaces
from gym.utils import seeding
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from stable_baselines3.common.env_checker import check_env

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class CAR1D(gym.Env):
    """
       ### Description
    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.
    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.
    ![Pendulum Coordinate System](./diagrams/pendulum.png)
    -  `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.
    ### Action Space
    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.
    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |
    ### Observation Space
    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.
    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |
    ### Rewards
    The reward function is defined as:
    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*
    where `$\theta$` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).
    ### Starting State
    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.
    ### Episode Truncation
    The episode truncates at 200 time steps.
    ### Arguments
    - `g`: acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
      The default value is g = 10.0 .
    ```
    gym.make('Pendulum-v1', g=9.81)
    ```
    ### Version History
    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    _np_random: Optional[np.random.Generator] = None
    def __init__(self, render_mode: Optional[str] = None, g=9.81, with_CBF=False):
        super().__init__()
        
        self.dt = 0.05
        self.screen_width = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        self.render_mode = render_mode

        self.max_speed = 2
        self.max_x = 5
        self.max_action = 2.0

        high = np.array([1.0, 1.0], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
   

    def step(self, u):
        start_time = time.time()
        u = u * self.max_action
        x, x_dot = self.state  # th := theta
        
        dt = self.dt
        costs = float(0.0)
        done = False
        
        costs += 0.1 * x ** 2 # + 0.05 * x_dot**2 + 0.001 * ((u[0]/6)**2)
        
        newxdot = x_dot + (-0.2 * x_dot + u) * dt 
        newxdot = np.clip(newxdot, -self.max_speed, self.max_speed)
        newx = x + newxdot * dt

     
        self.state = np.array([newx, newxdot]).flatten()

        if abs(self.state[0]) > self.max_x:
            # self.reset()
            costs += 50
            done = True

        self.reward = -costs

        end_time = time.time()
        self.step_executing_time = end_time - start_time

        return self._get_obs(), -costs, done, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # super().reset(seed=seed)
        

        high = np.array([2.2, 1.2])
        low = np.array([1.8, 0.8])  # We enforce symmetric limits.

        # find safe state
        self.state =  self.np_random.uniform(low=low, high=high)
        
        return self._get_obs()

    def _get_obs(self):
        x, xdot = self.state

        obs = np.clip(np.array([x/self.max_x, xdot/self.max_speed], dtype=np.float32), -1, 1)

        return obs

    def set_state(self, state):
        self.state = state

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_width)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_width))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = 5 * 2
        scale = self.screen_width / world_width
        cartwidth = 50.0
        cartheight = 30.0

        x, x_dot = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_width))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))


        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed."""
        if self._np_random is None:
            self._np_random, seed = seeding.np_random()
        return self._np_random

    


if __name__ == "__main__":
    my_car = CAR1D("human", g=9.81)

    check_env(my_car)

  
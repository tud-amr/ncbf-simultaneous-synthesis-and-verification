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


class MyPendulumEnv(gym.Env):
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
    def __init__(self, render_mode: Optional[str] = None, g=10.0, with_CBF=False):
        super().__init__()
        
        self.with_CBF = with_CBF
        self.break_safety = 0
        self.prefix = "with_CBF_" if self.with_CBF else "without_CBF_"
        self.max_speed = 8
        self.max_torque = 12.0
        self.dt = 0.05
        self.g = g
        self.m = 0.8
        self.l = 1.0

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.epoch_trajectory = []
        self.training_trajectories = []
        self.h = None

    def set_barrier_function(self, h):
        self.h = h

    def step(self, u):
        start_time = time.time()
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        costs = 0.0
        if self.with_CBF:
            if self.h is None:
                raise Exception(f"Please use self.set_barrier_function to set barrier function")
            
            device = self.h.device
            s = torch.from_numpy(self.state).float().reshape((-1,2)).to(device)
            
            hs = self.h(s)
            gradh = self.h.jacobian(hs, s)
            # if hs < 0:
            #     raise Exception(f"Current state [{self.state[0]}, {self.state[1]}] is unsafe, h(s)={hs}")
            
            u_ref = torch.from_numpy(u).float().reshape((-1,1)).to(device)            
            u_result, r_result = self.h.solve_CLF_QP(s, gradh, u_ref, epsilon=0.05)

            if r_result > 0.0:
                self.break_safety += 1
            #     raise Exception(f"The QP is infeasible, slack variable is {r_result}")

            if abs(u_result - u_ref) > 0.1:
                costs += 15

            u = u_result.cpu().numpy().squeeze(-1)

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs += angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * ((u/6)**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 1 / (m * l**2 / 3) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        
        is_unsafe = self.h.dynamic_system.unsafe_mask(torch.from_numpy(self.state).float().reshape((-1, 2)).to(self.h.device))
        if is_unsafe[0]:
            self.done = True
            costs += 15
            self.break_safety += 1
        else:
            self.done = False

        if self.render_mode == "human":
            self.render()

        self.reward = -costs
        normalized_theta = angle_normalize(self.state[0])
        self.epoch_trajectory.append( np.array([[normalized_theta], [self.state[1]]]) )

        end_time = time.time()
        self.step_executing_time = end_time - start_time

        return self._get_obs(), -costs, False, {"username": "wangxinyu"}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])

        # high = np.array([np.pi/5, 4])
        low = -high  # We enforce symmetric limits.

        # find safe state
        self.state =  self.np_random.uniform(low=low, high=high)
        if self.with_CBF:
            while self.h( torch.from_numpy(self.state).float().reshape((-1, 2)).to(self.h.device)) <= 0.01:
                self.state =  self.np_random.uniform(low=low, high=high)
        
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

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
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        scaler = 0.7
        down_offset = 20
        c1 = (250 - scaler* rod_length/np.sqrt(3), 250 - scaler * rod_length - down_offset)
        c2 = (250, 250 - down_offset)
        c3 = (250 + scaler * rod_length/np.sqrt(3), 250 - scaler * rod_length - down_offset)
        triangle_obstacle = [c1, c2, c3]
        gfxdraw.aapolygon(self.surf, triangle_obstacle, (100, 77, 77))
        gfxdraw.filled_polygon(self.surf, triangle_obstacle, (100, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        fname = "assets/clockwise.png"
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed."""
        if self._np_random is None:
            self._np_random, seed = seeding.np_random()
        return self._np_random

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


if __name__ == "__main__":
    my_ip_env = MyPendulumEnv("human", g=9.81)

    check_env(my_ip_env)

    print(angle_normalize(-5 * np.pi / 4))
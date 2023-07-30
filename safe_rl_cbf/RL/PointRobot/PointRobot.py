
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import pygame


class PointRobot():
    def __init__(self, mass, radius,state, space, scale, dt):
        self.mass = mass
        self.scale  = scale
        self.dt = dt
        self.radius = radius

        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
        x, y, v_x, v_y = state

        # create shape
        x = x * scale
        y = y * scale
        radius = self.radius * scale

        body.position = x, y
        body.velocity = Vec2d(v_x, v_y)
        body.angle = 0

        self.shape = pymunk.Circle(body, radius)
        space.add(body, self.shape)
    
    def step(self, action):
        a_x = action[0] * self.scale
        a_y = action[1] * self.scale
        
        v_x = self.shape.body.velocity[0] + a_x * self.dt
        v_y = self.shape.body.velocity[1] + a_y * self.dt

        self.shape.body.velocity = Vec2d(v_x, v_y)
        #pass
    
    def set_states(self, x, y):
        self.shape.body.position = Vec2d( x * self.scale, y * self.scale )
        self.shape.body.angle = 0
        self.shape.body.velocity = Vec2d(0, 0)
        self.shape.body.angular_velocity = 0


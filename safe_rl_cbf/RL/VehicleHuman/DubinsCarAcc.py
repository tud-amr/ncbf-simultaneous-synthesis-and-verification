
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import pygame


class DubinsCar():
    def __init__(self, mass, radius,state, space, scale, dt):
        self.mass = mass
        self.scale  = scale
        self.dt = dt
        self.radius = radius

        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
        x, y, theta, v, w = state

        # create shape
        x = x * scale
        y = y * scale
        radius = self.radius * scale

        body.position = x, y
        body.angle = theta

        self.shape = pymunk.Circle(body, radius)
        space.add(body, self.shape)
    
    def step(self, action):
        acc = action[0] 
        alpha = action[1]

        F1 = Vec2d(acc, 0) * self.mass * self.scale
        F2 = alpha * self.shape.body.moment / (2 * self.radius)
        
        self.shape.body.apply_force_at_local_point(F1, (0, 0))
        self.shape.body.apply_force_at_local_point((0, -F2), (-self.radius, 0))
        self.shape.body.apply_force_at_local_point((0, F2), (self.radius, 0))
        #pass
    
    def set_states(self, x, y, angle):
        self.shape.body.position = Vec2d( x * self.scale, y * self.scale )
        self.shape.body.angle = angle
        self.shape.body.velocity = Vec2d(0.1 * self.scale, 0).rotated(self.shape.body.angle)
        self.shape.body.angular_velocity = 0



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
        acc = action[0] * self.scale
        alpha = action[1]
        
        v = self.shape.body.velocity.length + acc * self.dt
        w = self.shape.body.angular_velocity + alpha * self.dt

        self.shape.body.velocity = Vec2d(v, 0).rotated(self.shape.body.angle) 
        self.shape.body.angular_velocity = w
        #pass
    
    def set_states(self, x, y, angle):
        self.shape.body.position = Vec2d( x * self.scale, y * self.scale )
        self.shape.body.angle = angle
        self.shape.body.velocity = Vec2d(0, 0)
        self.shape.body.angular_velocity = 0


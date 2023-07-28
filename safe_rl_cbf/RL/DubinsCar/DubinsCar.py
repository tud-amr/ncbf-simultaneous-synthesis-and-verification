
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import pygame


class DubinsCar():
    def __init__(self, mass, radius, x, y, angle, space, scale, dt):
        self.scale  = scale
        self.dt = dt
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment, body_type=pymunk.Body.KINEMATIC)
        
        # create shape
        x = x * scale
        y = y * scale
        radius = radius * scale

        body.position = x, y
        body.angle = angle

        self.shape = pymunk.Circle(body, radius)
        space.add(body, self.shape)
    
    def step(self, action):
        v = action[0] * self.scale
        w = action[1]

        self.shape.body.position += Vec2d(v * self.dt, 0).rotated(self.shape.body.angle) 
        self.shape.body.angle += w * self.dt
        #pass
    
    def set_states(self, x, y, angle):
        self.shape.body.position = Vec2d( x * self.scale, y * self.scale )
        
        self.shape.body.angle = angle
        
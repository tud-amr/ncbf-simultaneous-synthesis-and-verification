
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import pygame


class DubinsCarRotate():
    def __init__(self, mass, radius, v, state, space, scale, dt, color=(66, 135, 245)):
        self.mass = mass
        self.scale  = scale
        self.dt = dt
        self.radius = radius
        self.v = v

        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
        x, y, theta= state

        # create shape
        x = x * scale
        y = y * scale
        radius = self.radius * scale

        body.position = x, y
        body.angle = theta

        self.shape = pymunk.Circle(body, radius)
        self.shape.color = pygame.Color(color)
        space.add(body, self.shape)
    
    def step(self, action):
      
        omega = action[0]
        self.shape.body.velocity = Vec2d(self.v * self.scale, 0).rotated(self.shape.body.angle) 
        self.shape.body.angular_velocity = omega
        #pass
    
    def set_states(self, x, y, angle):
        self.shape.body.position = Vec2d( x * self.scale, y * self.scale )
        self.shape.body.angle = angle
        self.shape.body.velocity = Vec2d(0, 0)
        self.shape.body.angular_velocity = 0


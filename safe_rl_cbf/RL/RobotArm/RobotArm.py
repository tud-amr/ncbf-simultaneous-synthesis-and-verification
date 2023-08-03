
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import pygame


class RobotArm():
    def __init__(self, space, scale, dt):
        mass = 1
        width = 2
        height = 0.5
        origin_point = Vec2d(4, 2)
        self.scale  = scale
        self.dt = dt
        
        self.shapes = []
        self.joints = []

        ###################### create l1 ######################
        self.l1 = pymunk.Poly.create_box(None, size=(width * self.scale, height * self.scale/2))

        # configure body
        moment = pymunk.moment_for_poly(mass, self.l1.get_vertices())
        l1_body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)

        self.l1_init_x = (origin_point[0] + width/2) * self.scale
        self.l1_init_y = (origin_point[1]) * self.scale
        

        l1_body.position = self.l1_init_x, self.l1_init_y
        l1_body.angle = 0

        # attach body to shape
        self.l1.body = l1_body
        self.l1.sensor = True
        self.l1.color = pygame.Color((66, 135, 245))

        space.add(l1_body, self.l1)

        ###################### create l2 ######################
        self.l2 = pymunk.Poly.create_box(None, size=(width * self.scale, height * self.scale/2))

        # configure body
        moment = pymunk.moment_for_poly(mass, self.l2.get_vertices())
        l2_body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)

        self.l2_init_x = (origin_point[0] + width * 3 / 2) * self.scale
        self.l2_init_y = (origin_point[1]) * self.scale
        

        l2_body.position = self.l2_init_x, self.l2_init_y
        l2_body.angle = 0

        # attach body to shape
        self.l2.body = l2_body
        self.l2.sensor = True
        self.l2.color = pygame.Color((132, 94, 194))

        space.add(l2_body, self.l2)

        ###################### ignore collision ######################
        self.l1.filter = pymunk.ShapeFilter(group=1)
        self.l2.filter = pymunk.ShapeFilter(group=1)


        ##################### create joint ######################
        world = pymunk.Body(body_type=pymunk.Body.STATIC)
        c1 = pymunk.PivotJoint(world, self.l1.body, origin_point * self.scale)
        c2 = pymunk.PinJoint(self.l1.body, self.l2.body, (width / 2 * self.scale, 0), (-width / 2 * self.scale, 0))

        space.add(c1)
        space.add(c2)

        ##################### create motor ######################
        self.motor_1 = pymunk.SimpleMotor(world, self.l1.body, 0)
        self.motor_2 = pymunk.SimpleMotor(self.l1.body, self.l2.body, 0)

        space.add(self.motor_1)
        space.add(self.motor_2)
    
    def step(self, action):
        a_x = action[0] * self.scale  + (np.random.rand() * 0.6 - 0.3) * self.scale
        a_y = action[1] * self.scale  + (np.random.rand() * 0.6 - 0.3) * self.scale
        
        v_x = self.l1.body.velocity[0] + a_x * self.dt
        v_y = self.l1.body.velocity[1] + a_y * self.dt

        self.l1.body.velocity = Vec2d(v_x, v_y)
        #pass
    
    def reset(self):
        self.l1.body.position = self.l1_init_x, self.l1_init_y
        self.l1.body.velocity = Vec2d(0, 0)
        self.l1.body.angle = 0
        self.l1.body.angular_velocity = 0

        self.l2.body.position = self.l2_init_x, self.l2_init_y
        self.l2.body.velocity = Vec2d(0, 0)
        self.l2.body.angle = 0
        self.l2.body.angular_velocity = 0


from safe_rl_cbf.Dynamics.Car import Car
from safe_rl_cbf.Dynamics.InvertedPendulum import InvertedPendulum
from safe_rl_cbf.Dynamics.CartPole import CartPole
from safe_rl_cbf.Dynamics.DubinsCar import DubinsCar
from safe_rl_cbf.Dynamics.DubinsCarRotate import DubinsCarRotate
from safe_rl_cbf.Dynamics.DubinsCarAcc import DubinsCarAcc
from safe_rl_cbf.Dynamics.PointRobot import PointRobot
from safe_rl_cbf.Dynamics.PointRobotDisturbance import PointRobotDisturbance
from safe_rl_cbf.Dynamics.PointRobotsDisturbance import PointRobotsDisturbance
from safe_rl_cbf.Dynamics.RobotArm2D import RobotArm2D
from safe_rl_cbf.Dynamics.TwoVehicleAvoidance import TwoVehicleAvoidance
from safe_rl_cbf.Dynamics.VehicleAndHuman import VehicleAndHuman

import torch

####################### create an one-D car object ######################
car1 = Car(ns=2, nu=1)

domain_lower_bd = torch.Tensor([-2, -2]).float()
domain_upper_bd = -domain_lower_bd

control_lower_bd =torch.Tensor([-1]).float()
control_upper_bd = -control_lower_bd
    
def rou(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.unsqueeze(s[:, 0] + 1, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + 1, dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] + 1, dim=1)
    rou_4 = torch.unsqueeze( -s[:, 1] + 1, dim=1)
    return torch.hstack( (rou_1, rou_2, rou_3, rou_4) ) 

def rou_n(s: torch.Tensor) -> torch.Tensor:
    s_norm = torch.norm(s, dim=1).unsqueeze(dim=-1)

    return - s_norm + 0.6

car1.set_domain_limits(domain_lower_bd, domain_upper_bd)
car1.set_control_limits(control_lower_bd, control_upper_bd)
car1.set_state_constraints(rou)
car1.set_nominal_state_constraints(rou_n)

######################## creat inverted pendulum object ##################


domain_lower_bd2 = torch.Tensor([-torch.pi, -5]).float()
domain_upper_bd2 = -domain_lower_bd2

control_lower_bd2 =torch.Tensor([-12]).float()
control_upper_bd2 = -control_lower_bd2
    
def rou2(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.unsqueeze(s[:, 0] + torch.pi * 5 / 6, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + torch.pi * 5 / 6 , dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] + 4, dim=1)
    rou_4 = torch.unsqueeze( - s[:, 1] + 4 , dim=1)
    return torch.hstack( (rou_1, rou_2, rou_3, rou_4) ) 

def rou_n2(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.pi * 3 / 4 -  torch.norm(s, dim=-1)
    
    return rou_1.unsqueeze(dim=1)


inverted_pendulum_1 = InvertedPendulum(m=1, b=0.1)

inverted_pendulum_1.set_domain_limits(domain_lower_bd2, domain_upper_bd2)
inverted_pendulum_1.set_control_limits(control_lower_bd2, control_upper_bd2)
inverted_pendulum_1.set_state_constraints(rou2)
inverted_pendulum_1.set_nominal_state_constraints(rou_n2)

######################## create cart pole object ######################


cart_pole_1 = CartPole()

domain_lower_bd = torch.Tensor([-2.5, -5, -torch.pi * 3 / 2 , -5]).float()
domain_upper_bd = -domain_lower_bd

control_lower_bd =torch.Tensor([-15]).float()
control_upper_bd = -control_lower_bd
    
def rou(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.unsqueeze(s[:, 0] + 2, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + 2, dim=1)
    
    return torch.hstack( (rou_1, rou_2) ) 

def rou_n(s: torch.Tensor) -> torch.Tensor:
    s_norm = torch.norm(s, dim=1, keepdim=True)

    return - s_norm + 0.6

cart_pole_1.set_domain_limits(domain_lower_bd, domain_upper_bd)
cart_pole_1.set_control_limits(control_lower_bd, control_upper_bd)
cart_pole_1.set_state_constraints(rou)
cart_pole_1.set_nominal_state_constraints(rou_n)

######################## create dubins car object ######################

dubins_car = DubinsCar()

domain_lower_bd = torch.Tensor([-1, -1, -torch.pi]).float()
domain_upper_bd = torch.Tensor([9, 9, torch.pi]).float()

control_lower_bd = torch.Tensor([-1, -1]).float()
control_upper_bd = -control_lower_bd
    
def rou(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.unsqueeze(s[:, 0] + 0, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + 8, dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] + 0, dim=1)
    rou_4 = torch.unsqueeze( -s[:, 1] + 8, dim=1)
    rou_5 = torch.norm(s[:, 0:2] - torch.tensor([5,5]).to(s.device).reshape(1, 2), dim=1, keepdim=True) - 1.4
    return torch.hstack( (rou_1, rou_2, rou_3, rou_4, rou_5) ) 

def rou_n(s: torch.Tensor) -> torch.Tensor:
    s_norm = torch.norm(s, dim=1, keepdim=True)

    return - s_norm + 0.6

dubins_car.set_domain_limits(domain_lower_bd, domain_upper_bd)
dubins_car.set_control_limits(control_lower_bd, control_upper_bd)
dubins_car.set_state_constraints(rou)
dubins_car.set_nominal_state_constraints(rou_n)


########################## create dubins car rotate object ######################

dubins_car_rotate = DubinsCarRotate(v=0.4, dt=0.05)

domain_lower_bd = torch.Tensor([-1, -1, -4]).float()
domain_upper_bd = torch.Tensor([9, 9, 4]).float()

control_lower_bd = torch.Tensor([-1]).float()
control_upper_bd = -control_lower_bd
    
def rou(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.unsqueeze(s[:, 0] - 0.3, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + 7.7, dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] - 0.3, dim=1)
    rou_4 = torch.unsqueeze( -s[:, 1] + 7.7, dim=1)
    rou_5 = torch.norm(s[:, 0:2] - torch.tensor([5,5]).to(s.device).reshape(1, 2), dim=1, keepdim=True) - 1.8
    return torch.hstack( (rou_1, rou_2, rou_3, rou_4, rou_5) ) 

def rou_n(s: torch.Tensor) -> torch.Tensor:
    s_norm = torch.norm(s, dim=1, keepdim=True)

    return - s_norm + 0.6

dubins_car_rotate.set_domain_limits(domain_lower_bd, domain_upper_bd)
dubins_car_rotate.set_control_limits(control_lower_bd, control_upper_bd)
dubins_car_rotate.set_state_constraints(rou)
dubins_car_rotate.set_nominal_state_constraints(rou_n)


######################## create dubins car acc object ######################

dubins_car_acc = DubinsCarAcc(dt=0.1)

domain_lower_bd = torch.Tensor([-1, -1, -4, -0.7, -0.7]).float()
domain_upper_bd = torch.Tensor([9, 9, 4, 0.7, 0.7]).float()

control_lower_bd = torch.Tensor([-1, -1]).float()
control_upper_bd = -control_lower_bd
    
    
def rou(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.unsqueeze(s[:, 0] - 0.3, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + 7.7, dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] - 0.3, dim=1)
    rou_4 = torch.unsqueeze( -s[:, 1] + 7.7, dim=1)
    rou_5 = torch.norm(s[:, 0:2] - torch.tensor([5,5]).to(s.device).reshape(1, 2), dim=1, keepdim=True) - 1.8
    rou_6 = torch.unsqueeze(s[:, 3] + 0.5, dim=1)
    rou_7 = torch.unsqueeze( - s[:, 3] + 0.5, dim=1)
    rou_8 = torch.unsqueeze(s[:, 4] + 0.5, dim=1)
    rou_9 = torch.unsqueeze( -s[:, 4] + 0.5, dim=1)
    
    return torch.hstack( (rou_1, rou_2, rou_3, rou_4, rou_5, rou_6, rou_7, rou_8, rou_9) )

def rou_n(s: torch.Tensor) -> torch.Tensor:
    s_norm = torch.norm(s, dim=1, keepdim=True)

    return - s_norm + 0.6

dubins_car_acc.set_domain_limits(domain_lower_bd, domain_upper_bd)
dubins_car_acc.set_control_limits(control_lower_bd, control_upper_bd)
dubins_car_acc.set_state_constraints(rou)
dubins_car_acc.set_nominal_state_constraints(rou_n)


######################## create point robot object ######################

point_robot = PointRobot()

domain_lower_bd = torch.Tensor([-1, -1, -1.2, -1.2]).float()
domain_upper_bd = torch.Tensor([9, 9, 1.2, 1.2]).float()

control_lower_bd = torch.Tensor([-1, -1]).float()
control_upper_bd = -control_lower_bd
    
    
def rou(s: torch.Tensor) -> torch.Tensor:
    
    rou_1 = torch.unsqueeze(s[:, 0] - 0.1, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + 7.9, dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] - 0.1, dim=1)
    rou_4 = torch.unsqueeze( -s[:, 1] + 7.9, dim=1)
    rou_5 = torch.unsqueeze( -s[:, 2] + 1, dim=1)
    rou_6 = torch.unsqueeze(s[:, 2] + 1, dim=1)
    rou_7 = torch.unsqueeze( -s[:, 3] + 1, dim=1)
    rou_8 = torch.unsqueeze(s[:, 3] + 1, dim=1)

    dist = torch.abs( s[:, 0:2] - torch.tensor([5,5]).to(s.device).reshape(1, 2) )
    rou_9 = torch.max(dist[:, 0], dist[:, 1]).reshape(-1, 1) - 1.1
    # rou_9 = torch.norm(s[:, 0:2] - torch.tensor([5,5]).to(s.device).reshape(1, 2), dim=1, keepdim=True) - 1.9

    return torch.hstack( (rou_1, rou_2, rou_3, rou_4, rou_5, rou_6, rou_7, rou_8, rou_9) )

def rou_n(s: torch.Tensor) -> torch.Tensor:
    s_norm = torch.norm(s, dim=1, keepdim=True)

    return - s_norm + 0.6

point_robot.set_domain_limits(domain_lower_bd, domain_upper_bd)
point_robot.set_control_limits(control_lower_bd, control_upper_bd)
point_robot.set_state_constraints(rou)
point_robot.set_nominal_state_constraints(rou_n)


######################### create point robot disturbance object ######################

point_robot_dis = PointRobotDisturbance(dt=0.05)

domain_lower_bd = torch.Tensor([-1, -1, -1.2, -1.2]).float()
domain_upper_bd = torch.Tensor([9, 9, 1.2, 1.2]).float()

control_lower_bd = torch.Tensor([-1, -1]).float()
control_upper_bd = -control_lower_bd

disturbance_lower_bd = torch.Tensor([-0.3, -0.3]).float()
disturbance_upper_bd = -disturbance_lower_bd
    
def rou(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.unsqueeze(s[:, 0] - 0.3, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + 7.7, dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] - 0.3, dim=1)
    rou_4 = torch.unsqueeze( -s[:, 1] + 7.7, dim=1)
    rou_5 = torch.norm(s[:, 0:2] - torch.tensor([5,5]).to(s.device).reshape(1, 2), dim=1, keepdim=True) - 1.8
    
    rou_6 = torch.unsqueeze(s[:, 2] + 1, dim=1)
    rou_7 = torch.unsqueeze( - s[:, 2] + 1, dim=1)
    rou_8 = torch.unsqueeze(s[:, 3] + 1, dim=1)
    rou_9 = torch.unsqueeze( -s[:, 3] + 1, dim=1)
    
    return torch.hstack( (rou_1, rou_2, rou_3, rou_4, rou_5, rou_6, rou_7, rou_8, rou_9) ) 

def rou_n(s: torch.Tensor) -> torch.Tensor:
    s_norm = torch.norm(s, dim=1, keepdim=True)

    return - s_norm + 0.6

point_robot_dis.set_domain_limits(domain_lower_bd, domain_upper_bd)
point_robot_dis.set_control_limits(control_lower_bd, control_upper_bd)
point_robot_dis.set_disturbance_limits(disturbance_lower_bd, disturbance_upper_bd)
point_robot_dis.set_state_constraints(rou)
point_robot_dis.set_nominal_state_constraints(rou_n)

################################# create point robots  object ##############################

point_robots_dis = PointRobotsDisturbance()

domain_lower_bd = torch.Tensor([-1, -1, -1.2, -1.2, -1, -1, -1.2, -1.2]).float()
domain_upper_bd = torch.Tensor([9, 9, 1.2, 1.2, 9, 9, 1.2, 1.2]).float()

control_lower_bd = torch.Tensor([-1, -1]).float()
control_upper_bd = -control_lower_bd

disturbance_lower_bd = torch.Tensor([-0.5, -0.5]).float()
disturbance_upper_bd = -disturbance_lower_bd
    
def rou(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.unsqueeze(s[:, 0] - 0.3, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + 7.7, dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] - 0.3 , dim=1)
    rou_4 = torch.unsqueeze( -s[:, 1] + 7.7, dim=1)
    rou_5 = torch.norm(s[:, 0:2] - torch.tensor([5,5]).to(s.device).reshape(1, 2), dim=1, keepdim=True) - 1.8
    
    rou_6 = torch.norm(s[:, 0:2] - s[:, 4:6], dim=1, keepdim=True) - 0.8

    rou_7 = torch.unsqueeze(s[:, 2] + 1, dim=1)
    rou_8 = torch.unsqueeze( - s[:, 2] + 1, dim=1)
    rou_9 = torch.unsqueeze(s[:, 3] + 1, dim=1)
    rou_10 = torch.unsqueeze( -s[:, 3] + 1, dim=1)

    return torch.hstack( (rou_1, rou_2, rou_3, rou_4, rou_5, rou_6) )

def rou_n(s: torch.Tensor) -> torch.Tensor:
    s_norm = torch.norm(s, dim=1, keepdim=True)

    return - s_norm + 0.6

point_robots_dis.set_domain_limits(domain_lower_bd, domain_upper_bd)
point_robots_dis.set_control_limits(control_lower_bd, control_upper_bd)
point_robots_dis.set_disturbance_limits(disturbance_lower_bd, disturbance_upper_bd)
point_robots_dis.set_state_constraints(rou)
point_robots_dis.set_nominal_state_constraints(rou_n)


############################# create robot arm 2d object ##############################

robot_arm_2d = RobotArm2D()

domain_lower_bd = torch.Tensor([-4, -4]).float()
domain_upper_bd = torch.Tensor([4, 4]).float()

control_lower_bd = torch.Tensor([-0.1, -0.1]).float()
control_upper_bd = -control_lower_bd
    
def rou(s: torch.Tensor) -> torch.Tensor:
    theta_1 = s[:, 0]
    theta_2 = s[:, 1]
    l1 = 2
    l2 = 2
    x_e = l2 * torch.cos(theta_1 + theta_2) + l1 * torch.cos(theta_1)
    y_e = l2 * torch.sin(theta_1 + theta_2) + l1 * torch.sin(theta_1) 

    rou_1 = torch.unsqueeze(y_e + 0.5, dim=1)
    rou_2 = torch.unsqueeze( - y_e + 3, dim=1)
    rou_3 = torch.unsqueeze( theta_1 + 3, dim=1)
    rou_4 = torch.unsqueeze( - theta_1 + 3, dim=1)
    rou_5 = torch.unsqueeze( theta_2 + 3, dim=1)
    rou_6 = torch.unsqueeze( - theta_2 + 3, dim=1)
    
    return torch.hstack( (rou_1, rou_2, rou_3, rou_4, rou_5, rou_6) )

def rou_n(s: torch.Tensor) -> torch.Tensor:
    s_norm = torch.norm(s, dim=1, keepdim=True)

    return - s_norm + 0.6

robot_arm_2d.set_domain_limits(domain_lower_bd, domain_upper_bd)
robot_arm_2d.set_control_limits(control_lower_bd, control_upper_bd)
robot_arm_2d.set_state_constraints(rou)
robot_arm_2d.set_nominal_state_constraints(rou_n)

########################## create two vehicle avoidance object ##############################

two_vehicle_avoidance = TwoVehicleAvoidance(dt=0.05)

domain_lower_bd = torch.Tensor([-1, -1, -4, -1, -1, -4]).float()
domain_upper_bd = torch.Tensor([3, 3, 4, 3, 3, 4]).float()

control_lower_bd = torch.Tensor([-1]).float()
control_upper_bd = -control_lower_bd

disturbance_lower_bd = torch.Tensor([-1]).float()
disturbance_upper_bd = -disturbance_lower_bd
    
def rou(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.unsqueeze(s[:, 0] - 0.3, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + 7.7, dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] - 0.3, dim=1)
    rou_4 = torch.unsqueeze( -s[:, 1] + 7.7, dim=1)
    
    # rou_5 = torch.norm(s[:, 0:2] - torch.tensor([5,5]).to(s.device).reshape(1, 2), dim=1, keepdim=True) - 1.8
    
    rou_6 = torch.norm(s[:, 0:2] - s[:, 3:5], dim=1, keepdim=True) - 0.5

    return torch.hstack( (rou_6,) ) 

def rou_n(s: torch.Tensor) -> torch.Tensor:
    s_norm = torch.norm(s, dim=1, keepdim=True)

    return - s_norm + 0.6

two_vehicle_avoidance.set_domain_limits(domain_lower_bd, domain_upper_bd)
two_vehicle_avoidance.set_control_limits(control_lower_bd, control_upper_bd)
two_vehicle_avoidance.set_disturbance_limits(disturbance_lower_bd, disturbance_upper_bd)
two_vehicle_avoidance.set_state_constraints(rou)
two_vehicle_avoidance.set_nominal_state_constraints(rou_n)


######################## create vehicle and human object ######################
vehicle_human = VehicleAndHuman()

domain_lower_bd = torch.Tensor([-1, -1, -4, -1.2, -1.2, -1, -1, -1.2, -1.2]).float()
domain_upper_bd = torch.Tensor([3, 3, 4, 1.2, 1.2, 3, 3, 1.2, 1.2]).float()

control_lower_bd = torch.Tensor([-1, -1]).float()
control_upper_bd = -control_lower_bd

disturbance_lower_bd = torch.Tensor([-0.2, -0.2]).float()
disturbance_upper_bd = -disturbance_lower_bd
    
def rou(s: torch.Tensor) -> torch.Tensor:
    rou_1 = torch.unsqueeze(s[:, 0] - 0.3, dim=1)
    rou_2 = torch.unsqueeze( - s[:, 0] + 7.7, dim=1)
    rou_3 = torch.unsqueeze(s[:, 1] - 0.3, dim=1)
    rou_4 = torch.unsqueeze( -s[:, 1] + 7.7, dim=1)
    rou_5 = torch.norm(s[:, 0:2] - torch.tensor([5,5]).to(s.device).reshape(1, 2), dim=1, keepdim=True) - 1.8
    
    rou_6 = torch.norm(s[:, 0:2] - s[:, 5:7], dim=1, keepdim=True) - 0.5

    return torch.hstack( (rou_6,) ) 

def rou_n(s: torch.Tensor) -> torch.Tensor:
    s_norm = torch.norm(s, dim=1, keepdim=True)

    return - s_norm + 0.6

vehicle_human.set_domain_limits(domain_lower_bd, domain_upper_bd)
vehicle_human.set_control_limits(control_lower_bd, control_upper_bd)
vehicle_human.set_disturbance_limits(disturbance_lower_bd, disturbance_upper_bd)
vehicle_human.set_state_constraints(rou)
vehicle_human.set_nominal_state_constraints(rou_n)

import json
import os
import importlib.util
import sys
from safe_rl_cbf.Dataset.DataModule import DataModule
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule
from safe_rl_cbf.Dataset.TestingDataModule import TestingDataModule

# from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1, dubins_car, dubins_car_rotate ,dubins_car_acc, point_robot ,point_robots_dis, two_vehicle_avoidance
from safe_rl_cbf.Dataset.SqlDataSet import SqlDataSet
from safe_rl_cbf.Models.NeuralCBF import NeuralCBF

from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem
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


def read_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def select_dynamic_system(system_name, constraints_name):
    if system_name == "InvertedPendulum":
        
        system = InvertedPendulum(m=1, b=0.1)

    elif system_name == "PointRobot":
        system = PointRobot()
    
    else:
        pass

    # import constraints function
    constraints_path = os.path.join("safe_rl_cbf/Dynamics/Constraints", constraints_name + ".py")
    spec = importlib.util.spec_from_file_location(constraints_name, constraints_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[constraints_name] = module
    spec.loader.exec_module(module)
    
    domain_lower_bd = getattr(module, "domain_lower_bd")
    domain_upper_bd = getattr(module, "domain_upper_bd")
    control_lower_bd = getattr(module, "control_lower_bd")
    control_upper_bd = getattr(module, "control_upper_bd")
    rou = getattr(module, "rou")
   

    # set constraints
    system.set_domain_limits(domain_lower_bd, domain_upper_bd)
    system.set_control_limits(control_lower_bd, control_upper_bd)
    system.set_state_constraints(rou)

    return system

def select_RL_env(system_name):
    if system_name == "InvertedPendulum":
        from safe_rl_cbf.RL.InvertedPendulum.MyPendulum import MyPendulumEnv
        from safe_rl_cbf.RL.InvertedPendulum.inverted_pendulum_callback import CustomCallback
        return MyPendulumEnv, CustomCallback
    elif system_name == "PointRobot":
        from safe_rl_cbf.RL.PointRobot.PointRobotEnv import PointRobotEnv
        from safe_rl_cbf.RL.PointRobot.point_robot_callback import CustomCallback
        return PointRobotEnv, CustomCallback
    else:
        pass

import json
from safe_rl_cbf.Dataset.DataModule import DataModule
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule
from safe_rl_cbf.Dataset.TestingDataModule import TestingDataModule
from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem
from safe_rl_cbf.Dynamics.dynamic_system_instances import inverted_pendulum_1, dubins_car, dubins_car_rotate ,dubins_car_acc, point_robot ,point_robots_dis, two_vehicle_avoidance
from safe_rl_cbf.Dataset.SqlDataSet import SqlDataSet
from safe_rl_cbf.Models.NeuralCBF import NeuralCBF

def read_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def select_dynamic_system(name):
    if name == "inverted_pendulum_1":
        return inverted_pendulum_1
    elif name == "dubins_car":
        return dubins_car
    
import numpy as np
import os

import matplotlib.pyplot as plt
import torch


from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import scipy.io as sio

from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import *
from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1, cart_pole_1, dubins_car, dubins_car_acc, point_robot
from safe_rl_cbf.Dataset.DataModule import DataModule



system = inverted_pendulum_1

test_results = torch.load("test_results.pt")

domain_limit_lb, domain_limit_ub = system.domain_limits


h_shape_s = []
h_shape_val = []
s_unsafe_violation = []
s_unsafe_violation_val = []

descent_violation = []

inadmissible_boundary_state = []

for batch_id in range(len(test_results)):
    h_shape_s.append(test_results[batch_id]["shape_h"]["state"])
    h_shape_val.append(test_results[batch_id]["shape_h"]["val"])
    s_unsafe_violation.append(test_results[batch_id]["unsafe_violation"]["state"])
    descent_violation.append(test_results[batch_id]["descent_violation"]["state"])
    inadmissible_boundary_state.append(test_results[batch_id]["inadmissible_boundary"]["state"])


h_shape_s = torch.vstack(h_shape_s)
h_shape_val = torch.vstack(h_shape_val)
s_unsafe_violation = torch.vstack(s_unsafe_violation)
descent_violation = torch.vstack(descent_violation)
inadmissible_boundary_state = torch.vstack(inadmissible_boundary_state)
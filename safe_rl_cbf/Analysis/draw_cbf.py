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


def draw_cbf(system):

    ############### create folder #################

    if not os.path.exists("test_fig"):
        os.makedirs("test_fig")

    x_index = 0
    y_index = 1

    ##################### read data #######################


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


    ########################## start to plot #############################

    ############################### plot shape of function h(x) ##############################

    X = h_shape_s[:, x_index].detach().cpu().numpy()
    Y = h_shape_s[:, y_index].detach().cpu().numpy()
    H = h_shape_val.squeeze(dim=1).detach().cpu().numpy()

    H_positive_mask = H > 0


    x = X[H_positive_mask]
    y = Y[H_positive_mask]


    plt.figure()
    plt.scatter(x, y, s=10, c='b')

    X = inadmissible_boundary_state[:, x_index].detach().cpu().numpy()
    Y = inadmissible_boundary_state[:, y_index].detach().cpu().numpy()
    plt.scatter(X, Y, s=2, c='y')

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.xlim(domain_limit_lb[x_index], domain_limit_ub[x_index])
    plt.ylim(domain_limit_lb[y_index], domain_limit_ub[y_index])
    plt.title("shape of 0-superlevel set")
    plt.savefig("logs/test_fig/shape_of_cbf.png")

    ############################### plot unsafe violation ##############################

    ############################### plot descent violation ##############################

    X_descent = descent_violation[:, x_index].detach().cpu().numpy()
    Y_descent = descent_violation[:, y_index].detach().cpu().numpy()

    print(f"descent violation: {Y_descent.shape[0]}")

    plt.figure()

    plt.scatter(x, y, s=10, c='b')
    plt.scatter(X_descent, Y_descent, s=10, c='r')
    plt.scatter(X, Y, s=2, c='y')


    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.xlim(domain_limit_lb[x_index], domain_limit_ub[x_index])
    plt.ylim(domain_limit_lb[y_index], domain_limit_ub[y_index])
    plt.title("shape of 0-superlevel set")

    plt.savefig("logs/test_fig/descent_violation.png")
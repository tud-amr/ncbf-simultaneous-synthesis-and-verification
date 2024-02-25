import numpy as np
import os

import matplotlib.pyplot as plt
import torch


from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import scipy.io as sio

from safe_rl_cbf.Models.NeuralCBF import *
from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1, cart_pole_1, dubins_car, dubins_car_acc, point_robot
from safe_rl_cbf.Dataset.DataModule import DataModule


def draw_cbf(system, log_dir = "logs"):

    ############### create folder #################

    if not os.path.exists( os.path.join(log_dir, "test_fig")):
        os.makedirs(os.path.join(log_dir, "test_fig") )

    save_dir = os.path.join(log_dir, "test_fig")

    x_index = 0
    y_index = 1

    ##################### read data #######################


    test_results = torch.load( log_dir + "/test_results.pt")

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
    

    mat_contents = sio.loadmat("RA_result/extraOuts.mat")
    # print(mat_contents['a0'].shape)

    hVS_XData = mat_contents['a0']
    hVS_YData = mat_contents['a1']
    hVS_ZData = mat_contents['a2']
    hVS0_XData = mat_contents['a3']
    hVS0_YData = mat_contents['a4']
    hVS0_ZData = mat_contents['a5']

    ########################## start to plot #############################

    ############################### plot shape of function h(x) ##############################

   
    X = h_shape_s[:, x_index].detach().cpu().numpy()
    Y = h_shape_s[:, y_index].detach().cpu().numpy()
    H = h_shape_val.squeeze(dim=1).detach().cpu().numpy()

    H_positive_mask = H > 0


    x = X[H_positive_mask]
    y = Y[H_positive_mask]

    plt.figure()

    # Create contour lines or level curves using matpltlib.pyplt module
    # contours = plt.contourf(hVS_XData, hVS_YData, hVS_ZData, levels=[-0.1, 0, 1], colors=['w','y','w'], extend='both')

    # contours2 = plt.contour(hVS0_XData, hVS0_YData, hVS0_ZData, levels=[0], colors='grey', linewidth=5)


    plt.scatter(x, y, s=1, c='b')

    X_in = inadmissible_boundary_state[:, x_index].detach().cpu().numpy()
    Y_in = inadmissible_boundary_state[:, y_index].detach().cpu().numpy()
    plt.scatter(X_in, Y_in, s=1, c='y')

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.xlim(domain_limit_lb[x_index], domain_limit_ub[x_index])
    plt.ylim(domain_limit_lb[y_index], domain_limit_ub[y_index])
    plt.title("shape of 0-superlevel set")


    legend_elements = [
                        Line2D([0], [0], color='grey', lw=2, label='Obstacles'),
                        Patch(facecolor='y', edgecolor='y',
                            label='Invariant Set from RA'),
                        Patch(facecolor='b', edgecolor='b',
                            label='Invariant Set from neural CBF')
                    ]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1.1),loc='upper right')


    plt.savefig( os.path.join(save_dir, "shape_of_cbf.png"))


    ############################### plot descent violation ##############################

    X_descent = descent_violation[:, x_index].detach().cpu().numpy()
    Y_descent = descent_violation[:, y_index].detach().cpu().numpy()

    print(f"descent violation: {Y_descent.shape[0]}")

    plt.figure()

    plt.scatter(x, y, s=1, c='b')
    plt.scatter(X_descent, Y_descent, s=1, c='r')
    plt.scatter(X_in, Y_in, s=1, c='y')


    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.xlim(domain_limit_lb[x_index], domain_limit_ub[x_index])
    plt.ylim(domain_limit_lb[y_index], domain_limit_ub[y_index])
    plt.title("shape of 0-superlevel set")

    plt.savefig( os.path.join(save_dir, "descent_violation.png"))

    ############################### plot training points ##############################
    s_training = torch.load( os.path.join(log_dir,"s_training.pt") )

    plt.figure()
    plt.scatter(x, y, s=1, c='b')
    plt.scatter(X_descent, Y_descent, s=1, c='r')
    plt.scatter(X_in, Y_in, s=1, c='y')


    plt.scatter(s_training[:,0], s_training[:,1], marker='X', s=10, c='k')


    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.xlim(domain_limit_lb[x_index], domain_limit_ub[x_index])
    plt.ylim(domain_limit_lb[y_index], domain_limit_ub[y_index])
    plt.title("shape of 0-superlevel set")

    plt.savefig( os.path.join(save_dir,  "shape_of_cbf_with_training_points.png"))


########################## plot contour of h(x) #############################


    fig1,ax1=plt.subplots(1,1)
    cp = ax1.contourf(X.reshape((math.gcd(X.shape[0], 1000), -1)), Y.reshape((math.gcd(X.shape[0], 1000), -1)), H.reshape((math.gcd(X.shape[0], 1000), -1)))
    fig1.colorbar(cp) # Add a colorbar to a plot
    ax1.set_title('Filled Contours Plot')
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    ax1.set_title("contour of CBF")
    plt.savefig( os.path.join(save_dir, "contour_of_cbf.png"))

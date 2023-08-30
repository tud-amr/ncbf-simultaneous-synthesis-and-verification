import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch


from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D

import scipy.io as sio

from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import *
from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1, cart_pole_1, dubins_car, dubins_car_acc, point_robot
from safe_rl_cbf.Dataset.DataModule import DataModule



############### create folder #################

if not os.path.exists("fig/prove_continuous_satisfaction"):
    os.makedirs("fig/prove_continuous_satisfaction")

x_index = 0
y_index = 1

##################### read data #######################
system = inverted_pendulum_1    

ncbf_stage_1_test_results = torch.load("ncbf_stage_1_test_results.pt")
optimal_ncbf_test_results = torch.load("optimal_ncbf_test_results.pt")

domain_limit_lb, domain_limit_ub = system.domain_limits


h_shape_s = []
h_shape_val = []
s_unsafe_violation = []
s_unsafe_violation_val = []

descent_violation = []

inadmissible_boundary_state = []

for batch_id in range(len(ncbf_stage_1_test_results)):
    h_shape_s.append(ncbf_stage_1_test_results[batch_id]["shape_h"]["state"])
    h_shape_val.append(ncbf_stage_1_test_results[batch_id]["shape_h"]["val"])
    s_unsafe_violation.append(ncbf_stage_1_test_results[batch_id]["unsafe_violation"]["state"])
    descent_violation.append(ncbf_stage_1_test_results[batch_id]["descent_violation"]["state"])
    inadmissible_boundary_state.append(ncbf_stage_1_test_results[batch_id]["inadmissible_boundary"]["state"])


h_shape_s_ncbf_1 = torch.vstack(h_shape_s)
h_shape_val_ncbf_1= torch.vstack(h_shape_val)
s_unsafe_violation_ncbf_1 = torch.vstack(s_unsafe_violation)
descent_violation_ncbf_1 = torch.vstack(descent_violation)
inadmissible_boundary_state_ncbf_1 = torch.vstack(inadmissible_boundary_state)

##########
h_shape_s = []
h_shape_val = []
s_unsafe_violation = []
s_unsafe_violation_val = []

descent_violation = []

inadmissible_boundary_state = []

for batch_id in range(len(optimal_ncbf_test_results)):
    h_shape_s.append(optimal_ncbf_test_results[batch_id]["shape_h"]["state"])
    h_shape_val.append(optimal_ncbf_test_results[batch_id]["shape_h"]["val"])
    s_unsafe_violation.append(optimal_ncbf_test_results[batch_id]["unsafe_violation"]["state"])
    descent_violation.append(optimal_ncbf_test_results[batch_id]["descent_violation"]["state"])
    inadmissible_boundary_state.append(optimal_ncbf_test_results[batch_id]["inadmissible_boundary"]["state"])


h_shape_s_ncbf = torch.vstack(h_shape_s)
h_shape_val_ncbf = torch.vstack(h_shape_val)
s_unsafe_violation_ncbf = torch.vstack(s_unsafe_violation)
descent_violation_ncbf = torch.vstack(descent_violation)
inadmissible_boundary_state_ncbf = torch.vstack(inadmissible_boundary_state)

#####
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


X_ncbf_1 = h_shape_s_ncbf_1[:, x_index].detach().cpu().numpy()
Y_ncbf_1 = h_shape_s_ncbf_1[:, y_index].detach().cpu().numpy()
H_ncbf_1 = h_shape_val_ncbf_1.squeeze(dim=1).detach().cpu().numpy()

H_positive_mask_ncbf_1 = H_ncbf_1 > 0.01


x_ncbf_1 = X_ncbf_1[H_positive_mask_ncbf_1]
y_ncbf_1 = Y_ncbf_1[H_positive_mask_ncbf_1]

X_descent_ncbf_1 = descent_violation_ncbf_1[:, x_index].detach().cpu().numpy()
Y_descent_ncbf_1 = descent_violation_ncbf_1[:, y_index].detach().cpu().numpy()



####
X_ncbf = h_shape_s_ncbf[:, x_index].detach().cpu().numpy()
Y_ncbf = h_shape_s_ncbf[:, y_index].detach().cpu().numpy()
H_ncbf = h_shape_val_ncbf.squeeze(dim=1).detach().cpu().numpy()

H_positive_mask_ncbf = H_ncbf > 0.01


x_ncbf = X_ncbf[H_positive_mask_ncbf]
y_ncbf = Y_ncbf[H_positive_mask_ncbf]


X_descent_ncbf = descent_violation_ncbf[:, x_index].detach().cpu().numpy()
Y_descent_ncbf = descent_violation_ncbf[:, y_index].detach().cpu().numpy()

######################

plt.figure()

# Create contour lines or level curves using matpltlib.pyplt module
contours = plt.contourf(hVS_XData, hVS_YData, hVS_ZData, levels=[-0.1, 0, 1], colors=['w','#a7f790','w'], extend='both')

# contours2 = plt.contour(hVS0_XData, hVS0_YData, hVS0_ZData, levels=[0], colors='grey', linewidth=5)



plt.scatter(x_ncbf_1, y_ncbf_1, s=10, c='#63b2ee')
plt.scatter(X_descent_ncbf_1, Y_descent_ncbf_1, s=10, c='#ff866f')

X = inadmissible_boundary_state_ncbf[:, x_index].detach().cpu().numpy()
Y = inadmissible_boundary_state_ncbf[:, y_index].detach().cpu().numpy()
plt.scatter(X, Y, s=2, c='#7cd6cf')

plt.xlabel(r"$\theta$(rad)")
plt.ylabel(r"$\dot{\theta}$(rad/s)")
plt.xlim(domain_limit_lb[x_index], domain_limit_ub[x_index])
plt.ylim(domain_limit_lb[y_index], domain_limit_ub[y_index])
# plt.title("shape of 0-superlevel set")


legend_elements = [
                    Line2D([0], [0], color='#7cd6cf', lw=2, label='Obstacles'),
                    Patch(facecolor='#a7f790', edgecolor='#a7f790',
                        label='LST'),
                    Patch(facecolor='#63b2ee', edgecolor='#63b2ee',
                        label='ours'),
                    Patch(facecolor='#ff866f', edgecolor='#ff866f',
                    label='violation')
                ]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), fontsize="8", loc='upper right')


plt.savefig("fig/prove_continuous_satisfaction/1.png")

#######################
plt.figure()



# Create contour lines or level curves using matpltlib.pyplt module
contours = plt.contourf(hVS_XData, hVS_YData, hVS_ZData, levels=[-0.1, 0, 1], colors=['w','#a7f790','w'], extend='both')


plt.scatter(x_ncbf, y_ncbf, s=10, c='#63b2ee')
plt.scatter(X_descent_ncbf, Y_descent_ncbf, s=10, c='#ff866f')


X = inadmissible_boundary_state_ncbf[:, x_index].detach().cpu().numpy()
Y = inadmissible_boundary_state_ncbf[:, y_index].detach().cpu().numpy()
plt.scatter(X, Y, s=2, c='#7cd6cf')

plt.xlabel(r"$\theta$(rad)")
plt.ylabel(r"$\dot{\theta}$(rad/s)")
plt.xlim(domain_limit_lb[x_index], domain_limit_ub[x_index])
plt.ylim(domain_limit_lb[y_index], domain_limit_ub[y_index])
# plt.title("shape of 0-superlevel set")


legend_elements = [
                    Line2D([0], [0], color='#7cd6cf', lw=2, label='Obstacles'),
                    Patch(facecolor='#a7f790', edgecolor='#a7f790',
                        label='LST'),
                    Patch(facecolor='#63b2ee', edgecolor='#63b2ee',
                        label='ours'),
                    Patch(facecolor='#ff866f', edgecolor='#ff866f',
                    label='violation')
                ]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), fontsize="8", loc='upper right')


plt.savefig("fig/prove_continuous_satisfaction/2.png")

#######################

s_training = torch.load("saved_models/inverted_pendulum_umax_12/s_training.pt")

plt.figure()



# Create contour lines or level curves using matpltlib.pyplt module
contours = plt.contourf(hVS_XData, hVS_YData, hVS_ZData, levels=[-0.1, 0, 1], colors=['w','#a7f790','w'], extend='both')

plt.scatter(s_training[:,0], s_training[:,1], s=5, c='#3a4856')

plt.scatter(x_ncbf, y_ncbf, s=10, c='#63b2ee', alpha=0.02)
# plt.scatter(X_descent_ncbf, Y_descent_ncbf, s=10, c='#ff866f')


X = inadmissible_boundary_state_ncbf[:, x_index].detach().cpu().numpy()
Y = inadmissible_boundary_state_ncbf[:, y_index].detach().cpu().numpy()
plt.scatter(X, Y, s=2, c='#7cd6cf')

# sns.kdeplot(x=s_training[:,0], y=s_training[:,1], cmap="Blues", fill=True, thresh=0)


plt.xlabel(r"$\theta$(rad)")
plt.ylabel(r"$\dot{\theta}$(rad/s)")
plt.xlim(domain_limit_lb[x_index], domain_limit_ub[x_index])
plt.ylim(domain_limit_lb[y_index], domain_limit_ub[y_index])
# plt.title("shape of 0-superlevel set")


legend_elements = [
                    Line2D([0], [0], color='#7cd6cf', lw=2, label='Obstacles'),
                    Patch(facecolor='#a7f790', edgecolor='#a7f790',
                        label='LST'),
                    Patch(facecolor='#63b2ee', edgecolor='#63b2ee',
                        label='ours'),
                    Line2D([], [], color='#3a4856', marker='.', linestyle='None',
                          markersize=5, label='training data')
                ]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), fontsize="8", loc='upper right')


plt.savefig("fig/prove_continuous_satisfaction/3.png")
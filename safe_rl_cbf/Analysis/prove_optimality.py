import numpy as np
import os

import matplotlib.pyplot as plt
import torch


from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import scipy.io as sio
from scipy.interpolate import griddata

from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import *
from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1, cart_pole_1, dubins_car, dubins_car_acc, point_robot
from safe_rl_cbf.Dataset.DataModule import DataModule



############### create folder #################

if not os.path.exists("fig"):
    os.makedirs("fig")

x_index = 0
y_index = 1

##################### read data #######################
system = inverted_pendulum_1    

neural_clbf_test_results = torch.load("neural_clbf_test_results.pt")
optimal_ncbf_test_results = torch.load("test_results.pt")

domain_limit_lb, domain_limit_ub = system.domain_limits


h_shape_s = []
h_shape_val = []
s_unsafe_violation = []
s_unsafe_violation_val = []

descent_violation = []

inadmissible_boundary_state = []

for batch_id in range(len(neural_clbf_test_results)):
    h_shape_s.append(neural_clbf_test_results[batch_id]["shape_h"]["state"])
    h_shape_val.append(neural_clbf_test_results[batch_id]["shape_h"]["val"])
    s_unsafe_violation.append(neural_clbf_test_results[batch_id]["unsafe_violation"]["state"])
    descent_violation.append(neural_clbf_test_results[batch_id]["descent_violation"]["state"])
    inadmissible_boundary_state.append(neural_clbf_test_results[batch_id]["inadmissible_boundary"]["state"])
   
h_shape_s_neural_clbf = torch.vstack(h_shape_s)
h_shape_val_neural_clbf = torch.vstack(h_shape_val)
s_unsafe_violation_neural_clbf = torch.vstack(s_unsafe_violation)
descent_violation_neural_clbf = torch.vstack(descent_violation)
inadmissible_boundary_state_neural_clbf = torch.vstack(inadmissible_boundary_state)

##########
h_shape_s = []
h_shape_val = []
s_unsafe_violation = []
s_unsafe_violation_val = []

descent_violation = []

inadmissible_boundary_state = []
inadmissible_area_state = []
admissible_area_state = []

for batch_id in range(len(optimal_ncbf_test_results)):
    h_shape_s.append(optimal_ncbf_test_results[batch_id]["shape_h"]["state"])
    h_shape_val.append(optimal_ncbf_test_results[batch_id]["shape_h"]["val"])
    s_unsafe_violation.append(optimal_ncbf_test_results[batch_id]["unsafe_violation"]["state"])
    descent_violation.append(optimal_ncbf_test_results[batch_id]["descent_violation"]["state"])
    inadmissible_boundary_state.append(optimal_ncbf_test_results[batch_id]["inadmissible_boundary"]["state"])
    inadmissible_area_state.append(optimal_ncbf_test_results[batch_id]["inadmissible_area"]["state"])
    admissible_area_state.append(optimal_ncbf_test_results[batch_id]["admissible_area"]["state"])

h_shape_s_ncbf = torch.vstack(h_shape_s)
h_shape_val_ncbf = torch.vstack(h_shape_val)
s_unsafe_violation_ncbf = torch.vstack(s_unsafe_violation)
descent_violation_ncbf = torch.vstack(descent_violation)
inadmissible_boundary_state_ncbf = torch.vstack(inadmissible_boundary_state)
inadmissible_area_state_ncbf = torch.vstack(inadmissible_area_state)
admissible_area_state_ncbf = torch.vstack(admissible_area_state)

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


X_neural_clbf = h_shape_s_neural_clbf[:, x_index].detach().cpu().numpy()
Y_neural_clbf = h_shape_s_neural_clbf[:, y_index].detach().cpu().numpy()
H_neural_clbf = h_shape_val_neural_clbf.squeeze(dim=1).detach().cpu().numpy()

H_positive_mask_neural_clbf = H_neural_clbf > 0


x_neural_clbf = X_neural_clbf[H_positive_mask_neural_clbf]
y_neural_clbf = Y_neural_clbf[H_positive_mask_neural_clbf]


####
X_ncbf = h_shape_s_ncbf[:, x_index].detach().cpu().numpy()
Y_ncbf = h_shape_s_ncbf[:, y_index].detach().cpu().numpy()
H_ncbf = h_shape_val_ncbf.squeeze(dim=1).detach().cpu().numpy()

H_positive_mask_ncbf = H_ncbf > 0.01


x_ncbf = X_ncbf[H_positive_mask_ncbf]
y_ncbf = Y_ncbf[H_positive_mask_ncbf]

plt.figure()


X_inadmissible_area = inadmissible_area_state_ncbf[:, x_index].detach().cpu().numpy()
Y_inadmissible_area = inadmissible_area_state_ncbf[:, y_index].detach().cpu().numpy()
plt.scatter(X_inadmissible_area, Y_inadmissible_area, s=1, c='#939393')

X_admissible_area = admissible_area_state_ncbf[:, x_index].detach().cpu().numpy()
Y_admissible_area = admissible_area_state_ncbf[:, y_index].detach().cpu().numpy()
plt.scatter(X_admissible_area, Y_admissible_area, s=1, c='#B2EAAB')

# Create contour lines or level curves using matpltlib.pyplt module

x = np.linspace(min(hVS_XData.flatten()), max(hVS_XData.flatten()), 1000)
y = np.linspace(min(hVS_YData.flatten()), max(hVS_YData.flatten()), 1000)
xi, yi = np.meshgrid(x, y)

zi = griddata((hVS_XData.flatten(), hVS_YData.flatten()), hVS_ZData.flatten(), (xi, yi), method='cubic')

LST_x = []
LST_y = []

for i in range(xi.shape[0]-1):
    for j in range(xi.shape[1]-1):
        if zi[i,j] > 0:
            LST_x.append(xi[i,j])
            LST_y.append(yi[i,j])
plt.scatter(LST_x, LST_y, s=1, c='#F0B2E1')

# contours = plt.contourf(hVS_XData, hVS_YData, hVS_ZData, levels=[-0.1, 0, 1], colors=['w','#a7f790','w'], extend='both')

# contours2 = plt.contour(hVS0_XData, hVS0_YData, hVS0_ZData, levels=[0], colors='grey', linewidth=5)


plt.scatter(x_ncbf, y_ncbf, s=1, c='#3171AD')
plt.scatter(x_neural_clbf, y_neural_clbf, s=1, c='#EAE159')

X = inadmissible_boundary_state_ncbf[:, x_index].detach().cpu().numpy()
Y = inadmissible_boundary_state_ncbf[:, y_index].detach().cpu().numpy()
# plt.scatter(X, Y, s=2, c='#469C76')



plt.xlabel(r"$\theta$(rad)")
plt.ylabel(r"$\dot{\theta}$(rad/s)")
plt.xlim(domain_limit_lb[x_index]+0.3, domain_limit_ub[x_index]-0.3)
plt.ylim(domain_limit_lb[y_index]+0.7, domain_limit_ub[y_index]-0.7)
# plt.title("shape of 0-superlevel set")


legend_elements = [
                    Patch(facecolor='#939393', edgecolor='#939393',
                        label='Inadmissible'),
                    Patch(facecolor='#B2EAAB', edgecolor='#B2EAAB',
                        label='Admissible'),
                    Patch(facecolor='#F0B2E1', edgecolor='#F0B2E1',
                        label='LST'),
                    Patch(facecolor='#EAE159', edgecolor='#EAE159',
                        label='NeuralCLBF'),
                    Patch(facecolor='#3171AD', edgecolor='#3171AD',
                        label='Ours')
                ]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), fontsize="10", loc='upper right')

plt.xlabel(r"$\theta$ (rad)", fontsize="15")
plt.ylabel(r"$\dot{\theta}$ (rad/s)", fontsize="15")
plt.xticks(fontsize="15")
plt.yticks(fontsize="15")

plt.savefig("fig/prove_optimality.png", dpi=300)

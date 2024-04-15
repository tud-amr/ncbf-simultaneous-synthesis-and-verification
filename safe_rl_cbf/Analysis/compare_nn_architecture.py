import numpy as np
import os
# import seaborn as sns
import matplotlib.pyplot as plt
import torch


from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D

import scipy.io as sio

from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Models.custom_header import *



############### create folder #################

if not os.path.exists("fig/compare_nn_architecture"):
    os.makedirs("fig/compare_nn_architecture")

x_index = 0
y_index = 1

##################### read data #######################
system = select_dynamic_system("InvertedPendulum", "constraints_inverted_pendulum")    


ncbf_4_test_results = torch.load("logs/CBF_logs/IP_4_08_Mar/test_results.pt")
ncbf_8_test_results = torch.load("logs/CBF_logs/IP_4_depth_3/test_results.pt")
ncbf_32_test_results = torch.load("logs/CBF_logs/IP_8_depth_10/test_results.pt")

domain_limit_lb, domain_limit_ub = system.domain_limits


h_shape_s = []
h_shape_val = []
s_unsafe_violation = []
s_unsafe_violation_val = []

descent_violation = []

inadmissible_boundary_state = []
inadmissible_area_state = []
admissible_area_state = []

for batch_id in range(len(ncbf_4_test_results)):
    h_shape_s.append(ncbf_4_test_results[batch_id]["shape_h"]["state"])
    h_shape_val.append(ncbf_4_test_results[batch_id]["shape_h"]["val"])
    s_unsafe_violation.append(ncbf_4_test_results[batch_id]["unsafe_violation"]["state"])
    descent_violation.append(ncbf_4_test_results[batch_id]["descent_violation"]["state"])
    inadmissible_boundary_state.append(ncbf_4_test_results[batch_id]["inadmissible_boundary"]["state"])
    inadmissible_area_state.append(ncbf_4_test_results[batch_id]["inadmissible_area"]["state"])
    admissible_area_state.append(ncbf_4_test_results[batch_id]["admissible_area"]["state"])

h_shape_s_ncbf_4 = torch.vstack(h_shape_s)
h_shape_val_ncbf_4= torch.vstack(h_shape_val)
s_unsafe_violation_ncbf_4 = torch.vstack(s_unsafe_violation)
descent_violation_ncbf_4 = torch.vstack(descent_violation)
inadmissible_area_state_ncbf_4 = torch.vstack(inadmissible_area_state)
inadmissible_boundary_state_ncbf_4 = torch.vstack(inadmissible_boundary_state)
admissible_area_state_ncbf_4 = torch.vstack(admissible_area_state)

##########
h_shape_s = []
h_shape_val = []
s_unsafe_violation = []
s_unsafe_violation_val = []

descent_violation = []

inadmissible_boundary_state = []
inadmissible_area_state = []
admissible_area_state = []

for batch_id in range(len(ncbf_8_test_results)):
    h_shape_s.append(ncbf_8_test_results[batch_id]["shape_h"]["state"])
    h_shape_val.append(ncbf_8_test_results[batch_id]["shape_h"]["val"])
    s_unsafe_violation.append(ncbf_8_test_results[batch_id]["unsafe_violation"]["state"])
    descent_violation.append(ncbf_8_test_results[batch_id]["descent_violation"]["state"])
    inadmissible_boundary_state.append(ncbf_8_test_results[batch_id]["inadmissible_boundary"]["state"])
    inadmissible_area_state.append(ncbf_8_test_results[batch_id]["inadmissible_area"]["state"])
    admissible_area_state.append(ncbf_8_test_results[batch_id]["admissible_area"]["state"])


h_shape_s_ncbf_8 = torch.vstack(h_shape_s)
h_shape_val_ncbf_8 = torch.vstack(h_shape_val)
s_unsafe_violation_ncbf_8 = torch.vstack(s_unsafe_violation)
descent_violation_ncbf_8 = torch.vstack(descent_violation)
inadmissible_boundary_state_ncbf_8 = torch.vstack(inadmissible_boundary_state)
inadmissible_area_state_ncbf_8 = torch.vstack(inadmissible_area_state)
admissible_area_state_ncbf_8 = torch.vstack(admissible_area_state)


##########
h_shape_s = []
h_shape_val = []
s_unsafe_violation = []
s_unsafe_violation_val = []

descent_violation = []

inadmissible_boundary_state = []
inadmissible_area_state = []
admissible_area_state = []

for batch_id in range(len(ncbf_32_test_results)):
    h_shape_s.append(ncbf_32_test_results[batch_id]["shape_h"]["state"])
    h_shape_val.append(ncbf_32_test_results[batch_id]["shape_h"]["val"])
    s_unsafe_violation.append(ncbf_32_test_results[batch_id]["unsafe_violation"]["state"])
    descent_violation.append(ncbf_32_test_results[batch_id]["descent_violation"]["state"])
    inadmissible_boundary_state.append(ncbf_32_test_results[batch_id]["inadmissible_boundary"]["state"])
    inadmissible_area_state.append(ncbf_32_test_results[batch_id]["inadmissible_area"]["state"])
    admissible_area_state.append(ncbf_32_test_results[batch_id]["admissible_area"]["state"])


h_shape_s_ncbf_32 = torch.vstack(h_shape_s)
h_shape_val_ncbf_32 = torch.vstack(h_shape_val)
s_unsafe_violation_ncbf_32 = torch.vstack(s_unsafe_violation)
descent_violation_ncbf_32 = torch.vstack(descent_violation)
inadmissible_boundary_state_ncbf_32 = torch.vstack(inadmissible_boundary_state)
inadmissible_area_state_ncbf_32 = torch.vstack(inadmissible_area_state)
admissible_area_state_ncbf_32 = torch.vstack(admissible_area_state)


########################## start to plot #############################

############################### plot shape of function h(x) ##############################


X_ncbf_4 = h_shape_s_ncbf_4[:, x_index].detach().cpu().numpy()
Y_ncbf_4 = h_shape_s_ncbf_4[:, y_index].detach().cpu().numpy()
H_ncbf_4 = h_shape_val_ncbf_4.squeeze(dim=1).detach().cpu().numpy()

H_positive_mask_ncbf_4 = H_ncbf_4 > 0


x_ncbf_4 = X_ncbf_4[H_positive_mask_ncbf_4]
y_ncbf_4 = Y_ncbf_4[H_positive_mask_ncbf_4]

X_descent_ncbf_4 = descent_violation_ncbf_4[:, x_index].detach().cpu().numpy()
Y_descent_ncbf_4 = descent_violation_ncbf_4[:, y_index].detach().cpu().numpy()



####
X_ncbf_8 = h_shape_s_ncbf_8[:, x_index].detach().cpu().numpy()
Y_ncbf_8 = h_shape_s_ncbf_8[:, y_index].detach().cpu().numpy()
H_ncbf_8 = h_shape_val_ncbf_8.squeeze(dim=1).detach().cpu().numpy()

H_positive_mask_ncbf_8 = H_ncbf_8 > 0.01


x_ncbf_8 = X_ncbf_8[H_positive_mask_ncbf_8]
y_ncbf_8 = Y_ncbf_8[H_positive_mask_ncbf_8]


X_descent_ncbf_8 = descent_violation_ncbf_8[:, x_index].detach().cpu().numpy()
Y_descent_ncbf_8 = descent_violation_ncbf_8[:, y_index].detach().cpu().numpy()



####
X_ncbf_32 = h_shape_s_ncbf_32[:, x_index].detach().cpu().numpy()
Y_ncbf_32 = h_shape_s_ncbf_32[:, y_index].detach().cpu().numpy()
H_ncbf_32 = h_shape_val_ncbf_32.squeeze(dim=1).detach().cpu().numpy()

H_positive_mask_ncbf_32 = H_ncbf_32 > 0.01


x_ncbf_32 = X_ncbf_32[H_positive_mask_ncbf_32]
y_ncbf_32 = Y_ncbf_32[H_positive_mask_ncbf_32]


X_descent_ncbf_32 = descent_violation_ncbf_32[:, x_index].detach().cpu().numpy()
Y_descent_ncbf_32 = descent_violation_ncbf_32[:, y_index].detach().cpu().numpy()

######################

plt.figure()


X_inadmissible_area_4 = inadmissible_area_state_ncbf_4[:, x_index].detach().cpu().numpy()
Y_inadmissible_area_4 = inadmissible_area_state_ncbf_4[:, y_index].detach().cpu().numpy()
plt.scatter(X_inadmissible_area_4, Y_inadmissible_area_4, s=1, c='#939393')

X_admissible_area_4 = admissible_area_state_ncbf_4[:, x_index].detach().cpu().numpy()
Y_admissible_area_4 = admissible_area_state_ncbf_4[:, y_index].detach().cpu().numpy()
plt.scatter(X_admissible_area_4, Y_admissible_area_4, s=1.2, c='#B2EAAB')

plt.scatter(x_ncbf_4, y_ncbf_4, s=1, c='#3171AD')
plt.scatter(X_descent_ncbf_4, Y_descent_ncbf_4, s=10, c='#C66526')

X = inadmissible_boundary_state_ncbf_4[:, x_index].detach().cpu().numpy()
Y = inadmissible_boundary_state_ncbf_4[:, y_index].detach().cpu().numpy()
plt.scatter(X, Y, s=1, c='#939393')

X = s_unsafe_violation_ncbf_4[:, x_index].detach().cpu().numpy()
Y = s_unsafe_violation_ncbf_4[:, y_index].detach().cpu().numpy()
plt.scatter(X, Y, s=10, c='#C66526')

plt.xlabel(r"$\theta$(rad)", fontsize="15")
plt.ylabel(r"$\dot{\theta}$(rad/s)", fontsize="15")
plt.xticks(fontsize="15")
plt.yticks(fontsize="15")
plt.xlim(domain_limit_lb[x_index]+0.3, domain_limit_ub[x_index]-0.3)
plt.ylim(domain_limit_lb[y_index]+0.7, domain_limit_ub[y_index]-0.7)
# plt.title("shape of 0-superlevel set")


legend_elements = [
                    Patch(facecolor='#939393', edgecolor='#939393',
                        label='Inadmissible'),
                    Patch(facecolor='#B2EAAB', edgecolor='#B2EAAB',
                        label='Admissible'),
                    Patch(facecolor='#3171AD', edgecolor='#3171AD',
                        label='0 superlevel set'),
                    Patch(facecolor='#C66526', edgecolor='#C66526',
                    label='Violation')
                ]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), fontsize="12", loc='upper right')
plt.xlabel(r"$\theta$ (rad)", fontsize="15")
plt.ylabel(r"$\dot{\theta}$ (rad/s)", fontsize="15")
plt.xticks(fontsize="15")
plt.yticks(fontsize="15")

plt.savefig("fig/compare_nn_architecture/1.png", dpi=300)

#######################
plt.figure()


X_inadmissible_area_8 = inadmissible_area_state_ncbf_8[:, x_index].detach().cpu().numpy()
Y_inadmissible_area_8 = inadmissible_area_state_ncbf_8[:, y_index].detach().cpu().numpy()
plt.scatter(X_inadmissible_area_8, Y_inadmissible_area_8, s=1, c='#939393')

X_admissible_area_8 = admissible_area_state_ncbf_8[:, x_index].detach().cpu().numpy()
Y_admissible_area_8 = admissible_area_state_ncbf_8[:, y_index].detach().cpu().numpy()
plt.scatter(X_admissible_area_8, Y_admissible_area_8, s=1, c='#B2EAAB')



plt.scatter(x_ncbf_8, y_ncbf_8, s=1, c='#3171AD')
plt.scatter(X_descent_ncbf_8, Y_descent_ncbf_8, s=10, c='#C66526')


X = inadmissible_boundary_state_ncbf_8[:, x_index].detach().cpu().numpy()
Y = inadmissible_boundary_state_ncbf_8[:, y_index].detach().cpu().numpy()
plt.scatter(X, Y, s=1, c='#939393')

X = s_unsafe_violation_ncbf_8[:, x_index].detach().cpu().numpy()
Y = s_unsafe_violation_ncbf_8[:, y_index].detach().cpu().numpy()
plt.scatter(X, Y, s=10, c='#C66526')

plt.xlabel(r"$\theta$ (rad)", fontsize="15")
plt.ylabel(r"$\dot{\theta}$ (rad/s)", fontsize="15")
plt.xticks(fontsize="15")
plt.yticks(fontsize="15")
plt.xlim(domain_limit_lb[x_index]+0.3, domain_limit_ub[x_index]-0.3)
plt.ylim(domain_limit_lb[y_index]+0.7, domain_limit_ub[y_index]-0.7)

# plt.title("shape of 0-superlevel set")


legend_elements = [
                    Patch(facecolor='#939393', edgecolor='#939393',
                        label='Inadmissible'),
                    Patch(facecolor='#B2EAAB', edgecolor='#B2EAAB',
                        label='Admissible'),
                    Patch(facecolor='#3171AD', edgecolor='#3171AD',
                        label='0 superlevel set'),
                    Patch(facecolor='#C66526', edgecolor='#C66526',
                    label='Violation')
                ]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), fontsize="12", loc='upper right')


plt.savefig("fig/compare_nn_architecture/2.png", dpi=300)

#######################


plt.figure()


X_inadmissible_area_32 = inadmissible_area_state_ncbf_32[:, x_index].detach().cpu().numpy()
Y_inadmissible_area_32 = inadmissible_area_state_ncbf_32[:, y_index].detach().cpu().numpy()
plt.scatter(X_inadmissible_area_32, Y_inadmissible_area_32, s=1, c='#939393')

X_admissible_area_32 = admissible_area_state_ncbf_32[:, x_index].detach().cpu().numpy()
Y_admissible_area_32 = admissible_area_state_ncbf_32[:, y_index].detach().cpu().numpy()
plt.scatter(X_admissible_area_32, Y_admissible_area_32, s=1.2, c='#B2EAAB')



# Create contour lines or level curves using matpltlib.pyplt module
# contours = plt.contourf(hVS_XData, hVS_YData, hVS_ZData, levels=[-0.1, 0, 1], colors=['w','#a7f790','w'], extend='both')


plt.scatter(x_ncbf_32, y_ncbf_32, s=1, c='#3171AD')
plt.scatter(X_descent_ncbf_32, Y_descent_ncbf_32, s=10, c='#ff866f')


X = inadmissible_boundary_state_ncbf_32[:, x_index].detach().cpu().numpy()
Y = inadmissible_boundary_state_ncbf_32[:, y_index].detach().cpu().numpy()
plt.scatter(X, Y, s=1, c='#939393')

X = s_unsafe_violation_ncbf_32[:, x_index].detach().cpu().numpy()
Y = s_unsafe_violation_ncbf_32[:, y_index].detach().cpu().numpy()
plt.scatter(X, Y, s=10, c='#C66526')

plt.xlabel(r"$\theta$ (rad)", fontsize="15")
plt.ylabel(r"$\dot{\theta}$ (rad/s)", fontsize="15")
plt.xticks(fontsize="15")
plt.yticks(fontsize="15")
plt.xlim(domain_limit_lb[x_index]+0.3, domain_limit_ub[x_index]-0.3)
plt.ylim(domain_limit_lb[y_index]+0.7, domain_limit_ub[y_index]-0.7)
# plt.title("shape of 0-superlevel set")


legend_elements = [
                    Patch(facecolor='#939393', edgecolor='#939393',
                        label='Inadmissible'),
                    Patch(facecolor='#B2EAAB', edgecolor='#B2EAAB',
                        label='Admissible'),
                    Patch(facecolor='#3171AD', edgecolor='#3171AD',
                        label='0 superlevel set'),
                    Patch(facecolor='#C66526', edgecolor='#C66526',
                    label='Violation')
                ]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), fontsize="12", loc='upper right')


plt.savefig("fig/compare_nn_architecture/3.png", dpi=300)
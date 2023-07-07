import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
import lightning.pytorch as pl
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import scipy.io as sio

from MyNeuralNetwork import *
from dynamic_system_instances import car1, inverted_pendulum_1
from DataModule import DataModule

##################### draw loss curve #######################

test_results = torch.load("test_results.pt")

domain_limit_lb, domain_limit_ub = inverted_pendulum_1.domain_limits
data_module = DataModule(system=inverted_pendulum_1, val_split=0, train_batch_size=64, test_batch_size=128, train_grid_gap=0.3, test_grid_gap=0.01)
data_module.prepare_data()

s_training = data_module.s_training
s_training = torch.load("s_training.pt")

######################## extract test results ###########################

h_shape_s = []
h_shape_val = []
g_shape_s = []
g_shape_val = []
s_safe_violation = []
s_safe_violation_val = []
s_unsafe_violation = []
s_unsafe_violation_val = []

descent_violation = []

safe_boundary_state = []

for batch_id in range(len(test_results)):
    h_shape_s.append(test_results[batch_id]["shape_h"]["state"])
    h_shape_val.append(test_results[batch_id]["shape_h"]["val"])
#     g_shape_s.append(test_results[batch_id]["shape_g"]["state"])
#     g_shape_val.append(test_results[batch_id]["shape_g"]["val"])
    s_safe_violation.append(test_results[batch_id]["safe_violation"]["state"])
    s_unsafe_violation.append(test_results[batch_id]["unsafe_violation"]["state"])
    descent_violation.append(test_results[batch_id]["descent_violation"]["state"])
    safe_boundary_state.append(test_results[batch_id]["safe_boundary"]["state"])

h_shape_s = torch.vstack(h_shape_s)
h_shape_val = torch.vstack(h_shape_val)
# g_shape_s = torch.vstack(g_shape_s)
# g_shape_val = torch.vstack(g_shape_val)
s_safe_violation = torch.vstack(s_safe_violation)
s_unsafe_violation = torch.vstack(s_unsafe_violation)
descent_violation = torch.vstack(descent_violation)
safe_boundary_state = torch.vstack(safe_boundary_state)


########################## start to plot #############################

############################### plot shape of function l(x) ##############################



mat_contents = sio.loadmat("RA_result/extraOuts.mat")
# print(mat_contents['a0'].shape)

hVS_XData = mat_contents['a0']
hVS_YData = mat_contents['a1']
hVS_ZData = mat_contents['a2']
hVS0_XData = mat_contents['a3']
hVS0_YData = mat_contents['a4']
hVS0_ZData = mat_contents['a5']

plt.figure()
# Provide a title for the contour plt
plt.title('Avoid Set Surface Function')

plt.xlabel(r"$\theta$")
plt.ylabel(r"$\dot{\theta}$")

# Create contour lines or level curves using matpltlib.pyplt module
contours = plt.contourf(hVS0_XData, hVS0_YData, hVS0_ZData, levels=[-0.1, 0, 1], colors=['grey','w','w'], extend='both')

contours2 = plt.contour(hVS0_XData, hVS0_YData, hVS0_ZData)

# Display z values on contour lines
plt.clabel(contours2, inline=1, fontsize=10)

############################### plot shape of barrier function ##############################

plt.figure()

# Create contour lines or level curves using matpltlib.pyplt module
contours = plt.contourf(hVS_XData, hVS_YData, hVS_ZData, levels=[-0.1, 0, 1], colors=['w','y','w'], extend='both')

contours2 = plt.contour(hVS0_XData, hVS0_YData, hVS0_ZData, levels=[0], colors='grey', linewidth=5)

# create h_shape data_frame

X = h_shape_s[:, 0].cpu().numpy()
U = h_shape_s[:, 1].cpu().numpy()
H = h_shape_val.squeeze(dim=1).cpu().numpy()

H_positive_mask = H > 0


x_pos = X[H_positive_mask]
u_pos = U[H_positive_mask]
x_neg = X[~H_positive_mask]
u_neg = U[~H_positive_mask]

# h_shape_df = pd.DataFrame( {"x": X, "u": U, "h": H, "h_pos_mask": H_positive_mask}, index=range(0, X.shape[0]) )
# fig, ax = plt.subplots()
# sns.scatterplot(data=h_shape_df, x="x", y="u", hue="h_pos_mask", ax=ax)

plt.scatter(x_pos, u_pos, s=10, c='b')
# plt.scatter(x_neg, u_neg, s=10, c='tab:gray')
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\dot{\theta}$")
plt.title("shape of 0-superlevel set")


u_unsafe = np.arange(domain_limit_lb[1],domain_limit_ub[1],0.1)
x_unsafe1 = np.ones(u_unsafe.shape[0]) * np.pi * 5 /6
x_unsafe2 = - np.ones(u_unsafe.shape[0]) * np.pi * 5 /6
# plt.plot(x_unsafe1, u_unsafe, c='y', linewidth=2)
# plt.plot(x_unsafe2, u_unsafe, c='y', linewidth=2)

safe_boundary_state = safe_boundary_state.cpu().numpy()
# plt.scatter(safe_boundary_state[:, 0], safe_boundary_state[:,1], s=0.5, c='y')

# plt.scatter(s_training[:,0], s_training[:,1], marker='X', s=10, c='k')


# custom legend

legend_elements = [
                    Line2D([0], [0], marker='X', color='w', label='Training samples',
                          markerfacecolor='k', markersize=10),
                    Line2D([0], [0], color='grey', lw=2, label='Obstacles'),
                    Patch(facecolor='y', edgecolor='y',
                         label='Invariant Set from RA'),
                    Patch(facecolor='b', edgecolor='b',
                         label='Invariant Set from neural CBF')
                   ]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1.1),loc='upper right')


plt.figure()
contours2 = plt.contour(hVS0_XData, hVS0_YData, hVS0_ZData, levels=[0])
plt.scatter(s_training[:,0], s_training[:,1], marker='X', s=10, c='k')
legend_elements = [
                    Line2D([0], [0], marker='X', color='w', label='Training samples',
                          markerfacecolor='k', markersize=10),
                    Line2D([0], [0], color='grey', lw=2, label='Obstacles'),
                   ]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1.1),loc='upper right')

fig1,ax1=plt.subplots(1,1)
cp = ax1.contourf(X.reshape((math.gcd(X.shape[0], 1000), -1)), U.reshape((math.gcd(X.shape[0], 1000), -1)), H.reshape((math.gcd(X.shape[0], 1000), -1)))
fig1.colorbar(cp) # Add a colorbar to a plot
ax1.set_title('Filled Contours Plot')
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\dot{\theta}$")
ax1.set_title("contour of CBF")


####################### plot safe violation point #####################
# create safe_violation data_frame

X_safe_vio = s_safe_violation[:, 0].cpu().numpy()
U_safe_vio = s_safe_violation[:, 1].cpu().numpy()

print(f"there are {X_safe_vio.shape[0]} point violate safe reagion")

s_safe_violation_df = pd.DataFrame({"x": X_safe_vio, "u": U_safe_vio}, index=range(0, X_safe_vio.shape[0]))

# plt.figure()
# plt.scatter(x_pos, u_pos, s=10, c='b', label="forward invariant set")
# plt.scatter(x_neg, u_neg, s=10, c='tab:gray')
# plt.xlabel(r"$\theta$")
# plt.ylabel(r"$\dot{\theta}$")
# plt.title("safe violation area")
# plt.legend(bbox_to_anchor=(1, 1.1),loc='upper right')

u_unsafe = np.arange(domain_limit_lb[1],domain_limit_ub[1],0.1)
x_unsafe1 = np.ones(u_unsafe.shape[0]) * np.pi * 5 /6
x_unsafe2 = - np.ones(u_unsafe.shape[0]) * np.pi * 5 /6
# plt.plot(x_unsafe1, u_unsafe, c='y', linewidth=2)
# plt.plot(x_unsafe2, u_unsafe, c='y', linewidth=2)

# plt.scatter(safe_boundary_state[:, 0], safe_boundary_state[:,1], s=0.5, c='y')

# plt.scatter(X_safe_vio, U_safe_vio, marker='X', c='r')

# plt.figure()
# sns.scatterplot(data=s_safe_violation_df, x="x", y="u", marker="X")
# plt.title("safe violation states")

######################## plot unsafe violation points ##########################
# create unsafe_violation data_frame

X_unsafe_vio = s_unsafe_violation[:, 0].cpu().numpy()
U_unsafe_vio = s_unsafe_violation[:, 1].cpu().numpy()

print(f"there are {X_unsafe_vio.shape[0]} point violate unsafe reagion")

s_unsafe_violation_df = pd.DataFrame({"x": X_unsafe_vio, "u": U_unsafe_vio}, index=range(0, X_unsafe_vio.shape[0]))



plt.figure()
plt.scatter(x_pos, u_pos, s=10, c='b', label="forward invariant set")
plt.scatter(x_neg, u_neg, s=10, c='tab:gray')
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\dot{\theta}$")
plt.title("unsafe violation area")
plt.legend(bbox_to_anchor=(1, 1.1),loc='upper right')

u_unsafe = np.arange(domain_limit_lb[1],domain_limit_ub[1],0.1)
x_unsafe1 = np.ones(u_unsafe.shape[0]) * np.pi * 5 /6
x_unsafe2 = - np.ones(u_unsafe.shape[0]) * np.pi * 5 /6
plt.plot(x_unsafe1, u_unsafe, c='y', linewidth=2)
plt.plot(x_unsafe2, u_unsafe, c='y', linewidth=2)

# plt.scatter(safe_boundary_state[:, 0], safe_boundary_state[:,1], s=0.5, c='y')

plt.scatter(X_unsafe_vio, U_unsafe_vio, marker='X', c='r')


# plt.figure()
# sns.scatterplot(data=s_unsafe_violation_df, x="x", y="u", marker="X")
# plt.title("unsafe violation states")


########################## plot descent violation points #####################
# create descent_violation data_frame


X_descent = descent_violation[:, 0].cpu().numpy()
U_descent = descent_violation[:, 1].cpu().numpy()

print(f"there are {X_descent.shape[0]} points violate descent condition")

plt.figure()
plt.scatter(x_pos, u_pos, s=10, c='b', label="forward invariant set")
plt.scatter(x_neg, u_neg, s=10, c='tab:gray')
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\dot{\theta}$")
plt.title("CBC violation area")
plt.legend(bbox_to_anchor=(1, 1.1),loc='upper right')

u_unsafe = np.arange(domain_limit_lb[1],domain_limit_ub[1],0.1)
x_unsafe1 = np.ones(u_unsafe.shape[0]) * np.pi * 5 /6
x_unsafe2 = - np.ones(u_unsafe.shape[0]) * np.pi * 5 /6
plt.plot(x_unsafe1, u_unsafe, c='y', linewidth=2)
plt.plot(x_unsafe2, u_unsafe, c='y', linewidth=2)

# plt.scatter(safe_boundary_state[:, 0], safe_boundary_state[:,1], s=0.5, c='y')

plt.scatter(X_descent, U_descent, s=10, c='r')

# descent_violation_df = pd.DataFrame({"x": X_descent, "u": U_descent}, index=range(0, X_descent.shape[0]))
# fig, ax = plt.subplots()
# sns.scatterplot(data=descent_violation_df, x="x", y="u", marker="X", ax=ax)
# plt.figure()
# plt.scatter(X_descent, U_descent, s=10, c='y')
# plt.title("descent violation states")
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)



#################### plot shape of g(x) ###############
# X = g_shape_s[:, 0].cpu().numpy()
# X_g = X.reshape((math.gcd(X.shape[0], 1000), -1))
# U_g = g_shape_s[:, 1].cpu().numpy().reshape((math.gcd(X.shape[0], 1000), -1))
# G = g_shape_val.squeeze(dim=1).cpu().numpy().reshape((math.gcd(X.shape[0], 1000), -1))


# fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})

# ax3.set_xlim(-np.pi, np.pi)
# ax3.set_ylim(-5, 5)
# ax3.set_zlim(-1, 3)

# ax3.set_title("shape of CBF")
# ax3.set_xlabel(r"$\theta$")
# ax3.set_ylabel(r"$\dot{\theta}$")
# ax3.set_zlabel(r"$g(x)$")

# ax3.xaxis._axinfo["grid"].update({"linewidth": 0})
# ax3.yaxis._axinfo["grid"].update({"linewidth": 0})
# ax3.zaxis._axinfo["grid"].update({"linewidth": 0})

# ax3.plot_surface(X_g, U_g, G, color='#FF00FF', alpha=0.5)


plt.show()
############### end #####################






# x = np.arange(-2, 2, 0.1)
# u = np.arange(-2, 2, 0.1)

# X, U = np.meshgrid(x, u)

# print(X.shape)
# print(U.shape)

# H = []
# V = []
# with torch.no_grad():
#     for col in range(X.shape[1]):
#         x_c = X[:, col].reshape((-1, 1))
#         u_c = U[:, col].reshape((-1, 1))
        
#         s_c = np.hstack((x_c, u_c))
#         s_c_tensor = torch.from_numpy(s_c).float().to(device)
#         h_s_c_gpu, v_s_c_gpu = NN(s_c_tensor)
#         h_s_c = h_s_c_gpu.cpu().numpy()
#         v_s_c = v_s_c_gpu.cpu().numpy()
#         H.append(h_s_c)
#         V.append(v_s_c)



# H = np.hstack(H)
# V = np.hstack(V)

# H_b = (H >= 0)






# fig = plt.figure()
# ax2 = plt.axes(projection='3d')
# ax2.contour3D(X, U, H, 50, cmap='binary')
# ax2.set_xlabel('x')
# ax2.set_ylabel('u')
# ax2.set_zlabel('h')
# ax2.set_title("the shape of barrier function")

# #H_b = (H >= 0)


# fig1,ax1=plt.subplots(1,1)
# cp = ax1.contourf(X, U, H)
# fig1.colorbar(cp) # Add a colorbar to a plot
# ax1.set_title('Filled Contours Plot')
# ax1.set_xlabel('x')
# ax1.set_ylabel('u')
# ax1.set_title("the color map of barrier function on x-u 2D plain")
# ax1.plot(x, y)




# plt.show()





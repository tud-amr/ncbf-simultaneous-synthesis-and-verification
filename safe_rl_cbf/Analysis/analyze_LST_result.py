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




# compute gradient from the neighborhood of the points from the grid map
def compute_gradient(xi, yi, zi, x):
    # find the nearest point
    x_index = np.argmin(np.abs(xi[0, :] - x[0]))
    y_index = np.argmin(np.abs(yi[:, 0] - x[1]))

    # compute gradient
    grad_x = (zi[y_index, x_index + 1] - zi[y_index, x_index - 1]) / (xi[0, x_index + 1] - xi[0, x_index - 1])
    grad_y = (zi[y_index + 1, x_index] - zi[y_index - 1, x_index]) / (yi[y_index + 1, 0] - yi[y_index - 1, 0])

    return np.array([grad_x, grad_y])



mat_contents = sio.loadmat("RA_result/LST_0.05/extraOuts.mat")
# print(mat_contents['a0'].shape)

hVS_XData = mat_contents['a0']
hVS_YData = mat_contents['a1']
hVS_ZData = mat_contents['a2']
hVS0_XData = mat_contents['a3']
hVS0_YData = mat_contents['a4']
hVS0_ZData = mat_contents['a5']

x = np.linspace(min(hVS_XData.flatten()), max(hVS_XData.flatten()), 1000)
y = np.linspace(min(hVS_YData.flatten()), max(hVS_YData.flatten()), 1000)
xi, yi = np.meshgrid(x, y)

zi = griddata((hVS_XData.flatten(), hVS_YData.flatten()), hVS_ZData.flatten(), (xi, yi), method='cubic')

# print(xi.shape)
# print(yi.shape)
# print(zi.shape)

# test the gradient computation
# x = np.array([5*np.pi / 6, 0.0])
# print(compute_gradient(xi, yi, zi, x))

# enumerate the points in the grid map
count = 0
violation_x_list = []
for i in range(xi.shape[0]-1):
    for j in range(xi.shape[1]-1):
        print(f"current point: {i}, {j}")

        x = torch.tensor([xi[i, j], yi[i, j]]).reshape(1, -1).float()
        unsafe_mask = inverted_pendulum_1.unsafe_mask(x)
        if unsafe_mask == 1:
            # print("unsafe")
            # print("unsafe_mask: ", unsafe_mask)
            # exit()
            pass
        else:
            x_np = np.array([xi[i, j], yi[i, j]])
            grad_np = compute_gradient(xi, yi, zi, x_np)
            gradh = torch.tensor(grad_np).reshape(1, 1, -1).float()
            # print("gradh: ", gradh.shape)

            h_x = torch.tensor(zi[i, j]).reshape(1, -1).float()

            f = inverted_pendulum_1.f(x)
            g = inverted_pendulum_1.g(x)

            # print("f: ", f.shape)
            # print("g: ", g.shape)

            Lf_h = torch.bmm(gradh, f.unsqueeze(dim=-1)).squeeze(1)
            Lg_h = torch.bmm(gradh, g).squeeze(1)

            # print("Lf_h: ", Lf_h.shape)
            # print("Lg_h: ", Lg_h.shape)


            K = torch.ones(1, 1, 2) * torch.unsqueeze(inverted_pendulum_1.K, dim=0)
            
            # print("K: ", K.shape)
            
            u = -torch.bmm(K, x.unsqueeze(dim=-1))
            u = u.squeeze(dim=-1)

            # print("u: ", u.shape)

            u_lower_bd ,  u_upper_bd = inverted_pendulum_1.control_limits
            u = torch.clip(1000 * u, u_lower_bd, u_upper_bd)
        
            Lg_h_u = Lg_h * u
            dt = Lf_h + Lg_h_u.sum(dim=1, keepdim=True)

            # print("dt: ", dt.shape)

            result = dt + 0.5 * h_x
            if h_x >= 0 :
                # print("u: ", u)
                # print("x: ", x)
                # print("h_x: ", h_x)
                # print("result: ", result)
                if result < 0:
                    count += 1
                    violation_x_list.append(x_np)

print("count: ", count)
violation_x_list = np.array(violation_x_list)
np.save("RA_result/LST_0.05/violation_x_list.npy", violation_x_list)
        


# plt.figure()

# # Create contour lines or level curves using matpltlib.pyplt module
# # contours = plt.contourf(hVS_XData, hVS_YData, hVS_ZData, levels=[-0.1, 0, 1], colors=['w','y','w'], extend='both')
# contours = plt.contourf(xi, yi, zi, levels=[-0.1, 0, 1], colors=['w','y','w'], extend='both')
# plt.savefig("RA_result/LST_0.1/contour.png")
# plt.show()
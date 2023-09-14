import numpy as np
import os

import matplotlib.pyplot as plt
import torch


from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import scipy.io as sio

from safe_rl_cbf.NeuralCBF.MyNeuralNetwork import *
from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1, cart_pole_1, dubins_car, dubins_car_acc, point_robot
from safe_rl_cbf.Dataset.TestingDataModule import TestingDataModule


test_results = torch.load("test_results.pt")


inadmissible_boundary_state = []

for batch_id in range(len(test_results)):
    inadmissible_boundary_state.append(test_results[batch_id]["inadmissible_boundary"]["state"])


inadmissible_boundary_state = torch.vstack(inadmissible_boundary_state)
inadmissible_boundary_state = inadmissible_boundary_state.detach().cpu().numpy()


system = point_robot
checkpoint_path = "saved_models/point_robot/checkpoints/epoch=68-step=11247.ckpt"
data_module = TestingDataModule(system=system, test_batch_size=1024, test_points_num=int(5e2), test_index={0: None, 1: None, 2: 0.3, 3:0.0})


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

NN = NeuralNetwork.load_from_checkpoint(checkpoint_path, dynamic_system=system, data_module=data_module)
NN.to(device)



x = np.linspace(-1, 9, 500)
y = np.linspace(-1, 9, 500)
dx = 0.5
dy = 0

X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

x_control_invariant = []
y_control_invariant = []

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        s = torch.tensor([X[i, j], Y[i, j], dx, dy], dtype=torch.float32).to(device)
        hs = NN.h(s).detach().cpu().numpy()
        Z[i, j] = hs

        if abs(hs) < 0.05:
            x_control_invariant.append(X[i, j])
            y_control_invariant.append(Y[i, j])

x_control_invariant = np.array(x_control_invariant)
y_control_invariant = np.array(y_control_invariant)


plt.imshow(Z, extent=[-1, 9, -1, 9], origin='lower', cmap='Blues', alpha=1)
plt.colorbar()
plt.scatter(x_control_invariant, y_control_invariant, s=1, c='b', alpha=1)
plt.scatter(inadmissible_boundary_state[:, 0], inadmissible_boundary_state[:, 1], s=1, c='#7cd6cf', alpha=1)

legend_elements = [
                    Line2D([0], [0], color='#7cd6cf', lw=2, label='Obstacles'),
                    Line2D([0], [0], color='b', lw=2, label='h(s) = 0'),
                ]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), fontsize="8", loc='upper right')

plt.xlabel("x")
plt.ylabel("y")
plt.savefig("fig/contour_cbf.png", dpi=100)
plt.show()

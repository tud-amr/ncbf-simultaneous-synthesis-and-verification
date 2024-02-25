import numpy as np
import os

import matplotlib.pyplot as plt
import torch


from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import scipy.io as sio

from safe_rl_cbf.Models.NeuralCBF import *
from safe_rl_cbf.Dynamics.dynamic_system_instances import car1, inverted_pendulum_1, cart_pole_1, dubins_car, dubins_car_acc, point_robot
from safe_rl_cbf.Dataset.TestingDataModule import TestingDataModule




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
inadmissible_boundary_state = inadmissible_boundary_state.detach().cpu().numpy()


system = inverted_pendulum_1
checkpoint_path = "saved_models/inverted_pendulum_umax_12/checkpoints/epoch=4-step=5.ckpt"
data_module = TestingDataModule(system=system, test_batch_size=1024, test_points_num=int(1e3), test_index={0: None, 1: None})


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

NN = NeuralNetwork.load_from_checkpoint(checkpoint_path, dynamic_system=system, data_module=data_module)
NN.to(device)


x = np.linspace(-5*np.pi/6, 5*np.pi/6, 100)
y = np.linspace(-4, 4, 100)

X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)


x_control_invariant = []
y_control_invariant = []

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        s = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).to(device)
        hs = NN.h(s).detach().cpu().numpy()
        Z[i, j] = hs

        if hs > 0:
            x_control_invariant.append(X[i, j])
            y_control_invariant.append(Y[i, j])


x_control_invariant = np.array(x_control_invariant)
y_control_invariant = np.array(y_control_invariant)

ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
ax.scatter(x_control_invariant, y_control_invariant, zs=0, zdir='z', color='#63b2ee', s=1)
ax.scatter(inadmissible_boundary_state[:, 0], inadmissible_boundary_state[:, 1], zs=0, zdir='z', s=1, c='#666E7A', alpha=1)

ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\dot{\theta}$')
ax.set_zlabel('h(x)')



legend_elements = [
                    Line2D([0], [0], color='#666E7A', lw=2, label='Obstacles'),
                    Patch(facecolor='#63b2ee', edgecolor='#63b2ee',
                        label='control invariant set')
                ]

plt.legend(handles=legend_elements, bbox_to_anchor=(1.2, 1), loc='upper right')

plt.savefig("fig/3D_cbf.png", dpi=300,bbox_inches='tight')

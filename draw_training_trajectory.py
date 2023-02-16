import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pickle

with open('./masterthesis_test/stable_baseline_logs/run1/with_CBF_trajectory.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f)

print(f"the len of data is {len(data)}")

plt.figure()

# plot trajectory
for epoch_num in range(len(data)):
    trajectory = data[epoch_num]

    # print(trajectory.shape)

    x = np.arange(0, trajectory.shape[1])
    y = trajectory[0, :]

    X_Y_Spline = make_interp_spline(x, y)

    X_ = np.linspace(x.min(), x.max(), 1000)
    Y_ = X_Y_Spline(X_)


    plt.plot(X_, Y_, color='b', linewidth='0.5')

# plot unsafe bound
x = np.arange(0, trajectory.shape[1])
y1 = np.ones(trajectory.shape[1]) * np.pi * 5/6
y2 = np.ones(trajectory.shape[1]) * - np.pi * 5/6

plt.plot(x, y1, y2, color='r', linewidth='2')

plt.xlabel('time-step')
plt.ylabel(r'$\theta$')
plt.title("trajectories during learning (with CBF)")
plt.show()
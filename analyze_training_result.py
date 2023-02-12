import numpy as np
import math
import matplotlib.pyplot as plt

import pickle

with open('./masterthesis_test/stable_baseline_logs/trajectory.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f)

print(f"the len of data is {len(data)}")

plt.figure()
for epoch_num in range(len(data)):
    trajectory = data[epoch_num]

    # print(trajectory.shape)
    if epoch_num == 1:
        plt.plot(np.arange(0, trajectory.shape[1]), trajectory[0,:], color='b', linewidth='0.5')

plt.show()
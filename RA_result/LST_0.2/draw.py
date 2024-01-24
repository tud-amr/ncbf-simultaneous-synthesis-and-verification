import numpy as np
import matplotlib.pyplot as plt

violation_x = np.load("RA_result/LST_0.1/violation_x_list.npy")

plt.figure()
plt.scatter(violation_x[:, 0], violation_x[:, 1], s=1)
plt.show()

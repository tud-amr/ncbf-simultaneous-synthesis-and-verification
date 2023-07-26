# libraries & dataset
import numpy as np
import matplotlib.pyplot as plt
import pickle

import scipy.io as sio

mat_contents = sio.loadmat("RA_result/extraOuts.mat")
# print(mat_contents['a0'].shape)

hVS_XData = mat_contents['a0']
hVS_YData = mat_contents['a1']
hVS_ZData = mat_contents['a2']
hVS0_XData = mat_contents['a3']
hVS0_YData = mat_contents['a4']
hVS0_ZData = mat_contents['a5']

# Provide a title for the contour plt
plt.title('Forward Invariant Set from RA')

plt.xlabel(r"$\theta$")
plt.ylabel(r"$\dot{\theta}$")

# Create contour lines or level curves using matpltlib.pyplt module

contours = plt.contourf(hVS_XData, hVS_YData, hVS_ZData, levels=[-0.1, 0, 1], colors=['w','y','w'], extend='both')
# contours2 = plt.contourf(hVS0_XData, hVS0_YData, hVS0_ZData, levels=[-0.1, 0, 1], colors=['grey','w','w'], extend='both')
contours3 = plt.contour(hVS0_XData, hVS0_YData, hVS0_ZData, levels=[0], colors='grey')

# Display z values on contour lines
# plt.clabel(contours3, inline=1, fontsize=10)

# Display the contour plt
plt.show()
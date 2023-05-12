import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import scipy.io as sio


mat_contents = sio.loadmat("RA_result/extraOuts_new.mat")
# print(mat_contents['a0'].shape)

hVS_XData = mat_contents['a0']
hVS_YData = mat_contents['a1']
hVS_ZData = mat_contents['a2']
hVS0_XData = mat_contents['a3']
hVS0_YData = mat_contents['a4']
hVS0_ZData = mat_contents['a5']

fig2, ax2 = plt.subplots()

def init2():
    ax2.set_title("0 super-level set of CBF ")
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\dot{\theta}$")
    

def update2(frame):
    xdata = hVS_XData
    ydata = hVS_YData
    if frame > 2:
        zdata = hVS0_ZData - (hVS0_ZData - hVS_ZData)* (frame -2)
    else: 
        zdata = hVS0_ZData
    ax2.contourf(xdata, ydata, zdata, levels=[-0.1, 0, 1], colors=['w','y','w'], extend='both')
    ax2.contour(hVS0_XData, hVS0_YData, hVS0_ZData, linestyles=['solid'], linewidths=[3], levels=[0], colors='k')

ani2 = FuncAnimation(fig2, update2, frames=np.linspace(0.1, 3, 60), init_func=init2, interval=300, blit=False, repeat=True)

file_name = "zero_level_set.gif"
writergif = animation.PillowWriter(fps=60)
ani2.save(file_name, writer=writergif)


fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})

def init3():
    ax3.set_xlim(-np.pi, np.pi)
    ax3.set_ylim(-5, 5)
    ax3.set_zlim(-1, 3)

    ax3.set_title("shape of CBF")
    ax3.set_xlabel(r"$\theta$")
    ax3.set_ylabel(r"$\dot{\theta}$")
    ax3.set_zlabel(r"$h(x)$")

    ax3.xaxis._axinfo["grid"].update({"linewidth": 0})
    ax3.yaxis._axinfo["grid"].update({"linewidth": 0})
    ax3.zaxis._axinfo["grid"].update({"linewidth": 0})
    

def update3(frame):
    xdata = hVS_XData
    ydata = hVS_YData
    ax3.clear()
    init3()
    if frame > 2:
        zdata = hVS0_ZData - (hVS0_ZData - hVS_ZData)* (frame -2)
        ax3.plot_surface(xdata, ydata, np.full_like(zdata, 0), color='#FF00FF', alpha=0.5)
        ax3.plot_surface(xdata, ydata, zdata, color='#FF00FF', alpha=0.7)
        
    elif frame > 1:
        zdata = hVS0_ZData*(frame-1)
        

        ax3.plot_surface(xdata, ydata, np.where(hVS0_ZData >= 0, hVS0_ZData, 0), color='#FF00FF', alpha=0.7)
        # ax3.plot_surface(xdata, ydata, np.full_like(zdata, 0), color='white', alpha=0.5)
        ax3.plot_surface(xdata, ydata, np.where(zdata <= 0, zdata, 0), color='#FF00FF', alpha=0.7)
      
    else: 
        zdata = hVS0_ZData*frame
        ax3.plot_surface(xdata, ydata, np.where(zdata >= 0, zdata, 0), color='#FF00FF', alpha=0.7)
        # ax3.plot_surface(xdata, ydata, np.full_like(zdata, 0), color='white', alpha=0.5)
    # ax3.contour(xdata, ydata, hVS0_ZData, levels= [0], colors='y', linewidths=[5])


ani3 = FuncAnimation(fig3, update3, frames=np.linspace(0.1, 3, 60), init_func=init3, interval=300, blit=False, repeat=False)


file_name = "shape_cbf.gif"
writergif = animation.PillowWriter(fps=60)
ani3.save(file_name, writer=writergif)

# plt.show()
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pickle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def draw_box(center, hight, width, color, alpha, linewidth='2'):
    x = center[0]
    y = center[1]

    x1 = x - width/2
    y1 = y - hight/2
    x2 = x + width/2
    y2 = y + hight/2

    plt.plot([x1, x2], [y1, y1], color=color, linewidth=linewidth, alpha=alpha)
    plt.plot([x1, x2], [y2, y2], color=color, linewidth=linewidth, alpha=alpha)
    plt.plot([x1, x1], [y1, y2], color=color, linewidth=linewidth, alpha=alpha)
    plt.plot([x2, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha)


# a function that draws a rectangle with color filled
def draw_rectangle(center, hight, width, color, alpha):
    x = center[0]
    y = center[1]

    x1 = x - width/2
    y1 = y - hight/2
    x2 = x + width/2
    y2 = y + hight/2

    plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], color=color, alpha=alpha)


# a function that draws a circle with color filled
def draw_circle(center, radius, color, alpha):
    circle = plt.Circle(center, radius, color=color, alpha=alpha)
    plt.gca().add_patch(circle)

with open('logs/stable_baseline_logs_20_Oct/point_robot_with/run3/with_CBF_trajectory.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f)

print(f"the len of data is {len(data)}")

plt.figure()

# plot trajectory
for epoch_num in range(len(data)):
    if epoch_num > 0:
        trajectory = data[epoch_num]["traj"]
        collision = data[epoch_num]["collision"]
    
        print(trajectory.shape)
        
        x = trajectory[0, :]
        y = trajectory[1, :]


        # X_Y_Spline = make_interp_spline(x, y)

        # X_ = np.linspace(x.min(), x.max(), 1000)
        # Y_ = X_Y_Spline(X_)

        if collision:
            plt.plot(x, y, color='#976f38', linewidth='0.15', alpha=0.6)
        else:
            plt.plot(x, y, color='b', linewidth='0.15', alpha=0.6)
   
draw_rectangle([2, 1], 2, 1, '#939393', 1)
draw_box([2, 2], 4, 4, '#939393', 1,  linewidth='5')
draw_circle([3, 3], 0.2, '#ffc30b', 1)



legend_elements = [
                    Line2D([0], [0], color='#939393', lw=2, label='Obstacles'),
                    Line2D([], [], color='#ffc30b', marker='.', linestyle='None',
                    markersize=20, label='Goal'),
                    Line2D([0], [0], color='b', lw=2, label='collision-free trajectory'),
                    Line2D([0], [0], color='#976f38', lw=2, label='collision trajectory')
                ]

plt.legend(handles=legend_elements, fontsize="12", loc='lower right')


plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig('fig/trajectories_with_cbf.png', dpi=300)

plt.show()
# bbox_to_anchor=(1, 1)
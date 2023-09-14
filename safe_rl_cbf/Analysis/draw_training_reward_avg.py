import numpy as np
import math
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns
import pickle

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


root_dir ="logs/stable_baseline_logs/"

runlog = pd.DataFrame(columns=['epoch', 'reward', 'reward_type'])

for i in range(0, 5):
    file_path_with_CBF = root_dir + "point_robot_with/run" + str(i) + "/with_CBF_reward.pickle"
    file_path_without_CBF = root_dir + "point_robot_without/run" + str(i) + "/without_CBF_reward.pickle"


    with open(file_path_with_CBF, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data1 = pickle.load(f)
   

    with open(file_path_without_CBF, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data2 = pickle.load(f)

    data1 = np.array(data1)
    data2 = np.array(data2)


    for step, data in enumerate(data1):
        epoch = step
        reward = data
        reward_type = "with_nCBF"
        r = {'epoch': epoch, 'average reward': reward, ' ': reward_type}
        runlog = runlog.append(r, ignore_index=True)
    
    for step, data in enumerate(data2):
        epoch = step
        reward = data
        reward_type = "without_nCBF"
        r = {'epoch': epoch, 'average reward': reward, ' ': reward_type}
        runlog = runlog.append(r, ignore_index=True)


sns.set_style('whitegrid')

ax = sns.relplot(data=runlog, x="epoch", y="average reward", hue=' ' , kind="line", aspect=(19.2/14.4), legend=False)
legend_elements = [
                    Line2D([0], [0], color='b', lw=2, label='with_nCBF'),
                    Line2D([0], [0], color='r', lw=2, label='withou_nCBF'),
                ]
ax.ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1), fontsize="8", loc='upper right')
plt.savefig("fig/RL_average_reward.png", dpi=300)
plt.show()

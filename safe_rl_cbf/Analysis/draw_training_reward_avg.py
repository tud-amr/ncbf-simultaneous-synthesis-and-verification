import numpy as np
import math
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns
import pickle

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

root_dir ="logs/stable_baseline_logs_IP_23_Oct/"

runlog = pd.DataFrame(columns=['epoch', 'reward', 'reward_type'])


for i in range(0, 2):
    file_path_with_CBF = root_dir + "IP_with/run" + str(i) + "/with_CBF_reward.pickle"

    with open(file_path_with_CBF, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
        data1 = pickle.load(f)

    data1 = np.array(data1)
    print(data1.shape)

    weight = np.repeat(1.0, 3)/3
    data1 = np.convolve(data1, weight, 'valid')

    for step, data in enumerate(data1):
        epoch = step * 2048
        reward = data
        reward_type = "with_nCBF"
        r = {'steps': epoch, 'average reward': reward, ' ': reward_type}
        runlog = runlog.append(r, ignore_index=True)
    

for i in range(0, 5):
   
    file_path_without_CBF = root_dir + "IP_without/run" + str(i) + "/without_CBF_reward.pickle"

    # with open(file_path_without_CBF, 'rb') as f:
    #     # The protocol version used is detected automatically, so we do not
    #     # have to specify it.
    #     data2 = pickle.load(f)

    
    data2 = np.array(data2)
    print(data2.shape)

    weight = np.repeat(1.0, 3)/3
    data2 = np.convolve(data2, weight, 'valid')

    
    for step, data in enumerate(data2):
        epoch = step * 2048
        reward = data
        reward_type = "without_nCBF"
        r = {'steps': epoch, 'average reward': reward, ' ': reward_type}
        runlog = runlog.append(r, ignore_index=True)




sns.set_style('whitegrid')

ax = sns.relplot(data=runlog, x="steps", y="average reward", hue=' ' , kind="line", palette=['b', '#6e912a'] , aspect=(19.2/14.4), legend=False)
legend_elements = [
                    Line2D([0], [0], color='#6e912a', lw=2, label='without_nCBF'),
                    Line2D([0], [0], color='b', lw=2, label='with_nCBF')
                ]
ax.ax.legend(handles=legend_elements, fontsize="14", loc='lower right')
plt.xlabel("step number" ,fontsize=15)
plt.ylabel("average reward",fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig("fig/RL_average_reward.png", dpi=300, bbox_inches='tight')
plt.show()

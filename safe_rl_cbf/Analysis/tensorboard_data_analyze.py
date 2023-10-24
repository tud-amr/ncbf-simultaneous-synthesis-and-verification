
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

file_path = "logs/tensorboard/"

# get all the files in the directory
all_files = glob.glob(os.path.join(file_path, "*.csv"))

# read all the files and put them into one dataframe
all_df = []
for file in all_files:
    df = pd.read_csv(file, index_col=None, header=0)
    # compute the moving average reward
    df['moving_average_reward'] = df['Value'].rolling(50).mean()  
    all_df.append(df)

# concatenate all the dataframes
merged_df = pd.concat(all_df, axis=0, ignore_index=True)

sns.set_style('whitegrid')

ax = sns.relplot(data=merged_df, x="Step", y="moving_average_reward", kind="line", aspect=(19.2/14.4), legend=False)
legend_elements = [
                    Line2D([0], [0], color='b', lw=2, label='with_nCBF'),
                    Line2D([0], [0], color='r', lw=2, label='without_nCBF'),
                ]
ax.ax.legend(handles=legend_elements, fontsize="14", loc='lower right')
plt.xlabel("epoch" ,fontsize=15)
plt.ylabel("average reward",fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig("fig/RL_average_reward.png", dpi=300)
plt.show()
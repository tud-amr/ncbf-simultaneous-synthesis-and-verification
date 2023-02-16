import numpy as np
import math
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

root_dir ="./masterthesis_test/CBF_logs/epoch_time/"

runlog = pd.DataFrame(columns=['name', 'average time for one epoch (s)', 'type'])

for i in range(5, 15):
    file_path_with_OptNet = root_dir + "run" + str(i) + "_with_OptNet_lightning_logs_version_0.json"
    file_path_without_OptNet = root_dir + "run" + str(i) + "_without_OptNet_lightning_logs_version_0.json"

    f1 = open(file_path_with_OptNet)
    data1 = json.load(f1)

    f2 = open(file_path_without_OptNet)
    data2 = json.load(f2)

    data1 = np.array(data1)
    data2 = np.array(data2)

    for step, data in enumerate(data1):
        name = "average epoch time"
        time = data[2]
        type = "with_OptNet"
        r = {'name': name, 'average time for one epoch (s)': time, 'type': type}
        runlog = runlog.append(r, ignore_index=True)
    
    for step, data in enumerate(data2):
        name = "average epoch time"
        time = data[2]
        type = "without_OptNet"
        r = {'name': name, 'average time for one epoch (s)': time, 'type': type}
        runlog = runlog.append(r, ignore_index=True)

    f1.close()
    f2.close()

sns.set_theme()


sns.displot(data=runlog, x="average time for one epoch (s)", hue='type', kind="kde",  fill=True )
plt.title("distribution of epoch time")
plt.legend(loc="upper right")
plt.xlim((5, 60))
plt.ylim((-0.01, 1))
plt.show()


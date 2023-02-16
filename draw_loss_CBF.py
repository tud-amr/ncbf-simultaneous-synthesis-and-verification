import numpy as np
import math
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

root_dir ="./masterthesis_test/CBF_logs/loss/"

runlog = pd.DataFrame(columns=['epoch', 'loss', 'loss_type'])

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
        epoch = data[1] // 70
        loss = data[2]
        loss_type = "with_OptNet"
        r = {'epoch': epoch, 'loss': loss, 'loss_type': loss_type}
        runlog = runlog.append(r, ignore_index=True)
    
    for step, data in enumerate(data2):
        epoch = data[1] // 70
        loss = data[2]
        loss_type = "without_OptNet"
        r = {'epoch': epoch, 'loss': loss, 'loss_type': loss_type}
        runlog = runlog.append(r, ignore_index=True)

    f1.close()
    f2.close()

sns.set_theme()


sns.relplot(data=runlog, x="epoch", y="loss", hue='loss_type' , kind="line")
plt.title("loss curve")
plt.show()


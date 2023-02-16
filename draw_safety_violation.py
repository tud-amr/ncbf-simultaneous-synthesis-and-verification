import numpy as np
import math
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

root_dir ="./masterthesis_test/stable_baseline_logs/safety_violation/"

runlog = pd.DataFrame(columns=['epoch', 'violation_num', 'violation_type'])

for i in range(0, 5):
    file_path_with_CBF = root_dir + "run" + str(i) + "_with_CBF__1.json"
    file_path_without_CBF = root_dir + "run" + str(i) + "_without_CBF__1.json"

    f1 = open(file_path_with_CBF)
    data1 = json.load(f1)

    f2 = open(file_path_without_CBF)
    data2 = json.load(f2)

    data1 = np.array(data1)
    data2 = np.array(data2)


    for step, data in enumerate(data1):
        epoch = data[1] // 2048
        violation_num = data[2]
        violation_type = "with_CBF"
        r = {'epoch': epoch, 'violation_num': violation_num, 'violation_type': violation_type}
        runlog = runlog.append(r, ignore_index=True)
    
    for step, data in enumerate(data2):
        epoch = data[1] // 2048
        violation_num = data[2]
        violation_type = "without_CBF"
        r = {'epoch': epoch, 'violation_num': violation_num, 'violation_type': violation_type}
        runlog = runlog.append(r, ignore_index=True)

    f1.close()
    f2.close()


sns.set_theme()


sns.relplot(data=runlog, x="epoch", y="violation_num", hue='violation_type' , kind="line")
plt.title("safety violation curve")
plt.show()

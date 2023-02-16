import numpy as np
import math
import matplotlib.pyplot as plt
import json

violation_with_CBF_file_path = "./masterthesis_test/stable_baseline_logs/with_CBF_run1_5/train_reward_with_CBF.json"
violation_without_CBF_file_path = "./masterthesis_test/stable_baseline_logs/without_CBF_run1_1/training_reward_without_CBF.json"


f1 = open(violation_with_CBF_file_path)
data1 = json.load(f1)

f2 = open(violation_without_CBF_file_path)
data2 = json.load(f2)

data1 = np.array(data1)
data2 = np.array(data2)
print(data1.shape)
print(data2.shape)


plt.figure()
plt.plot(data1[:, 1], data1[:, 2], color='b', linewidth='2', label='with CBF')
plt.plot(data2[:, 1], data2[:, 2], color='r', linewidth='2', label='without CBF')
plt.legend()
plt.xlabel("time-step")
plt.ylabel("reward")
plt.title("the reward curve")
plt.show()
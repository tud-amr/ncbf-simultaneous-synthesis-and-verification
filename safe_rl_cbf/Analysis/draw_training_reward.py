import numpy as np
import math
import matplotlib.pyplot as plt
import pickle


reward_with_CBF_file_path = "logs/stable_baseline_logs_backup/point_robot_with/run5/with_CBF_reward.pickle"
reward_without_CBF_file_path = "logs/stable_baseline_logs_backup/point_robot_without/run5/without_CBF_reward.pickle"


with open(reward_with_CBF_file_path, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data1 = pickle.load(f)


with open(reward_without_CBF_file_path, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data2 = pickle.load(f)

data1 = np.array(data1)
data2 = np.array(data2)
print(data1.shape)
print(data2.shape)

epoch_num_1 = np.arange(data1.shape[0])
epoch_num_2 = np.arange(data2.shape[0])

plt.figure()
plt.plot(epoch_num_1, data1, color='b', linewidth='2', label='with CBF')
plt.plot(epoch_num_2, data2, color='r', linewidth='2', label='without CBF')
plt.legend(loc='lower right',  fontsize="10")
plt.xlabel("time-step")
plt.ylabel("reward")
plt.title("the reward curve")
plt.show()
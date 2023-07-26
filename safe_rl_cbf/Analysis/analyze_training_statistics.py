import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
import lightning.pytorch as pl
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


training_statistics_results = torch.load("training_record_dict.pt")



# performance_grad =  torch.hstack(training_statistics_results["performance_grad"])
# safety_grad =  torch.hstack(training_statistics_results["safety_grad"])
descent_grad_mean =  torch.hstack(training_statistics_results["descent_grad_mean"])
descent_grad_max =  torch.hstack(training_statistics_results["descent_grad_max"])
descent_loss =  training_statistics_results["descent_loss"]
# violation_states_list = training_statistics_results["violation_states"]
# violation_value_list = training_statistics_results["violation_value"]

# y1 = performance_grad.cpu().numpy()
# y2 = safety_grad.cpu().numpy()
descent_loss = np.array(descent_loss)
descent_grad_mean = descent_grad_mean.cpu().numpy()
descent_grad_max = descent_grad_max.cpu().numpy()
x = np.arange(0, descent_grad_mean.shape[0], 1)

plt.figure()
# plt.plot(x, y1, label="performance_grad")
# plt.plot(x[500:], y2[500:], label="safety_grad")
plt.plot(x, descent_grad_mean, label="descent_grad_mean")
plt.legend()
plt.savefig("training_fig/descent_grad_mean.png")


plt.figure()
# plt.plot(x, y1, label="performance_grad")
# plt.plot(x[500:], y2[500:], label="safety_grad")
plt.plot(x, descent_grad_max, label="descent_grad_max")
plt.legend()
plt.savefig("training_fig/descent_grad_max.png")

plt.figure()
plt.plot(x, descent_loss, label="descent_loss")
plt.legend()
plt.savefig("training_fig/descent_loss.png")



# angles=  torch.hstack(training_statistics_results["angles"])
# angles2=  torch.hstack(training_statistics_results["angles2"])
# angles = angles.cpu().numpy()
# plt.figure()
# plt.plot(x, angles)
# plt.savefig("training_fig/angles.png")


# angles2 = angles2.cpu().numpy()
# plt.figure()
# plt.plot(x, angles2)
# plt.savefig("training_fig/angles2.png")

# violation_states = np.array([0,0]).reshape(-1, 2)
# violation_value = np.array([0])
# for i in range(len(violation_states_list)):
   
#     violation_states = np.vstack((violation_states, violation_states_list[i].cpu().numpy()))
#     violation_value = np.hstack((violation_value, violation_value_list[i].detach().cpu().numpy()))

# plt.figure()
# plt.scatter(violation_states[:, 0], violation_states[:, 1])
# plt.xlim(-4, 4)
# plt.ylim(-5, 5)
# plt.savefig("training_fig/violation_location.png")


# plt.figure()
# plt.hist(violation_value)
# plt.savefig("training_fig/violation_value.png")

plt.show()

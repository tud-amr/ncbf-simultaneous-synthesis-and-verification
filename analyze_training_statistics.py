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



performance_grad =  torch.hstack(training_statistics_results["performance_grad"])
safety_grad =  torch.hstack(training_statistics_results["safety_grad"])
descent_grad =  torch.hstack(training_statistics_results["descent_grad"])

y1 = performance_grad.cpu().numpy()
y2 = safety_grad.cpu().numpy()
y3 = descent_grad.cpu().numpy()
x = np.arange(0, y1.shape[0], 1)

plt.figure()
# plt.plot(x, y1, label="performance_grad")
plt.plot(x, y2, label="safety_grad")
plt.plot(x, y3, label="descent_grad")
plt.legend()
plt.savefig("training_fig/grad.png")
plt.show()



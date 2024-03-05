import numpy as np
from typing import Tuple, List, Optional
import json
import time
import os
import datetime
import copy
import itertools
from itertools import product
import sqlite3
from abc import ABC, abstractmethod
from termcolor import colored
import json
import scipy.io as sio
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import argparse


import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB
from qpth.qp import QPFunction

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from collections import defaultdict

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt


def print_error(message):
    print(colored(message, 'red'))

def print_warning(message):
    print(colored(message, 'yellow'))

def print_info(message):
    print(colored(message, 'blue'))
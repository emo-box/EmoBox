import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
from itertools import repeat
from copy import deepcopy
from collections import OrderedDict
from sys import stderr
from torch import Tensor
from torch.nn.utils import weight_norm

class SuperbBaseModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.pre_net = nn.Linear(input_size, hidden_size)
        self.post_net = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, padding_mask = None):
        x = self.relu(self.pre_net(x))
        
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).float()
            x = x.sum(dim=1) / padding_mask.float().sum(dim=1, keepdim=True)  # Compute average
        else:
            x = x.mean(dim=1)
        x = self.post_net(x)
        return x



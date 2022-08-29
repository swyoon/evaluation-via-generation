import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OutlierExposure(nn.Module):
    """implementation of <deep anomaly detection with outlier exposure> (Hendricks et al., 2019)"""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def predict(self, x):
        output = self.network(x)

        smax = F.softmax(output, dim=1)
        score = -torch.max(smax, axis=1).values
        #         score = to_np((output.mean(1) - torch.logsumexp(output, dim=1)))
        return score

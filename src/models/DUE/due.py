import gpytorch
import torch.nn as nn
import torch.nn.functional as F

from models.DUE.duelib import dkl
from models.DUE.duelib.sngp import Laplace
from models.DUE.duelib.wide_resnet import WideResNet


class DUE(nn.Module):
    def __init__(self, net, likelihood):
        super().__init__()
        self.net = net
        self.likelihood = likelihood

    def predict(self, x):
        with gpytorch.settings.num_likelihood_samples(32):
            y_pred = self.net(x).to_data_independent_dist()
            output = self.likelihood(y_pred).probs.mean(0)

        uncertainty = -(output * output.log()).sum(1)

        # logits = self.net(x)
        # output = F.softmax(logits, dim=1)

        # # Dempster-Shafer uncertainty for SNGP
        # # From: https://github.com/google/uncertainty-baselines/blob/main/baselines/cifar/ood_utils.py#L22
        # num_classes = logits.shape[1]
        # belief_mass = logits.exp().sum(1)
        # uncertainty = num_classes / (belief_mass + num_classes)
        return uncertainty


class SNGP(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def predict(self, x):
        self.net.eval()
        logits = self.net(x)
        output = F.softmax(logits, dim=1)

        # Dempster-Shafer uncertainty for SNGP
        # From: https://github.com/google/uncertainty-baselines/blob/main/baselines/cifar/ood_utils.py#L22
        num_classes = logits.shape[1]
        belief_mass = logits.exp().sum(1)
        uncertainty = num_classes / (belief_mass + num_classes)
        return uncertainty

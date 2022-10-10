import torch
import torch.nn as nn

from .models import modules_ibp as ibp


class Prood(nn.Module):
    def __init__(self, classifier, detector, num_classes):
        super().__init__()
        self.net = ibp.JointModel(classifier, detector, classes=num_classes)

    def classify(self, x):
        return self.net(x)

    def predict(self, x):
        """returns confidence score for out-of-distribution detection
        returns large value for outlier"""
        pred = torch.log_softmax(self.classify(x), dim=1)
        c, pr_cls = pred.max(1)
        return -c.exp()

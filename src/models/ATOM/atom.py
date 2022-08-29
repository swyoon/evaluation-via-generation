import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_ood_score(inputs, model):
    with torch.no_grad():
        outputs = model(inputs)

    # scores = -1.0 * (F.softmax(outputs, dim=1)[:,-1]).float().detach().cpu()# .numpy()
    # return large value for outlier
    scores = (F.softmax(outputs, dim=1)[:, -1]).float().detach()  # .cpu()# .numpy()
    return scores


def get_rowl_score(inputs, model, method_args, raw_score=False):
    num_classes = method_args["num_classes"]
    with torch.no_grad():
        outputs = model(inputs)

    # scores = -1.0 * F.softmax(outputs, dim=1)[:, num_classes].float().detach().cpu()# .numpy()
    # return large value for outlier
    scores = (
        F.softmax(outputs, dim=1)[:, num_classes].float().detach()
    )  # .cpu()# .numpy()
    return scores


class ATOM(nn.Module):
    def __init__(self, net, num_classes=10):
        super().__init__()
        self.net = net
        self.num_classes = num_classes

    def predict(self, x):
        # for supervised learning
        # outputs = F.softmax(model(x)[:, :num_classes], dim=1)
        # outputs = outputs.detach().cpu().numpy()
        # preds = np.argmax(outputs, axis=1)
        # confs = np.max(outputs, axis=1)
        scores = get_ood_score(x, self.net)
        return scores


class ROWL(nn.Module):
    def __init__(self, net, num_classes=10):
        super().__init__()
        self.net = net
        self.num_classes = num_classes

    def predict(self, x):
        # for supervised learning
        # outputs = F.softmax(model(x)[:, :num_classes], dim=1)
        # outputs = outputs.detach().cpu().numpy()
        # preds = np.argmax(outputs, axis=1)
        # confs = np.max(outputs, axis=1)
        scores = get_rowl_score(
            x, self.net, {"num_classes": self.num_classes}, raw_score=True
        )
        return scores

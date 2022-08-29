import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ODIN(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, x, y, optimizer):
        self.train()

        optimizer.zero_grad()

        outputs = self.net(x)
        L = self.criterion(outputs, y)
        acc = 100 * (
            torch.sum((torch.argmax(outputs, dim=1) == y).float()) / outputs.shape[0]
        )
        L.backward()
        optimizer.step()

        return {"loss": L.item(), "train_acc_": acc.item()}

    def validation_step(self, x, y):
        self.eval()

        outputs = self.net(x)
        L = self.criterion(outputs, y)
        acc = 100 * (
            torch.sum((torch.argmax(outputs, dim=1) == y).float()) / outputs.shape[0]
        )
        return {"loss": L.item(), "val_acc_": acc.item()}

    def softmax_score(self, outputs, T=1.0):
        temp1 = torch.max(torch.exp(outputs / T), dim=1)[0]
        temp2 = torch.sum(torch.exp(outputs / T), dim=1)
        score = temp1 / temp2
        return score

    def image_perturbation(self, inputs, gradient, noiseMagnitude):
        return inputs + noiseMagnitude * gradient

    def predict(self, x, temperature=1000.0, noiseMagnitude=0.0014):
        """return OOD score (the higher the more likely to be an outlier"""
        self.eval()
        x.requires_grad = True
        outputs = self.net(x)

        score_before_manipulation = self.softmax_score(outputs)  # naive max softmax
        score_temp1000 = self.softmax_score(outputs, T=temperature)  # only temperature

        L = torch.sum(torch.log(self.softmax_score(outputs)))
        L.backward()
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        outputs_of_perturbed_inputs = self.net(
            self.image_perturbation(x, gradient, noiseMagnitude)
        )
        score_odin = self.softmax_score(outputs_of_perturbed_inputs)

        return -score_odin

import argparse
import json
import pathlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from .cnn_duq import CNN_DUQ
from .resnet_duq import ResNet_DUQ


class DUQ(nn.Module):
    def __init__(
        self,
        net,
        num_classes=10,
        centroid_size=512,
        model_output_size=512,
        length_scale=0.1,
        gamma=0.999,
        l_gradient_penalty=0.5,
        learnable_length_scale=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.centroid_size = centroid_size
        self.model_output_size = model_output_size
        self.length_scale = length_scale
        self.gamma = gamma
        self.l_gradient_penalty = l_gradient_penalty
        if net.lower() == "resnet_duq":
            self.model = ResNet_DUQ(
                None, num_classes, centroid_size, model_output_size, length_scale, gamma
            )  # it does not uses input_sizes
        elif net.lower() == "cnn_duq":
            self.model = CNN_DUQ(
                None,
                num_classes,
                centroid_size,
                learnable_length_scale,
                length_scale,
                gamma,
            )

    def forward(self, x):
        return self.model(x)

    def bce_loss_fn(self, y_pred, y):
        bce = F.binary_cross_entropy(y_pred, y, reduction="sum").div(
            self.num_classes * y_pred.shape[0]
        )
        return bce

    def output_transform_bce(self, output):
        y_pred, y, x = output

        y = F.one_hot(y, self.num_classes).float()

        return y_pred, y

    def output_transform_acc(self, output):
        y_pred, y, x = output

        return y_pred, y

    def output_transform_gp(self, output):
        y_pred, y, x = output

        return x, y_pred

    def calc_gradients_input(self, x, y_pred):
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        return gradients

    def calc_gradient_penalty(self, x, y_pred):
        gradients = self.calc_gradients_input(x, y_pred)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty

    def train_step(self, x, y, optimizer):
        self.train()

        optimizer.zero_grad()

        if self.l_gradient_penalty > 0:
            x.requires_grad_(True)

        z, y_pred = self(x)
        y = F.one_hot(y, self.num_classes).float()

        loss = self.bce_loss_fn(y_pred, y)

        if self.l_gradient_penalty > 0:
            loss += self.l_gradient_penalty * self.calc_gradient_penalty(x, y_pred)

        loss.backward()
        optimizer.step()

        x.requires_grad_(False)

        with torch.no_grad():
            self.eval()
            self.model.update_embeddings(x, y)

        return {"loss": loss.item()}

    def validation_step(self, x, y):
        self.model.eval()

        x.requires_grad_(True)

        z, y_pred = self.model(x)

        return {"y_pred": y_pred.detach().cpu(), "y": y, "x": x}

    def predict(self, x):
        z, y_pred = self.model(x)
        kernel_distance, pred = y_pred.max(1)
        score = -kernel_distance
        return score

    def classify(self, x):
        z, y_pred = self.model(x)
        kernel_distance, pred = y_pred.max(1)
        return pred

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet152, wide_resnet50_2


class ResNetClassifier(nn.Module):
    """A ResNet-based image classifier with the last layer of ResNet replaced."""

    def __init__(self, pretrained=False, n_class=1, net="resnet18", **kwargs):
        super().__init__()
        if net == "resnet18":
            resnet = resnet18(pretrained=pretrained)
            n_hidden = 512
        elif net == "resnet50":
            resnet = resnet50(pretrained=pretrained)
            n_hidden = 2048
        elif net == "resnet152":
            resnet = resnet152(pretrained=pretrained)
            n_hidden = 2048
        elif net == "wide_resnet50_2":
            resnet = wide_resnet50_2(pretrained=pretrained)
            n_hidden = 2048
        else:
            raise NotImplementedError
        self.n_class = n_class
        feature_part = list(resnet.children())[:-1]
        feature_part.append(nn.Conv2d(n_hidden, n_class, 1, 1))
        self.net = nn.Sequential(*feature_part)

    def forward(self, x):
        return self.net(x).squeeze(3).squeeze(2)

    def train_step(self, x, y, optimizer, clip_grad=None):
        self.train()
        optimizer.zero_grad()
        pred = self(x)
        if self.n_class == 1:
            loss = F.binary_cross_entropy_with_logits(pred, y)
        else:
            loss = F.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def validation_step(self, x, y):
        self.eval()
        with torch.no_grad():
            pred = self(x)
            loss = F.binary_cross_entropy_with_logits(pred, y)
        return {"loss": loss.item(), "pred": pred.detach().cpu()}

    def predict(self, x, logit=True):
        """
        logit: 'predict' method will return logit if True. Otherwise, it returns probability.
        """
        # with torch.no_grad():
        pred = self(x)
        if not logit:
            pred = torch.sigmoid(pred)
        return pred

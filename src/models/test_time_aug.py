import torch
import torch.nn as nn
from torchvision.transforms.functional import hflip, pad, rotate


class TestTimeAug(nn.Module):
    """Test-time augmentation of horizontal flipping"""

    def __init__(self, model, hflip=True, rot90=True, rot180=True, rot270=True):
        super().__init__()
        self.model = model
        self.hflip = hflip
        self.rot90 = rot90
        self.rot180 = rot180
        self.rot270 = rot270

    def forward(self, x, **kwargs):
        l_out = []
        l_out.append(self.model(x))
        if self.hflip:
            l_out.append(self.model(hflip(x)))
        if self.rot90:
            l_out.append(self.model(rotate(x, 90)))
        if self.rot180:
            l_out.append(self.model(rotate(x, 180)))
        if self.rot270:
            l_out.append(self.model(rotate(x, 270)))
        return torch.stack(l_out).mean(dim=0)

    def predict(self, x, **kwargs):
        return self(x)

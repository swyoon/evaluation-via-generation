import torch
from torch.optim import Adam

from models.energybased import EnergyBasedModel
from models.modules import FCNet


def test_EBM():
    net = FCNet(3, 1, l_hidden=(100,))
    model = EnergyBasedModel(net)

    X = torch.rand(10, 3)
    opt = Adam(model.parameters(), lr=0.001)
    model.train_step(X, opt)

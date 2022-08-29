import torch
from torch.optim import Adam

from models.pixelcnn import PixelCNN, PixelCNN_OC
from models.pixelcnn_layers import discretized_mix_logistic_loss


def test_pixelcnn():
    model = PixelCNN()
    loss_op = lambda real, fake: discretized_mix_logistic_loss(real, fake)  # noqa
    opt = Adam(model.parameters())
    X = torch.rand(2, 3, 32, 32)

    opt.zero_grad()
    out = model(X)
    loss = loss_op(X, out)
    loss.backward()
    opt.step()


def test_pixelcnn_oc():
    model = PixelCNN_OC()
    opt = Adam(model.parameters())
    X = torch.rand(2, 3, 32, 32)

    pred = model.predict(X)
    assert len(pred) == len(X)
    assert len(pred.shape) == 1

    d_loss = model.train_step(X, opt)
    assert "loss" in d_loss

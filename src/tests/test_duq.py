import pytest
import torch
from torch.optim import SGD

from models.duq.duq import DUQ


@pytest.mark.parametrize("dataset", ["mnist", "cifar"])
def test_duq(dataset):
    if dataset == "mnist":
        model = DUQ(net="cnn_duq")
        x = torch.rand(2, 1, 28, 28, dtype=torch.float)
    elif dataset == "cifar":
        model = DUQ(net="resnet_duq")
        x = torch.rand(2, 3, 32, 32, dtype=torch.float)
    y = torch.ones(2, dtype=torch.long)
    opt = SGD(model.parameters(), lr=1e-4)
    d_train = model.train_step(x, y, opt)

    assert isinstance(d_train, dict)
    assert "loss" in d_train

    score = model.predict(x)
    assert len(score) == len(y)
    pred_class = model.classify(x)
    assert len(pred_class) == len(y)

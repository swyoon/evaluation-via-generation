import pytest
import torch
from torch.optim import Adam

from models.cls import ResNetClassifier


@pytest.mark.parametrize(
    "net", ["resnet18", "resnet50", "resnet152", "wide_resnet50_2"]
)
def test_resnet_classifier(net):
    model = ResNetClassifier(n_class=1, net=net, pretrained=False)
    x = torch.rand(5, 3, 32, 32)
    y = torch.randint(2, size=(5, 1), dtype=torch.float)
    f = model(x)
    model.train()
    opt = Adam(model.parameters(), lr=1e-4)
    model.train_step(x, y, opt)

    model.eval()
    model.validation_step(x, y)

    pred = model.predict(x, logit=True)
    assert len(pred) == len(x)

    pred = model.predict(x, logit=False)
    assert (pred <= 1).all() and (pred >= 0).all()

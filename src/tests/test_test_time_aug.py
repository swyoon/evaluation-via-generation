import torch

from models.cls import ResNetClassifier
from models.test_time_aug import TestTimeAug


def test_test_time_aug():
    net = ResNetClassifier(pretrained=False, n_class=1, net="resnet18")
    net = TestTimeAug(net)
    net.eval()

    N = 3
    x = torch.rand(N, 3, 32, 32)
    out = net(x)
    assert out.shape == (N, 1)

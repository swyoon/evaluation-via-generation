import pytest
import torch
from torch.optim import Adam

from models.OE.allconv import AllConvNet
from models.OE.outlier_exposure import OutlierExposure
from models.OE.wrn import WideResNet


@pytest.mark.parametrize("dataset", ["cifar10"])
def test_likelihood_regret(dataset):
    if dataset == "mnist":
        x_dim = 1
        x = torch.randn(5, 1, 28, 28)
    elif dataset == "cifar10":
        x_dim = 3
        x = torch.randn(5, 3, 32, 32)
    else:
        raise ValueError(x)

    net1 = AllConvNet(10)
    state_allconv = torch.load(
        "./pretrained/cifar_ood_msp/allconv/cifar10_allconv_baseline_epoch_99.pt",
        map_location="cpu",
    )
    net1.load_state_dict(state_allconv)
    net1.eval()
    model1 = OutlierExposure(net1)

    net2 = WideResNet(40, 10, 2, 0.3)
    state_wrn = torch.load(
        "./pretrained/cifar_ood_msp/wrn/cifar10_wrn_baseline_epoch_99.pt",
        map_location="cpu",
    )
    net2.load_state_dict(state_wrn)
    net2.eval()
    model2 = OutlierExposure(net2)

    net3 = AllConvNet(10)
    state_allconv = torch.load(
        "./pretrained/cifar_ood_oe_tune/allconv/cifar10_allconv_oe_tune_epoch_9.pt",
        map_location="cpu",
    )
    net3.load_state_dict(state_allconv)
    net3.eval()
    model3 = OutlierExposure(net3)

    net4 = WideResNet(40, 10, 2, 0.3)
    state_wrn = torch.load(
        "./pretrained/cifar_ood_oe_tune/wrn/cifar10_wrn_oe_tune_epoch_9.pt",
        map_location="cpu",
    )
    net4.load_state_dict(state_wrn)
    net4.eval()
    model4 = OutlierExposure(net4)

    net5 = AllConvNet(10)
    state_allconv = torch.load(
        "./pretrained/cifar_ood_oe_scratch/allconv/cifar10_allconv_oe_scratch_epoch_99.pt",
        map_location="cpu",
    )
    net5.load_state_dict(state_allconv)
    net5.eval()
    model5 = OutlierExposure(net5)

    net6 = WideResNet(40, 10, 2, 0.3)
    state_wrn = torch.load(
        "./pretrained/cifar_ood_oe_scratch/wrn/cifar10_wrn_oe_scratch_epoch_99.pt",
        map_location="cpu",
    )
    net6.load_state_dict(state_wrn)
    net6.eval()
    model6 = OutlierExposure(net6)

    pred1 = model1.predict(x)
    assert len(pred1) == len(x)

    pred2 = model2.predict(x)
    assert len(pred2) == len(x)

    pred3 = model3.predict(x)
    assert len(pred3) == len(x)

    pred4 = model4.predict(x)
    assert len(pred4) == len(x)

    pred5 = model5.predict(x)
    assert len(pred5) == len(x)

    pred6 = model6.predict(x)
    assert len(pred6) == len(x)

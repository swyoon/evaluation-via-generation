import torch

from models.prood import Prood
from models.prood.models import provable_classifiers, resnet


def test_prood():
    dset_in_name = "CIFAR10"
    arch_size = "S"
    use_last_bias = True
    num_classes = 1
    last_layer_neg = True
    bias_shift = 3
    detector = provable_classifiers.CNN_IBP(
        dset_in_name=dset_in_name,
        size=arch_size,
        last_bias=use_last_bias,
        num_classes=num_classes,
        last_layer_neg=last_layer_neg,
    )

    if last_layer_neg and use_last_bias:
        with torch.no_grad():
            detector.layers[-1].bias.data += bias_shift

    classifier = resnet.get_ResNet(dset=dset_in_name)
    model = Prood(classifier, detector, num_classes=10)

    x = torch.randn(1, 3, 32, 32)
    pred = model.predict(x)
    assert pred.shape == (1,)

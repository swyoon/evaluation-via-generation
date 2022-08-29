import pytest
import torch

from attacks.grad import GradientBasedAttack
from models.ae import AE
from models.modules import ConvNet2, DeConvNet2


@pytest.mark.parametrize("initial", ["random", "x0"])
def test_grad_attack(initial):
    N = 4
    adv = GradientBasedAttack(n_step=10, stepsize=0.1, bound=(0.0, 1.0), initial="rand")

    # model to be attacked
    encoder = ConvNet2(in_chan=1, out_chan=16)
    decoder = DeConvNet2(in_chan=16, out_chan=1)
    ae = AE(encoder, decoder)

    # perform attack
    if initial == "random":
        d_result = adv.attack(
            model_fn=ae.predict, n_sample=N, shape=(1, 28, 28), device="cpu"
        )
    elif initial == "x0":
        x0 = torch.rand(N, 1, 28, 28)
        d_result = adv.attack(model_fn=ae.predict, x0=x0)

    x = d_result["x"]
    assert x.shape == (N, 1, 28, 28)

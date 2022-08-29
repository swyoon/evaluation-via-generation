import pytest
import torch

from models.mcmc import LangevinSampler
from models.modules import FCNet


@pytest.mark.parametrize("return_min", [True, False])
@pytest.mark.parametrize("push_min", [True, False])
def test_max_langevin(return_min, push_min):
    energy = FCNet(2, 1, out_activation="linear")
    sampler_x = LangevinSampler(
        n_step=2,
        stepsize=0.1,
        noise_std=0.1,
        noise_anneal=1.0,
        bound=(-5, 5),
        buffer_size=10000,
        replay_ratio=0.0,
        reject_boundary=False,
        mh=True,
        initial_dist="uniform",
        sample_shape=(2,),
        return_min=return_min,
        push_min=push_min,
    )
    d_sample = sampler_x.sample(energy, n_sample=5, device="cpu", replay=False)

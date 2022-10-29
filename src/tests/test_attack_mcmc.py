import torch

from attacks.mcmc import (
    CoordinateDescentSampler,
    LangevinSampler,
    MHSampler,
    RandomSampler,
    sample_discrete_gibbs,
    sample_mh,
)


def test_sample_gibbs():
    B = 3
    K = 10
    x0 = torch.randint(K, size=(B, 4, 4))
    n_step = 2
    T = 1.0

    def energy_fn(z):
        return (z**2).flatten(1).sum(axis=-1).to(torch.float)

    d_result = sample_discrete_gibbs(x0, energy_fn, n_step, K, T=T, order="sequential")
    seq_len = n_step * 4 * 4
    assert d_result["x"].shape == (B, 4, 4)
    assert d_result["l_x"].shape == (seq_len + 1, B, 4, 4)
    assert d_result["l_E"].shape == (seq_len + 1, B)
    assert d_result["l_accept"].shape == (seq_len, B)


def test_mh():
    B = 5
    D = 1
    n_step = 10
    stepsize = 0.1
    energy_fn = lambda x: 0.5 * (x**2).flatten(1).sum(axis=1)
    x0 = torch.rand(B, D)
    d_result = sample_mh(x0, energy_fn, n_step, stepsize, T=1, bound=None)

    assert d_result["x"].shape == (B, D)
    assert d_result["l_x"].shape == (n_step + 1, B, D)
    assert d_result["l_E"].shape == (n_step + 1, B)
    assert d_result["l_accept"].shape == (n_step, B)


def test_mh_block():
    B = 5
    D = 1
    n_step = 10
    stepsize = 0.1
    energy_fn = lambda x: 0.5 * (x**2).flatten(1).sum(axis=1)
    x0 = torch.rand(B, D)
    d_result = sample_mh(x0, energy_fn, n_step, stepsize, T=1, bound=None, block=3)

    assert d_result["x"].shape == (B, D)
    assert d_result["l_x"].shape == (n_step + 1, B, D)
    assert d_result["l_E"].shape == (n_step + 1, B)
    assert d_result["l_accept"].shape == (n_step, B)


def test_random_sampler():
    energy_fn = lambda x: 0.5 * (x**2).flatten(1).sum(axis=1)
    sampler = RandomSampler(
        sample_shape=(3, 32, 32), n_step=10, bound=(0, 1), initial_dist="uniform"
    )
    d_sample = sampler.sample(energy_fn, n_sample=10, device="cpu")
    assert "x" in d_sample
    assert "l_x" in d_sample
    assert "l_E" in d_sample


def test_mh_sampler():
    energy_fn = lambda x: 0.5 * (x**2).flatten(1).sum(axis=1)
    sampler = MHSampler(
        sample_shape=(3, 32, 32),
        stepsize=0.1,
        n_step=10,
        bound=(0, 1),
        initial_dist="uniform",
    )
    d_sample = sampler.sample(energy_fn, n_sample=10, device="cpu")
    assert "x" in d_sample
    assert "l_x" in d_sample
    assert "l_E" in d_sample


def test_langevin_sampler():
    energy_fn = lambda x: 0.5 * (x**2).flatten(1).sum(axis=1)
    sampler = LangevinSampler(
        sample_shape=(3, 32, 32),
        stepsize=0.1,
        n_step=10,
        bound=(0, 1),
        initial_dist="uniform",
    )
    d_sample = sampler.sample(energy_fn, n_sample=10, device="cpu")
    assert "x" in d_sample
    assert "l_x" in d_sample
    assert "l_E" in d_sample


def test_coordinate_descent():
    energy_fn = lambda x: 0.5 * (x**2).flatten(1).sum(axis=1)
    sampler = CoordinateDescentSampler(
        sample_shape=None, n_step=10, bound=(0, 1), Linf=0.01, h=0.1, stepsize=0.01
    )
    d_sample = sampler.sample(energy_fn, x0=torch.rand(2, 3, 32, 32))
    assert "x" in d_sample
    assert "l_x" in d_sample
    assert "l_E" in d_sample

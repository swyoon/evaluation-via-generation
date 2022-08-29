import torch
import torch.nn as nn
from torch.optim import Adam

from attacks.policy import GaussianPolicy
from attacks.REINFORCE import REINFORCEAttack, SoftREINFORCEAttack


def test_reinforce():
    # reward
    def reward(x):
        return ((x > 0) & (x < 1)).sum()

    policy = GaussianPolicy(mu=1.5, sigma=1.0)

    # reinforce class
    # attacker = REINFORCEAttack(policy)
    attacker = SoftREINFORCEAttack(policy)
    optim = Adam(attacker.parameters(), lr=1e-3)

    d_train = attacker.train_step(reward, optim)
    assert "loss" in d_train

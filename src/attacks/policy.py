import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianPolicy(nn.Module):
    def __init__(self, mu, sigma):
        super().__init__()
        self.register_parameter("mu", nn.Parameter(torch.tensor(mu, dtype=torch.float)))
        self.register_parameter(
            "sigma", nn.Parameter(torch.tensor(sigma, dtype=torch.float))
        )

    def sample(self, n_sample=1, device="cpu"):
        return Normal(self.mu, self.sigma).sample(sample_shape=(n_sample,)).to(device)

    def forward(self, x):
        """computes log probability"""
        return Normal(self.mu, self.sigma).log_prob(x)

    def entropy(self, x):
        return Normal(self.mu, self.sigma).entropy()

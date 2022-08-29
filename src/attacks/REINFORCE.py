import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from models.LSTM import LSTM


class REINFORCEAttack(nn.Module):
    def __init__(self, policy_network):
        super().__init__()
        self.policy = policy_network

    def train_step(self, reward_fn, optimizer, n_sample=1):
        optimizer.zero_grad()
        sample = self.attack(n_sample=n_sample)
        reward = reward_fn(sample)
        log_p = self.policy(sample)

        loss = -(log_p * reward).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def attack(self, n_sample=1, device="cpu"):
        return self.policy.sample(n_sample=n_sample, device="cpu")


class SoftREINFORCEAttack(nn.Module):
    def __init__(self, policy_network):
        super().__init__()
        self.policy = policy_network

    def train_step(self, reward_fn, optimizer, n_sample=1, temperature=0.5):
        optimizer.zero_grad()
        sample = self.attack(n_sample=n_sample)
        reward = reward_fn(sample)
        log_p = self.policy(sample)

        # entropy = self.policy.entropy(sample)
        entropy = -(log_p * (1 + log_p).detach()).mean()
        loss = -(log_p * reward).mean() - temperature * entropy
        loss.backward()
        optimizer.step()
        # return {'loss': loss.item()}
        return {"loss": loss}

    def attack(self, n_sample=1, device="cpu"):
        return self.policy.sample(n_sample=n_sample, device="cpu")


class PPOAttack(nn.Module):
    def __init__(self, policy_network, eps_clip):
        super().__init__()
        self.policy = policy_network

        self.eps_clip = eps_clip
        self.memory = ReplayBuffer()

    def train_step(self, reward_fn, optimizer, n_sample=1, temperature=0.5, epochs=5):
        optimizer.zero_grad()

        sample = self.attack(n_sample=n_sample)
        reward = reward_fn(sample)
        old_log_p = self.policy(sample).detach()

        for _ in range(epochs):
            log_p = self.policy(sample)
            ratios = torch.exp(log_p - old_log_p)
            # entropy = self.policy.entropy(sample)
            entropy = -(log_p * (1 + log_p).detach()).mean()

            surr1 = ratios * reward
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * reward
            loss = -torch.min(surr1, surr2) - temperature * entropy

            loss.mean().backward()
            optimizer.step()

        # return {'loss': loss.item()}
        return {"loss": loss}

    def attack(self, n_sample=1, device="cpu"):
        return self.policy.sample(n_sample=n_sample, device="cpu")


class ReplayBuffer:
    def __init__(self, buffer_size=None):
        self.buffer = list() if buffer_size is None else deque(maxlen=buffer_size)
        self.first_store = True

    def store(self, sample, reward, log_p):
        if self.first_store:
            self.first_store = False

        for s, r, lp in zip(sample, reward, log_p):
            self.buffer.append((s, r, lp))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        sample = np.stack([b[0] for b in batch], axis=0)
        reward = np.stack([b[1] for b in batch], axis=0)
        log_p = np.stack([b[2] for b in batch], axis=0)

        return (sample, reward, log_p)

    def rollout(self):
        sample = np.stack([b[0] for b in self.buffer], axis=0)
        reward = np.stack([b[1] for b in self.buffer], axis=0)
        log_p = np.stack([b[2] for b in self.buffer], axis=0)

        self.clear()

        return (sample, reward, log_p)

    def clear(self):
        self.buffer.clear()

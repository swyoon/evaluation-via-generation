import torch
import torch.autograd as autograd
import torch.nn as nn

from models.mcmc import clip_vector_norm


class GradientBasedAttack(nn.Module):
    def __init__(self, n_step, stepsize, bound=None, initial="rand"):
        """
        n_step: the number of gradient steps to proceed
        stepsize: the step size of gradient step
        bound: bound of data. e.g. (0, 1)
        """
        self.n_step = n_step
        self.stepsize = stepsize
        self.bound = bound
        self.initial = initial
        # self.noise = noise

    def train_step(self, **kwargs):
        """no training is required"""
        pass

    def attack(
        self,
        model_fn,
        n_sample=None,
        shape=None,
        device=None,
        x0=None,
        clip_grad=None,
        n_step=None,
        stepsize=None,
    ):
        """
        performs gradient descent attack
        either x0 or (n_sample, shape, device) should be given.
        if x0 is given, (n_sample, shape, device) will be ignored.

        model_fn: OOD detection function. The higher model_fn, the more likely that an input is OOD.
        """
        if x0 is None:
            x0 = self.initial_sample(n_sample, shape, device)
        if n_step is None:
            n_step = self.n_step
        if stepsize is None:
            stepsize = self.stepsize

        x = x0
        x.requires_grad = True
        l_x = []
        l_f = []  # detector function values
        for i_step in range(n_step):
            out = model_fn(x)
            grad = autograd.grad(out.sum(), x, only_inputs=True)[0]

            if clip_grad is not None:
                grad = clip_vector_norm(grad, max_norm=clip_grad)
            l_f.append(out.detach().cpu())
            l_x.append(x.detach().cpu())

            x = x - stepsize * grad
            if self.bound is not None:
                x.clamp_(*self.bound)
        l_x.append(x.detach().cpu())
        return {
            "x": x.detach().cpu(),
            "x0": x0.detach().cpu(),
            "l_x": l_x,
            "l_f": torch.stack(l_f),
        }

    def initial_sample(self, n_sample, shape, device):
        """generate initial point from a random distribution"""
        if self.initial == "rand":
            x0 = torch.rand((n_sample,) + shape, device=device, dtype=torch.float)
        elif self.initial == "randn":
            x0 = torch.randn((n_sample,) + shape, device=device, dtype=torch.float)
            if self.bound is not None:
                x0.clamp_(*self.bound)
        else:
            raise ValueError(f"invalid initial distribution {self.initia}")
        return x0


class LangevinAttack(nn.Module):
    """TODO: to code Langevin dynamics-based gradient attack"""

    pass

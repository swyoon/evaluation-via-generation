from itertools import product

import numpy as np
import torch
import torch.autograd as autograd
from tqdm import tqdm


def sample_discrete_gibbs(x0, energy_fn, n_step, K, T, order="sequential"):
    """gibbs sampler for discrete variable
    x0: initial tensor. [B, H, W]
    energy_fn: function which returns the energy
    K: int. codebook size
    T: float. temperature"""
    B, H, W = x0.shape
    device = x0.device
    x = x0
    E = energy_fn(x)
    l_x = [x.detach().cpu()]
    l_accept = []
    l_E = [E.detach().cpu()]
    for i_iter in range(n_step):
        for i, j in product(range(H), range(W)):
            x_new = x.clone()
            x_new[:, i, j] = torch.randint(K, size=(B,))
            E_new = energy_fn(x_new)
            p_accept = torch.exp(-(E_new - E) / T)
            accept = p_accept >= torch.rand(len(p_accept), device=device)
            x_new[~accept, i, j] = x[~accept, i, j]
            E_new[~accept] = E[~accept]
            x = x_new
            E = E_new
            l_x.append(x.detach().cpu())
            l_E.append(E.detach().cpu())
            l_accept.append(accept.detach().cpu())
    l_x = torch.stack(l_x)
    l_E = torch.stack(l_E)
    l_accept = torch.stack(l_accept)
    return {"x": x, "l_x": l_x, "l_E": l_E, "l_accept": l_accept}


def sample_mh(
    x0,
    energy_fn,
    n_step,
    stepsize,
    T,
    bound=None,
    rng=None,
    block=None,
    writer=None,
    log_interval=100,
    mh=True,
):
    """Metropolis-Hastings MCMC algorithm"""
    B, D = x0.shape[:2]
    device = x0.device
    if rng is None:
        rng = np.random.default_rng()  # to avoid CSI random seed fixing effect
    x = x0
    E = energy_fn(x)
    l_x = [x.detach().cpu()]
    l_accept = []
    l_E = [E.detach().cpu()]
    l_p_accept = []
    for i_iter in tqdm(range(n_step)):
        # proposal
        if block is None:
            randn = torch.tensor(
                rng.standard_normal(size=x.shape), device=device, dtype=torch.float
            )
            x_new = x + randn * stepsize
        else:
            x_new = x.clone()
            if stepsize is None:  # uniform sampling
                rand = torch.tensor(
                    rng.uniform(low=0.0, high=1.0, size=x.shape),
                    device=device,
                    dtype=torch.float,
                )[:, :block]
                dim = torch.randperm(D)[:block]
                x_new[:, dim] = rand
            else:
                randn = torch.tensor(
                    rng.standard_normal(size=x.shape), device=device, dtype=torch.float
                )[:, :block]
                dim = torch.randperm(D)[:block]
                x_new[:, dim] = x[:, dim] + randn * stepsize

        if bound is not None:
            if bound == "spherical":
                x_new = x_new / x_new.norm(2, dim=1, keepdim=True)
            elif hasattr(bound, "__iter__"):  # not the ideal way to check if iterable
                x_new = x_new.clamp(bound[0], bound[1])
            else:
                x_new = x_new.clamp(-bound, bound)

        E_new = energy_fn(x_new)
        # M-H accept
        p_accept = torch.exp(-(E_new - E) / T)
        if mh:
            rand = torch.tensor(
                rng.uniform(size=len(p_accept)), device=device, dtype=torch.float
            )
        else:
            rand = torch.zeros(len(p_accept), device=device, dtype=torch.float)
        # accept = p_accept >= torch.rand(len(p_accept), device=device)
        accept = p_accept >= rand
        x_new[~accept] = x[~accept]
        E_new[~accept] = E[~accept]
        x = x_new
        E = E_new
        l_x.append(x.detach().cpu())
        l_E.append(E.detach().cpu())
        l_accept.append(accept.detach().cpu())
        l_p_accept.append(p_accept.detach().cpu())

        if writer is not None and i_iter % log_interval == 0:
            for jj in range(i_iter - log_interval, i_iter + 1):
                if jj < 0:
                    continue
                writer.add_scalar("mean_energy", l_E[jj].mean(), jj)

    l_x = torch.stack(l_x)
    l_E = torch.stack(l_E)
    l_accept = torch.stack(l_accept)
    l_p_accept = torch.stack(l_p_accept)
    return {
        "x": x,
        "l_x": l_x,
        "l_E": l_E,
        "l_accept": l_accept,
        "l_p_accept": l_p_accept,
    }


class MHSampler:
    def __init__(
        self,
        sample_shape=None,
        n_step=None,
        stepsize=None,
        bound=None,
        rng=None,
        T=1.0,
        block=None,
        initial_dist="uniform",
        writer=None,
        log_interval=100,
        mh=True,
    ):
        self.sample_shape = tuple(sample_shape)
        self.n_step = n_step
        self.stepsize = stepsize
        self.bound = bound  # if isinstance(bound, str) else tuple(bound)
        self.T = T
        self.block = block
        self.writer = writer
        self.log_interval = log_interval
        self.initial_dist = initial_dist
        self.mh = mh
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def initial_sample(self, n_sample, device):
        shape = (n_sample,) + self.sample_shape
        if self.initial_dist == "gaussian":
            x0_new = torch.randn(shape, dtype=torch.float)
        elif self.initial_dist == "uniform":
            bound = tuple(self.bound)
            x0_new = torch.rand(shape, dtype=torch.float)
            x0_new = x0_new * (bound[1] - bound[0]) + bound[0]
        elif self.initial_dist == "uniform_sphere":
            x0_new = torch.randn(shape, dtype=torch.float)
            x0_new = x0_new / (x0_new).norm(dim=1, keepdim=True)
        else:
            raise ValueError(f"Invalid initial_dist: {self.initial_dist}")

        return x0_new.to(device)

    def sample(self, energy, x0=None, n_sample=None, device=None):
        if x0 is None:
            x0 = self.initial_sample(n_sample=n_sample, device=device)
        d_sample = sample_mh(
            x0,
            energy,
            n_step=self.n_step,
            stepsize=self.stepsize,
            T=self.T,
            block=self.block,
            bound=self.bound,
            rng=self.rng,
            writer=self.writer,
            log_interval=self.log_interval,
            mh=self.mh,
        )
        return d_sample


class RandomSampler:
    def __init__(
        self,
        sample_shape=None,
        n_step=None,
        bound=None,
        rng=None,
        initial_dist="uniform",
        writer=None,
        log_interval=100,
    ):
        self.sample_shape = tuple(sample_shape)
        self.n_step = n_step
        self.bound = tuple(bound)
        self.writer = writer
        self.log_interval = log_interval
        self.initial_dist = initial_dist
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def initial_sample(self, n_sample, device):
        shape = (n_sample,) + self.sample_shape
        if self.initial_dist == "gaussian":
            x0_new = torch.randn(shape, dtype=torch.float)
        elif self.initial_dist == "uniform":
            x0_new = torch.rand(shape, dtype=torch.float)
            x0_new = x0_new * (self.bound[1] - self.bound[0]) + self.bound[0]
        elif self.initial_dist == "uniform_sphere":
            x0_new = torch.randn(shape, dtype=torch.float)
            x0_new = x0_new / (x0_new).norm(dim=1, keepdim=True)
        else:
            raise ValueError(f"Invalid initial_dist: {self.initial_dist}")

        return x0_new.to(device)

    def sample(self, energy, x0=None, n_sample=None, device=None):
        l_x = []
        l_E = []
        for i in range(self.n_step):
            x = self.initial_sample(n_sample=n_sample, device=device)
            l_E.append(energy(x).detach().cpu())
            l_x.append(x.detach().cpu())
        l_x = torch.stack(l_x)
        l_E = torch.stack(l_E)
        return {"l_x": l_x, "l_E": l_E, "x": l_x[-1]}


# def coordinate_descent(x0, energy, h, stepsize, n_step, bound, deterministic=True,):
#     D = x0.shape[1]
#     x = x0
#     E = energy(x)
#     l_x = [x.detach().cpu()]; l_E = [E.detach().cpu()]
#     for i in tqdm(range(n_step)):
#         dd = i % D
#         e = torch.zeros_like(x)
#         e[:, dd] = 1 * h
#         Enew = energy((x + e).clamp(bound[0], bound[1]))
#         dE = (Enew - E) / (h)
#         x -= e * dE[:,None] * stepsize
#         x = x.clamp(0, 1)
#         E = energy(x.detach())
#         l_E.append(E.detach().cpu())
#         l_x.append(x.detach().cpu())
#     l_x = torch.stack(l_x)
#     l_E = torch.stack(l_E)
#     d_result = {'l_x': l_x, 'l_E':l_E, 'x': l_x[-1]}
#     return d_result


def coordinate_descent(
    x0,
    energy,
    h,
    stepsize,
    n_step,
    bound,
    momentum=None,
    writer=None,
    log_interval=100,
    Linf=0.01,
    half_every=None,
    save_only_min=False,
    rng=None,
):
    """stochastic coordinate descent with momentum
    recommended parameters for image Linf attack
    Linf = 0.01
    h = 0.005
    stepsize = 0.01
    momentum = 0.999"""
    if rng is None:
        rng = np.random.default_rng()  # to avoid CSI random seed fixing effect
    is_image = True if len(x0.shape) == 4 else False
    D = np.prod(x0.shape[1:])
    x = x0.clone()
    min_x = x0.clone()
    E = energy(x)
    min_E = E.detach()
    v = 0
    l_x = [x.detach().cpu()]
    l_E = [E.detach().cpu()]
    for i_iter in tqdm(range(n_step)):
        #         dd = i % D
        #         dd = torch.randperm(D)[0]
        #         e = torch.zeros_like(x).view(len(x), -1)
        #         e[:, dd] = 1 * h
        randn = torch.tensor(
            rng.standard_normal(size=x.shape), device=x.device, dtype=torch.float
        )
        e = randn.view(len(x), -1) * h

        e = e.view(*x.shape)
        Enew = energy((x + e).clamp(bound[0], bound[1]))
        dE = (Enew - E) / (h)
        if is_image:
            grad = e * dE[:, None, None, None]
        else:
            grad = e * dE[:, None]
        v = stepsize * grad + momentum * v
        x = x - v
        if Linf is not None:
            x = x.clamp(x0 - Linf, x0 + Linf)
        x = x.clamp(bound[0], bound[1])
        E = energy(x.detach())

        l_E.append(E.detach().cpu())
        if save_only_min:
            min_idx = min_E > E
            min_x[min_idx] = x[min_idx]
            min_E[min_idx] = E[min_idx]
        else:
            l_x.append(x.detach().cpu())

        if i_iter > 0 and half_every is not None and i_iter % half_every == 0:
            stepsize /= 2.0
            h /= 2.0

        if writer is not None and i_iter % log_interval == 0:
            for jj in range(i_iter - log_interval, i_iter + 1):
                if jj < 0:
                    continue
                writer.add_scalar("mean_energy", l_E[jj].mean(), jj)

    l_E = torch.stack(l_E)
    d_result = {"l_x": l_x, "l_E": l_E, "x": l_x[-1]}
    if save_only_min:
        d_result["min_x"] = min_x
        d_result["min_E"] = min_E
        d_result["min_img"] = min_x
    else:
        l_x = torch.stack(l_x)
        d_result["l_x"] = l_x

    return d_result


class CoordinateDescentSampler:
    def __init__(
        self,
        sample_shape=None,
        n_step=None,
        bound=None,
        rng=None,
        initial_dist="uniform",
        writer=None,
        log_interval=100,
        Linf=None,
        momentum=0,
        h=None,
        stepsize=None,
        half_every=None,
        save_only_min=False,
    ):
        self.sample_shape = tuple(sample_shape) if sample_shape is not None else None
        self.n_step = n_step
        self.bound = tuple(bound)
        self.writer = writer
        self.log_interval = log_interval
        self.initial_dist = initial_dist
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        # coordinate descent parameters
        self.Linf = Linf
        self.momentum = momentum
        self.h = h
        self.stepsize = stepsize
        self.half_every = half_every
        self.save_only_min = save_only_min

    def initial_sample(self, n_sample, device):
        shape = (n_sample,) + self.sample_shape
        if self.initial_dist == "gaussian":
            x0_new = torch.randn(shape, dtype=torch.float)
        elif self.initial_dist == "uniform":
            x0_new = torch.rand(shape, dtype=torch.float)
            x0_new = x0_new * (self.bound[1] - self.bound[0]) + self.bound[0]
        elif self.initial_dist == "uniform_sphere":
            x0_new = torch.randn(shape, dtype=torch.float)
            x0_new = x0_new / (x0_new).norm(dim=1, keepdim=True)
        else:
            raise ValueError(f"Invalid initial_dist: {self.initial_dist}")

        return x0_new.to(device)

    def sample(self, energy, x0=None, n_sample=None, device=None):
        if x0 is None:
            x0 = self.initial_sample(n_sample=n_sample, device=device)
        d_sample = coordinate_descent(
            x0,
            energy,
            h=self.h,
            stepsize=self.stepsize,
            n_step=self.n_step,
            bound=self.bound,
            momentum=self.momentum,
            writer=self.writer,
            log_interval=self.log_interval,
            half_every=self.half_every,
            Linf=self.Linf,
            save_only_min=self.save_only_min,
        )
        if not self.save_only_min:
            # compute minimum energy
            argmin = d_sample["l_E"].argmin(dim=0)
            d_sample["min_x"] = d_sample["l_x"][argmin, range(len(argmin)), :]
            d_sample["min_E"] = d_sample["l_E"][argmin, range(len(argmin))]
            d_sample["min_img"] = d_sample["min_x"]
        return d_sample


def sample_langevin_v2(
    x,
    model,
    stepsize,
    n_step,
    noise_scale=None,
    bound=None,
    clip_grad=None,
    reject_boundary=False,
    noise_anneal=None,
    noise_anneal_v2=None,
    mh=False,
    temperature=None,
):
    """Langevin Monte Carlo
    x: torch.Tensor, initial points
    model: An energy-based model. returns energy
    stepsize: float
    n_step: integer
    noise_scale: Optional. float. If None, set to np.sqrt(stepsize * 2)
    bound : tuple (start, end) or None boundary of square domain
    reject_boundary: Reject out-of-domain samples if True. otherwise clip.
    """
    assert not (
        (stepsize is None) and (noise_scale is None)
    ), "stepsize and noise_scale cannot be None at the same time"
    if noise_scale is None:
        noise_scale = np.sqrt(stepsize * 2)
    if stepsize is None:
        stepsize = (noise_scale**2) / 2
    noise_scale_ = noise_scale
    stepsize_ = stepsize
    if temperature is None:
        temperature = 1.0

    # initial data
    x.requires_grad = True
    E_x = model(x)
    grad_E_x = autograd.grad(E_x.sum(), x, only_inputs=True)[0]
    if clip_grad is not None:
        # grad_E_x = clip_vector_norm(grad_E_x, max_norm=clip_grad)
        grad_E_x.clamp_(-clip_grad, clip_grad)
    E_y = E_x
    grad_E_y = grad_E_x

    l_sample = [x.detach().to("cpu")]
    l_dynamics = []
    l_drift = []
    l_diffusion = []
    l_accept = []
    l_E = [E_x.detach().cpu()]
    for i_step in tqdm(range(n_step)):
        noise = torch.randn_like(x) * noise_scale_
        dynamics = -stepsize_ * grad_E_x / temperature + noise
        y = x + dynamics
        reject = torch.zeros(len(y), dtype=torch.bool)

        if bound == "spherical":
            y = y / y.norm(dim=1, p=2, keepdim=True)
        elif bound is not None:
            if reject_boundary:
                accept = ((y >= bound[0]) & (y <= bound[1])).view(len(x), -1).all(dim=1)
                reject = ~accept
                y[reject] = x[reject]
            else:
                y = torch.clamp(y, bound[0], bound[1])

        # y_accept = y[~reject]
        # E_y[~reject] = model(y_accept)
        # grad_E_y[~reject] = autograd.grad(E_y.sum(), y_accept, only_inputs=True)[0]
        E_y = model(y)
        grad_E_y = autograd.grad(E_y.sum(), y, only_inputs=True)[0]

        if clip_grad is not None:
            # grad_E_y = clip_vector_norm(grad_E_y, max_norm=clip_grad)
            grad_E_y.clamp_(-clip_grad, clip_grad)

        if mh:
            y_to_x = ((grad_E_x + grad_E_y) * stepsize_ - noise).view(len(x), -1).norm(
                p=2, dim=1, keepdim=True
            ) ** 2
            x_to_y = (noise).view(len(x), -1).norm(dim=1, keepdim=True, p=2) ** 2
            transition = -(y_to_x - x_to_y) / 4 / stepsize_  # B x 1
            prob = -E_y + E_x
            accept_prob = torch.exp((transition + prob) / temperature)[:, 0]  # B
            reject = torch.rand_like(accept_prob) > accept_prob  # | reject
            y[reject] = x[reject]
            E_y[reject] = E_x[reject]
            grad_E_y[reject] = grad_E_x[reject]
            x = y
            E_x = E_y
            grad_E_x = grad_E_y
            l_accept.append(~reject)

        x = y
        E_x = E_y
        grad_E_x = grad_E_y

        if noise_anneal is not None:
            noise_scale_ = noise_scale / (1 + i_step)
        if noise_anneal_v2 is not None:
            noise_scale_ = noise_scale / (1 + i_step)
            stepsize_ = stepsize / ((1 + i_step) ** 2)

        l_dynamics.append(dynamics.detach().cpu())
        l_drift.append((-stepsize * grad_E_x).detach().cpu())
        l_diffusion.append(noise.detach().cpu())
        l_sample.append(x.detach().cpu())
        l_E.append(E_x.detach().cpu())

    return {
        "x": x.detach(),
        "l_x": torch.stack(l_sample),
        "l_dynamics": l_dynamics,
        "l_drift": l_drift,
        "l_diffusion": l_diffusion,
        "l_accept": l_accept,
        "l_E": torch.stack(l_E),
    }


class LangevinSampler:
    """class for Langevin Monte Carlo"""

    def __init__(
        self,
        n_step=None,
        stepsize=None,
        noise_std=None,
        noise_anneal=None,
        bound=None,
        clip_langevin_grad=None,
        reject_boundary=False,
        mh=False,
        initial_dist="uniform",
        sample_shape=None,
        T=1.0,
    ):
        """
        n_step: the number of MCMC steps
        stepsize, noise_std: gradient step size and noise standard deviation.
        buffer_size: the size of buffer
        bound: (lower, upper) or None. The range of valid value of x
        reject_boundary: reject samples if it moves outside of bound
        sample_shape: the shape of samples.
                      set by NAE.set_x_shape() method
        return_min: return minimum along the trajectory
        """

        self.n_step = n_step
        self.stepsize = stepsize
        self.noise_std = noise_std
        self.noise_anneal = noise_anneal
        self.bound = bound
        self.clip_langevin_grad = clip_langevin_grad
        self.reject_boundary = reject_boundary
        self.mh = mh
        self.initial_dist = initial_dist
        self.sample_shape = tuple(sample_shape)
        self.T = T

    def sample(self, energy, n_sample=None, device=None, x0=None, replay=True, T=None):
        """replay: If False, temporarily disable sample replay buffer"""
        if x0 is None:
            x0 = self.initial_sample(n_sample=n_sample, device=device)
        if T is None:
            T = self.T
        d_sample_result = sample_langevin_v2(
            x0.detach(),
            energy,
            stepsize=self.stepsize,
            n_step=self.n_step,
            noise_scale=self.noise_std,
            bound=self.bound,
            noise_anneal=self.noise_anneal,
            clip_grad=self.clip_langevin_grad,
            mh=self.mh,
            reject_boundary=self.reject_boundary,
            temperature=self.T,
        )
        sample_result = d_sample_result["x"]
        d_sample_result["x_0"] = x0
        return d_sample_result

    def initial_sample(self, n_sample, device):
        l_sample = []

        shape = (n_sample,) + self.sample_shape
        x0_new = initial_sample(self.initial_dist, shape, device)
        l_sample.append(x0_new)
        return torch.cat(l_sample).to(device)

    def initial_sample(self, n_sample, device):
        shape = (n_sample,) + self.sample_shape
        if self.initial_dist == "gaussian":
            x0_new = torch.randn(shape, dtype=torch.float)
        elif self.initial_dist == "uniform":
            x0_new = torch.rand(shape, dtype=torch.float)
            x0_new = x0_new * (self.bound[1] - self.bound[0]) + self.bound[0]
        elif self.initial_dist == "uniform_sphere":
            x0_new = torch.randn(shape, dtype=torch.float)
            x0_new = x0_new / (x0_new).norm(dim=1, keepdim=True)
        else:
            raise ValueError(f"Invalid initial_dist: {self.initial_dist}")

        return x0_new.to(device)

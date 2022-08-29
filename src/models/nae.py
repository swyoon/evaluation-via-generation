import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torchvision.utils import make_grid

from models.ae import AE
from models.energybased import SampleBuffer, SampleBufferV2
from models.mcmc import sample_langevin, sample_langevin_v2
from models.modules import IsotropicGaussian, IsotropicLaplace, SphericalActivation
from models.utils import weight_norm


class FFEBM(nn.Module):
    """feed-forward energy-based model"""

    def __init__(
        self,
        net,
        x_step=None,
        x_stepsize=None,
        x_noise_std=None,
        x_noise_anneal=None,
        x_noise_anneal_v2=None,
        x_bound=None,
        x_clip_langevin_grad=None,
        l2_norm_reg=None,
        buffer_size=10000,
        replay_ratio=0.95,
        replay=True,
        gamma=1,
        sampling="x",
        initial_dist="gaussian",
        temperature=1.0,
        temperature_trainable=False,
        mh=False,
        reject_boundary=False,
    ):
        super().__init__()
        self.net = net

        self.x_bound = x_bound
        self.l2_norm_reg = l2_norm_reg
        self.gamma = gamma
        self.sampling = sampling

        self.x_step = x_step
        self.x_stepsize = x_stepsize
        self.x_noise_std = x_noise_std
        self.x_noise_anneal = x_noise_anneal
        self.x_noise_anneal_v2 = x_noise_anneal_v2
        self.x_bound = x_bound
        self.x_clip_langevin_grad = x_clip_langevin_grad
        self.mh = mh
        self.reject_boundary = reject_boundary

        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.replay = replay

        self.buffer = SampleBufferV2(max_samples=buffer_size, replay_ratio=replay_ratio)

        self.x_shape = None
        self.initial_dist = initial_dist
        temperature = np.log(temperature)
        self.temperature_trainable = temperature_trainable
        if temperature_trainable:
            self.register_parameter(
                "temperature_",
                nn.Parameter(torch.tensor(temperature, dtype=torch.float)),
            )
        else:
            self.register_buffer(
                "temperature_", torch.tensor(temperature, dtype=torch.float)
            )

    @property
    def temperature(self):
        return torch.exp(self.temperature_)

    @property
    def sample_shape(self):
        return self.x_shape

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return self.forward(x)

    def energy(self, x):
        return self.forward(x)

    def energy_T(self, x):
        return self.energy(x) / self.temperature

    def min_energy(self, x):
        return -self.decoder.max_log_likelihood(x)

    def sample(self, x0=None, n_sample=None, device=None, replay=None):
        """sampling factory function. takes either x0 or n_sample and device"""
        if x0 is not None:
            n_sample = len(x0)
            device = x0.device
        if replay is None:
            replay = self.replay

        if self.sampling == "x":
            return self.sample_x(n_sample, device, replay=replay)
        elif self.sampling == "cd":
            return self.sample_x(n_sample, device, x0=x0, replay=False)
        elif self.sampling == "on_manifold":
            return self.sample_v1(n_sample, device, replay=replay)

    def sample_x(self, n_sample=None, device=None, x0=None, replay=False):
        if x0 is None:
            x0 = self.initial_sample(n_sample, device=device)
        d_sample_result = sample_langevin_v2(
            x0.detach(),
            self.energy,
            stepsize=self.x_stepsize,
            n_step=self.x_step,
            noise_scale=self.x_noise_std,
            bound=self.x_bound,
            noise_anneal=self.x_noise_anneal,
            noise_anneal_v2=self.x_noise_anneal_v2,
            clip_grad=self.x_clip_langevin_grad,
            mh=self.mh,
            temperature=self.temperature,
            reject_boundary=self.reject_boundary,
        )
        sample_result = d_sample_result["sample"]
        if replay:
            self.buffer.push(sample_result)
        d_sample_result["sample_x"] = sample_result
        d_sample_result["sample_x0"] = x0
        return d_sample_result

    def initial_sample(self, n_samples, device):
        l_sample = []
        if not self.replay or len(self.buffer) == 0:
            n_replay = 0
        else:
            n_replay = (np.random.rand(n_samples) < self.replay_ratio).sum()
            l_sample.append(self.buffer.get(n_replay))

        shape = (n_samples - n_replay,) + self.sample_shape
        if self.initial_dist == "gaussian":
            x0_new = torch.randn(shape, dtype=torch.float)
        elif self.initial_dist == "uniform":
            x0_new = torch.rand(shape, dtype=torch.float)
            if self.sampling != "on_manifold" and self.x_bound is not None:
                x0_new = x0_new * (self.x_bound[1] - self.x_bound[0]) + self.x_bound[0]
            elif self.sampling == "on_manifold" and self.z_bound is not None:
                x0_new = x0_new * (self.z_bound[1] - self.z_bound[0]) + self.z_bound[0]

        l_sample.append(x0_new)
        return torch.cat(l_sample).to(device)

    def _set_x_shape(self, x):
        if self.x_shape is not None:
            return
        self.x_shape = x.shape[1:]

    def weight_norm(self, net):
        norm = 0
        for param in net.parameters():
            norm += (param**2).sum()
        return norm

    def train_step(self, x, opt):
        self._set_x_shape(x)

        # negative sample
        d_sample = self.sample(x)
        x_neg = d_sample["sample_x"]

        opt.zero_grad()
        neg_e = self.energy(x_neg)

        # ae recon pass
        pos_e = self.energy(x)

        loss = (pos_e.mean() - neg_e.mean()) / self.temperature

        if self.gamma is not None:
            loss += self.gamma * (pos_e**2 + neg_e**2).mean()

        # weight regularization
        l2_norm = self.weight_norm(self.net)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * l2_norm

        loss.backward()
        opt.step()

        d_result = {
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "x_neg": x_neg.detach().cpu(),
            "x_neg_0": d_sample["sample_x0"].detach().cpu(),
            "loss": loss.item(),
            "sample": x_neg.detach().cpu(),
            "l2_norm": l2_norm.item(),
        }
        return d_result

    def validation_step(self, x, y=None):
        pos_e = self.energy(x)
        loss = pos_e.mean().item()
        predict = pos_e.detach().cpu().flatten()
        return {"loss": pos_e, "predict": predict}


class NAE(FFEBM):
    """Normalized Autoencoder"""

    def __init__(
        self,
        encoder,
        decoder,
        z_step=50,
        z_stepsize=0.2,
        z_noise_std=0.2,
        z_noise_anneal=None,
        x_step=50,
        x_stepsize=10,
        x_noise_std=0.05,
        x_noise_anneal=None,
        x_noise_anneal_v2=None,
        sigma=0.1,
        sigma_trainable=False,
        x_bound=(0, 1),
        z_bound=None,
        z_clip_langevin_grad=None,
        x_clip_langevin_grad=None,
        l2_norm_reg=None,
        l2_norm_reg_en=None,
        spherical=True,
        z_norm_reg=None,
        buffer_size=10000,
        replay_ratio=0.95,
        replay=True,
        gamma=None,
        delta=None,
        recon_error="l2",
        ebae_coef=1.0,
        sampling="on_manifold",
        temperature=1.0,
        temperature_trainable=True,
        initial_dist="gaussian",
        deterministic=False,
        mh=False,
        mh_z=False,
        reject_boundary=False,
        reject_boundary_z=False,
    ):
        if recon_error == "l2":
            decoder = IsotropicGaussian(
                decoder,
                sigma=sigma,
                sigma_trainable=sigma_trainable,
                deterministic=deterministic,
            )
        elif recon_error == "l1":
            decoder = IsotropicLaplace(
                decoder, sigma=sigma, sigma_trainable=sigma_trainable
            )
        elif recon_error == "asis":
            pass
        else:
            raise ValueError(f"{recon_error}")

        super(NAE, self).__init__(
            net=None,
            x_step=x_step,
            x_stepsize=x_stepsize,
            x_noise_std=x_noise_std,
            x_noise_anneal=x_noise_anneal,
            x_noise_anneal_v2=x_noise_anneal_v2,
            x_bound=x_bound,
            x_clip_langevin_grad=x_clip_langevin_grad,
            l2_norm_reg=l2_norm_reg,
            buffer_size=buffer_size,
            replay_ratio=replay_ratio,
            replay=replay,
            gamma=gamma,
            sampling=sampling,
            initial_dist=initial_dist,
            temperature=temperature,
            temperature_trainable=temperature_trainable,
            mh=mh,
            reject_boundary=reject_boundary,
        )
        self.encoder = encoder
        self.decoder = decoder
        self.delta = delta
        self.z_step = z_step
        self.z_stepsize = z_stepsize
        self.z_noise_std = z_noise_std
        self.z_noise_anneal = z_noise_anneal
        self.z_clip_langevin_grad = z_clip_langevin_grad
        self.mh_z = mh_z
        self.reject_boundary_z = reject_boundary_z

        self.z_bound = z_bound
        self.l2_norm_reg = l2_norm_reg  # decoder
        self.l2_norm_reg_en = l2_norm_reg_en
        self.spherical = spherical
        self.ebae_coef = ebae_coef
        self.sampling = sampling
        self.z_norm_reg = z_norm_reg

        self.z_shape = None
        self.x_shape = None

    @property
    def sample_shape(self):
        if self.sampling == "on_manifold":
            return self.z_shape
        else:
            return self.x_shape

    def forward(self, x):
        D = np.prod(x.shape[1:])
        z = self.encode(x)
        return -self.decoder.log_likelihood(x, z) / D

    def energy_with_z(self, x):
        D = np.prod(x.shape[1:])
        z = self.encode(x)
        return -self.decoder.log_likelihood(x, z) / D, z

    def min_energy(self, x):
        D = np.prod(x.shape[1:])
        return -self.decoder.max_log_likelihood(x) / D

    def normalize(self, z):
        """normalize to unit length"""
        if self.spherical:
            if len(z.shape) == 4:
                z = z / z.view(len(z), -1).norm(dim=-1)[:, None, None, None]
            else:
                z = z / z.view(len(z), -1).norm(dim=1, keepdim=True)
            return z
        else:
            return z

    def encode(self, x):
        if self.spherical:
            return self.normalize(self.encoder(x))
        else:
            return self.encoder(x)

    def sample_v1(self, n_sample, device, replay=False):
        """using on-manifold initialization"""
        # Step 1: On-manifold initialization: LMC on Z space
        z0 = self.initial_sample(n_sample, device)
        if self.spherical:
            z0 = self.normalize(z0)
        d_sample_z = self.sample_z(z0=z0, replay=replay)
        sample_z = d_sample_z["sample"]

        sample_x_1 = self.decoder.sample(sample_z).detach()
        if self.x_bound is not None:
            sample_x_1.clamp_(self.x_bound[0], self.x_bound[1])

        # Step 2: LMC on X space
        d_sample_x = self.sample_x(x0=sample_x_1, replay=False)
        sample_x_2 = d_sample_x["sample_x"]
        return {
            "sample_x": sample_x_2,
            "sample_z": sample_z.detach(),
            "sample_x0": sample_x_1,
            "sample_z0": z0.detach(),
        }

    def sample_z(self, n_sample=None, device=None, replay=False, z0=None):
        if z0 is None:
            z0 = self.initial_sample(n_sample, device)
        energy = lambda z: self.energy(self.decoder.sample(z))
        d_sample_result = sample_langevin_v2(
            z0,
            energy,
            stepsize=self.z_stepsize,
            n_step=self.z_step,
            noise_scale=self.z_noise_std,
            bound=self.z_bound,
            clip_grad=self.z_clip_langevin_grad,
            mh=self.mh_z,
            temperature=self.temperature,
            reject_boundary=self.reject_boundary_z,
        )
        sample_z = d_sample_result["sample"]
        if replay:
            self.buffer.push(sample_z)
        return d_sample_result

    def _set_z_shape(self, x):
        if self.z_shape is not None:
            return
        # infer z_shape by computing forward
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        z_shape = dummy_z.shape
        self.z_shape = z_shape[1:]

    def reconstruct(self, x):
        z = self.encode(x)
        return self.decoder.sample(z)

    def train_step_ae(self, x, opt, clip_grad=None):
        opt.zero_grad()
        z = self.encode(x)
        recon = self.decoder(z)
        # error = self.decoder.error(x, recon)
        z_norm = (z**2).mean()
        x_dim = np.prod(x.shape[1:])
        recon_error = -self.decoder.log_likelihood(x, z).mean() / x_dim
        loss = recon_error

        # weight regularization
        decoder_norm = self.weight_norm(self.decoder)
        encoder_norm = self.weight_norm(self.encoder)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                opt.param_groups[0]["params"], max_norm=clip_grad
            )
        opt.step()
        d_result = {
            "loss": loss.item(),
            "z_norm": z_norm.item(),
            "recon_error_": recon_error.item(),
            "decoder_norm_": decoder_norm.item(),
            "encoder_norm_": encoder_norm.item(),
        }
        return d_result

    def train_step(self, x, opt, clip_grad=None):
        self._set_z_shape(x)
        self._set_x_shape(x)

        # negative sample
        d_sample = self.sample(x)
        x_neg = d_sample["sample_x"]

        opt.zero_grad()
        neg_e, neg_z = self.energy_with_z(x_neg)

        # ae recon pass
        pos_e, pos_z = self.energy_with_z(x)

        loss = pos_e.mean() - neg_e.mean() * self.ebae_coef  # / self.temperature

        if self.temperature_trainable:
            loss = (
                loss
                + (pos_e.mean() - neg_e.mean() * self.ebae_coef).detach()
                / self.temperature
            )

        if self.gamma is not None:
            min_e = self.min_energy(x).detach()
            gamma_term = ((neg_e - min_e) ** 2).mean()
            loss += self.gamma * gamma_term
        if self.delta is not None:
            delta_term = ((neg_e - pos_e.mean().detach()).clamp_(0, None) ** 2).mean()
            loss += self.delta * delta_term  # / self.temperature.detach()

        # weight regularization
        decoder_norm = self.weight_norm(self.decoder)
        encoder_norm = self.weight_norm(self.encoder)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        if self.z_norm_reg is not None:
            z_norm = (torch.cat([pos_z, neg_z]) ** 2).mean()
            loss = loss + self.z_norm_reg * z_norm

        loss.backward()
        opt.step()

        # for debugging
        x_neg_0 = d_sample["sample_x0"]
        neg_e_x0 = self.energy(x_neg_0)  # energy of samples from latent chain
        recon_neg = self.reconstruct(x_neg)
        d_result = {
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "x_neg": x_neg.detach().cpu(),
            "recon_neg": recon_neg.detach().cpu(),
            "loss": loss.item(),
            "sample": x_neg.detach().cpu(),
            "decoder_norm": decoder_norm.item(),
            "encoder_norm": encoder_norm.item(),
            "neg_e_x0": neg_e_x0.mean().item(),
            "x_neg_0": x_neg_0.detach().cpu(),
            "temperature": self.temperature.item(),
            "sigma": self.decoder.sigma.item(),
            "pos_z": pos_z.detach().cpu(),
            "neg_z": neg_z.detach().cpu(),
        }
        if self.delta is not None:
            d_result["delta_term"] = delta_term.item()
        if self.gamma is not None:
            d_result["gamma_term"] = gamma_term.item()
        if "sample_z0" in d_sample:
            x_neg_z0 = self.decoder.mean(d_sample["sample_z0"])
            d_result["neg_e_z0"] = self.energy(x_neg_z0).mean().item()
        return d_result

    def validation_step(self, x, y=None):
        z = self.encode(x)
        energy = -self.decoder.log_likelihood(x, z)
        loss = energy.mean().item()
        recon = self.decoder.sample(z)
        pos_e = self.energy(x)
        recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        input_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
        return {"loss": loss, "pos_e": loss, "recon@": recon_img, "input@": input_img}


class FFEBMV2(nn.Module):
    """Another refactored version of EBM. MCMCSampler is separated"""

    def __init__(self, net, sampler, gamma=1.0, sampling="x", l2_norm_reg=None):
        """
        net: nn.Module
        sampler: MCMCSampler
        gamma: regularizer for negative energy
        sampling: sampling method ('x', 'cd')
        """
        super().__init__()
        self.net = net
        self.sampler = sampler
        self.gamma = gamma
        self.sampling = sampling
        self.l2_norm_reg = l2_norm_reg

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return self.forward(x)

    def energy(self, x):
        return self.forward(x)

    def sample(self, x0=None, n_sample=None, device=None, replay=True, sampling=None):
        """takes either x0 or n_sample and device"""
        if x0 is not None:
            n_sample = len(x0)
            device = x0.device
        sampling = self.sampling if sampling is None else sampling

        if sampling == "x":
            d_sample = self.sampler.sample(
                self.energy, n_sample=n_sample, device=device, replay=replay
            )
        elif sampling == "cd":
            d_sample = self.sampler.sample(self.energy, x0=x0, replay=False)
        else:
            raise ValueError(f"Invalid sampling {self.sampling}")
        d_sample["sample_x"] = d_sample["sample"]
        d_sample["sample_x0"] = d_sample["sample_0"]
        return d_sample

    def _set_x_shape(self, x):
        if self.sampler.sample_shape is not None:
            return
        self.sampler.sample_shape = x.shape[1:]

    def train_step(self, x, opt):
        self._set_x_shape(x)

        # negative sample
        d_sample = self.sample(x)
        x_neg = d_sample["sample_x"]

        opt.zero_grad()
        neg_e = self.energy(x_neg)

        # ae recon pass
        pos_e = self.energy(x)

        loss = pos_e.mean() - neg_e.mean()

        if self.gamma is not None:
            loss += self.gamma * (pos_e**2 + neg_e**2).mean()

        # weight regularization
        l2_norm = weight_norm(self.net)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * l2_norm

        loss.backward()
        opt.step()

        d_result = {
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "x_neg": x_neg.detach().cpu(),
            "x_neg_0": d_sample["sample_x0"].detach().cpu(),
            "loss": loss.item(),
            "sample": x_neg.detach().cpu(),
            "l2_norm": l2_norm.item(),
        }
        return d_result

    def validation_step(self, x, y=None):
        pos_e = self.energy(x)
        loss = pos_e.mean().item()
        predict = pos_e.detach().cpu().flatten()
        return {"loss": pos_e, "predict": predict}


class NAE_L2_base(nn.Module):
    """Base class for Normalized Autoencoders"""

    def __init__(
        self,
        encoder,
        decoder,
        T=1.0,
        T_trainable=False,
        spherical=False,
        l2_norm_reg_en=None,
        l2_norm_reg_de=None,
        zhat_norm_reg=None,
        encoding_noise=None,
    ):
        """
        encoder: An encoder network, an instance of nn.Module.
        decoder: A decoder network, an instance of nn.Module.

        **Regularization Parameters**
        gamma: The coefficient for regularizing the negative sample energy.
        l2_norm_reg: The coefficient for L2 norm of decoder weights.
        l2_norm_reg_en: The coefficient for L2 norm of encoder weights.

        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        T_ = np.log(T)
        if T_trainable:
            self.register_parameter(
                "T_", nn.Parameter(torch.tensor(T_, dtype=torch.float))
            )
        else:
            self.register_buffer("T_", torch.tensor(T_))
        self.spherical = spherical
        self.l2_norm_reg_en = l2_norm_reg_en
        self.l2_norm_reg_de = l2_norm_reg_de
        self.zhat_norm_reg = zhat_norm_reg
        self.encoding_noise = encoding_noise

    @property
    def T(self):
        return torch.exp(self.T_)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return ((x - recon) ** 2).view(len(x), -1).mean(dim=-1)

    def reconstruct(self, x):
        return self.decoder(self.encode(x))

    def energy(self, x):
        return self.forward(x)

    def energy_T(self, x):
        return self.forward(x) / self.T

    def predict(self, x):
        return self.forward(x)

    def forward_training(self, x):
        z = self.encoder(x)
        if self.encoding_noise is not None:
            z += torch.randn_like(z) * self.encoding_noise
            z = self.project(z)
        recon = self.decoder(z)
        return ((x - recon) ** 2).view(len(x), -1).mean()

    def project(self, zhat):
        if self.spherical:
            return zhat / zhat.norm(p=2, dim=1, keepdim=True)
        else:
            return zhat

    def encode(self, x):
        return self.project(self.encoder(x))

    def decode(self, x):
        return self.decoder(x)

    def train_step_ae(self, x, opt, clip_grad=None):
        opt.zero_grad()
        recon_error = self.forward_training(x)
        loss = recon_error

        # weight regularization
        encoder_norm, decoder_norm = self.weight_norm()
        if self.l2_norm_reg_de is not None:
            loss = loss + self.l2_norm_reg_de * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        if self.zhat_norm_reg is not None and hasattr(self.encoder, "pre_activation"):
            zhat_norm = (self.encoder.pre_activation**2).sum(dim=1).mean()
            loss = loss + self.zhat_norm_reg * zhat_norm
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                opt.param_groups[0]["params"], max_norm=clip_grad
            )
        opt.step()
        d_result = {
            "loss": loss.item(),
            "recon_error_": recon_error.item(),
            "l2_norm_de_": decoder_norm.item(),
            "l2_norm_en_": encoder_norm.item(),
        }
        if self.zhat_norm_reg is not None:
            d_result["zhat_norm_"] = zhat_norm.item()
        return d_result

    def validation_step_ae(self, x, y=None, visualize=True):
        recon = self.reconstruct(x)
        recon_error = ((recon - x) ** 2).mean()
        d_result = {
            "loss": recon_error.item(),
            "predict": recon.detach().cpu().flatten(),
        }
        if visualize:
            input_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
            d_result["input@"] = input_img
            d_result["recon@"] = recon_img
        return d_result

    def weight_norm(self):
        decoder_norm = weight_norm(self.decoder)
        encoder_norm = weight_norm(self.encoder)
        return encoder_norm, decoder_norm


class NAE_L2_CD(NAE_L2_base):
    """Normalized Autoencoder trained via k-Contrastive Divergence (CD) or Persistent CD (PCD)"""

    def __init__(
        self,
        encoder,
        decoder,
        sampler,
        gamma=1.0,
        sampling="x",
        l2_norm_reg_en=None,
        l2_norm_reg_de=None,
        T=1.0,
        T_trainable=False,
        **kwargs,
    ):
        """
        sampling: 'x' (PCD), 'cd' (CD)
        """
        super().__init__(
            encoder,
            decoder,
            T=T,
            T_trainable=T_trainable,
            l2_norm_reg_en=l2_norm_reg_en,
            l2_norm_reg_de=l2_norm_reg_de,
            **kwargs,
        )
        self.sampler = sampler
        self.gamma = gamma
        self.sampling = sampling

    def sample(self, x0=None, n_sample=None, device=None, replay=None, sampling=None):
        """takes either x0 or n_sample and device"""
        if x0 is not None:
            n_sample = len(x0)
            device = x0.device

        sampling = self.sampling if sampling is None else sampling

        if sampling == "x":
            d_sample = self.sampler.sample(
                self.energy_T, n_sample=n_sample, device=device, replay=replay
            )
        elif sampling == "cd":
            d_sample = self.sampler.sample(self.energy_T, x0=x0, replay=False)
        else:
            raise ValueError(f"Invalid sampling {self.sampling}")
        d_sample["sample_x"] = d_sample["sample"]
        d_sample["sample_x0"] = d_sample["sample_0"]
        return d_sample

    def _set_x_shape(self, x):
        if self.sampler.sample_shape is not None:
            return
        self.sampler.sample_shape = x.shape[1:]

    def train_step(self, x, opt):
        self._set_x_shape(x)

        # negative sample
        d_sample = self.sample(x0=x)
        x_neg = d_sample["sample_x"]

        opt.zero_grad()
        neg_e = self.energy(x_neg)

        # ae recon pass
        pos_e = self.energy(x)

        loss = pos_e.mean() - neg_e.mean()

        if self.gamma is not None:
            loss += self.gamma * (pos_e**2 + neg_e**2).mean()

        # weight regularization
        encoder_norm, decoder_norm = self.weight_norm()
        if self.l2_norm_reg_de is not None:
            loss = loss + self.l2_norm_reg_de * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        loss.backward()
        opt.step()

        d_result = {
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "x_neg": x_neg.detach().cpu(),
            "x_neg_0": d_sample["sample_x0"].detach().cpu(),
            "loss": loss.item(),
            "sample": x_neg.detach().cpu(),
            "l2_norm_en": encoder_norm.item(),
            "l2_norm_de": decoder_norm.item(),
        }
        return d_result


class NAE_L2_NCE(NAE_L2_base):
    """NAE by Noise Contrastive Estimation"""

    def __init__(
        self,
        encoder,
        decoder,
        sampler,
        l2_norm_reg_en=None,
        l2_norm_reg_de=None,
        T=1.0,
        T_trainable=False,
        **kwargs,
    ):
        super().__init__(encoder, decoder, T=T, T_trainable=T_trainable, **kwargs)
        self.sampler = sampler  # sampler for noise.
        self.register_parameter(
            "logz", nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        )
        self.l2_norm_reg_en = l2_norm_reg_en
        self.l2_norm_reg_de = l2_norm_reg_de

    def sample_noise(self, n_sample, device):
        return self.sampler.sample(n_sample=n_sample, device=device)

    def train_step(self, x, opt):
        opt.zero_grad()
        xn = self.sample_noise(n_sample=len(x), device=x.device)
        pos_e = self.energy(x)
        neg_e = self.energy(xn)
        logpx = -torch.cat([pos_e, neg_e]) / self.T - self.logz
        logpn = self.sampler.log_prob(torch.cat([x, xn]))
        logit = -logpn + logpx
        target = torch.cat([torch.ones_like(pos_e), torch.zeros_like(neg_e)])
        nce_loss = F.binary_cross_entropy_with_logits(logit, target)

        loss = nce_loss

        # weight regularization
        encoder_norm, decoder_norm = self.weight_norm()
        if self.l2_norm_reg_de is not None:
            loss = loss + self.l2_norm_reg_de * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm

        loss.backward()
        opt.step()
        d_train = {
            "loss": loss.item(),
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "l2_norm_en_": encoder_norm.item(),
            "l2_norm_de_": decoder_norm.item(),
            "nce_loss": nce_loss.item(),
            "x_neg": xn.detach().cpu(),
        }
        return d_train


class NAE_L2_OMI(NAE_L2_base):
    """On-manifold Initialization"""

    def __init__(
        self,
        encoder,
        decoder,
        sampler_z,
        sampler_x,
        l2_norm_reg_en=None,
        l2_norm_reg_de=None,
        zhat_norm_reg=None,
        T=1.0,
        T_trainable=False,
        gamma=1.0,
        spherical=False,
        **kwargs,
    ):
        super().__init__(
            encoder,
            decoder,
            T=T,
            T_trainable=T_trainable,
            l2_norm_reg_en=l2_norm_reg_en,
            l2_norm_reg_de=l2_norm_reg_de,
            zhat_norm_reg=zhat_norm_reg,
            spherical=spherical,
            **kwargs,
        )
        self.sampler_x = sampler_x
        self.sampler_z = sampler_z
        self.gamma = gamma

    def _set_x_shape(self, x):
        if self.sampler_x.sample_shape is not None:
            return
        self.sampler_x.sample_shape = x.shape[1:]

    def _set_z_shape(self, x):
        if self.sampler_z.sample_shape is not None:
            return
        z = self.encoder(x[[0]])
        self.sampler_z.sample_shape = z.shape[1:]

    def sample(self, x0=None, n_sample=None, device=None, replay=True):
        if x0 is not None:
            n_sample = len(x0)
            device = x0.device
        # d_sample_z = self.sampler_z.sample(self.energy_z, n_sample=n_sample, device=device, T=self.T, replay=replay)
        d_sample_z = self.sampler_z.sample(
            self.energy_z, n_sample=n_sample, device=device, replay=replay
        )
        z_sample = d_sample_z["sample"]
        x0 = self.decoder(z_sample)
        d_sample_x = self.sampler_x.sample(self.energy, x0=x0, T=self.T)
        d_sample = {
            "sample_x": d_sample_x["sample"],
            "sample_x0": d_sample_x["sample_0"],
            "d_sample_x": d_sample_x,
            "sample_z": d_sample_z["sample"],
            "sample_z0": d_sample_z["sample_0"],
            "d_sample_z": d_sample_z,
        }
        if "l_accept" in d_sample_z and len(d_sample_z["l_accept"]) > 0:
            d_sample["accept_z"] = (
                torch.stack(d_sample_z["l_accept"]).to(torch.float).mean()
            )
        if "l_accept" in d_sample_x and len(d_sample_x["l_accept"]) > 0:
            d_sample["accept_x"] = (
                torch.stack(d_sample_x["l_accept"]).to(torch.float).mean()
            )
        return d_sample

    def energy_z(self, z):
        return self.energy(self.decoder(z))

    def train_step(self, x, opt, clip_grad=None):
        self._set_x_shape(x)
        self._set_z_shape(x)

        # negative sample
        d_sample = self.sample(x0=x)
        x_neg = d_sample["sample_x"]

        opt.zero_grad()
        neg_e = self.energy(x_neg)

        # ae recon pass
        pos_e = self.energy(x)

        loss = (pos_e.mean() - neg_e.mean()) / self.T

        if self.gamma is not None:
            loss += self.gamma * (neg_e**2).mean()

        # weight regularization
        encoder_norm, decoder_norm = self.weight_norm()
        if self.l2_norm_reg_de is not None:
            loss = loss + self.l2_norm_reg_de * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        loss.backward()
        opt.step()

        sample_img = make_grid(x_neg.detach().cpu(), nrow=10, range=(0, 1))
        x_neg_0 = d_sample["sample_x0"]
        sample_img_0 = make_grid(x_neg_0.detach().cpu(), nrow=10, range=(0, 1))
        neg_e_0 = self.energy(x_neg_0)

        d_result = {
            "nae/pos_e_": pos_e.mean().item(),
            "nae/neg_e_": neg_e.mean().item(),
            "nae/neg_e_0_": neg_e_0.mean().item(),
            # 'nae/pos_recon_e_': pos_e.mean().item(), 'nae/neg_recon_e_': neg_e.mean().item(),
            "x_neg": x_neg.detach().cpu(),
            "x_neg_0": sample_img_0,
            "loss": loss.item(),
            "sample": x_neg.detach().cpu(),
            "nae/sample_img@": sample_img,
            "nae/sample_img_0@": sample_img_0,
            "l2_norm_en_": encoder_norm.item(),
            "l2_norm_de_": decoder_norm.item(),
            "d_sample": d_sample,
        }
        if "accept_x" in d_sample:
            d_result["mcmc/accept_x_"] = d_sample["accept_x"].item()
        if "accept_z" in d_sample:
            d_result["mcmc/accept_z_"] = d_sample["accept_z"].item()
        return d_result


class NAE_CL_base(nn.Module):
    """Base class for conditional likelihood formulation of NAE"""

    def __init__(
        self,
        encoder,
        decoder,
        varnet,
        T=1.0,
        T_trainable=False,
        varparam=None,
        encoding_noise=None,
        zhat_norm_reg=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.varnet = varnet
        self.varparam = varparam  # parametrization mode of variance
        T_ = np.log(T)
        if T_trainable:
            self.register_parameter(
                "T_", nn.Parameter(torch.tensor(T_, dtype=torch.float))
            )
        else:
            self.register_buffer("T_", torch.tensor(T_))
        self.encoding_noise = encoding_noise
        self.zhat_norm_reg = zhat_norm_reg

    @property
    def T(self):
        return torch.exp(self.T_)

    def forward(self, x, train=False):
        """train: Returns intermediate variables if true. Returns only energy when false."""
        z = self.encoder(x)
        recon = self.decoder(z)
        v = self.varnet(z).flatten()
        if self.varparam == "1pexp":
            var = torch.exp(v)
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / (1 + var) + torch.log1p(
                var
            )
        elif self.varparam == "1psq":
            var = v**2
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / (1 + var) + torch.log1p(
                var
            )
        elif self.varparam == "expabs":
            var = torch.exp(torch.abs(v))
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / var + torch.abs(v)
        elif self.varparam == "expsq":
            var = torch.exp(v**2)
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / var + v**2
        elif self.varparam == "novar":  # for ablation study
            var = torch.exp(v)
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) + v
        elif self.varparam == "nov":  # for ablation study
            var = torch.exp(v)
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / var
        elif self.varparam == "expprod":
            var = torch.exp(v)
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) * var
        else:
            var = torch.exp(v)
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / var + v

        if train:
            d_train = {
                "E": E,
                "recon": recon,
                "v": v,
                "recon_error": recon_error,
                "var": var,
            }
            if self.zhat_norm_reg is not None:
                d_train["zhat_norm"] = (
                    (self.encoder.pre_activation**2).sum(dim=1).mean()
                )
            return d_train
        else:
            return E

    def recon_error(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return ((recon - x) ** 2).view(len(x), -1).mean(dim=-1)

    def recon_error_noise(self, x):
        z = self.encoder(x)
        z += torch.randn_like(z) * self.encoding_noise
        if isinstance(self.encoder.out_activation, SphericalActivation) or isinstance(
            self.encoder.out_activation, SphericalActivationV2
        ):
            z = z / z.norm(dim=1, keepdim=True)
        recon = self.decoder(z)
        return ((recon - x) ** 2).view(len(x), -1).mean(dim=-1)

    def v(self, x):
        z = self.encoder(x)
        return self.varnet(z).flatten()

    def reconstruct(self, x):
        return self.decoder(self.encoder(x))

    def energy(self, x, train=False):
        return self.forward(x, train=train)

    def energy_T(self, x, train=False):
        return self.energy(x, train=train) / self.T

    def predict(self, x):
        return self.forward(x)

    def train_step_ae(self, x, opt, clip_grad=None):
        opt.zero_grad()
        if self.encoding_noise is not None:
            recon_error = self.recon_error_noise(x)
        else:
            recon_error = self.recon_error(x)

        loss = recon_error.mean()

        # weight regularization
        encoder_norm, decoder_norm, varnet_norm = self.weight_norm()
        if self.l2_norm_reg_de is not None:
            loss = loss + self.l2_norm_reg_de * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        if self.zhat_norm_reg is not None and hasattr(self.encoder, "pre_activation"):
            zhat_norm = (self.encoder.pre_activation**2).sum(dim=1).mean()
            loss = loss + self.zhat_norm_reg * zhat_norm
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                opt.param_groups[0]["params"], max_norm=clip_grad
            )
        opt.step()
        d_result = {
            "loss": loss.item(),
            "recon_error_": recon_error.mean().item(),
            "l2_norm_de_": decoder_norm.item(),
            "l2_norm_en_": encoder_norm.item(),
        }
        if self.zhat_norm_reg is not None:
            d_result["zhat_norm_"] = zhat_norm.item()
        return d_result

    def validation_step_ae(self, x, y=None, visualize=True):
        recon = self.reconstruct(x)
        recon_error = ((recon - x) ** 2).mean()
        d_result = {
            "loss": recon_error.item(),
            "predict": recon.detach().cpu().flatten(),
        }
        if visualize:
            input_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
            d_result["input@"] = input_img
            d_result["recon@"] = recon_img
        return d_result

    def weight_norm(self):
        decoder_norm = weight_norm(self.decoder)
        encoder_norm = weight_norm(self.encoder)
        varnet_norm = weight_norm(self.varnet)
        return encoder_norm, decoder_norm, varnet_norm


class NAE_CL_CD(NAE_CL_base):
    """Conditional Likelihood Formulation of NAE"""

    def __init__(self, encoder, decoder, varnet, sampler):
        # super().__init__(encoder, decoder
        pass


class NAE_CL_NCE(NAE_CL_base):
    def __init__(
        self,
        encoder,
        decoder,
        varnet,
        sampler,
        l2_norm_reg_en=None,
        l2_norm_reg_de=None,
        varparam=None,
        gamma_v=None,
    ):
        super().__init__(encoder, decoder, varnet, varparam=varparam)
        self.sampler = sampler
        self.register_parameter(
            "logz", nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        )
        self.l2_norm_reg_en = l2_norm_reg_en
        self.l2_norm_reg_de = l2_norm_reg_de
        self.gamma_v = gamma_v

    def sample_noise(self, n_sample, device):
        return self.sampler.sample(n_sample=n_sample, device=device)

    def train_step(self, x, opt):
        opt.zero_grad()
        xn = self.sample_noise(n_sample=len(x), device=x.device)
        d_pos = self.energy(x, train=True)
        d_neg = self.energy(xn, train=True)
        pos_e, neg_e = d_pos["E"], d_neg["E"]
        logpx = -torch.cat([pos_e, neg_e]) - self.logz
        logpn = self.sampler.log_prob(torch.cat([x, xn]))
        logit = -logpn + logpx
        target = torch.cat([torch.ones_like(pos_e), torch.zeros_like(neg_e)])
        nce_loss = F.binary_cross_entropy_with_logits(logit, target)

        loss = nce_loss

        # energy regularization
        if self.gamma_v is not None:
            pos_v, neg_v = d_pos["v"], d_neg["v"]
            loss += self.gamma_v * ((pos_v**2).mean() + (neg_v**2)).mean()

        # weight regularization
        encoder_norm, decoder_norm, varnet_norm = self.weight_norm()
        if self.l2_norm_reg_de is not None:
            loss = loss + self.l2_norm_reg_de * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm

        loss.backward()
        opt.step()
        d_train = {
            "loss": loss.item(),
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "l2_norm_en_": encoder_norm.item(),
            "l2_norm_de_": decoder_norm.item(),
            "nce_loss": nce_loss.item(),
            "x_neg": xn.detach().cpu(),
        }
        return d_train

    def train_step_v2(self, x, opt_ae, opt_var):
        """separate ae training and var training"""
        d_ae = self.train_step_ae(x, opt_ae)
        d_ae["loss_ae"] = d_ae.pop("loss")

        opt_var.zero_grad()
        xn = self.sample_noise(n_sample=len(x), device=x.device)
        pos_e = self.energy(x)
        neg_e = self.energy(xn)
        logpx = -torch.cat([pos_e, neg_e]) - self.logz
        logpn = self.sampler.log_prob(torch.cat([x, xn]))
        logit = -logpn + logpx
        target = torch.cat([torch.ones_like(pos_e), torch.zeros_like(neg_e)])
        nce_loss = F.binary_cross_entropy_with_logits(logit, target)

        loss = nce_loss

        # weight regularization
        encoder_norm, decoder_norm, varnet_norm = self.weight_norm()
        loss.backward()
        opt_var.step()
        d_train = {
            "loss": loss.item(),
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "nce_loss": nce_loss.item(),
            "x_neg": xn.detach().cpu(),
        }
        d_train.update(d_ae)
        return d_train


class NAE_CL_OMI(NAE_CL_base):
    """On-manifold Initialization"""

    def __init__(
        self,
        encoder,
        decoder,
        varnet,
        sampler_z,
        sampler_x,
        T=1.0,
        l2_norm_reg_en=None,
        l2_norm_reg_de=None,
        l2_norm_reg_var=None,
        zhat_norm_reg=None,
        gamma=None,
        gamma_neg=None,
        gamma_neg_recon=None,
        varparam=None,
        gamma_v=None,
        gamma_v2=None,
        gamma_var=None,
        gamma_var2=None,
        encoding_noise=None,
    ):
        super().__init__(
            encoder,
            decoder,
            varnet,
            T=T,
            varparam=varparam,
            encoding_noise=encoding_noise,
            zhat_norm_reg=zhat_norm_reg,
        )
        self.sampler_x = sampler_x
        self.sampler_z = sampler_z
        self.l2_norm_reg_en = l2_norm_reg_en
        self.l2_norm_reg_de = l2_norm_reg_de
        self.l2_norm_reg_var = l2_norm_reg_var
        self.gamma = gamma
        self.gamma_neg = gamma_neg
        self.gamma_neg_recon = gamma_neg_recon
        self.gamma_v = gamma_v
        self.gamma_v2 = gamma_v2
        self.gamma_var = gamma_var
        self.gamma_var2 = gamma_var2

    def _set_x_shape(self, x):
        if self.sampler_x.sample_shape is not None:
            return
        self.sampler_x.sample_shape = x.shape[1:]

    def _set_z_shape(self, x):
        if self.sampler_z.sample_shape is not None:
            return
        z = self.encoder(x[[0]])
        self.sampler_z.sample_shape = z.shape[1:]

    def sample(self, x0=None, n_sample=None, device=None, replay=True):
        if x0 is not None:
            n_sample = len(x0)
            device = x0.device
        d_sample_z = self.sampler_z.sample(
            self.energy_z, n_sample=n_sample, device=device, T=self.T, replay=replay
        )
        z_sample = d_sample_z["sample"]
        x0 = self.decoder(z_sample)
        d_sample_x = self.sampler_x.sample(self.energy, x0=x0, T=self.T)
        d_sample = {
            "sample_x": d_sample_x["sample"],
            "sample_x0": d_sample_x["sample_0"],
            "sample_z": d_sample_z["sample"],
            "sample_z0": d_sample_z["sample_0"],
            "d_sample_x": d_sample_x,
            "d_sample_z": d_sample_z,
        }
        return d_sample

    def energy_z(self, z):
        return self.energy(self.decoder(z))

    def energy_recon(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        logvar = self.varnet(z)
        std = torch.exp(logvar * 0.5)
        recon_error = ((recon - x) ** 2).view(len(x), -1).mean(-1)
        return (
            -Normal(loc=recon, scale=std).log_prob(x).view(len(x), -1).mean(dim=-1),
            recon_error,
        )

    def train_step(self, x, opt, clip_grad=None):
        self._set_x_shape(x)
        self._set_z_shape(x)

        # negative sample
        d_sample = self.sample(x0=x)
        x_neg = d_sample["sample_x"]

        opt.zero_grad()
        d_pos, d_neg = self.energy(x, train=True), self.energy(x_neg, train=True)
        pos_e, neg_e = d_pos["E"], d_neg["E"]
        pos_recon_error = d_pos["recon_error"]
        neg_recon_error = d_neg["recon_error"]

        loss = pos_e.mean() - neg_e.mean()  # / self.T

        if self.gamma is not None:
            loss += self.gamma * (pos_e**2 + neg_e**2).mean()
        if self.gamma_neg is not None:
            loss += self.gamma_neg * (neg_e**2).mean()
        if self.gamma_neg_recon is not None:
            neg_recon_error = d_neg["recon_error"]
            loss += self.gamma_neg_recon * (neg_recon_error**2).mean()
        if self.gamma_var is not None:
            pos_var, neg_var = d_pos["var"], d_neg["var"]
            loss += self.gamma_var * ((pos_var**2).mean() + (neg_var**2).mean())
        if self.gamma_var2 is not None:
            neg_var = d_neg["var"]
            loss += self.gamma_var2 * ((neg_var**2).mean())
        if self.gamma_v is not None:
            pos_v, neg_v = d_pos["v"], d_neg["v"]
            loss += self.gamma_v * ((pos_v**2).mean() + (neg_v**2).mean())
        if self.gamma_v2 is not None:
            neg_v = d_pos["v"]
            loss += self.gamma_v2 * ((neg_v**2).mean())

        # weight regularization
        encoder_norm, decoder_norm, varnet_norm = self.weight_norm()
        if self.l2_norm_reg_de is not None:
            loss = loss + self.l2_norm_reg_de * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        if self.l2_norm_reg_var is not None:
            loss = loss + self.l2_norm_reg_var * varnet_norm
        if self.zhat_norm_reg is not None and hasattr(self.encoder, "pre_activation"):
            # loss = loss + self.zhat_norm_reg * (d_pos['zhat_norm'] + d_neg['zhat_norm']) / 2
            # loss = loss + self.zhat_norm_reg * d_pos['zhat_norm']
            loss = loss + self.zhat_norm_reg * d_neg["zhat_norm"]

        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                opt.param_groups[0]["params"], max_norm=clip_grad
            )
        opt.step()

        sample_img = make_grid(x_neg, nrow=10, range=(0, 1))
        neg_e_x0 = d_sample["d_sample_x"]["l_E"][0].mean()
        neg_e_z0 = d_sample["d_sample_z"]["l_E"][0].mean()

        d_result = {
            "nae/pos_e_": pos_e.mean().item(),
            "nae/neg_e_": neg_e.mean().item(),
            "nae/pos_recon_e_": pos_recon_error.mean().item(),
            "nae/neg_recon_e_": neg_recon_error.mean().item(),
            "cl_v/pos_": d_pos["v"].mean().item(),
            "cl_v/neg_": d_neg["v"].mean().item(),
            "cl_var/pos_": d_pos["var"].mean().item(),
            "cl_var/neg_": d_neg["var"].mean().item(),
            "x_neg": x_neg.detach().cpu(),
            "x_neg_0": d_sample["sample_x0"].detach().cpu(),
            "loss": loss.item(),
            "nae/sample_img@": sample_img.detach().cpu(),
            "l2_norm/en_": encoder_norm.item(),
            "l2_norm/de_": decoder_norm.item(),
            "l2_norm/var_": varnet_norm.item(),
            "d_sample": d_sample,
            "nae/neg_e_0_": neg_e_x0.item(),
            "nae/nae_e_z0_": neg_e_z0.item(),
        }

        if self.zhat_norm_reg is not None:
            d_result["zhat_norm/pos_"] = d_pos["zhat_norm"].item()
            d_result["zhat_norm/neg_"] = d_neg["zhat_norm"].item()
        return d_result


class NAE_CLX_base(nn.Module):
    """Base class for conditional likelihood formulation of NAE.
    Varnet takes x as an input."""

    def __init__(
        self,
        encoder,
        decoder,
        varnet,
        T=1.0,
        T_trainable=False,
        varparam=None,
        encoding_noise=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.varnet = varnet
        self.varparam = varparam
        self.encoding_noise = encoding_noise  # magnitude of noise in encoding
        T_ = np.log(T)
        if T_trainable:
            self.register_parameter(
                "T_", nn.Parameter(torch.tensor(T_, dtype=torch.float))
            )
        else:
            self.register_buffer("T_", torch.tensor(T_))

    @property
    def T(self):
        return torch.exp(self.T_)

    def forward(self, x, train=False):
        z = self.encoder(x)
        recon = self.decoder(z)
        v = self.varnet(x).flatten()
        if self.varparam == "1pexp":
            var = torch.exp(v)
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / (1 + var) + torch.log1p(
                var
            )
        elif self.varparam == "1psq":
            var = v**2
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / (1 + var) + torch.log1p(
                var
            )
        elif self.varparam == "expabs":
            var = torch.exp(torch.abs(v))
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / var + torch.abs(v)
        elif self.varparam == "expsq":
            var = torch.exp(v**2)
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / var + v**2
        else:
            # std = torch.exp(v * 0.5)
            var = torch.exp(v)
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / var + v
            # E = - Normal(loc=recon, scale=std).log_prob(x).view(len(x), -1).mean(dim=-1)

        if train:
            return {
                "E": E,
                "recon": recon,
                "v": v,
                "recon_error": recon_error,
                "var": var,
            }
        else:
            return E

    def recon_error(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return ((recon - x) ** 2).view(len(x), -1).mean(dim=-1)

    def recon_error_noise(self, x):
        z = self.encoder(x)
        z += torch.randn_like(z) * self.encoding_noise
        if isinstance(self.encoder.out_activation, SphericalActivation):
            z = z / z.norm(dim=1, keepdim=True)
        recon = self.decoder(z)
        return ((recon - x) ** 2).view(len(x), -1).mean(dim=-1)

    def reconstruct(self, x):
        return self.decoder(self.encoder(x))

    def energy(self, x, train=False):
        return self.forward(x, train=train)

    def energy_T(self, x, train=False):
        return self.energy(x, train=train) / self.T

    def energy_with_z(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        logvar = self.varnet(x)
        std = torch.exp(logvar * 0.5)
        # return - Normal(loc=recon, scale=std.clip(0.01, None)).log_prob(x).view(len(x), -1).sum(dim=-1), z
        return (
            -Normal(loc=recon, scale=std).log_prob(x).view(len(x), -1).mean(dim=-1),
            z,
        )

    def predict(self, x):
        return self.forward(x)

    def train_step_ae(self, x, opt, clip_grad=None):
        opt.zero_grad()
        if self.encoding_noise is not None:
            recon_error = self.recon_error_noise(x)
        else:
            recon_error = self.recon_error(x)
        loss = recon_error.mean()

        # weight regularization
        encoder_norm, decoder_norm, varnet_norm = self.weight_norm()
        if self.l2_norm_reg_de is not None:
            loss = loss + self.l2_norm_reg_de * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                opt.param_groups[0]["params"], max_norm=clip_grad
            )
        opt.step()
        d_result = {
            "loss": loss.item(),
            "recon_error_": recon_error.mean().item(),
            "l2_norm_de_": decoder_norm.item(),
            "l2_norm_en_": encoder_norm.item(),
        }
        return d_result

    def validation_step_ae(self, x, y=None, visualize=True):
        recon = self.reconstruct(x)
        recon_error = ((recon - x) ** 2).mean()
        d_result = {
            "loss": recon_error.item(),
            "predict": recon.detach().cpu().flatten(),
        }
        if visualize:
            input_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
            d_result["input@"] = input_img
            d_result["recon@"] = recon_img
        return d_result

    def weight_norm(self):
        decoder_norm = weight_norm(self.decoder)
        encoder_norm = weight_norm(self.encoder)
        varnet_norm = weight_norm(self.varnet)
        return encoder_norm, decoder_norm, varnet_norm


class NAE_CLX_OMI(NAE_CLX_base):
    """On-manifold Initialization"""

    def __init__(
        self,
        encoder,
        decoder,
        varnet,
        sampler_z,
        sampler_x,
        T=1.0,
        l2_norm_reg_en=None,
        l2_norm_reg_de=None,
        l2_norm_reg_var=None,
        gamma=None,
        gamma_neg=None,
        gamma_neg_recon=None,
        varparam=None,
        gamma_v=None,
        gamma_v2=None,
        gamma_var=None,
        gamma_var2=None,
        encoding_noise=None,
    ):
        super().__init__(
            encoder,
            decoder,
            varnet,
            T=T,
            varparam=varparam,
            encoding_noise=encoding_noise,
        )
        self.sampler_x = sampler_x
        self.sampler_z = sampler_z
        self.l2_norm_reg_en = l2_norm_reg_en
        self.l2_norm_reg_de = l2_norm_reg_de
        self.l2_norm_reg_var = l2_norm_reg_var
        self.gamma = gamma
        self.gamma_neg = gamma_neg
        self.gamma_neg_recon = gamma_neg_recon
        self.gamma_v = gamma_v
        self.gamma_v2 = gamma_v2
        self.gamma_var = gamma_var
        self.gamma_var2 = gamma_var2

    def _set_x_shape(self, x):
        if self.sampler_x.sample_shape is not None:
            return
        self.sampler_x.sample_shape = x.shape[1:]

    def _set_z_shape(self, x):
        if self.sampler_z.sample_shape is not None:
            return
        z = self.encoder(x[[0]])
        self.sampler_z.sample_shape = z.shape[1:]

    def sample(self, x0=None, n_sample=None, device=None, replay=True):
        if x0 is not None:
            n_sample = len(x0)
            device = x0.device
        d_sample_z = self.sampler_z.sample(
            self.energy_z, n_sample=n_sample, device=device, T=self.T, replay=replay
        )
        z_sample = d_sample_z["sample"]
        x0 = self.decoder(z_sample)
        d_sample_x = self.sampler_x.sample(self.energy, x0=x0, T=self.T)
        d_sample = {
            "sample_x": d_sample_x["sample"],
            "sample_x0": d_sample_x["sample_0"],
            "d_sample_x": d_sample_x,
            "sample_z": d_sample_z["sample"],
            "sample_z0": d_sample_z["sample_0"],
            "d_sample_z": d_sample_z,
        }
        return d_sample

    def energy_z(self, z):
        return self.energy(self.decoder(z))

    def energy_recon(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        logvar = self.varnet(x)
        std = torch.exp(logvar * 0.5)
        recon_error = ((recon - x) ** 2).view(len(x), -1).mean(-1)
        return (
            -Normal(loc=recon, scale=std).log_prob(x).view(len(x), -1).mean(dim=-1),
            recon_error,
            logvar,
        )

    def train_step(self, x, opt, clip_grad=None):
        self._set_x_shape(x)
        self._set_z_shape(x)

        # negative sample
        d_sample = self.sample(x0=x)
        x_neg = d_sample["sample_x"]

        opt.zero_grad()
        d_pos, d_neg = self.energy(x, train=True), self.energy(x_neg, train=True)
        pos_e, neg_e = d_pos["E"], d_neg["E"]
        pos_recon_error = d_pos["recon_error"]
        neg_recon_error = d_neg["recon_error"]

        loss = pos_e.mean() - neg_e.mean()  # / self.T

        if self.gamma is not None:
            loss += self.gamma * (pos_e**2 + neg_e**2).mean()
        if self.gamma_neg is not None:
            loss += self.gamma_neg * (neg_e**2).mean()
        if self.gamma_neg_recon is not None:
            neg_recon_error = d_neg["recon_error"]
            loss += self.gamma_neg_recon * (neg_recon_error**2).mean()
        if self.gamma_var is not None:
            pos_var, neg_var = d_pos["var"], d_neg["var"]
            loss += self.gamma_var * ((pos_var**2).mean() + (neg_var**2).mean())
        if self.gamma_var2 is not None:
            neg_var = d_neg["var"]
            loss += self.gamma_var2 * ((neg_var**2).mean())
        if self.gamma_v is not None:
            pos_v, neg_v = d_pos["v"], d_neg["v"]
            loss += self.gamma_v * ((pos_v**2).mean() + (neg_v**2).mean())
        if self.gamma_v2 is not None:
            neg_v = d_pos["v"]
            loss += self.gamma_v2 * ((neg_v**2).mean())

        # weight regularization
        encoder_norm, decoder_norm, varnet_norm = self.weight_norm()
        if self.l2_norm_reg_de is not None:
            loss = loss + self.l2_norm_reg_de * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        if self.l2_norm_reg_var is not None:
            loss = loss + self.l2_norm_reg_var * varnet_norm

        loss.backward()
        opt.step()

        sample_img = make_grid(x_neg, nrow=10, range=(0, 1))
        x_neg_0 = d_sample["sample_x0"]
        with torch.no_grad():
            neg_e_x0 = self.energy(x_neg_0, train=False)
            # recon_neg = self.reconstruct(x_neg)

        d_result = {
            "nae/pos_e_": pos_e.mean().item(),
            "nae/neg_e_": neg_e.mean().item(),
            "nae/pos_recon_e_": pos_recon_error.mean().item(),
            "nae/neg_recon_e_": neg_recon_error.mean().item(),
            "cl_v/pos_": d_pos["v"].mean().item(),
            "cl_v/neg_": d_neg["v"].mean().item(),
            "cl_var/pos_": d_pos["var"].mean().item(),
            "cl_var/neg_": d_neg["var"].mean().item(),
            "x_neg": x_neg.detach().cpu(),
            "x_neg_0": d_sample["sample_x0"].detach().cpu(),
            "loss": loss.item(),
            "nae/sample_img@": sample_img.detach().cpu(),
            "l2_norm/en_": encoder_norm.item(),
            "l2_norm/de_": decoder_norm.item(),
            "l2_norm/var_": varnet_norm.item(),
            "d_sample": d_sample,
            "nae/neg_e_0_": neg_e_x0.mean().item(),
        }
        return d_result


class NAE_CLZX_base(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        varnetz,
        varnetx,
        T=1.0,
        T_trainable=False,
        varparam=None,
        encoding_noise=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.varnetx = varnetx
        self.varnetz = varnetz
        self.varparam = varparam
        self.encoding_noise = encoding_noise  # magnitude of noise in encoding
        T_ = np.log(T)
        if T_trainable:
            self.register_parameter(
                "T_", nn.Parameter(torch.tensor(T_, dtype=torch.float))
            )
        else:
            self.register_buffer("T_", torch.tensor(T_))

    @property
    def T(self):
        return torch.exp(self.T_)

    def forward(self, x, train=False):
        z = self.encoder(x)
        recon = self.decoder(z)
        vx = self.varnetx(x).flatten()
        vz = self.varnetz(z).flatten()
        if self.varparam == "expsq":
            v = vx**2 + vz**2
            var = torch.exp(v)
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / var + v
        else:
            v = vx + vz
            var = torch.exp(v)
            recon_error = (recon - x) ** 2
            E = (recon_error).view(len(x), -1).mean(dim=-1) / var + v

        if train:
            return {
                "E": E,
                "recon": recon,
                "v": v,
                "recon_error": recon_error,
                "var": var,
                "vx": vx,
                "vz": vz,
                "z": z,
            }
        else:
            return E

    def recon_error(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return ((recon - x) ** 2).view(len(x), -1).mean(dim=-1)

    def recon_error_noise(self, x):
        z = self.encoder(x)
        z += torch.randn_like(z) * self.encoding_noise
        if isinstance(self.encoder.out_activation, SphericalActivation):
            z = z / z.norm(dim=1, keepdim=True)
        recon = self.decoder(z)
        return ((recon - x) ** 2).view(len(x), -1).mean(dim=-1)

    def reconstruct(self, x):
        return self.decoder(self.encoder(x))

    def energy(self, x, train=False):
        return self.forward(x, train=train)

    def energy_T(self, x, train=False):
        return self.energy(x, train=train) / self.T

    def predict(self, x):
        return self.forward(x)

    def train_step_ae(self, x, opt, clip_grad=None):
        opt.zero_grad()
        if self.encoding_noise is not None:
            recon_error = self.recon_error_noise(x)
        else:
            recon_error = self.recon_error(x)
        loss = recon_error.mean()

        # weight regularization
        encoder_norm, decoder_norm, varnet_norm = self.weight_norm()
        if self.l2_norm_reg_de is not None:
            loss = loss + self.l2_norm_reg_de * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                opt.param_groups[0]["params"], max_norm=clip_grad
            )
        opt.step()
        d_result = {
            "loss": loss.item(),
            "recon_error_": recon_error.mean().item(),
            "l2_norm_de_": decoder_norm.item(),
            "l2_norm_en_": encoder_norm.item(),
        }
        return d_result

    def validation_step_ae(self, x, y=None, visualize=True):
        recon = self.reconstruct(x)
        recon_error = ((recon - x) ** 2).mean()
        d_result = {
            "loss": recon_error.item(),
            "predict": recon.detach().cpu().flatten(),
        }
        if visualize:
            input_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
            d_result["input@"] = input_img
            d_result["recon@"] = recon_img
        return d_result

    def weight_norm(self):
        decoder_norm = weight_norm(self.decoder)
        encoder_norm = weight_norm(self.encoder)
        varnetx_norm = weight_norm(self.varnetx)
        varnetz_norm = weight_norm(self.varnetz)
        return encoder_norm, decoder_norm, varnetz_norm, varnetx_norm


class NAE_CLZX_OMI(NAE_CLZX_base):
    """On-manifold Initialization"""

    def __init__(
        self,
        encoder,
        decoder,
        varnetz,
        varnetx,
        sampler_z,
        sampler_x,
        T=1.0,
        l2_norm_reg_en=None,
        l2_norm_reg_de=None,
        l2_norm_reg_varz=None,
        l2_norm_reg_varx=None,
        gamma=None,
        gamma_neg=None,
        gamma_neg_recon=None,
        varparam=None,
        gamma_v=None,
        gamma_v2=None,
        gamma_var=None,
        gamma_var2=None,
        encoding_noise=None,
    ):
        super().__init__(
            encoder,
            decoder,
            varnetz,
            varnetx,
            T=T,
            varparam=varparam,
            encoding_noise=encoding_noise,
        )
        self.sampler_x = sampler_x
        self.sampler_z = sampler_z
        self.l2_norm_reg_en = l2_norm_reg_en
        self.l2_norm_reg_de = l2_norm_reg_de
        self.l2_norm_reg_varx = l2_norm_reg_varx
        self.l2_norm_reg_varz = l2_norm_reg_varz
        self.gamma = gamma
        self.gamma_neg = gamma_neg
        self.gamma_neg_recon = gamma_neg_recon
        self.gamma_v = gamma_v
        self.gamma_v2 = gamma_v2
        self.gamma_var = gamma_var
        self.gamma_var2 = gamma_var2

    def _set_x_shape(self, x):
        if self.sampler_x.sample_shape is not None:
            return
        self.sampler_x.sample_shape = x.shape[1:]

    def _set_z_shape(self, x):
        if self.sampler_z.sample_shape is not None:
            return
        z = self.encoder(x[[0]])
        self.sampler_z.sample_shape = z.shape[1:]

    def sample(self, x0=None, n_sample=None, device=None, replay=True):
        if x0 is not None:
            n_sample = len(x0)
            device = x0.device
        d_sample_z = self.sampler_z.sample(
            self.energy_z, n_sample=n_sample, device=device, T=self.T, replay=replay
        )
        z_sample = d_sample_z["sample"]
        x0 = self.decoder(z_sample)
        d_sample_x = self.sampler_x.sample(self.energy, x0=x0, T=self.T)
        d_sample = {
            "sample_x": d_sample_x["sample"],
            "sample_x0": d_sample_x["sample_0"],
            "sample_z": d_sample_z["sample"],
            "sample_z0": d_sample_z["sample_0"],
        }
        return d_sample

    def energy_z(self, z):
        return self.energy(self.decoder(z))

    def train_step(self, x, opt, clip_grad=None):
        self._set_x_shape(x)
        self._set_z_shape(x)

        # negative sample
        d_sample = self.sample(x0=x)
        x_neg = d_sample["sample_x"]

        opt.zero_grad()
        d_pos, d_neg = self.energy(x, train=True), self.energy(x_neg, train=True)
        pos_e, neg_e = d_pos["E"], d_neg["E"]
        pos_recon_error = d_pos["recon_error"]
        neg_recon_error = d_neg["recon_error"]

        loss = pos_e.mean() - neg_e.mean()  # / self.T

        if self.gamma is not None:
            loss += self.gamma * (pos_e**2 + neg_e**2).mean()
        if self.gamma_neg is not None:
            loss += self.gamma_neg * (neg_e**2).mean()
        if self.gamma_neg_recon is not None:
            neg_recon_error = d_neg["recon_error"]
            loss += self.gamma_neg_recon * (neg_recon_error**2).mean()
        if self.gamma_var is not None:
            pos_var, neg_var = d_pos["var"], d_neg["var"]
            loss += self.gamma_var * ((pos_var**2).mean() + (neg_var**2).mean())
        if self.gamma_var2 is not None:
            neg_var = d_neg["var"]
            loss += self.gamma_var2 * ((neg_var**2).mean())
        if self.gamma_v is not None:
            pos_v, neg_v = d_pos["v"], d_neg["v"]
            loss += self.gamma_v * ((pos_v**2).mean() + (neg_v**2).mean())
        if self.gamma_v2 is not None:
            neg_v = d_pos["v"]
            loss += self.gamma_v2 * ((neg_v**2).mean())

        # weight regularization
        encoder_norm, decoder_norm, varnetx_norm, varnetz_norm = self.weight_norm()
        if self.l2_norm_reg_de is not None:
            loss = loss + self.l2_norm_reg_de * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        if self.l2_norm_reg_varx is not None:
            loss = loss + self.l2_norm_reg_varx * varnetx_norm
        if self.l2_norm_reg_varz is not None:
            loss = loss + self.l2_norm_reg_varz * varnetz_norm

        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                opt.param_groups[0]["params"], max_norm=clip_grad
            )
        opt.step()

        sample_img = make_grid(x_neg, nrow=10, range=(0, 1))
        x_neg_0 = d_sample["sample_x0"]
        with torch.no_grad():
            neg_e_x0 = self.energy(x_neg_0, train=False)
            # recon_neg = self.reconstruct(x_neg)

        d_result = {
            "nae/pos_e_": pos_e.mean().item(),
            "nae/neg_e_": neg_e.mean().item(),
            "nae/pos_recon_e_": pos_recon_error.mean().item(),
            "nae/neg_recon_e_": neg_recon_error.mean().item(),
            "cl_v/pos_": d_pos["v"].mean().item(),
            "cl_v/neg_": d_neg["v"].mean().item(),
            "cl_vx/pos_": d_pos["vx"].mean().item(),
            "cl_vx/neg_": d_neg["vx"].mean().item(),
            "cl_vz/pos_": d_pos["vz"].mean().item(),
            "cl_vz/neg_": d_neg["vz"].mean().item(),
            "cl_var/pos_": d_pos["var"].mean().item(),
            "cl_var/neg_": d_neg["var"].mean().item(),
            "x_neg": x_neg.detach().cpu(),
            "x_neg_0": d_sample["sample_x0"].detach().cpu(),
            "loss": loss.item(),
            "nae/sample_img@": sample_img.detach().cpu(),
            "l2_norm/en_": encoder_norm.item(),
            "l2_norm/de_": decoder_norm.item(),
            "l2_norm/varx_": varnetx_norm.item(),
            "l2_norm/varz_": varnetz_norm.item(),
            "d_sample": d_sample,
            "nae/neg_e_0_": neg_e_x0.mean().item(),
        }
        return d_result

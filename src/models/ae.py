"""
ae.py
=====
Autoencoders
"""
import warnings

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.utils import make_grid

from models.energybased import SampleBuffer
from models.mcmc import sample_langevin
from models.modules import (
    ConvClassifier,
    ConvMLP,
    ConvNet1,
    ConvNet2,
    ConvNet3,
    DCGANDecoder,
    DCGANDecoder2,
    DCGANDecoder_Resnet,
    DCGANEncoder,
    DCGANEncoder2,
    DeConvNet1,
    DeConvNet2,
    DeConvNet3,
    FCNet,
    IsotropicGaussian,
    IsotropicLaplace,
    ResNetEncoder18,
    ResNetEncoder34,
    ResNetEncoder50,
    ResNetEncoder101,
    ResNetEncoder152,
)


class GlobalConvNet(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(GlobalConvNet, self).__init__()
        self.in_chan, self.out_chan = in_chan, out_chan
        self.main = [
            nn.Conv2d(in_chan, in_chan, 4, 4, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_chan, out_chan, 4, 4, bias=True),
            nn.ReLU(),
            # nn.AdaptiveMaxPool2d(1),
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


class FullyClassifier(nn.Module):
    def __init__(self, in_chan, h_dim=128):
        super(FullyClassifier, self).__init__()
        self.main = [
            nn.Conv2d(in_chan, h_dim, 1),
            nn.ReLU(),
            nn.Conv2d(h_dim, 1, 1),
            nn.Sigmoid(),
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


def get_dagmm(gpu=True, n_chans=1, backbone="cnn", **kwargs):
    z_dim = kwargs.get("z_dim", 5)
    n_gmm = kwargs.get("n_gmm", 5)
    lambda_energy = kwargs.get("lambda_energy", 0.1)
    lambda_cov_diag = kwargs.get("lambda_cov_diag", 0.005)
    grad_clip = kwargs.get("grad_clip", True)
    rec_cosine = kwargs.get("rec_cosine", True)
    rec_euclidean = kwargs.get("rec_euclidean", True)
    likelihood_type = kwargs.get("likelihood_type", "isotropic_gaussian")

    if backbone == "cnn":
        # encoder = ConvNet(in_chan=n_chans, out_chan=z_dim)
        # decoder = DeConvNet(in_chan=z_dim, out_chan=n_chans)
        pass
    elif backbone == "cnn2":
        nh = kwargs.get("nh", 8)
        n_hidden = kwargs.get("n_hidden", 1024)
        out_activation = kwargs.get("out_activation", None)
        encoder = ConvNet2(
            in_chan=n_chans, out_chan=z_dim, nh=nh, out_activation=out_activation
        )
        decoder = DeConvNet2(
            in_chan=z_dim, out_chan=n_chans, nh=nh, likelihood_type=likelihood_type
        )
        latent_dim = z_dim
        print(rec_cosine, rec_euclidean)
        if rec_cosine:
            latent_dim += 1
        if rec_euclidean:
            latent_dim += 1

        estimator = FCNet(
            in_dim=latent_dim,
            out_dim=n_gmm,
            l_hidden=(n_hidden,),
            activation="sigmoid",
            out_activation="softmax",
        )
    elif backbone == "cnn3":
        nh = kwargs.get("nh", 8)
        # encoder = ConvNet3(in_chan=n_chans, out_chan=z_dim, nh=nh)
        # decoder = DeConvNet3(in_chan=z_dim, out_chan=n_chans, nh=nh)
    elif backbone == "dcgan":
        ndf = kwargs.get("ndf", 64)
        ngf = kwargs.get("ngf", 64)
        # encoder = DCGANEncoder(in_chan=n_chans, out_chan=z_dim, ndf=ndf)
        # decoder = DCGANDecoder(in_chan=z_dim, out_chan=n_chans, ngf=ngf)
    elif backbone == "dcgan2":
        pass
        # encoder = DCGANEncoder2(in_chan=n_chans, out_chan=z_dim)
        # decoder = DCGANDecoder2(in_chan=z_dim, out_chan=n_chans)
    elif backbone == "FC":  # fully connected
        n_hidden = kwargs.get("n_hidden", 1024)
        # encoder = FCNet(in_dim=n_chans, out_dim=z_dim, l_hidden=(n_hidden,), activation='relu', out_activation='tanh')
        # decoder = FCNet(in_dim=z_dim, out_dim=n_chans, l_hidden=(n_hidden,), activation='relu', out_activation='linear')
    else:
        raise ValueError(f"Invalid argument backbone: {backbone}")

    return DaGMM(
        encoder,
        decoder,
        estimator,
        gpu=gpu,
        lambda_energy=lambda_energy,
        lambda_cov_diag=lambda_cov_diag,
        grad_clip=grad_clip,
        rec_cosine=rec_cosine,
        rec_euclidean=rec_euclidean,
    )


class AE(nn.Module):
    """autoencoder"""

    def __init__(self, encoder, decoder):
        """
        encoder, decoder : neural networks
        """
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.own_optimizer = False

    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        z = self.encoder(x)
        return z

    def predict(self, x):
        """one-class anomaly prediction"""
        recon = self(x)
        if hasattr(self.decoder, "error"):
            predict = self.decoder.error(x, recon)
        else:
            predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return predict

    def predict_and_reconstruct(self, x):
        recon = self(x)
        if hasattr(self.decoder, "error"):
            recon_err = self.decoder.error(x, recon)
        else:
            recon_err = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return recon_err, recon

    def validation_step(self, x, **kwargs):
        recon = self(x)
        if hasattr(self.decoder, "error"):
            predict = self.decoder.error(x, recon)
        else:
            predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        loss = predict.mean()

        if kwargs.get("show_image", True):
            x_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        else:
            x_img, recon_img = None, None
        return {
            "loss": loss.item(),
            "predict": predict,
            "reconstruction": recon,
            "input@": x_img,
            "recon@": recon_img,
        }

    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        optimizer.zero_grad()
        recon_error = self.predict(x)
        loss = recon_error.mean()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()
        return {"loss": loss.item()}

    def reconstruct(self, x):
        return self(x)

    def sample(self, N, z_shape=None, device="cpu"):
        if z_shape is None:
            z_shape = self.encoder.out_shape

        rand_z = torch.rand(N, *z_shape).to(device) * 2 - 1
        sample_x = self.decoder(rand_z)
        return sample_x


class EBAE_V0(AE):
    """sampling on x space"""

    def __init__(
        self,
        encoder,
        decoder,
        step_size=50,
        sample_step=100,
        noise_std=0.2,
        noise_decay=None,
        sigma=0.001,
        sigma_trainable=False,
        x_bound_1=None,
        x_bound_2=None,
        neg_error_clip=None,
    ):
        super(EBAE_V0, self).__init__(
            encoder,
            IsotropicGaussian(decoder, sigma=sigma, sigma_trainable=sigma_trainable),
        )
        self.step_size = step_size
        self.sample_step = sample_step
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.x_bound_1 = x_bound_1
        self.x_bound_2 = x_bound_2
        self.neg_error_clip = neg_error_clip

    def sample(
        self,
        x_shape,
        device="cpu",
        intermediate=False,
        sample_step=None,
        step_size=None,
        noise_std=None,
    ):
        if sample_step is None:
            sample_step = self.sample_step
        if step_size is None:
            step_size = self.step_size
        if noise_std is None:
            noise_std = self.noise_std

        l_samples = []
        # x = torch.zeros(*x_shape, dtype=torch.float).to(device)
        x = torch.rand(*x_shape, dtype=torch.float).to(device)
        x.requires_grad = True
        for i_step in range(sample_step):
            l_samples.append(x.detach().cpu())
            E = self.predict(x)
            grad = autograd.grad(E.sum(), x, only_inputs=True)[0]
            if self.noise_decay is None:
                x = x - step_size * grad + torch.randn_like(x) * noise_std
            else:
                x = (
                    x
                    - step_size * grad
                    + torch.randn_like(x) * noise_std / (i_step + self.noise_decay)
                )

            if self.x_bound_2 is not None:
                x = torch.clamp(x, self.x_bound_2[0], self.x_bound_2[1])
        l_samples.append(x.detach().cpu())
        if intermediate:
            return x.detach(), torch.stack(l_samples)
        else:
            return x.detach()

    def train_step(self, x, opt):

        # negative sample
        x_neg = self.sample(x.shape, device=x.device)
        if self.x_bound_1 is not None:
            x_neg = torch.clamp(x_neg, self.x_bound_1[0], self.x_bound_1[1])

        opt.zero_grad()
        recon_neg = self.decoder(self.encoder(x_neg))
        # neg_e = self.decoder.square_error(x_neg, recon_neg)
        neg_e = (x_neg - recon_neg) ** 2
        if self.neg_error_clip is not None:
            neg_e = torch.clamp(neg_e, 0, self.neg_error_clip)

        # ae recon pass
        z = self.encoder(x)
        recon = self.decoder(z)
        pos_e = self.decoder.square_error(x, recon)

        loss = pos_e.mean() - neg_e.mean()
        loss.backward()
        opt.step()
        d_result = {
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "loss": loss.item(),
            "sample": x_neg.detach().cpu(),
        }
        return d_result


def clip_vector_norm(x, max_norm):
    norm = x.norm(dim=-1, keepdim=True)
    x = x * (
        (norm < max_norm).to(torch.float)
        + (norm > max_norm).to(torch.float) * max_norm / norm
        + 1e-6
    )
    return x


class EBAE_V1(AE):
    """Two-stage sampling"""

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
        x_noise_anneal=1,
        sigma=0.001,
        sigma_trainable=False,
        x_bound=(0, 1),
        z_bound=None,
        z_clip_langevin_grad=None,
        x_clip_langevin_grad=0.01,
        l2_norm_reg=None,
        l2_norm_reg_en=None,
        spherical=True,
        buffer_size=10000,
        replay_ratio=0.95,
        replay=True,
        gamma=1,
        recon_error="l2",
        ebae_coef=1.0,
        sampling="v1",
        error_scale=1.0,
    ):
        if recon_error == "l2":
            decoder = IsotropicGaussian(
                decoder, sigma=sigma, sigma_trainable=sigma_trainable
            )
        elif recon_error == "l1":
            decoder = IsotropicLaplace(
                decoder, sigma=sigma, sigma_trainable=sigma_trainable
            )
        elif recon_error == "asis":
            pass
        else:
            raise ValueError(f"{recon_error}")

        super(EBAE_V1, self).__init__(encoder, decoder)
        self.z_step = z_step
        self.z_stepsize = z_stepsize
        self.z_noise_std = z_noise_std
        self.z_noise_anneal = z_noise_anneal
        self.z_clip_langevin_grad = z_clip_langevin_grad
        self.x_step = x_step
        self.x_stepsize = x_stepsize
        self.x_noise_std = x_noise_std
        self.x_noise_anneal = x_noise_anneal
        self.x_clip_langevin_grad = x_clip_langevin_grad

        self.x_bound = x_bound
        self.z_bound = z_bound
        self.l2_norm_reg = l2_norm_reg  # decoder
        self.l2_norm_reg_en = l2_norm_reg_en
        self.spherical = spherical
        self.gamma = gamma
        self.ebae_coef = ebae_coef
        self.sampling = sampling
        self.error_scale = error_scale

        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.replay = replay

        bound = "spherical" if self.spherical else z_bound
        self.buffer = SampleBuffer(
            max_samples=buffer_size, replay_ratio=replay_ratio, bound=bound
        )
        self.buffer_x = SampleBuffer(
            max_samples=buffer_size, replay_ratio=1, bound=None
        )

        self.z_shape = None
        self.x_shape = None

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

    def encode(self, x, wo_att=False):
        if wo_att:
            return self.normalize(self.encoder.forward_wo_att(x))
        else:
            return self.normalize(self.encoder(x))

    def sample(self, *args, **kwargs):
        if self.sampling == "v1":
            return self.sample_v1(*args, **kwargs)
        elif self.sampling == "v2":
            return self.sample_v2(*args, **kwargs)

    def sample_v1(self, n_sample, device, replay=False):
        z0 = self.buffer.sample(
            (n_sample,) + self.z_shape, device=device, replay=replay
        )

        energy = lambda z: self.predict(self.decoder(z)) * self.error_scale
        sample_z = sample_langevin(
            z0,
            energy,
            stepsize=self.z_stepsize,
            n_step=self.z_step,
            noise_scale=self.z_noise_std,
            clip_x=None,
            clip_grad=self.z_clip_langevin_grad,
            intermediate_samples=False,
            spherical=self.spherical,
        )
        sample_x_1 = self.decoder(sample_z).detach()

        if replay:
            self.buffer.push(sample_z)

        x_energy = lambda x: self.predict(x) * self.error_scale
        sample_x_2 = sample_langevin(
            sample_x_1.detach(),
            x_energy,
            stepsize=self.x_stepsize,
            n_step=self.x_step,
            noise_scale=self.x_noise_std,
            intermediate_samples=False,
            clip_x=self.x_bound,
            noise_anneal=self.x_noise_anneal,
            clip_grad=self.x_clip_langevin_grad,
            spherical=False,
        )
        return {
            "sample_x": sample_x_2,
            "sample_z": sample_z.detach(),
            "sample_x0": sample_x_1,
        }

    def sample_v2(self, n_sample, device, replay=None):
        """use sample replay buffer in x space
        for replay_ratio, run only x chain
        for 1 - replay_ratio, start from z"""
        if len(self.buffer_x) == 0:
            n_replay = 0
        else:
            n_replay = (torch.rand(n_sample) < self.replay_ratio).sum()

        """start from latent chain"""
        n_from_z = n_sample - n_replay
        if n_from_z > 0:
            z0 = self.buffer.random((n_from_z,) + self.z_shape, device=device)

            energy = lambda z: self.predict(self.decoder(z))
            sample_z = sample_langevin(
                z0,
                energy,
                stepsize=self.z_stepsize,
                n_step=self.z_step,
                noise_scale=self.z_noise_std,
                clip_x=None,
                clip_grad=self.z_clip_langevin_grad,
                intermediate_samples=False,
                spherical=self.spherical,
            )
            sample_x_1 = self.decoder(sample_z).detach()
        else:
            sample_z = torch.tensor([]).to(device)
            sample_x_1 = torch.tensor([]).to(device)

        """start from x buffer"""
        if n_replay > 0:
            x0 = self.buffer_x.sample(
                (n_replay,) + self.x_shape, device=device, replay=True
            )
        else:
            x0 = torch.tensor([]).to(device)
        x0 = torch.cat([x0, sample_x_1]).detach()

        x_energy = lambda x: self.predict(x)
        sample_x_2 = sample_langevin(
            x0.detach(),
            x_energy,
            stepsize=self.x_stepsize,
            n_step=self.x_step,
            noise_scale=self.x_noise_std,
            intermediate_samples=False,
            clip_x=self.x_bound,
            noise_anneal=self.x_noise_anneal,
            clip_grad=self.x_clip_langevin_grad,
            spherical=False,
        )
        self.buffer_x.push(sample_x_2)
        return {"sample_x": sample_x_2, "sample_z": sample_z.detach(), "sample_x0": x0}

    def sample_z(
        self, n_sample, device, replay=False, z0=None, intermediate_samples=False
    ):
        if z0 is None:
            z0 = self.buffer.sample(
                (n_sample,) + self.z_shape, device=device, replay=replay
            )

        energy = lambda z: self.predict(self.decoder(z))
        sample_result = sample_langevin(
            z0,
            energy,
            stepsize=self.z_stepsize,
            n_steps=self.z_step,
            noise_scale=self.z_noise_std,
            clip_x=None,
            clip_grad=self.z_clip_langevin_grad,
            intermediate_samples=intermediate_samples,
            spherical=self.spherical,
        )
        # if intermediate_samples = True
        #     sample_result = (l_samples, l_dynamics)
        # if intermediate_samples = False
        #     sample_result = sample_z
        if replay:
            self.buffer.push(sample_z)
        return sample_result

    def sample_x(self, n_sample, device, x0, intermediate_samples=False):
        x_energy = lambda x: self.predict(x)
        sample_result = sample_langevin(
            x0.detach(),
            x_energy,
            stepsize=self.x_stepsize,
            n_steps=self.x_step,
            noise_scale=self.x_noise_std,
            intermediate_samples=intermediate_samples,
            clip_x=self.x_bound,
            noise_anneal=self.x_noise_anneal,
            clip_grad=self.x_clip_langevin_grad,
            spherical=False,
        )
        return sample_result

    def _set_z_shape(self, x):
        if self.z_shape is not None:
            return
        # infer z_shape by computing forward
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        z_shape = dummy_z.shape
        # self.register_buffer('z_shape', z_shape[1:])
        self.z_shape = z_shape[1:]

    def _set_x_shape(self, x):
        if self.x_shape is not None:
            return
        self.x_shape = x.shape[1:]

    def weight_norm(self, net):
        norm = 0
        for param in net.parameters():
            norm += (param**2).sum()
        return norm

    def train_step_ae(self, x, opt, clip_grad=None):
        opt.zero_grad()
        z = self.encode(x)
        recon = self.decoder(z)
        error = self.decoder.error(x, recon)
        z_norm = (z**2).mean()
        recon_error = error.mean()
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

    def train_step(self, x, opt):
        self._set_z_shape(x)
        self._set_x_shape(x)

        # negative sample
        d_sample = self.sample(len(x), x.device, replay=self.replay)
        x_neg = d_sample["sample_x"]

        opt.zero_grad()
        z_neg = self.encode(x_neg)
        recon_neg = self.decoder(z_neg)
        neg_e = self.decoder.error(x_neg, recon_neg)

        # ae recon pass
        z = self.encode(x)
        recon = self.decoder(z)
        pos_e = self.decoder.error(x, recon)

        loss = pos_e.mean() - neg_e.mean() * self.ebae_coef

        if self.gamma is not None:
            loss += self.gamma * (neg_e**2).mean()

        # regularization
        z_norm = (z**2).mean()
        z_neg_norm = (z_neg**2).mean()

        # weight regularization
        decoder_norm = self.weight_norm(self.decoder)
        encoder_norm = self.weight_norm(self.encoder)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm

        loss.backward()
        opt.step()

        # for debugging
        x_neg_0 = d_sample["sample_x0"]
        neg_e_0 = self.predict(x_neg_0)  # energy of samples from latent chain
        d_result = {
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "x_neg": x_neg.detach().cpu(),
            "recon_neg": recon_neg.detach().cpu(),
            "loss": loss.item(),
            "sample": x_neg.detach().cpu(),
            "z_norm": z_norm.item(),
            "z_neg_norm": z_neg_norm.item(),
            "decoder_norm": decoder_norm.item(),
            "encoder_norm": encoder_norm.item(),
            "neg_e_0": neg_e_0.mean().item(),
        }
        return d_result


class EBAE_V2(AE):
    """Use conditional perturbation"""

    def __init__(
        self,
        encoder,
        decoder,
        perturb,
        att,
        z_step=50,
        z_stepsize=0.2,
        z_noise_std=0.2,
        z_noise_anneal=None,
        x_step=50,
        x_stepsize=10,
        x_noise_std=0.05,
        x_noise_anneal=1,
        sigma=0.001,
        sigma_trainable=False,
        x_bound=(0, 1),
        z_bound=None,
        z_clip_langevin_grad=None,
        x_clip_langevin_grad=0.01,
        l2_norm_reg=None,
        l2_norm_reg_en=None,
        spherical=True,
        buffer_size=10000,
        replay_ratio=0.95,
        replay=True,
        gamma=1,
    ):
        super(EBAE_V2, self).__init__(
            encoder,
            IsotropicGaussian(decoder, sigma=sigma, sigma_trainable=sigma_trainable),
        )
        self.perturb = perturb
        self.att = att

        self.z_step = z_step
        self.z_stepsize = z_stepsize
        self.z_noise_std = z_noise_std
        self.z_noise_anneal = z_noise_anneal
        self.z_clip_langevin_grad = z_clip_langevin_grad
        self.x_step = x_step
        self.x_stepsize = x_stepsize
        self.x_noise_std = x_noise_std
        self.x_noise_anneal = x_noise_anneal
        self.x_clip_langevin_grad = x_clip_langevin_grad

        self.x_bound = x_bound
        self.z_bound = z_bound
        self.l2_norm_reg = l2_norm_reg  # decoder
        self.l2_norm_reg_en = l2_norm_reg_en
        self.spherical = spherical
        self.gamma = gamma

        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.replay = replay

        bound = "spherical" if self.spherical else z_bound
        self.buffer = SampleBuffer(
            max_samples=buffer_size, replay_ratio=replay_ratio, bound=bound
        )

        self.z_shape = None

        self.ebae = False  # flag for switching between AE-mode and EBAE-mode

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
        if self.ebae:
            z = self.normalize(self.encoder(x))
            att = self.att(z)
            perturb = self.perturb(z)
            return self.normalize(z + torch.sigmoid(att) * perturb)
        else:
            return self.normalize(self.encoder(x))

    def sample(self, n_sample, device, replay=False):
        z0 = self.buffer.sample(
            (n_sample,) + self.z_shape, device=device, replay=replay
        )

        energy = lambda z: self.predict(self.decoder(z))
        sample_z = sample_langevin(
            z0,
            energy,
            stepsize=self.z_stepsize,
            n_steps=self.z_step,
            noise_scale=self.z_noise_std,
            clip_x=None,
            clip_grad=self.z_clip_langevin_grad,
            intermediate_samples=False,
            spherical=self.spherical,
        )
        sample_x_1 = self.decoder(sample_z).detach()

        self.buffer.push(sample_z)

        x_energy = lambda x: self.predict(x)
        sample_x_2 = sample_langevin(
            sample_x_1.detach(),
            x_energy,
            stepsize=self.x_stepsize,
            n_steps=self.x_step,
            noise_scale=self.x_noise_std,
            intermediate_samples=False,
            clip_x=self.x_bound,
            noise_anneal=self.x_noise_anneal,
            clip_grad=self.x_clip_langevin_grad,
            spherical=False,
        )
        return {
            "sample_x": sample_x_2,
            "sample_z": sample_z.detach(),
            "sample_x0": sample_x_1,
        }

    def _set_z_shape(self, x):
        if self.z_shape is not None:
            return
        # infer z_shape by computing forward
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        z_shape = dummy_z.shape
        # self.register_buffer('z_shape', z_shape[1:])
        self.z_shape = z_shape[1:]

    def weight_norm(self, net):
        norm = 0
        for param in net.parameters():
            norm += (param**2).sum()
        return norm

    def train_step_ae(self, x, opt, clip_grad=None):
        opt.zero_grad()
        z = self.encode(x)
        recon = self.decoder(z)
        error = (x - recon) ** 2
        z_norm = (z**2).mean()
        recon_error = error.mean()
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
            "ae/z_norm_": z_norm.item(),
            "ae/recon_error_": recon_error.item(),
            "ae/decoder_norm_": decoder_norm.item(),
            "ae/encoder_norm_": encoder_norm.item(),
        }
        return d_result

    def train_step(self, x, opt):
        self._set_z_shape(x)

        # negative sample
        d_sample = self.sample(len(x), x.device, replay=self.replay)
        x_neg = d_sample["sample_x"]

        opt.zero_grad()
        z_neg = self.encode(x_neg)
        recon_neg = self.decoder(z_neg)
        neg_e = self.decoder.square_error(x_neg, recon_neg)

        # ae recon pass
        z = self.encode(x)
        recon = self.decoder(z)
        pos_e = self.decoder.square_error(x, recon)

        loss = pos_e.mean() - neg_e.mean()

        if self.gamma is not None:
            loss += self.gamma * (neg_e**2).mean()

        # regularization
        z_norm = (z**2).mean()
        z_neg_norm = (z_neg**2).mean()

        # weight regularization
        decoder_norm = self.weight_norm(self.decoder)
        encoder_norm = self.weight_norm(self.encoder)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm

        loss.backward()
        opt.step()
        d_result = {
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "x_neg": x_neg.detach().cpu(),
            "recon_neg": recon_neg.detach().cpu(),
            "loss": loss.item(),
            "sample": x_neg.detach().cpu(),
            "z_norm": z_norm.item(),
            "z_neg_norm": z_neg_norm.item(),
            "decoder_norm": decoder_norm.item(),
            "encoder_norm": encoder_norm.item(),
        }
        return d_result


class EBAE_V3(EBAE_V1):
    """temperature as learnable parameter"""

    def __init__(self, encoder, decoder, **kwargs):
        super(EBAE_V3, self).__init__(encoder, decoder, **kwargs)
        self.register_parameter(
            "temperature_", nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        )  # actually 1/T value

    @property
    def temperature(self):
        return torch.exp(self.temperature_)

    def sample_v1(self, n_sample, device, replay=False):
        z0 = self.buffer.sample(
            (n_sample,) + self.z_shape, device=device, replay=replay
        )

        energy = (
            lambda z: self.predict(self.decoder(z))
            * self.temperature
            * self.error_scale
        )
        sample_z = sample_langevin(
            z0,
            energy,
            stepsize=self.z_stepsize,
            n_steps=self.z_step,
            noise_scale=self.z_noise_std,
            clip_x=None,
            clip_grad=self.z_clip_langevin_grad,
            intermediate_samples=False,
            spherical=self.spherical,
        )
        sample_x_1 = self.decoder(sample_z).detach()

        if replay:
            self.buffer.push(sample_z)

        x_energy = lambda x: self.predict(x) * self.temperature * self.error_scale
        sample_x_2 = sample_langevin(
            sample_x_1.detach(),
            x_energy,
            stepsize=self.x_stepsize,
            n_steps=self.x_step,
            noise_scale=self.x_noise_std,
            intermediate_samples=False,
            clip_x=self.x_bound,
            noise_anneal=self.x_noise_anneal,
            clip_grad=self.x_clip_langevin_grad,
            spherical=False,
        )
        return {
            "sample_x": sample_x_2,
            "sample_z": sample_z.detach(),
            "sample_x0": sample_x_1,
        }

    def train_step(self, x, opt):
        self._set_z_shape(x)
        self._set_x_shape(x)

        # negative sample
        d_sample = self.sample(len(x), x.device, replay=self.replay)
        x_neg = d_sample["sample_x"]

        opt.zero_grad()
        z_neg = self.encode(x_neg)
        recon_neg = self.decoder(z_neg)
        neg_e = self.decoder.error(x_neg, recon_neg)

        # ae recon pass
        z = self.encode(x)
        recon = self.decoder(z)
        pos_e = self.decoder.error(x, recon)

        loss = (pos_e.mean() - neg_e.mean() * self.ebae_coef) * self.temperature

        if self.gamma is not None:
            loss += self.gamma * (neg_e**2).mean()

        # regularization
        z_norm = (z**2).mean()
        z_neg_norm = (z_neg**2).mean()

        # weight regularization
        decoder_norm = self.weight_norm(self.decoder)
        encoder_norm = self.weight_norm(self.encoder)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm

        loss.backward()
        opt.step()

        # for debugging
        x_neg_0 = d_sample["sample_x0"]
        neg_e_0 = self.predict(x_neg_0)  # energy of samples from latent chain
        d_result = {
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "x_neg": x_neg.detach().cpu(),
            "recon_neg": recon_neg.detach().cpu(),
            "loss": loss.item(),
            "sample": x_neg.detach().cpu(),
            "z_norm": z_norm.item(),
            "z_neg_norm": z_neg_norm.item(),
            "decoder_norm": decoder_norm.item(),
            "encoder_norm": encoder_norm.item(),
            "neg_e_0": neg_e_0.mean().item(),
            "temperature": self.temperature.item(),
        }
        return d_result


class EBAE_old(AE):
    def __init__(
        self,
        encoder,
        decoder,
        prior,
        alpha=1.0,
        step_size=10,
        sample_step=60,
        noise_std=0.005,
        buffer_size=10000,
        replay_ratio=0.95,
        langevin_clip_grad=0.01,
        z_bound=(0, 1),
        z_bound_2=(0, 1),
        neg_error_clip=0.1,
        reflect=False,
        sigma=0.1,
        sigma_trainable=True,
    ):
        super(EBAE_old, self).__init__(
            encoder,
            IsotropicGaussian(decoder, sigma=sigma, sigma_trainable=sigma_trainable),
        )
        self.own_optimizer = False
        self.prior = prior
        self.alpha = alpha
        self.step_size = step_size
        self.sample_step = sample_step
        self.noise_std = noise_std
        self.buffer_size = buffer_size
        self.buffer = SampleBuffer(max_samples=buffer_size, bound=z_bound_2)
        self.replay_ratio = replay_ratio
        self.replay = True if self.replay_ratio > 0 else False
        self.langevin_clip_grad = langevin_clip_grad
        self.bound = z_bound
        self.bound_2 = z_bound_2  # clamp sample after langevin dynamics
        self.neg_error_clip = neg_error_clip
        self.reflect = reflect

        self.z_shape = None  # tuple. shape excluding batch dimension

    def _set_z_shape(self, x):
        if self.z_shape is not None:
            return
        # infer z_shape by computing forward
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        z_shape = dummy_z.shape
        self.z_shape = z_shape[1:]

    def train_step(
        self,
        x,
        d_optimizer,
        clip_grad_ae=None,
        clip_grad_prior=None,
        clip_grad_neg=None,
    ):
        self.eval()
        self._set_z_shape(x)
        self.train()
        ae_opt = d_optimizer["ae"]
        prior_opt = d_optimizer["prior"]
        neg_opt = d_optimizer["neg"]

        # sampling
        z_shape = (len(x),) + self.z_shape
        neg_z = self.sample_prior(
            z_shape, x.device, intermediate=False, replay=self.replay
        )

        # ae negative pass
        # self.eval()
        neg_opt.zero_grad()
        if self.bound_2 is not None:
            neg_z = torch.clamp(neg_z, self.bound_2[0], self.bound_2[1])
        with torch.no_grad():
            neg_x = self.decoder.sample(neg_z)
        neg_recon = self.decoder(self.encoder(neg_x))
        # neg_recon_err = self.decoder.square_error(neg_x, neg_recon)
        # neg_loss = - torch.mean(torch.clamp((neg_recon - neg_images) ** 2, 0, 1))
        if self.neg_error_clip is not None:
            neg_e = torch.mean(
                torch.clamp((neg_recon - neg_x) ** 2, 0, self.neg_error_clip)
            )
            # neg_e = torch.mean(torch.clamp(neg_recon_err, 0, self.neg_error_clip))
            # error_clip = self.decoder.sigma.cpu().data * self.neg_error_clip
            # neg_e = torch.mean(torch.clamp(neg_recon_err, 0, error_clip))
        else:
            neg_recon_err = self.decoder.square_error(neg_x, neg_recon)
            neg_e = torch.mean(neg_recon_err)
        neg_obj = -neg_e
        neg_obj.backward()
        if clip_grad_neg is not None:
            torch.nn.utils.clip_grad_norm_(
                neg_opt.param_groups[0]["params"], max_norm=clip_grad_neg
            )
        neg_opt.step()

        # ae recon pass
        # self.train()
        ae_opt.zero_grad()
        z = self.encoder(x)
        recon = self.decoder(z)
        # pos_e = torch.mean((recon - x) ** 2) / (self.sigma ** 2)
        pos_e = self.decoder.square_error(x, recon)
        pos_e_max = pos_e.max().cpu().data
        pos_e = pos_e.mean()
        pos_e.backward()
        ae_opt.step()

        # prior pass
        prior_opt.zero_grad()
        prior_pos_e = self.prior(z.detach())
        prior_neg_e = self.prior(neg_z)
        prior_loss = (prior_pos_e).mean() - (prior_neg_e).mean()
        ebm_reg = (prior_pos_e**2).mean() + (prior_neg_e**2).mean()
        prior_obj = prior_loss + self.alpha * ebm_reg
        prior_obj.backward()
        if clip_grad_prior is not None:
            torch.nn.utils.clip_grad_norm_(
                self.prior.parameters(), max_norm=clip_grad_prior
            )
        prior_opt.step()

        ae_loss = pos_e - neg_e
        loss = ae_loss + prior_loss
        # loss.backward()
        # if clip_grad is not None:
        #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        # optimizer.step()

        return {
            "loss": loss.item(),
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "ae_loss": ae_loss.item(),
            "prior_loss": prior_loss.item(),
            "prior_pos_e": prior_pos_e.mean().item(),
            "prior_neg_e": prior_neg_e.mean().item(),
            "ebm_reg": ebm_reg.item(),
        }

    def train_step_z(self, x, optimizer, clip_grad=None):
        with torch.no_grad():
            z = self.encoder(x)
        neg_z = self.sample_prior(
            z.shape, z.device, intermediate=False, replay=self.replay
        )
        optimizer.zero_grad()
        pos_e = self.prior(z)
        neg_e = self.prior(neg_z)

        # prior_loss = (pos_e - neg_e).mean()
        # ebm_reg = (pos_e ** 2 + neg_e ** 2).mean()
        prior_loss = pos_e.mean() - neg_e.mean()
        ebm_reg = (pos_e**2).mean() + (neg_e**2).mean()
        loss = prior_loss + self.alpha * ebm_reg

        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.prior.parameters(), max_norm=clip_grad)
        optimizer.step()
        return {
            "loss": loss.item(),
            "pos_e": pos_e.mean().item(),
            "neg_e": neg_e.mean().item(),
            "prior_loss": prior_loss.item(),
            "ebm_reg": ebm_reg.item(),
        }

    def sample_prior(
        self,
        shape,
        device,
        intermediate=False,
        replay=True,
        step_size=None,
        sample_step=None,
    ):
        if step_size is None:
            step_size = self.step_size
        if sample_step is None:
            sample_step = self.sample_step
        # initialize
        x0 = self.buffer.sample(shape, device, replay=replay)
        # run langevin
        sample_x = sample_langevin(
            x0,
            self.prior,
            step_size,
            sample_step,
            noise_scale=self.noise_std,
            intermediate_samples=intermediate,
            clip_x=self.bound,
            clip_grad=self.langevin_clip_grad,
            reflect=self.reflect,
        )
        # push samples
        if replay:
            self.buffer.push(sample_x)
        return sample_x

    def sample(self, example_input=None, z_shape=None, device=None, replay=False):
        if example_input is not None:
            # infer z_shape by computing forward
            with torch.no_grad():
                dummy_z = self.encode(example_input[[0]])
                z_shape = dummy_z.shape
                device = example_input.device
                z_shape = (len(example_input),) + z_shape[1:]
        assert z_shape is not None and device is not None
        z = self.sample_prior(z_shape, device, intermediate=False, replay=replay)
        z = torch.clamp(z, self.bound_2[0], self.bound_2[1])
        return self.decoder.sample(z)

    def validation_step(self, x):
        pass


class KernelEntropyAE(AE):
    def __init__(self, encoder, decoder, reg=0.0, h=0.5):
        super(KernelEntropyAE, self).__init__(encoder, decoder)
        self.reg = reg
        self.h = h

    def train_step(self, x, optimizer):
        optimizer.zero_grad()

        z = self.encoder(x)
        recon = self.decoder(z)
        recon_loss = ((recon - x) ** 2).mean()
        entropy = self.entropy(z)

        loss = recon_loss - self.reg * entropy
        loss.backward()
        optimizer.step()
        return {"loss": recon_loss, "entropy": entropy}

    def entropy(self, x):
        """kernel biased estimate of entropy"""
        # x: (n_batch, n_chan, 1, 1)
        x_ = torch.squeeze(torch.squeeze(x, dim=3), dim=2)
        D = x_.shape[1]
        pdist = self.pdist(x_)
        K = torch.exp(-pdist / 2 / self.h**2) / ((np.sqrt(2 * np.pi) * (self.h)) ** D)
        return -torch.mean(torch.log(K.mean(dim=1)))

    def cdist(self, X, Z):
        """pairwise squared euclidean distance"""
        t1 = torch.diag(X.mm(X.t()))[:, None]
        t2 = torch.diag(Z.mm(Z.t()))[:, None]
        t3 = X.mm(Z.t())
        return (
            torch.mm(t1, torch.ones_like(t2.t()))
            + torch.mm(torch.ones_like(t1), t2.t())
            - t3 * 2
        )

    def pdist(self, X):
        """pairwise squared euclidean distance"""
        t3 = X.mm(X.t())
        sq = torch.diag(t3)[:, None]
        t1 = torch.mm(sq, torch.ones_like(sq.t()))
        return t1 + t1.t() - t3 * 2


class DAE(AE):
    """denoising autoencoder"""

    def __init__(self, encoder, decoder, sig=0.0, noise_type="gaussian"):
        super(DAE, self).__init__(encoder, decoder)
        self.sig = sig
        self.noise_type = noise_type

    def train_step(self, x, optimizer, y=None):
        optimizer.zero_grad()
        if self.noise_type == "gaussian":
            noise = torch.randn(*x.shape, dtype=torch.float32)
            noise = noise.to(x.device)
            recon = self(x + self.sig * noise)
        elif self.noise_type == "saltnpepper":
            x = self.salt_and_pepper(x)
            recon = self(x)
        else:
            raise ValueError(f"Invalid noise_type: {self.noise_type}")

        loss = torch.mean((recon - x) ** 2)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def salt_and_pepper(self, img):
        """salt and pepper noise for mnist"""
        # for salt and pepper noise, sig is probability of occurance of noise pixels.
        img = img.copy()
        prob = self.sig
        rnd = torch.random.rand(*img.shape).to(img.device)
        img[rnd < prob / 2] = 0.0
        img[rnd > 1 - prob / 2] = 1.0
        return img


class NNProjNet(AE):
    """Nearest Neighbor Projection Network"""

    def __init__(self, encoder, decoder, k=1, width=3, sig=0.1):
        super(NNProjNet, self).__init__(encoder, decoder)
        self.k = k
        self.width = width
        self.sig = sig

    def sample_background(self, x):
        r = torch.rand_like(x)
        return (r - 0.5) * self.width

    def train_step(self, x, optimizer):
        optimizer.zero_grad()
        U = self.sample_background(x)
        if len(x.shape) == 4:
            x = x.squeeze(3).squeeze(2)
            U = U.squeeze(3).squeeze(2)

        with torch.no_grad():
            dist = self.pdist(U, x)
            sort_idx = torch.argsort(dist, dim=1)

        l_loss = []
        d_proj_loss = {}
        if self.k == 0:
            d_proj_loss = {"k0": 0}
        for i_k in range(self.k):
            target_x = x[sort_idx[:, i_k]]
            recon = self(U)
            proj_loss = torch.mean((recon - target_x) ** 2)
            d_proj_loss["k" + str(i_k)] = proj_loss.item()
            l_loss.append(proj_loss)

        if self.sig is not None:
            if self.sig == 0:
                recon = self(x)

            else:
                noise = torch.randn_like(x)
                recon = self(x + self.sig * noise)
            denoise_loss = torch.mean((recon - x) ** 2)
            l_loss.append(denoise_loss)
            denoise_loss_ = denoise_loss.item()
        else:
            denoise_loss_ = 0

        loss = torch.mean(torch.stack(l_loss))
        loss.backward()
        optimizer.step()
        d_result = {"loss": loss.item(), "denoise": denoise_loss_}
        d_result.update(d_proj_loss)
        return d_result

    def pdist(self, X, Y):
        t1 = (X**2).sum(dim=1)[:, None]
        t2 = (Y**2).sum(dim=1)[None, :]
        t3 = X.mm(Y.t())
        return t1 + t2 - 2 * t3


class VAE(AE):
    def __init__(
        self,
        encoder,
        decoder,
        n_sample=1,
        use_mean=False,
        pred_method="recon",
        sigma_trainable=False,
    ):
        super(VAE, self).__init__(
            encoder,
            IsotropicGaussian(decoder, sigma=1, sigma_trainable=sigma_trainable),
        )
        self.n_sample = (
            n_sample  # the number of samples to generate for anomaly detection
        )
        self.use_mean = use_mean  # if True, does not sample from posterior distribution
        self.pred_method = pred_method  # which anomaly score to use
        self.z_shape = None

    def forward(self, x):
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        return self.decoder(z_sample)

    def sample_latent(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        if self.use_mean:
            return mu
        eps = torch.randn(*mu.shape, dtype=torch.float32)
        eps = eps.to(z.device)
        return mu + torch.exp(log_sig) * eps

    # def sample_marginal_latent(self, z_shape):
    #     return torch.randn(z_shape)

    def kl_loss(self, z):
        """analytic (positive) KL divergence between gaussians
        KL(q(z|x) | p(z))"""
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        mu_sq = mu**2
        sig_sq = torch.exp(log_sig) ** 2
        kl = mu_sq + sig_sq - torch.log(sig_sq) - 1
        # return 0.5 * torch.mean(kl.view(len(kl), -1), dim=1)
        return 0.5 * torch.sum(kl.view(len(kl), -1), dim=1)

    def train_step(self, x, optimizer, y=None, clip_grad=None):
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        nll = -self.decoder.log_likelihood(x, z_sample)

        kl_loss = self.kl_loss(z)
        loss = nll + kl_loss
        loss = loss.mean()
        nll = nll.mean()

        loss.backward()
        optimizer.step()
        return {
            "loss": nll.item(),
            "vae/kl_loss_": kl_loss.mean(),
            "vae/sigma_": self.decoder.sigma.item(),
        }

    def predict(self, x):
        """one-class anomaly prediction using the metric specified by self.anomaly_score"""
        if self.pred_method == "recon":
            return self.reconstruction_probability(x)
        elif self.pred_method == "lik":
            return -self.marginal_likelihood(x)  # negative log likelihood
        else:
            raise ValueError(f"{self.pred_method} should be recon or lik")

    def validation_step(self, x, y=None, **kwargs):
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        recon = self.decoder(z_sample)
        loss = torch.mean((recon - x) ** 2)
        predict = -self.decoder.log_likelihood(x, z_sample)

        if kwargs.get("show_image", True):
            x_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        else:
            x_img, recon_img = None, None

        return {
            "loss": loss.item(),
            "predict": predict,
            "reconstruction": recon,
            "input@": x_img,
            "recon@": recon_img,
        }

    def reconstruction_probability(self, x):
        l_score = []
        z = self.encoder(x)
        for i in range(self.n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = -self.decoder.log_likelihood(x, z_sample)
            score = recon_loss
            l_score.append(score)
        return torch.stack(l_score).mean(dim=0)

    def marginal_likelihood(self, x, n_sample=None):
        """marginal likelihood from importance sampling
        log P(X) = log \int P(X|Z) * P(Z)/Q(Z|X) * Q(Z|X) dZ"""
        if n_sample is None:
            n_sample = self.n_sample

        # check z shape
        with torch.no_grad():
            z = self.encoder(x)

            l_score = []
            for i in range(n_sample):
                z_sample = self.sample_latent(z)
                log_recon = self.decoder.log_likelihood(x, z_sample)
                log_prior = self.log_prior(z_sample)
                log_posterior = self.log_posterior(z, z_sample)
                l_score.append(log_recon + log_prior - log_posterior)
        score = torch.stack(l_score)
        logN = torch.log(torch.tensor(n_sample, dtype=torch.float, device=x.device))
        return torch.logsumexp(score, dim=0) - logN

    def marginal_likelihood_naive(self, x, n_sample=None):
        if n_sample is None:
            n_sample = self.n_sample

        # check z shape
        z_dummy = self.encoder(x[[0]])
        z = torch.zeros(len(x), *list(z_dummy.shape[1:]), dtype=torch.float).to(
            x.device
        )

        l_score = []
        for i in range(n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = -self.decoder.log_likelihood(x, z_sample)
            score = recon_loss
            l_score.append(score)
        score = torch.stack(l_score)
        return -torch.logsumexp(-score, dim=0)

    def elbo(self, x):
        l_score = []
        z = self.encoder(x)
        for i in range(self.n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = -self.decoder.log_likelihood(x, z_sample)
            kl_loss = self.kl_loss(z)
            score = recon_loss + kl_loss
            l_score.append(score)
        return torch.stack(l_score).mean(dim=0)

    def log_posterior(self, z, z_sample):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]

        log_p = torch.distributions.Normal(mu, torch.exp(log_sig)).log_prob(z_sample)
        log_p = log_p.view(len(z), -1).sum(-1)
        return log_p

    def log_prior(self, z_sample):
        log_p = torch.distributions.Normal(
            torch.zeros_like(z_sample), torch.ones_like(z_sample)
        ).log_prob(z_sample)
        log_p = log_p.view(len(z_sample), -1).sum(-1)
        return log_p

    def posterior_entropy(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        D = mu.shape[1]
        pi = torch.tensor(np.pi, dtype=torch.float32).to(z.device)
        term1 = D / 2
        term2 = D / 2 * torch.log(2 * pi)
        term3 = log_sig.view(len(log_sig), -1).sum(dim=-1)
        return term1 + term2 + term3

    def _set_z_shape(self, x):
        if self.z_shape is not None:
            return
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        dummy_z = self.sample_latent(dummy_z)
        z_shape = dummy_z.shape
        self.z_shape = z_shape[1:]

    def sample_z(self, n_sample, device):
        z_shape = (n_sample,) + self.z_shape
        return torch.randn(z_shape, device=device, dtype=torch.float)

    def sample(self, n_sample, device):
        z = self.sample_z(n_sample, device)
        return {"sample_x": self.decoder.sample(z)}


class VAE_ConstEnt(VAE):
    def __init__(self, encoder, decoder, n_sample=1, use_mean=False, sig=None):
        super(VAE_ConstEnt, self).__init__(encoder, decoder)
        self.n_sample = (
            n_sample  # the number of samples to generate for anomaly detection
        )
        self.use_mean = use_mean  # if True, does not sample from posterior distribution
        if sig is None:
            self.register_parameter(
                "sig", nn.Parameter(torch.tensor(1, dtype=torch.float))
            )
        else:
            self.register_buffer("sig", torch.tensor(sig, dtype=torch.float))

    def sample_latent(self, z):
        mu = z
        std = self.sig
        # half_chan = int(z.shape[1] / 2)
        # mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        if self.use_mean:
            return mu
        eps = torch.randn(*mu.shape, dtype=torch.float32)
        eps = eps.to(z.device)
        return mu + std * eps

    def kl_loss(self, z):
        """analytic (positive) KL divergence between gaussians"""
        mu = z
        mu_sq = mu**2
        sig_sq = self.sig**2
        kl = mu_sq + sig_sq - torch.log(sig_sq) - 1
        return 0.5 * torch.sum(kl.view(len(kl), -1), dim=1)

    def train_step(self, x, optimizer):
        d_result = super().train_step(x, optimizer)
        d_result["sig"] = self.sig
        return d_result

    def log_posterior(self, z, z_sample):
        mu = z
        log_p = torch.distributions.Normal(mu, self.sig).log_prob(z_sample)
        log_p = log_p.view(len(z), -1).sum(-1)
        return log_p


class VAE_PROJ(AE):
    def __init__(self, encoder, decoder, n_sample=1, use_mean=False, sample_proj=False):
        super(VAE_PROJ, self).__init__(encoder, decoder)
        self.n_sample = (
            n_sample  # the number of samples to generate for anomaly detection
        )
        self.use_mean = use_mean  # if True, does not sample from posterior distribution
        self.sample_proj = sample_proj

    def proj(self, x):
        return x / x.norm(dim=1, keepdim=True)

    def forward(self, x):
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        return self.decoder(z_sample)

    def sample_latent(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        mu = self.proj(mu)
        if self.use_mean:
            return mu
        eps = torch.randn(*mu.shape, dtype=torch.float32)
        eps = eps.to(z.device)
        z_sample = mu + torch.exp(log_sig) * eps
        if self.sample_proj:
            return self.proj(z_sample)
        else:
            return z_sample

    def kl_loss(self, z):
        """analytic (positive) KL divergence between gaussians"""
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        mu = self.proj(mu)
        mu_sq = mu**2
        mu_sq = mu_sq.detach()
        sig_sq = torch.exp(log_sig) ** 2
        kl = mu_sq + sig_sq - torch.log(sig_sq) - 1
        return 0.5 * torch.mean(kl.view(len(kl), -1), dim=1)

    def train_step(self, x, optimizer):
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        nll = -self.decoder.log_likelihood(x, z_sample)

        kl_loss = self.kl_loss(z)
        loss = nll + kl_loss
        loss = loss.mean()
        nll = nll.mean()

        loss.backward()
        optimizer.step()
        return {"loss": nll.item(), "kl_loss": kl_loss.mean()}

    def predict(self, x):
        """one-class anomaly prediction"""
        l_score = []
        z = self.encoder(x)
        for i in range(self.n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = -self.decoder.log_likelihood(x, z_sample)
            score = recon_loss
            l_score.append(score)
        return torch.stack(l_score).mean(dim=0)

    def marginal_likelihood(self, x, n_sample=None):
        if n_sample is None:
            n_sample = self.n_sample

        # check z shape
        z_dummy = self.encoder(x[[0]])
        z = torch.zeros(len(x), *list(z_dummy.shape[1:]), dtype=torch.float).to(
            x.device
        )

        l_score = []
        for i in range(n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = -self.decoder.log_likelihood(x, z_sample)
            score = recon_loss
            l_score.append(score)
        score = torch.stack(l_score)
        return -torch.logsumexp(-score, dim=0)

    def elbo(self, x):
        l_score = []
        z = self.encoder(x)
        for i in range(self.n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = -self.decoder.log_likelihood(x, z_sample)
            kl_loss = self.kl_loss(z)
            score = recon_loss + kl_loss
            l_score.append(score)
        return torch.stack(l_score).mean(dim=0)

    def log_posterior(self, z, z_sample):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]

        log_p = torch.distributions.Normal(mu, torch.exp(log_sig)).log_prob(z_sample)
        log_p = log_p.view(len(z), -1).sum(-1)
        return log_p

    def log_prior(self, z_sample):
        log_p = torch.distributions.Normal(
            torch.zeros_like(z_sample), torch.ones_like(z_sample)
        ).log_prob(z_sample)
        log_p = log_p.view(len(z_sample), -1).sum(-1)
        return log_p


class VAE_FLOW(VAE):
    def __init__(
        self, encoder, decoder, flow, n_sample=1, use_mean=False, n_kl_sample=1
    ):
        super(VAE_FLOW, self).__init__(
            encoder, decoder, n_sample=n_sample, use_mean=use_mean
        )
        self.flow = flow
        self.n_kl_sample = n_kl_sample

    def kl_loss(self, z):
        l_kl = []
        for i in range(self.n_kl_sample):
            z_sample = self.sample_latent(z)
            log_qz = self.log_posterior(z, z_sample)
            log_pz = self.flow.log_likelihood(z_sample)
            l_kl.append(log_qz - log_pz)
        return torch.stack(l_kl).mean(dim=0)

    def log_prior(self, z_sample):
        return self.flow.log_likelihood(z_sample)


class VAE_regret(VAE):
    def __init__(
        self,
        encoder,
        decoder,
        n_sample=1,
        use_mean=False,
        pred_method="recon",
        sigma_trainable=False,
    ):
        super(VAE_regret, self).__init__(
            encoder,
            IsotropicGaussian(decoder, sigma=1, sigma_trainable=sigma_trainable),
        )
        self.n_sample = (
            n_sample  # the number of samples to generate for anomaly detection
        )
        self.use_mean = use_mean  # if True, does not sample from posterior distribution
        self.pred_method = pred_method  # which anomaly score to use
        self.z_shape = None

    def train_step_fore(self, x, optimizer, y=None, clip_grad=None):
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        nll = -self.decoder.log_likelihood(x, z_sample)

        kl_loss = self.kl_loss(z)
        loss = nll + kl_loss
        loss = loss.mean()
        nll = nll.mean()

        loss.backward()
        optimizer.step()

        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        recon = self.decoder(z_sample)
        loss = torch.mean((recon - x) ** 2)
        predict = -self.decoder.log_likelihood(x, z_sample)
        return {
            "loss": loss.item(),
            "vae/kl_loss_": kl_loss.mean(),
            "vae/sigma_": self.decoder.sigma.item(),
            "predict": predict,
            "reconstruction": recon,
        }


class AAE(AE):
    """Adversarial Autoencoder"""

    def __init__(
        self,
        encoder,
        decoder,
        discriminator,
        gpu=True,
        lr=1e-4,
        wasserstein=False,
        n_dcsr_iter=5,
    ):
        super(AAE, self).__init__(encoder, decoder)
        self.dscr = discriminator
        self.en_solver = optim.Adam(self.encoder.parameters(), lr=lr)
        self.de_solver = optim.Adam(self.decoder.parameters(), lr=lr)
        self.dc_solver = optim.Adam(self.dscr.parameters(), lr=lr)
        self.n_dcsr_iter = n_dcsr_iter
        warnings.warn("AAE uses its own optimizers")
        self.wasserstein = wasserstein
        print(f"wasserstein {wasserstein}")

    def train_step(self, x, optimizer=None):
        """reconstruction phase
        optimizer has no effect"""
        self.reset_grad()
        recon_x = self(x)
        loss = torch.mean((recon_x - x) ** 2)
        loss.backward()
        self.en_solver.step()
        self.de_solver.step()

        """regularization phase"""
        for _ in range(self.n_dcsr_iter):
            # discriminator
            self.reset_grad()
            z_fake = self.encoder(x)
            z_real = torch.randn(
                x.shape[0], self.decoder.in_chan, z_fake.shape[2], z_fake.shape[3]
            )
            z_real = z_real.to(x.device)

            D_real = self.dscr(z_real)
            D_fake = self.dscr(z_fake)
            if self.wasserstein:
                D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
            else:
                D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
            D_loss.backward()
            self.dc_solver.step()

            # weight clipping
            if self.wasserstein:
                for p in self.dscr.parameters():
                    p.data.clamp_(-0.01, 0.01)

        # generator

        self.reset_grad()
        z_fake = self.encoder(x)
        D_fake = self.dscr(z_fake)
        if self.wasserstein:
            G_loss = -torch.mean(D_fake)
        else:
            G_loss = -torch.mean(torch.log(D_fake))

        G_loss.backward()
        self.en_solver.step()

        # gaussian likelihood
        z_fake = self.encoder(x)
        nll = torch.mean(
            0.5 * (z_fake**2).sum(dim=1)
        )  # unnormalized negative log likelihood

        return {"loss": loss, "D_loss": D_loss, "G_loss": G_loss, "gaussian_nll": nll}

    def reset_grad(self):
        self.en_solver.zero_grad()
        self.de_solver.zero_grad()
        self.dc_solver.zero_grad()


class WAE(AE):
    """Wassertstein Autoencoder with MMD loss"""

    def __init__(self, encoder, decoder, reg=1.0, bandwidth="median", prior="gaussian"):
        super().__init__(encoder, decoder)
        if not isinstance(bandwidth, str):
            bandwidth = float(bandwidth)
        self.bandwidth = bandwidth
        self.reg = reg  # coefficient for MMD loss
        self.prior = prior

    def train_step(self, x, optimizer, y=None, **kwargs):
        optimizer.zero_grad()
        # forward step
        z = self.encoder(x)
        recon = self.decoder(z)
        # recon_loss = torch.mean(self.decoder.square_error(x, recon))
        recon_loss = torch.mean((x - recon) ** 2)

        # MMD step
        z_prior = self.sample_prior(z)
        mmd_loss = self.mmd(z_prior, z)

        loss = recon_loss + mmd_loss * self.reg
        loss.backward()
        optimizer.step()
        return {
            "loss": loss.item(),
            "recon_loss": recon_loss,
            "mmd_loss": mmd_loss.item(),
        }

    def sample_prior(self, z):
        if self.prior == "gaussian":
            return torch.randn_like(z)
        elif self.prior == "uniform_tanh":
            return torch.rand_like(z) * 2 - 1
        else:
            raise ValueError(f"invalid prior {self.prior}")

    def mmd(self, X1, X2):
        if len(X1.shape) == 4:
            X1 = X1.view(len(X1), -1)
        if len(X2.shape) == 4:
            X2 = X2.view(len(X2), -1)

        N1 = len(X1)
        X1_sq = X1.pow(2).sum(1).unsqueeze(0)
        X1_cr = torch.mm(X1, X1.t())
        X1_dist = X1_sq + X1_sq.t() - 2 * X1_cr

        N2 = len(X2)
        X2_sq = X2.pow(2).sum(1).unsqueeze(0)
        X2_cr = torch.mm(X2, X2.t())
        X2_dist = X2_sq + X2_sq.t() - 2 * X2_cr

        X12 = torch.mm(X1, X2.t())
        X12_dist = X1_sq.t() + X2_sq - 2 * X12

        # median heuristic to select bandwidth
        if self.bandwidth == "median":
            X1_triu = X1_dist[torch.triu(torch.ones_like(X1_dist), diagonal=1) == 1]
            bandwidth1 = torch.median(X1_triu)
            X2_triu = X2_dist[torch.triu(torch.ones_like(X2_dist), diagonal=1) == 1]
            bandwidth2 = torch.median(X2_triu)
            bandwidth_sq = ((bandwidth1 + bandwidth2) / 2).detach()
        else:
            bandwidth_sq = self.bandwidth**2

        C = -0.5 / bandwidth_sq
        K11 = torch.exp(C * X1_dist)
        K22 = torch.exp(C * X2_dist)
        K12 = torch.exp(C * X12_dist)
        K11 = (1 - torch.eye(N1).to(X1.device)) * K11
        K22 = (1 - torch.eye(N2).to(X1.device)) * K22
        mmd = K11.sum() / N1 / (N1 - 1) + K22.sum() / N2 / (N2 - 1) - 2 * K12.mean()
        return mmd

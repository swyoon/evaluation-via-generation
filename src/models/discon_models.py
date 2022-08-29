import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from models.vqvae.modules import VQEmbedding
from optimizers import get_optimizer


class STOneHot(Function):
    """Straight-Through gradient for argmax one-hot"""

    @staticmethod
    def forward(ctx, input):
        # result = torch.squeeze(torch.squeeze(torch.zeros_like(input), -1), -1)
        # index_max = torch.squeeze(torch.squeeze(torch.argmax(input, dim=1), -1), -1)
        # index_tensor = index_max.view(*index_max.size(), -1)
        # return result.scatter(len(index_max.size()), index_tensor, 1.0)
        result = torch.zeros_like(input)
        index_max = torch.argmax(input, dim=1)
        index_tensor = index_max.view(*index_max.size(), -1)
        return result.scatter(len(index_max.size()), index_tensor, 1.0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_inputs = grad_output.clone()
        return grad_inputs


class SoftmaxNet(nn.Module):
    """Autoencoder with discrete representation"""

    def __init__(self, encoder_disc, decoder, gpu=True, entropy_reg=0.0, eps=1e-5):
        super(SoftmaxNet, self).__init__()
        self.encoder_disc = encoder_disc  # last activation should be linear
        self.decoder = decoder
        self.gpu = gpu
        self.one_hot = STOneHot.apply
        # self.one_hot = self.naive_one_hot
        self.entropy_reg = entropy_reg
        self.eps = torch.tensor(eps, dtype=torch.float32)
        if gpu:
            self.eps = self.eps.cuda()

    def forward(self, x):
        z_softmax = F.softmax(self.encoder_disc(x), dim=1)
        z = self.one_hot(z_softmax)
        self.z = z
        self.z_softmax = z_softmax
        return self.decoder(z)

    def encode(self, x):
        z_softmax = F.softmax(self.encoder_disc(x), dim=1)
        z = self.one_hot(z_softmax)
        self.z = z
        self.z_softmax = z_softmax
        return z

    def predict(self, x):
        recon = self(x)
        return ((recon - x) ** 2).view(len(x), -1).mean(dim=1)

    def entropy(self, z_softmax):
        z_softmax_ = z_softmax.view((len(z_softmax), -1))
        return -(torch.log(z_softmax_ + self.eps) * z_softmax_).sum(dim=1)

    def train_step(self, x, optimizer):
        optimizer.zero_grad()
        recon = self(x)
        loss = torch.mean((recon - x) ** 2)
        if self.entropy_reg > 0:
            entropy = self.entropy(self.z_softmax).mean()
            loss -= self.entropy_reg * entropy
        else:
            entropy = torch.tensor(0)
        loss.backward()
        optimizer.step()
        return {"loss": loss, "entropy": entropy}

    def validation_step(self, x, reconstruction):
        recon = self(x)
        loss = torch.mean((recon - x) ** 2)
        predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)

        if not reconstruction:
            return {"loss": loss, "predict": predict}
        else:
            return {"loss": loss, "predict": predict, "reconstruction": recon}

    def naive_one_hot(self, z):
        """does not propagate gradients"""
        result = torch.zeros(z.shape)
        if self.gpu:
            result = result.cuda()
        index_max = torch.argmax(F.softmax(z, dim=1), dim=1)
        index_tensor = index_max.view(*index_max.size(), -1)
        return result.scatter(len(index_max.size()), index_tensor, 1.0)


class SoftmaxConcatNet(nn.Module):
    def __init__(
        self, encoder_disc, encoder_cont, decoder, gpu=True, entropy_reg=0.0, eps=1e-20
    ):
        super(SoftmaxConcatNet, self).__init__()
        self.encoder_cont = encoder_cont
        self.encoder_disc = encoder_disc
        self.decoder = decoder
        self.gpu = gpu
        self.one_hot = STOneHot.apply
        # self.one_hot = self.naive_one_hot
        self.entropy_reg = entropy_reg
        self.eps = torch.tensor(eps, dtype=torch.float32)
        if gpu:
            self.eps = self.eps.cuda()

    def forward(self, x):
        z_softmax = F.softmax(self.encoder_disc(x), dim=1)
        z_disc = self.one_hot(z_softmax)
        z_cont = self.encoder_cont(x)
        z = torch.cat([z_disc, z_cont], dim=1)
        self.z_disc = z_disc
        self.z_cont = z_cont
        self.z_softmax = z_softmax
        self.z = z
        return self.decoder(z)

    def encode(self, x):
        z_softmax = F.softmax(self.encoder_disc(x), dim=1)
        z_disc = self.one_hot(z_softmax)
        z_cont = self.encoder_cont(x)
        z = torch.cat([z_disc, z_cont], dim=1)
        self.z_disc = z_disc
        self.z_cont = z_cont
        self.z_softmax = z_softmax
        self.z = z
        return z

    def predict(self, x):
        recon = self(x)
        return ((recon - x) ** 2).view(len(x), -1).mean(dim=1)

    def entropy(self, z_softmax):
        z_softmax_ = z_softmax.view((len(z_softmax), -1))
        return -(torch.log(z_softmax_ + self.eps) * z_softmax_).sum(dim=1)

    def train_step(self, x, optimizer):
        optimizer.zero_grad()
        recon = self(x)
        loss = torch.mean((recon - x) ** 2)
        if self.entropy_reg > 0:
            entropy = self.entropy(self.z_softmax).mean()
            loss -= self.entropy_reg * entropy
        else:
            entropy = torch.tensor(0)
        loss.backward()
        optimizer.step()
        return {"loss": loss, "entropy": entropy}

    def validation_step(self, x, reconstruction):
        recon = self(x)
        loss = torch.mean((recon - x) ** 2)
        predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)

        if not reconstruction:
            return {"loss": loss, "predict": predict}
        else:
            return {"loss": loss, "predict": predict, "reconstruction": recon}

    def naive_one_hot(self, z):
        """does not propagate gradients"""
        result = torch.zeros(z.shape)
        if self.gpu:
            result = result.cuda()
        index_max = torch.argmax(F.softmax(z, dim=1), dim=1)
        index_tensor = index_max.view(*index_max.size(), -1)
        return result.scatter(len(index_max.size()), index_tensor, 1.0)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def st_gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class GumbelNet(nn.Module):
    """Autoencoder with Gumbel Softmax"""

    def __init__(
        self, encoder_disc, decoder, gpu=True, tau=0.1, entropy_reg=0.0, eps=1e-20
    ):
        """tau: temperature"""
        super(GumbelNet, self).__init__()
        self.encoder_disc = encoder_disc  # last activation should be linear
        self.decoder = decoder
        self.gpu = gpu
        self.one_hot = STOneHot.apply
        tau = torch.tensor(tau, dtype=torch.float32)
        if gpu:
            tau = tau.cuda()
        self.tau = tau
        self.entropy_reg = entropy_reg
        self.eps = torch.tensor(eps, dtype=torch.float32)
        if gpu:
            self.eps = self.eps.cuda()

    def forward(self, x, tau=None):
        if tau is None:
            tau = self.tau

        logit = self.encoder_disc(x)
        z = st_gumbel_softmax(logit, tau)
        self.z = z
        self.logit = logit
        return self.decoder(z)

    def encode(self, x, tau=None):
        if tau is None:
            tau = self.tau

        logit = self.encoder_disc(x).view(len(x), -1)
        z = st_gumbel_softmax(logit, tau)
        self.z = z
        self.logit = logit
        return z

    def entropy(self, z_softmax):
        z_softmax_ = z_softmax.view((len(z_softmax), -1))
        return -(torch.log(z_softmax_ + self.eps) * z_softmax_).sum(dim=1)

    def predict(self, x):
        recon = self(x, tau=0.1)  # temperature is set to 0 when prediction
        return ((recon - x) ** 2).view(len(x), -1).mean(dim=1)

    def train_step(self, x, optimizer):
        optimizer.zero_grad()
        recon = self(x)
        loss = torch.mean((recon - x) ** 2)
        if self.entropy_reg > 0:
            entropy = self.entropy(F.softmax(self.logit, dim=1)).mean()
            loss -= self.entropy_reg * entropy
        else:
            entropy = torch.tensor(0)
        loss.backward()
        optimizer.step()
        return {"loss": loss, "entropy": entropy}

    def validation_step(self, x, reconstruction):
        recon = self(x, tau=0.1)  # temperature is set to 0 when prediction
        loss = torch.mean((recon - x) ** 2)
        predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)

        if not reconstruction:
            return {"loss": loss, "predict": predict}
        else:
            return {"loss": loss, "predict": predict, "reconstruction": recon}


class ConcatNet(nn.Module):
    """Naive concatenation of discrete and continuous feature"""

    def __init__(
        self,
        encoder_vq,
        encoder,
        decoder,
        dim=10,
        gpu=True,
        beta=1.0,
        K=512,
        decoder_2d=False,
    ):
        super(ConcatNet, self).__init__()
        self.encoder_vq = encoder_vq
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.codebook = VQEmbedding(K, dim)
        self.gpu = gpu
        self.K = K
        self.decoder_2d = decoder_2d

    def to_4d(self, x):
        """convert 2D tensor to 4D tensor."""
        if len(x.shape) == 2:
            return x[:, :, None, None]
        else:
            return x

    def to_decoder_shape(self, x):
        """convert 4D tensor to 2D tensor."""
        if self.decoder_2d and len(x.shape) == 4:
            return torch.squeeze(torch.squeeze(x, 3), 2)
        else:
            return x

    def encode(self, x):
        z_e_x = self.encoder_vq(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder_vq(z_q_x)
        return x_tilde

    def forward(self, x):
        # VQVAE pass
        z_e_x = self.encoder_vq(x)
        z_e_x = self.to_4d(z_e_x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        z_q_x_st, z_q_x = self.to_decoder_shape(z_q_x_st), self.to_decoder_shape(z_q_x)
        # x_tilde = self.decoder_vq(z_q_x_st)  # reconstruction from discrete AE

        # continuous autoencoder pass
        z_cont = self.encoder(x)
        z = torch.cat([z_q_x_st, z_cont], dim=1)

        x_recon = self.decoder(z)  # reconstuction from continuous AE
        self.z = z
        self.z_e_x = self.to_decoder_shape(z_e_x)
        self.z_q_x = z_q_x
        self.z_q_x_st = z_q_x_st
        return x_recon

    def predict(self, x):
        z_e_x = self.encoder_vq(x)
        z_e_x = self.to_4d(z_e_x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        z_q_x_st, z_q_x = self.to_decoder_shape(z_q_x_st), self.to_decoder_shape(z_q_x)

        z_cont = self.encoder(x)
        z = torch.cat([z_q_x_st, z_cont], dim=1)
        return -self.decoder.log_likelihood(x, z)

    def train_step(self, x, optimizer):
        optimizer.zero_grad()
        x_recon = self(x)
        z_e_x, z_q_x_st, z_q_x, z = self.z_e_x, self.z_q_x_st, self.z_q_x, self.z

        # Reconstruction loss
        # nll = - self.decoder.log_likelihood(x, z_q_x_st).mean() / torch.prod(torch.tensor(x.shape[1:]))
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        # continuous autoencoder loss
        continuous_nll = -self.decoder.log_likelihood(x, z).mean() / torch.prod(
            torch.tensor(x.shape[1:])
        )

        loss = loss_vq + self.beta * loss_commit + continuous_nll
        loss.backward()
        optimizer.step()
        return {"quantization": loss_vq, "commit": loss_commit, "loss": continuous_nll}

    def validation_step(self, x, reconstruction):
        recon, z_e_x, z_q_x, z_q_x_st = self(x)
        predict = -self.decoder.log_likelihood(x, z_q_x)
        loss = predict.mean()

        if not reconstruction:
            return {"loss": loss, "predict": predict}
        else:
            return {"loss": loss, "predict": predict, "reconstruction": recon}


class AEwGAN(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        dscr,
        z_dim,
        gpu=True,
        wasserstein=False,
        gan_weight=1.0,
        recon_weight=1.0,
    ):
        super(AEwGAN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dscr = dscr
        self.gpu = gpu
        self.z_dim = z_dim
        self.wasserstein = wasserstein
        self.gan_weight = gan_weight
        self.recon_weight = recon_weight

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def log_likelihood(self, x):
        z = self.encoder(x)
        return self.decoder.log_likelihood(x, z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def predict(self, x):
        recon = self(x)
        return ((recon - x) ** 2).view(len(x), -1).mean(dim=1)

    def train_step(self, x, opt_e, opt_d, opt_dscr):
        # autoencoder training
        if self.recon_weight > 0:
            opt_e.zero_grad()
            opt_d.zero_grad()
            # recon = self(x)
            # loss = torch.mean((recon - x) ** 2)
            loss = -self.log_likelihood(x).mean() * self.recon_weight
            loss.backward()
            opt_e.step()
            opt_d.step()
        else:
            loss = torch.zeros(1)

        # gan training
        if self.gan_weight > 0:
            # generator (or decoder)
            opt_d.zero_grad()
            if len(x.shape) == 2:
                noise = (
                    torch.rand((len(x), self.z_dim), dtype=torch.float32) * 2 - 1
                )  # [-1, 1]
            else:
                noise = (
                    torch.rand((len(x), self.z_dim, 1, 1), dtype=torch.float32) * 2 - 1
                )  # [-1, 1]
            if self.gpu:
                noise = noise.cuda()

            x_fake = self.decoder(noise)
            D_fake = self.dscr(x_fake)
            if self.wasserstein:
                G_loss = -torch.mean(D_fake)
            else:
                G_loss = -torch.mean(torch.log(D_fake))
            G_loss *= self.gan_weight
            G_loss.backward()
            opt_d.step()

            # discriminator
            opt_dscr.zero_grad()
            D_fake = self.dscr(x_fake.detach())
            D_real = self.dscr(x)

            if self.wasserstein:
                D_loss = -torch.mean(D_real) + torch.mean(D_fake)
            else:
                D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
            D_loss *= self.gan_weight
            D_loss.backward()
            opt_dscr.step()

            if self.wasserstein:
                for p in self.dscr.parameters():
                    p.data.clamp_(-0.01, 0.01)

        else:
            D_loss = torch.zeros(1)
            G_loss = torch.zeros(1)

        return {"loss": loss, "D_loss": D_loss, "G_loss": G_loss}

    def validation_step(self, x, reconstruction):
        recon = self(x)
        loss = torch.mean((recon - x) ** 2)
        predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)

        if not reconstruction:
            return {"loss": loss, "predict": predict}
        else:
            return {"loss": loss, "predict": predict, "reconstruction": recon}


class EM_AE_V1(nn.Module):
    def __init__(self, l_ae, fit_sigma=False, sample_E=False, sigma=0.01):
        super(EM_AE_V1, self).__init__()
        self.l_ae = nn.ModuleList(l_ae)
        self.K = len(l_ae)
        self.own_optimizer = True
        self.fit_sigma = fit_sigma
        self.sample_E = sample_E

        if self.fit_sigma:
            self.sigma = nn.Parameter(sigma)
        else:
            self.sigma = torch.tensor(sigma, requires_grad=False, dtype=torch.float)

        self.prior_logit = nn.Parameter(torch.ones(self.K))

    def forward(self, x):
        """reconstruct input"""
        recon_mse, cluster_assignment = self.predict(x)
        x_out = torch.zeros_like(x)
        for k, ae in enumerate(self.l_ae):
            c_mask = cluster_assignment == k
            if c_mask.sum() == 0:
                continue
            recon = ae.forward(x[c_mask])
            x_out.masked_scatter_(c_mask.reshape((-1, 1)).repeat(1, x.shape[1]), recon)
        return x_out

    def predict(self, x, return_recon=False):
        """return reconstruction error and cluster assignment"""
        l_cluster_prob = []
        l_recon = []
        for ae in self.l_ae:
            recon_err, recon = ae.predict_and_reconstruct(x)
            cluster_prob = -recon_err  # negative recon error
            l_cluster_prob.append(cluster_prob)
            l_recon.append(recon)

        l_cluster_prob = torch.stack(l_cluster_prob)
        max_prob, cluster_assignment = torch.max(l_cluster_prob, dim=0)
        recon_mse = -max_prob

        if return_recon:
            l_recon = torch.stack(l_recon)
            data_dim = x.shape[1:]
            one_data_dim = np.ones(len(data_dim), dtype=int)
            index = cluster_assignment.reshape((1, -1, *one_data_dim)).repeat(
                1, 1, *data_dim
            )
            nearest_recon = l_recon.gather(0, index).squeeze(0)
            return recon_mse, cluster_assignment, nearest_recon
        else:
            return recon_mse, cluster_assignment

    def posterior(self, x):
        """returns unnormalized log posterior prob"""
        l_cluster_prob = []
        #         prior = self.prior()
        for ae in self.l_ae:
            cluster_prob = -ae.predict(x)  # negative recon error
            #             cluster_prob = ae.prob(x) #+ prior
            l_cluster_prob.append(cluster_prob)
        l_cluster_prob = torch.stack(l_cluster_prob)  # K x B, log probs
        log_posterior = torch.exp(l_cluster_prob.t() / (self.sigma**2))
        # TODO : numerical stability
        return log_posterior  # B x K

    def train_step(self, x, l_opt):
        assert len(l_opt) == self.K
        # E-step
        log_posterior = self.posterior(x)
        if self.sample_E:
            # probabilities are normalized automatically
            # TODO: numerical stability
            cluster_assignment = torch.multinomial(
                torch.exp(log_posterior), 1
            ).flatten()
        else:
            cluster_assignment = torch.argmax(log_posterior, dim=1)

        # M-step
        l_loss = []
        for k in range(self.K):
            ae = self.l_ae[k]
            opt = l_opt[k]
            cluster_x = x[cluster_assignment == k]
            if len(cluster_x) == 0:
                l_loss.append(np.nan)
                continue
            d_step = ae.train_step(cluster_x, opt)
            l_loss.append(d_step["loss"])
        return {
            "loss_": l_loss,
            "loss": np.nanmean(l_loss),
            "cluster": cluster_assignment,
        }

    def get_optimizer(self, opt_cfg):
        l_opt = []
        for ae in self.l_ae:
            opt = get_optimizer(opt_cfg, ae.parameters())
            l_opt.append(opt)
        return l_opt

    def validation_step(self, x, reconstruction=True):
        recon_mse, cluster_assignment, nearest_recon = self.predict(
            x, return_recon=True
        )

        if not reconstruction:
            return {"loss": recon_mse.mean(), "predict": recon_mse}
        else:
            return {
                "loss": recon_mse.mean(),
                "predict": recon_mse,
                "reconstruction": nearest_recon,
                "cluster": cluster_assignment,
            }


class DisconAE_V1(nn.Module):
    """classifier with K autoencoders. regularized by possible reconstruction"""

    def __init__(self, l_ae, classifier, reg1=0.1, info=True, **kwargs):
        super(DisconAE_V1, self).__init__()
        self.l_ae = nn.ModuleList(l_ae)
        self.classifier = classifier
        self.K = len(l_ae)
        self.own_optimizer = True
        self.reg1 = reg1
        self.info = info

    def get_optimizer(self, opt_cfg):
        l_opt = []
        for ae in self.l_ae:
            opt = get_optimizer(opt_cfg, ae.parameters())
            l_opt.append(opt)
        opt = get_optimizer(opt_cfg, self.classifier.parameters())
        return {"ae": l_opt, "classifier": opt}

    def forward(self, x, random_assign=False):
        """reconstruct input"""
        if random_assign:
            assignment = torch.randint(0, self.K, (len(x),)).to(x.device)
        else:
            cls_logit = self.classifier(x)
            max_prob, assignment = torch.max(cls_logit, dim=1)

        x_out = torch.zeros_like(x)
        ones = [1 for i in range(len(x_out.shape) - 1)]
        for k, ae in enumerate(self.l_ae):
            c_mask = assignment == k
            if c_mask.sum() == 0:
                continue

            recon = ae.forward(x[c_mask])
            x_out.masked_scatter_(c_mask.reshape(-1, *ones), recon)
        return x_out, assignment

    def validation_step(self, x):

        batch_size = x.shape[0]

        # ------------------
        # Reconstruction
        # ------------------

        recon, assignment = self(x)
        # print(label_x, code_x, recon)
        loss_recon = torch.mean((recon - x) ** 2, dim=1)

        return {
            "loss": loss_recon.mean().item(),
            "predict": loss_recon,
            "reconstruction": recon,
        }

    def sample(self, N, out_dim, device="cpu"):

        sampled_labels = torch.randint(0, self.K, (N,)).to(device)
        x_out = torch.zeros(*out_dim).to(device)
        ones = [1 for i in range(len(x_out.shape) - 1)]
        for k, ae in enumerate(self.l_ae):
            c_mask = sampled_labels == k
            if c_mask.sum() == 0:
                continue
            recon = ae.sample(c_mask.sum(), device=device)
            x_out.masked_scatter_(c_mask.reshape(-1, *ones), recon)
        return x_out, sampled_labels

    def train_step(self, x, d_optimizer, random_assign=False):
        l_ae_opt = d_optimizer["ae"]
        cls_opt = d_optimizer["classifier"]
        batch_size = x.shape[0]
        d_loss = {}

        # ------------------
        # Information Loss
        # ------------------

        if self.info:
            #
            # train classifier
            #
            cls_opt.zero_grad()
            samples, cluster_assignment = self.sample(
                batch_size, x.shape, device=x.device
            )

            cls_pred = self.classifier(samples)
            cls_loss = F.cross_entropy(cls_pred, cluster_assignment)
            cls_loss.backward()
            cls_opt.step()
            d_loss["cls_loss"] = cls_loss.item()

            #
            # train decoders (generators)
            #
            for ae in self.l_ae:
                ae.zero_grad()
            samples, cluster_assignment = self.sample(
                batch_size, x.shape, device=x.device
            )

            cls_pred = self.classifier(samples)
            cls_loss = F.cross_entropy(cls_pred, cluster_assignment)
            cls_loss.backward()
            for ae_opt in l_ae_opt:
                ae_opt.step()
            d_loss["decode_loss"] = cls_loss.item()

        # ------------------
        # Reconstruction Loss
        # ------------------
        recon, assignment = self.forward(x, random_assign=random_assign)
        loss_recon = F.mse_loss(recon, x)
        loss_recon.backward()
        d_loss["loss"] = loss_recon.item()

        for ae_opt in l_ae_opt:
            ae_opt.step()

        return d_loss

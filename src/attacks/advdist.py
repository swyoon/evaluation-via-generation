import functools

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize
from torchvision.utils import make_grid

from attacks.mcmc import sample_discrete_gibbs, sample_mh
from attacks.transforms import apply_affine_batch, apply_colortransform_batch


class AdversarialDistribution(nn.Module):
    def __init__(self, model, mode="reinforce", flow=False, T=1.0):
        super().__init__()
        self.model = model
        self.T = T
        assert mode in {"reinforce", "reverseKL"}
        self.mode = mode
        self.flow = flow

    def forward(self, x):
        """computes log probability"""
        return self.model.log_likelihood(x)

    def log_likelihood(self, x):
        return self.model.log_likelihood(x)

    def sample(self, n_sample, device):
        d_sample = self.model.sample(n_sample=n_sample, device=device)
        return d_sample

    def train_step(self, n_sample, device, detector, opt, clip_grad=None):
        opt.zero_grad()
        # sample
        d_sample = self.sample(n_sample=n_sample, device=device)
        sample = d_sample["x"]

        # Entropy
        if self.flow:
            #             logp = d_sample['logp']
            logp = self.log_likelihood(sample.detach())
            entropy_term = d_sample["logdet"].mean()
        else:
            logp = self.log_likelihood(sample.detach())
            entropy_term = self.entropy(logp).mean()

        # Energy (reward)
        if self.mode == "reinforce":
            with torch.no_grad():
                detector_score = detector.predict(sample)
            energy_term = logp * detector_score
        else:
            energy_term = detector.predict(sample)
            detector_score = energy_term.detach()
        energy_term = energy_term.mean()

        # Gradient step
        loss = -entropy_term * self.T + energy_term  # / self.T
        loss = loss.mean()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        opt.step()
        d_result = {
            "loss": loss.item(),
            "energy_term_": energy_term.item(),
            "entropy_term_": entropy_term.item(),  # 'x': d_sample['x'].detach().cpu(),
            "logp": logp.detach().cpu(),
            "detector_score": detector_score.cpu(),
            "entropy_est_": -logp.mean().detach().cpu(),
        }
        return d_result

    def entropy(self, logp):
        """Note: This is not actually entropy.
        However, its gradient yields the gradient for entropy"""
        return -logp * (1 + logp).detach()

    def validation_step(self, n_sample, device, detector):
        sample = self.sample(n_sample=n_sample, device=device).detach()
        logp = self.log_likelihood(sample)
        energy = detector.predict(sample).mean()
        entropy = -logp.mean()
        loss = -entropy + energy / self.T
        img = make_grid(sample.cpu(), nrow=8, normalize=True, range=(0, 1))
        d_result = {
            "loss": loss.item(),
            "val_energy_": energy.item(),
            "val_entropy_": entropy.item(),
            "sample@": img,
        }
        return d_result


class AdversarialDistributionVQVAE(nn.Module):
    """Adversarial Distribution defined on a discrete latent space formed by VQVAE"""

    def __init__(
        self,
        model,
        vqvae,
        T=1.0,
        classifier=None,
        classifier_thres=None,
        z_shape=None,
        K=None,
        n_step=None,
        detector=None,
    ):
        super().__init__()
        self.model = model
        self.vqvae = vqvae
        self.T = T
        self.register_buffer("baseline", torch.tensor(0.0))
        self.momentum = 0.99
        self.classifier = classifier
        self.classifier_thres = classifier_thres
        if self.classifier_thres is not None:
            self.classifier_thres_logit = np.log(
                classifier_thres / (1 - classifier_thres)
            )

        # for gibbs sampling
        self.z_shape = z_shape  # shape of tensor to be sampled
        self.K = K
        self.n_step = n_step
        self.detector = detector

    def forward(self, z):
        """computes log probability"""
        return self.model.log_likelihood(z)

    def log_likelihood(self, z):
        return self.model.log_likelihood(z)

    def sample(self, n_sample, device, reject=False):
        if self.model == "gibbs":
            d_sample = self._sample_gibbs(n_sample, device)
        else:
            d_sample = self._sample_model(n_sample, device)
        sample = d_sample.pop("x")

        d_result = {"d_sample": d_sample}  # for saving other outputs for debugging
        sample_z = sample.to(torch.long)
        sample_x = self.vqvae.decode(sample_z).contiguous()
        d_result["x"] = sample_x
        d_result["z"] = sample_z

        if reject and (self.classifier is not None):
            prob = self.classifier.predict(sample_x, logit=False)
            accept = (prob <= self.classifier_thres).flatten()
            d_result["x_accept"] = sample_x[accept]
            d_result["z_accept"] = sample_z[accept]
        return d_result

    def _sample_model(self, n_sample, device):
        d_sample = self.model.sample(n_sample=n_sample, device=device)
        return d_sample

    def _sample_gibbs(self, n_sample, device):
        shape = (n_sample,) + self.z_shape
        z0 = torch.randint(self.K, size=shape, device=device)
        return sample_discrete_gibbs(z0, self._energy_z, self.n_step, self.K, self.T)

    def train_step(self, n_sample, device, detector, opt, clip_grad=None):
        opt.zero_grad()
        # sample
        d_sample = self.sample(n_sample=n_sample, device=device, reject=False)
        sample_x, sample_z = d_sample["x"], d_sample["z"]

        # Entropy
        logp = self.log_likelihood(sample_z.detach())
        entropy_term = self.entropy(logp).mean()

        # Energy (reward)
        with torch.no_grad():
            detector_score = detector.predict(sample_x)
        ood_barrier = self._ood_barrier(sample_x)
        energy = detector_score + ood_barrier

        energy_term = (logp * (energy - self.baseline)).mean()

        # update baseline
        self.baseline = (
            self.baseline * self.momentum + (1 - self.momentum) * energy.mean()
        )
        # self.baseline = self.baseline * self.momentum + (1 - self.momentum) * detector_score.mean()

        # Gradient step
        loss = -entropy_term * self.T + energy_term  # / self.T
        loss = loss.mean()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        opt.step()
        d_result = {
            "loss": loss.item(),
            "train/energy_term_": energy_term.item(),
            "train/entropy_term_": entropy_term.item(),
            "x": d_sample["x"].detach().cpu(),
            "z": d_sample["z"].detach().cpu(),
            "logp": logp.detach().cpu(),
            "detector_score": detector_score.cpu(),
            "train/entropy_est_": -logp.mean().detach().cpu(),
            "train/baseline_": self.baseline.item(),
            "train/ood_barrier_": ood_barrier.mean().item(),
        }
        return d_result

    def _ood_barrier(self, x):
        if self.classifier is not None:
            ood_barrier = torch.relu(
                self.classifier.predict(x, logit=True) - self.classifier_thres_logit
            )
        else:
            ood_barrier = torch.zeros(len(x))
        return ood_barrier

    def _energy_z(self, z):
        """for gibbs sampling"""
        with torch.no_grad():
            x = self.vqvae.decode(z.to(torch.long)).contiguous()
            detector_score = self.detector.predict(x).flatten()
        ood_barrier = self._ood_barrier(x).flatten()
        energy = detector_score + ood_barrier
        return energy

    def entropy(self, logp):
        """Note: This is not actually entropy.
        However, its gradient yields the gradient for entropy"""
        return -logp * (1 + logp).detach()

    def validation_step(self, n_sample, device, detector):
        d_sample = self.sample(n_sample=n_sample, device=device, reject=True)

        # samples before rejection
        sample_x, sample_z = d_sample["x"], d_sample["z"]
        logp = self.log_likelihood(sample_z)
        energy = detector.predict(sample_x)
        mean_energy = energy.mean()
        entropy = -logp.mean()
        loss = -entropy + mean_energy / self.T
        img = make_grid(sample_x.cpu(), nrow=8, normalize=True, range=(0, 1))

        d_result = {
            "loss": loss.item(),
            "valid/energy_": mean_energy.item(),
            "valid/entropy_": entropy.item(),
            "valid/sample@": img,
            "valid/score": energy.detach().cpu().numpy(),
        }

        # samples after rejection
        if self.classifier is not None:
            sample_x, sample_z = d_sample["x_accept"], d_sample["z_accept"]
            logp = self.log_likelihood(sample_z)
            energy = detector.predict(sample_x)
            mean_energy = energy.mean()
            entropy = -logp.mean()
            loss = -entropy + mean_energy / self.T
            img = make_grid(sample_x.cpu(), nrow=8, normalize=True, range=(0, 1))

            d_accept = {
                "accept/energy_": mean_energy.item(),
                "accept/entropy_": entropy.item(),
                "accept/sample@": img,
                "accept/score": energy.detach().cpu().numpy(),
            }
            d_result.update(d_accept)
        return d_result


class AdversarialDistributionAE(nn.Module):
    """Adversarial Distribution defined on a continuous latent space formed by AE, VAE, ..."""

    def __init__(
        self,
        model,
        ae,
        T=1.0,
        barrier="norm",
        classifier=None,
        classifier_thres=None,
        classifier_thres_logit=None,
        classifier_mean=None,
        classifier_std=None,
        z_shape=None,
        n_step=None,
        detector=None,
        stepsize=None,
        z_bound=None,
        rng=None,
        ood_barrier_const=0.0,
        block=None,
    ):
        """
        model: sampling method. should be 'mh'
        """
        super().__init__()
        self.model = model
        self.ae = ae
        self.T = T
        self.register_buffer("baseline", torch.tensor(0.0))
        self.momentum = 0.99
        self.barrier = barrier
        self.classifier = classifier
        self.classifier_thres = classifier_thres
        if self.classifier_thres is not None:
            self.classifier_thres_logit = np.log(
                classifier_thres / (1 - classifier_thres)
            )
        else:
            self.classifier_thres_logit = classifier_thres_logit
        self.classifier_mean = classifier_mean
        self.classifier_std = classifier_std
        self.ood_barrier_const = ood_barrier_const

        # for MCMC
        self.z_shape = z_shape
        self.n_step = n_step
        self.stepsize = stepsize
        self.detector = detector
        self.z_bound = z_bound  # [z_bound, -z_bound]
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        self.block = block  # block-Gibbs sampling. interger or None

    def forward(self, z):
        """computes log probability"""
        return self.model.log_likelihood(z)

    def log_likelihood(self, z):
        return self.model.log_likelihood(z)

    def classifier_predict(self, x):
        with torch.no_grad():
            if self.classifier_mean is not None:
                x = normalize(x, mean=self.classifier_mean, std=self.classifier_std)
            return self.classifier.predict(x, logit=True)

    def sample(self, z0=None, n_sample=None, device=None, reject=False):
        assert (z0 is None) != (n_sample is None and device is None)
        if self.model == "mh":
            d_sample = self._sample_mh(z0=z0, n_sample=n_sample, device=device)
        else:
            d_sample = self._sample_model(n_sample, device)
        sample_z = d_sample.pop("x")
        d_result = {"d_sample": d_sample}
        sample_x = self.ae.decoder(sample_z)
        d_result["x"] = sample_x
        d_result["z"] = sample_z

        if reject and (self.classifier is not None):
            logit = self.classifier_predict(sample_x)
            accept = (logit <= self.classifier_thres_logit).flatten()
            d_result["x_accept"] = sample_x[accept]
            d_result["z_accept"] = sample_z[accept]
        return d_result

    def _sample_model(self, n_sample, device):
        d_sample = self.model.sample(n_sample=n_sample, device=device)
        return d_sample

    def _initial_sample(self, n_sample, device):
        shape = (n_sample,) + tuple(self.z_shape)
        return (torch.rand(shape, device=device) * 2 - 1) * self.z_bound

    def _sample_mh(self, z0=None, n_sample=None, device=None):
        assert (z0 is None) != (n_sample is None and device is None)
        if z0 is None:
            z0 = self._initial_sample(n_sample, device)
        print(self.block)
        return sample_mh(
            z0,
            self._energy_z,
            n_step=self.n_step,
            stepsize=self.stepsize,
            T=self.T,
            bound=self.z_bound,
            rng=self.rng,
            block=self.block,
        )

    def train_step(self, n_sample, device, detector, opt, clip_grad=None):
        opt.zero_grad()
        # sample
        d_sample = self.sample(n_sample=n_sample, device=device, reject=False)
        sample_x, sample_z = d_sample["x"], d_sample["z"]

        # Entropy
        logp = self.log_likelihood(sample_z.detach())
        entropy_term = self.entropy(logp).mean()

        # Energy (reward)
        # with torch.no_grad():
        detector_score = detector.predict(sample_x).to(device)
        ood_barrier = self._ood_barrier(sample_x)

        energy = detector_score + ood_barrier

        energy_term = (logp * (energy - self.baseline).detach()).mean()

        # update baseline
        self.baseline = (
            self.baseline * self.momentum + (1 - self.momentum) * energy.detach().mean()
        )
        # self.baseline = self.baseline * self.momentum + (1 - self.momentum) * detector_score.mean()

        # barrier
        if self.barrier == "norm":
            sample_barrier = torch.norm(sample_z, dim=1)
        elif self.barrier == "abs":
            sample_barrier = torch.abs(sample_z).max(1).values.to(sample_z)

        # Gradient step
        loss = (
            -entropy_term * self.T
            + energy_term
            + 10 * torch.relu(sample_barrier - 1) ** 2
        )  # / self.T
        loss = loss.mean()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        opt.step()

        d_result = {
            "loss": loss.item(),
            "train/energy_term_": energy_term.item(),
            "train/entropy_term_": entropy_term.item(),
            "x": d_sample["x"].detach().cpu(),
            "z": d_sample["z"].detach().cpu(),
            "logp": logp.detach().cpu(),
            "detector_score": detector_score.cpu(),
            "train/entropy_est_": -logp.mean().detach().cpu(),
            "train/baseline_": self.baseline.item(),
            "train/ood_barrier_": ood_barrier.mean().item(),
            "train/sample_barrier_": sample_barrier.mean().item(),
        }
        return d_result

    def _ood_barrier(self, x):
        if self.classifier is not None:
            diff = self.classifier_predict(x) - self.classifier_thres_logit
            active = (diff > 0).to(torch.float)
            ood_barrier = (torch.relu(diff) + self.ood_barrier_const) * active
            # ood_barrier = torch.relu(self.classifier.predict(x, logit=True) - self.classifier_thres_logit)
        else:
            ood_barrier = torch.zeros(len(x)).to(x)
        return ood_barrier

    def _energy_z(self, z):
        # with torch.no_grad():
        if self.z_bound is not None:
            x = self.ae.decoder(z.clamp(-self.z_bound, self.z_bound))
        else:
            x = self.ae.decoder(z)
        detector_score = self.detector.predict(x).flatten()
        ood_barrier = self._ood_barrier(x).flatten()
        energy = detector_score + ood_barrier
        if self.z_bound is not None:
            energy += 10 * (torch.relu(z.abs() - self.z_bound) ** 2).flatten(1).sum(
                axis=1
            )

        return energy

    def entropy(self, logp):
        """Note: This is not actually entropy.
        However, its gradient yields the gradient for entropy"""
        return -logp * (1 + logp).detach()

    def validation_step(self, n_sample, device, detector):
        d_sample = self.sample(n_sample=n_sample, device=device, reject=True)

        # samples before rejection
        sample_x, sample_z = d_sample["x"], d_sample["z"]
        logp = self.log_likelihood(sample_z)
        energy = detector.predict(sample_x)
        mean_energy = energy.mean()
        entropy = -logp.mean()
        loss = -entropy + mean_energy / self.T
        img = make_grid(sample_x.cpu(), nrow=8, normalize=True, range=(0, 1))

        d_result = {
            "loss": loss.item(),
            "valid/energy_": mean_energy.item(),
            "valid/entropy_": entropy.item(),
            "valid/sample@": img,
            "valid/score": energy.detach().cpu().numpy(),
        }

        # samples after rejection
        if self.classifier is not None:
            sample_x, sample_z = d_sample["x_accept"], d_sample["z_accept"]
            logp = self.log_likelihood(sample_z)
            energy = detector.predict(sample_x)
            mean_energy = energy.mean()
            entropy = -logp.mean()
            loss = -entropy + mean_energy / self.T
            img = make_grid(sample_x.cpu(), nrow=8, normalize=True, range=(0, 1))

            d_accept = {
                "accept/energy_": mean_energy.item(),
                "accept/entropy_": entropy.item(),
                "accept/sample@": img,
                "accept/score": energy.detach().cpu().numpy(),
            }
            d_result.update(d_accept)
        return d_result


class AdversarialDistributionTransform:
    def __init__(self, detector=None, sampler=None, transform="affineV0", z_bound=None):
        """
        sampler: 'mh', 'random', 'gd'
        """
        self.detector = detector
        self.sampler = sampler
        self.transform_name = transform
        if transform == "affineV0":
            self.D = 5
            self.transform = functools.partial(
                apply_affine_batch,
                a_bound=(-45, 45),
                tx_bound=(-10, 10),
                ty_bound=(-10, 10),
                scale_bound=(0.9, 1.5),
                shear_bound=(-30, 30),
            )
        elif transform == "colorV0":
            self.D = 4
            self.transform = functools.partial(
                apply_colortransform_batch,
                b_bound=(0.5, 2),
                c_bound=(0, 2),
                s_bound=(0, 2),
                h_bound=(-0.5, 0.5),
            )
        elif transform == "colorV1":
            """reduce range for brightness and contrast -- colorV0 makes human face unrecognizable"""
            self.D = 4
            self.transform = functools.partial(
                apply_colortransform_batch,
                b_bound=(0.5, 1.5),
                c_bound=(0.5, 1.5),
                s_bound=(0, 2),
                h_bound=(-0.5, 0.5),
            )
        else:
            raise ValueError(f"Invalid transform name: {transform}")

        self.z_bound = z_bound

    def energy(self, z, img):
        """
        img: a batch of images [N, 3, H, W]
        z: a batch of transform parameters [N, D]
        """
        return self.detector.predict(self.transform(img, z))

    def sample(self, img, z0=None, writer=None):
        energy = functools.partial(self.energy, img=img)
        d_sample = self.sampler.sample(energy, n_sample=len(img), device=img.device)

        # compute minimum energy
        argmin = d_sample["l_E"].argmin(dim=0)
        d_sample["min_x"] = d_sample["l_x"][argmin, range(len(argmin)), :]
        d_sample["min_E"] = d_sample["l_E"][argmin, range(len(argmin))]
        d_sample["min_img"] = self.transform(img, d_sample["min_x"])
        d_sample["last_img"] = self.transform(img, d_sample["x"])

        return d_sample


class AdversarialDistributionLinf:
    def __init__(self, detector=None, sampler=None):
        """Linf is passed to sampler"""
        self.detector = detector
        self.sampler = sampler

    def energy(self, img):
        return self.detector.predict(img)

    def sample(self, img, writer=None):
        d_sample = self.sampler.sample(self.energy, x0=img)
        d_sample["last_img"] = d_sample["x"]
        return d_sample


class AdversarialDistributionStyleGAN2:
    def __init__(
        self,
        generator,
        detector=None,
        sampler=None,
        truncation_psi=1.0,
        truncation_cutoff=None,
    ):
        """StyleGAN2 generator"""
        self.detector = detector
        self.sampler = sampler
        self.generator = generator
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

    def generate(self, z, truncation_psi=None, truncation_cutoff=None):
        with torch.no_grad():
            i = self.generator(
                z,
                None,
                truncation_psi=self.truncation_psi
                if truncation_psi is None
                else truncation_psi,
                truncation_cutoff=self.truncation_cutoff
                if truncation_cutoff is None
                else truncation_cutoff,
            )
        return ((i.detach() + 1) / 2).clamp(0, 1)

    def energy(self, z):
        return self.detector.predict(self.generate(z))

    def sample(self, z0=None, n_sample=None, device=None, img=None):
        if img is not None:
            n_sample = len(img)
            device = img.device
        d_sample = self.sampler.sample(
            self.energy, x0=z0, n_sample=n_sample, device=device
        )
        # compute minimum energy
        argmin = d_sample["l_E"].argmin(dim=0)
        d_sample["min_x"] = d_sample["l_x"][argmin, range(len(argmin)), :]
        d_sample["min_E"] = d_sample["l_E"][argmin, range(len(argmin))]
        d_sample["min_img"] = self.generate(d_sample["min_x"].to(device))
        d_sample["last_img"] = self.generate(d_sample["x"].to(device))
        return d_sample

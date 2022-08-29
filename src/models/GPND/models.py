import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.special import loggamma
from torch import nn
from torch.optim import Adam

from .novelty_detector import extract_statistics_V2, r_pdf
from .utils.jacobian import compute_jacobian


class GPNDModel(nn.Module):
    """Generative Probabilistic Novelty Detection"""

    def __init__(self, encoder, decoder, z_discr, discr):
        super().__init__()
        self.E = encoder
        self.G = decoder
        self.ZD = z_discr
        self.D = discr
        self.own_optimizer = True

        self.BCE_loss = nn.BCELoss()
        self.z_shape = None  # shape of z vector, excluding the batch dimension

        self.register_buffer("counts", torch.zeros(30))
        self.register_buffer("bin_edges", torch.zeros(31))
        self.register_buffer("gennorm_param", torch.zeros(3, self.G.z_size))

    def get_optimizer(self, opt_cfg):
        lr = opt_cfg.get("lr", 1e-4)
        G_optimizer = Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        D_optimizer = Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        GE_optimizer = Adam(
            list(self.E.parameters()) + list(self.G.parameters()),
            lr=lr,
            betas=(0.5, 0.999),
        )
        ZD_optimizer = Adam(self.ZD.parameters(), lr=lr, betas=(0.5, 0.999))
        return {
            "G_optimizer": G_optimizer,
            "D_optimizer": D_optimizer,
            "GE_optimizer": GE_optimizer,
            "ZD_optimizer": ZD_optimizer,
        }

    def _get_z_shape(self, x):
        """obatain z shape by running a dummy run"""
        with torch.no_grad():
            z = self.E(x[[0]])
        self.z_shape = z.shape[1:]

    def forward(self, x):
        return self.G(self.E(x))

    def train_step(self, x, opt):
        if self.z_shape is None:
            self._get_z_shape(x)
        G_optimizer = opt["G_optimizer"]
        D_optimizer = opt["D_optimizer"]
        GE_optimizer = opt["GE_optimizer"]
        ZD_optimizer = opt["ZD_optimizer"]
        self.G.train()
        self.D.train()
        self.E.train()
        self.ZD.train()
        # noqa

        y_real_ = torch.ones(x.shape[0]).to(x.device)
        y_fake_ = torch.zeros(x.shape[0]).to(x.device)

        # y_real_z = torch.ones(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0]).cuda()
        # y_fake_z = torch.zeros(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0]).cuda()
        y_real_z = torch.ones(x.shape[0]).to(x.device)
        y_fake_z = torch.zeros(x.shape[0]).to(x.device)

        #############################################

        self.D.zero_grad()

        D_result = self.D(x).squeeze()
        D_real_loss = self.BCE_loss(D_result, y_real_)

        z = torch.randn((x.shape[0],) + self.z_shape).to(x.device)

        x_fake = self.G(z).detach()
        D_result = self.D(x_fake).squeeze()
        D_fake_loss = self.BCE_loss(D_result, y_fake_)

        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()

        D_optimizer.step()

        #############################################

        self.G.zero_grad()

        z = torch.randn((x.shape[0],) + self.z_shape).to(x.device)

        x_fake = self.G(z)
        D_result = self.D(x_fake).squeeze()

        G_train_loss = self.BCE_loss(D_result, y_real_)

        G_train_loss.backward()
        G_optimizer.step()

        #############################################

        self.ZD.zero_grad()

        z = torch.randn((x.shape[0],) + self.z_shape).to(x.device)

        ZD_result = self.ZD(z).squeeze()
        ZD_real_loss = self.BCE_loss(ZD_result, y_real_z)

        z = self.E(x).detach()

        ZD_result = self.ZD(z).squeeze()
        ZD_fake_loss = self.BCE_loss(ZD_result, y_fake_z)

        ZD_train_loss = ZD_real_loss + ZD_fake_loss
        ZD_train_loss.backward()

        ZD_optimizer.step()

        # #############################################

        self.E.zero_grad()
        self.G.zero_grad()

        z = self.E(x)
        x_d = self.G(z)

        ZD_result = self.ZD(z).squeeze()

        E_train_loss = self.BCE_loss(ZD_result, y_real_z) * 1.0

        Recon_loss = F.binary_cross_entropy(x_d, x.detach()) * 2.0

        (Recon_loss + E_train_loss).backward()

        GE_optimizer.step()
        return {
            "loss": Recon_loss.item(),
            "D_train_loss": D_train_loss.item(),
            "G_train_loss": G_train_loss.item(),
            "ZD_train_loss": ZD_train_loss.item(),
        }

    def validation_step(self, x):
        recon = self(x)
        recon_loss = ((x - recon) ** 2).mean()
        return {
            "loss": recon_loss,
            "predict": torch.tensor(0),
            "reconstruction": recon.detach(),
        }

    def predict(self, x):
        """compute anomaly score"""
        if self.z_shape is None:
            self._get_z_shape(x)
        self._check_params_in_cpu()
        mul = 0.2  # set as the original code
        x_shape = x.shape
        z_dim = self.z_shape[0]
        # N = (cfg.MODEL.INPUT_IMAGE_SIZE * cfg.MODEL.INPUT_IMAGE_SIZE - cfg.MODEL.LATENT_SIZE) * mul
        N = (x.shape[2] * x.shape[2] - z_dim) * mul
        logC = loggamma(N / 2.0) - (N / 2.0) * np.log(2.0 * np.pi)

        def logPe_func(x, bin_edges, counts):
            # p_{\|W^{\perp}\|} (\|w^{\perp}\|)
            # \| w^{\perp} \|}^{m-n}
            return logC - (N - 1) * np.log(x) + np.log(r_pdf(x, bin_edges, counts))

        x = x.view(len(x), -1)
        # x.requires_grad = True
        x.requires_grad_(True)

        z = self.E(x.view(-1, x_shape[1], x_shape[2], x_shape[3]))
        recon_batch = self.G(z)
        z = z.squeeze(3).squeeze(2)

        include_jacobian = True
        if include_jacobian:
            # from pudb import set_trace; set_trace()
            J = compute_jacobian(x, z)
            J = J.cpu().numpy()

        z = z.cpu().detach().numpy()

        recon_batch = recon_batch.cpu().detach().numpy()
        x = x.cpu().detach().numpy()

        result = []
        for i in range(x.shape[0]):
            if include_jacobian:
                u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                logD = -np.sum(np.log(np.abs(s)))  # | \mathrm{det} S^{-1} |
                # logD = np.log(np.abs(1.0/(np.prod(s))))
            else:
                logD = 0

            p = scipy.stats.gennorm.pdf(
                z[i],
                self.gennorm_param_[0, :],
                self.gennorm_param_[1, :],
                self.gennorm_param_[2, :],
            )
            logPz = np.sum(np.log(p))

            # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
            # In this case, just assign some large negative value to make sure that the sample
            # is classified as unknown.
            if not np.isfinite(logPz):
                logPz = -1000

            distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())

            logPe = logPe_func(distance, self.bin_edges_, self.counts_)

            P = logD + logPz + logPe

            result.append(P)

        result = np.asarray(result, dtype=np.float32)

        return torch.tensor(result)

    def reset_grad(self):
        self.en_solver.zero_grad()
        self.de_solver.zero_grad()
        self.dc_solver.zero_grad()

    def _check_params_in_cpu(self):
        self.counts_ = self.counts.cpu().numpy()
        self.bin_edges_ = self.bin_edges.cpu().numpy()
        self.gennorm_param_ = self.gennorm_param.cpu().numpy()

    def prepare(self, train_dl):
        """compute statistics of training data offline to prepare novelty detection
        train_dl: DataLoader
        """
        counts, bin_edges, gennorm_param = extract_statistics_V2(
            train_dl, self, device=next(self.parameters()).device
        )
        self.register_buffer("counts", torch.tensor(counts))
        self.register_buffer("bin_edges", torch.tensor(bin_edges))
        self.register_buffer("gennorm_param", torch.tensor(gennorm_param))

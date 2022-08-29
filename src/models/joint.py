import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedAE(nn.Module):
    """autoencoder"""

    def __init__(self, encoder, decoder, predictor, recon_weight=1.0):
        """
        encoder, decoder : neural networks
        """
        super(SupervisedAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.recon_weight = recon_weight
        self.own_optimizer = False

    def forward(self, x, mode="both"):
        z = self.encode(x)
        if mode == "both":
            recon = self.decoder(z)
            pred = self.predictor(z)
            if len(pred.shape) == 4:
                pred = pred.squeeze(3).squeeze(2)
            return pred, recon
        elif mode == "pred":
            pred = self.predictor(z)
            if len(pred.shape) == 4:
                pred = pred.squeeze(3).squeeze(2)
            return pred
        elif mode == "ae":
            recon = self.decoder(z)
            return recon
        else:
            raise ValueError(f"mode should be one of both, ae, pred")

    def encode(self, x):
        z = self.encoder(x)
        return z

    def validation_step(self, x, y):
        pred, recon = self(x, mode="both")
        pred_lbl = torch.argmax(pred, dim=1)
        pred_loss = F.cross_entropy(pred, y).mean()
        recon_loss = F.mse_loss(recon, x).mean()
        loss = pred_loss + self.recon_weight * recon_loss
        return {
            "loss": loss.item(),
            "pred_prob": pred.detach().cpu(),
            "reconstruction": recon,
            "y": y.cpu(),
            "pred_lbl": pred_lbl.detach().cpu(),
        }

    def train_step(self, x, y, optimizer, mode="both", clip_grad=None):
        """
        mode : pred - train encoder and predictor
               both - train all parameters
               ae - train encoder and decoder
        """
        optimizer.zero_grad()
        if mode == "both":
            pred, recon = self.forward(x, mode=mode)
            pred_loss = F.cross_entropy(pred, y).mean()
            recon_loss = F.mse_loss(recon, x).mean()
            loss = pred_loss + self.recon_weight * recon_loss
        elif mode == "pred":
            pass
        elif mode == "ae":
            pass
        else:
            raise ValueError(f"mode should be one of both, ae, pred")
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()
        return {
            "loss": loss.item(),
            "pred_loss": pred_loss.item(),
            "recon_loss": recon_loss.item(),
        }

    def sample(self, N, z_shape=None, device="cpu"):
        if z_shape is None:
            z_shape = self.encoder.out_shape

        rand_z = torch.rand(N, *z_shape).to(device) * 2 - 1
        sample_x = self.decoder(rand_z)
        return sample_x

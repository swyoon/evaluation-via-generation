import torch
import torch.nn as nn
from transformers import ViTForImageClassification


class ViT_HF_MD(nn.Module):
    """Wrapper of HuggingFace Vision Transformer for
    out-of-distribution detection with mahalanobis distance"""

    def __init__(self, maha_statistic=None):
        super().__init__()
        self.net = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=10
        )

        if maha_statistic is not None:
            maha_statistic = torch.load(maha_statistic)
            # self.all_means = maha_statistic['all_means']
            # self.invcov = maha_statistic['invcov']
            self.register_buffer("all_means", maha_statistic["all_means"])
            self.register_buffer("invcov", maha_statistic["invcov"])
        else:
            self.all_means = torch.zeros(1, 768, 10)
            self.invcov = torch.randn(768, 768)

    def predict(self, x):
        assert isinstance(x, torch.Tensor)
        z = self.forward_prelogit(x)
        return self.forward_maha(z)

    def forward_prelogit(self, x):
        """extract represenation from ViT"""
        out = self.net(x, output_hidden_states=True)
        out = out.hidden_states[-1][:, 0, :]
        out = self.net.vit.layernorm(out)
        return out

    def forward_maha(self, z):
        """mahalanobis distance"""
        z = z.unsqueeze(-1)
        z = z - self.all_means
        op1 = torch.einsum("ijk,jl->ilk", z, self.invcov)
        op2 = torch.einsum("ijk,ijk->ik", op1, z)

        return torch.min(op2, dim=1).values

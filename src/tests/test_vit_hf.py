import torch

from models.ViT_HF.vit_ood import ViT_HF_MD


def test_vit_hf_mahalanobis():
    """Test the mahalanobis distance in the ViT-HF model."""
    model = ViT_HF_MD()
    x = torch.rand(2, 3, 224, 224)
    y = model.predict(x)
    assert y.shape == (2,)

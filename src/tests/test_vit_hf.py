import torch

from models.ViT_HF.vit_ood import ViT_HF_MD


def test_vit_hf_mahalanobis():
    """Test the mahalanobis distance in the ViT-HF model."""
    model = ViT_HF_MD()
    x = torch.rand(2, 3, 224, 224)
    y = model.predict(x)
    assert y.shape == (2,)


def test_vit_hf_mahalanobis_float_bug():
    """
    there was a strange bug that the mahalanobis distance of a sample
    is dependent on the other samples in the batch.
    it seems like it is related to floating point precision.
    the error is only reproducible with the checkpoints.
    """
    model = ViT_HF_MD(maha_statistic="pretrained/cifar_ood_vit/hf/maha-statistic.pkl")
    model.net.load_state_dict(
        torch.load("pretrained/cifar_ood_vit/hf/pytorch_model.bin")
    )
    x = torch.rand(10, 3, 224, 224) * 2 - 1
    z = model.forward_prelogit(x)
    maha1 = model.forward_maha(z)[[0]]
    maha2 = model.forward_maha(z[[0]])
    assert torch.allclose(maha1, maha2)

    from torch.optim import Adam

    opt = Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = torch.sum(maha1)
    loss.backward()
    opt.step()

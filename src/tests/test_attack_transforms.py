import torch

from attacks.transforms import apply_affine_batch, apply_colortransform_batch


def test_colortransform():
    img = torch.rand(32, 3, 32, 32)
    x = torch.rand(32, 4)
    transformed = apply_colortransform_batch(img, x)


def test_affine():
    img = torch.rand(32, 3, 32, 32)
    x = torch.rand(32, 5)
    transformed = apply_affine_batch(img, x)

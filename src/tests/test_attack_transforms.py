import time

import torch

from attacks.transforms import (
    apply_affine_batch,
    apply_affine_kornia,
    apply_colortransform_batch,
    apply_colortransform_kornia,
)


def test_colortransform():
    img = torch.rand(32, 3, 32, 32)
    x = torch.rand(32, 4)
    s = time.time()
    transformed = apply_colortransform_batch(img, x)
    e = time.time()
    print(e - s)


def test_colortransform_kornia():
    img = torch.rand(32, 3, 32, 32)
    x = torch.rand(32, 4)
    s = time.time()
    transformed = apply_colortransform_kornia(img, x)
    e = time.time()
    print(e - s)
    assert transformed.shape == img.shape


def test_affine():
    img = torch.rand(32, 3, 32, 32)
    x = torch.rand(32, 5)
    s = time.time()
    transformed = apply_affine_batch(img, x)
    e = time.time()
    # print(e - s)


def test_affine_kornia():
    img = torch.rand(32, 3, 32, 32)
    x = torch.rand(32, 5)
    s = time.time()
    transformed = apply_affine_kornia(img, x)
    e = time.time()
    # print(e - s)
    assert transformed.shape == img.shape

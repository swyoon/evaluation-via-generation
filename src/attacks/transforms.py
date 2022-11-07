"""
transforms.py
=============

"""
import kornia
import numpy as np
import torch
import torchvision.transforms.functional as TF


def apply_colortransform_batch(
    img,
    x,
    b_bound=(0.5, 1.5),
    c_bound=(0.5, 1.5),
    s_bound=(0.0, 2.0),
    h_bound=(-0.5, 0.5),
):
    """
    img: [N, 3, H, W], torch.Tensor
    x: [N, 4], torch.Tensor. bounded to [0, 1]
    b_bound, c_bound, s_bound, h_bound: bounds of brightness_factor, contrast_factor, saturation_factor, hue_factor
    brightness, contrast, saturation, hue: scalar or 1D tensor that has the length N
    """
    bb = x[:, 0] * (b_bound[1] - b_bound[0]) + b_bound[0]
    cc = x[:, 1] * (c_bound[1] - c_bound[0]) + c_bound[0]
    ss = x[:, 2] * (s_bound[1] - s_bound[0]) + s_bound[0]
    hh = x[:, 3] * (h_bound[1] - h_bound[0]) + h_bound[0]
    l_batch = []
    for img_, b, c, s, h in zip(img, bb, cc, ss, hh):
        img_ = TF.adjust_brightness(img_, brightness_factor=b)
        img_ = TF.adjust_contrast(img_, contrast_factor=c)
        img_ = TF.adjust_saturation(img_, saturation_factor=s)
        img_ = TF.adjust_hue(img_, hue_factor=h)
        l_batch.append(img_)
    return torch.stack(l_batch)


def apply_colortransform_kornia(
    img,
    x,
    b_bound=(0.5, 1.5),
    c_bound=(0.5, 1.5),
    s_bound=(0.0, 2.0),
    h_bound=(-0.5, 0.5),
):
    """
    color transform using kornia implementation.
    Kornia implementation of color transforms are different from torchvision.
    note that the hue is in radian.

    img: [N, 3, H, W], torch.Tensor
    x: [N, 4], torch.Tensor. bounded to [0, 1]
    b_bound, c_bound, s_bound, h_bound: bounds of brightness_factor, contrast_factor, saturation_factor, hue_factor
    """
    bb = x[:, 0] * (b_bound[1] - b_bound[0]) + b_bound[0]
    cc = x[:, 1] * (c_bound[1] - c_bound[0]) + c_bound[0]
    ss = x[:, 2] * (s_bound[1] - s_bound[0]) + s_bound[0]
    hh = x[:, 3] * (h_bound[1] - h_bound[0]) + h_bound[0]

    img = kornia.enhance.adjust_brightness_accumulative(img, bb)
    img = kornia.enhance.adjust_contrast_with_mean_subtraction(img, cc)
    img = kornia.enhance.adjust_saturation_with_gray_subtraction(img, ss)
    img = kornia.enhance.adjust_hue(img, hh * 2 * np.pi)
    return img


def apply_affine_batch(
    img,
    x,
    a_bound=(-90, 90),
    tx_bound=(-10, 10),
    ty_bound=(-10, 10),
    scale_bound=(0.9, 1.5),
    shear_bound=(-30, 30),
):
    """
    img: [N, 3, H, W], torch.Tensor
    x: [N, 5], bounded to [0, 1]
    """
    aa = x[:, 0] * (a_bound[1] - a_bound[0]) + a_bound[0]
    tx = x[:, 1] * (tx_bound[1] - tx_bound[0]) + tx_bound[0]
    ty = x[:, 2] * (ty_bound[1] - ty_bound[0]) + ty_bound[0]
    sc = x[:, 3] * (scale_bound[1] - scale_bound[0]) + scale_bound[0]
    sh = x[:, 4] * (shear_bound[1] - shear_bound[0]) + shear_bound[0]
    l_batch = []
    for img_, a_, tx_, ty_, sc_, sh_ in zip(img, aa, tx, ty, sc, sh):
        img_ = TF.affine(
            img_,
            angle=a_.item(),
            translate=[tx_.item(), ty_.item()],
            scale=sc_.item(),
            shear=sh_.item(),
        )
        l_batch.append(img_)
    return torch.stack(l_batch)


def apply_affine_kornia(
    img,
    x,
    a_bound=(-90, 90),
    tx_bound=(-10, 10),
    ty_bound=(-10, 10),
    scale_bound=(0.9, 1.5),
    shear_bound=(-1 / np.sqrt(3), 1 / np.sqrt(3)),
):

    """
    affine transform using kornia.
    Differentiable and supports batch processing.

    img: [N, 3, H, W], torch.Tensor
    x: [N, 5], bounded to [0, 1]
    a_bound: rotations angle bound
    """
    assert img.shape[0] == x.shape[0]
    assert len(img.shape) == 4
    assert img.shape[2] == img.shape[3]
    center = float(img.shape[2] - 1) / 2

    aa = x[:, 0] * (a_bound[1] - a_bound[0]) + a_bound[0]
    tx = x[:, 1] * (tx_bound[1] - tx_bound[0]) + tx_bound[0]
    ty = x[:, 2] * (ty_bound[1] - ty_bound[0]) + ty_bound[0]
    sc = x[:, 3] * (scale_bound[1] - scale_bound[0]) + scale_bound[0]
    sh = x[:, 4] * (shear_bound[1] - shear_bound[0]) + shear_bound[0]

    matrix = kornia.geometry.get_affine_matrix2d(
        translations=torch.stack([tx, ty], dim=1),
        center=torch.ones(img.shape[0], 2, device=x.device) * center,
        scale=torch.stack([sc, sc], dim=1),
        angle=aa,
        sx=sh,
        sy=torch.zeros_like(sh),
    )
    return kornia.geometry.warp_affine(img, matrix[:, :2, :3], dsize=img.shape[2:])


def apply_cutout(
    img, x, cx_bound=(0, 32), cy_bound=(0, 32), w_bound=(1, 32), h_bound=(1, 32)
):
    """WIP:
    apply cutout to image batch
    img: [N, 3, H, W], torch.Tensor
    x: [N, 5], bounded to [0, 1]
    """
    cx = x[:, 0] * (cx_bound[1] - cx_bound[0]) + cx_bound[0]
    cy = x[:, 1] * (cy_bound[1] - cy_bound[0]) + cy_bound[0]
    w = x[:, 2] * (w_bound[1] - w_bound[0]) + w_bound[0]
    h = x[:, 3] * (h_bound[1] - h_bound[0]) + h_bound[0]
    l_batch = []
    for img_, cx_, cy_, w_, h_ in zip(img, cx, cy, w, h):
        img_ = TF.erase(
            img_,
            i=int(cx_.item()),
            j=int(cy_.item()),
            h=int(h_.item()),
            w=int(w_.item()),
        )
        l_batch.append(img_)
    return torch.stack(l_batch)

# import jax
import numpy as np
import torch

# from models.ViT.vit_ood import ViT_Maha, ViT_Maha_torch


# def test_vit_maha():
#     vit_checkpoint = "pretrained/cifar100_ood_vit/L_16/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0--cifar100-steps_2k-lr_0.01-res_384.npz"
#     mahalanobis_statistic = (
#         "pretrained/cifar100_ood_vit/L_16/maha_intermediate_dict.pkl"
#     )
#
#     model = ViT_Maha(
#         vit_checkpoint=vit_checkpoint,
#         mahalanobis_statistic=mahalanobis_statistic,
#         relative_maha=True,
#     )
#
#     batch_x = np.random.rand(2, 32, 32, 3)
#     pred = model.predict(batch_x)
#     assert isinstance(pred, jax.numpy.ndarray)
#     assert pred.shape == (2,)
#
#
# def test_vit_maha_torch():
#     vit_checkpoint = "pretrained/cifar100_ood_vit/L_16/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0--cifar100-steps_2k-lr_0.01-res_384.npz"
#     mahalanobis_statistic = (
#         "pretrained/cifar100_ood_vit/L_16/maha_intermediate_dict.pkl"
#     )
#
#     model = ViT_Maha_torch(
#         vit_checkpoint=vit_checkpoint,
#         mahalanobis_statistic=mahalanobis_statistic,
#         relative_maha=True,
#     )
#
#     batch_x = torch.randn(2, 3, 32, 32)
#     pred = model.predict(batch_x)
#     assert pred.shape == (2,)
#     assert isinstance(pred, torch.Tensor)

import os

from .vit_ood import ViT_Maha_torch


def load_pretrained_vit_tf(cfg, root, identifier, ckpt_file, **kwargs):
    """load vit implemented in tensorflow. It is wrapped with pytorch"""
    from models.ViT.vit_ood import ViT_Maha_torch

    cfg = cfg["model"]
    cfg.pop("arch")
    checkpoint = os.path.join(root, identifier, ckpt_file)
    maha_stat_file = os.path.join(root, identifier, "maha_intermediate_dict.pkl")
    model = ViT_Maha_torch(
        vit_checkpoint=checkpoint, mahalanobis_statistic=maha_stat_file, **cfg
    )
    model.eval()
    return model, cfg

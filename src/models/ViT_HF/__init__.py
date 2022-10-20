import os

import torch

from .vit_ood import ViT_HF_MD


def load_pretrained_vit_hf(cfg, root, identifier, ckpt_file):

    cfg = cfg["model"]
    cfg.pop("arch")
    checkpoint = os.path.join(root, identifier, ckpt_file)
    maha_statistic = os.path.join(root, identifier, cfg["maha-statistic"])
    model = ViT_HF_MD(maha_statistic=maha_statistic)
    model.net.load_state_dict(torch.load(checkpoint))
    model.eval()
    return model, cfg

"""
evaluate OOD detection performance through AUROC score

Example:
    python evaluate_cifar_ood.py --dataset FashionMNIST_OOD \
            --ood MNIST_OOD,ConstantGray_OOD \
            --resultdir results/fmnist_ood_vqvae/Z7K512/e300 \
            --ckpt model_epoch_280.pkl \
            --config Z7K512.yml \
            --device 1
"""
import argparse
import copy
import os
from time import time

import numpy as np
import scipy
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils import data

from attacks import get_detector
from attacks.grad_based_attacks import batch_attack
from gpu_utils import AutoGPUAllocation
from loader import get_dataloader
from models import get_model, load_pretrained
from utils import batch_run, mkdir_p, parse_nested_args, parse_unknown_args, roc_btw_arr

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="path to detector config")
parser.add_argument(
    "--ood", type=str, help="list of OOD datasets, separated by comma", default=None
)
parser.add_argument("--device", type=str, help="device")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["CIFAR10_OOD", "RImgNet"],
    help="inlier dataset dataset",
)

parser.add_argument(
    "--in_split",
    type=str,
    help="inlier dataset split",
    choices=["training", "validation", "evaluation", "training_full"],
    default="evaluation",
)
parser.add_argument(
    "--out_split",
    type=str,
    help="outlier dataset split",
    choices=["training", "validation", "evaluation", "training_full"],
    default="evaluation",
)
parser.add_argument("--n_sample", type=int, help="number of samples", default=400)
parser.add_argument("--normalize", action="store_true", default=False)
args, unknown = parser.parse_known_args()


"""load model"""
if args.device == "cpu":
    device = f"cpu"
elif args.device == "auto":
    gpu_allocation = AutoGPUAllocation()
    device = gpu_allocation.device
else:
    device = f"cuda:{args.device}"

cfg_detector = OmegaConf.load(args.config)
model = get_detector(**cfg_detector, device=device, normalize=args.normalize)

"""output directory setting"""
result_dir = os.path.join("results", args.dataset.split("_")[0], cfg_detector.alias)
mkdir_p(result_dir)

"""load dataset"""
if args.dataset in {"MNIST_OOD", "FashionMNIST_OOD"}:
    size = 28
    channel = 1
else:
    size = 32
    channel = 3
data_dict = {
    "path": "datasets",
    "size": size,
    "channel": channel,
    "batch_size": 64,
    "n_workers": 4,
}

data_dict_ = copy.copy(data_dict)
data_dict_["dataset"] = args.dataset
data_dict_["split"] = args.in_split
in_dl = get_dataloader(data_dict_)

l_ood = [s.strip() for s in args.ood.split(",")] if args.ood is not None else []
l_ood_dl = []
for ood_name in l_ood:
    data_dict_ = copy.copy(data_dict)
    data_dict_["dataset"] = ood_name
    data_dict_["split"] = args.out_split
    dl = get_dataloader(data_dict_, subset=range(0, args.n_sample))
    dl.name = ood_name
    l_ood_dl.append(dl)

"""Compute AAUC with Graident-Based Attacks (PGD-type 1, 2, 3)"""
time_s = time()
in_pred = batch_run(model, in_dl, device=device, no_grad=False)
print(f"{time() - time_s:.3f} sec for inlier inference")
if args.in_split == "evaluation":
    in_score_file = os.path.join(result_dir, "IN_score.pkl")
else:
    in_score_file = os.path.join(result_dir, f"IN_{args.in_split}_score.pkl")
torch.save(in_pred, in_score_file)

for ood_name, dl in zip(l_ood, l_ood_dl):
    time_s = time()
    out_pred = batch_run(model, dl, device=device, no_grad=False)
    print(f"{time() - time_s:.3f} sec for outlier inference for OOD {ood_name}")

    time_s = time()
    attacked_out_pred = batch_attack(
        model, dl, device=device, no_grad=False, use_label=False
    )
    print(f"{time() - time_s:.3f} sec for outlier attack for OOD {ood_name}")
    auc = roc_btw_arr(out_pred, in_pred)
    aauc = roc_btw_arr(attacked_out_pred, in_pred)
    with open(os.path.join(result_dir, f"{ood_name}_AAUC.txt"), "w") as f:
        # f.write("auc: " + str(auc) + ", aauc: " + str(aauc))
        f.write(str(aauc))
    print(ood_name, auc, aauc)

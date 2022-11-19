"""
benchmark experiment for additional langevin optimization
"""
import sys

sys.path.append("../../")
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from attacks import get_advdist, get_detector
from gpu_utils import AutoGPUAllocation
from loader import get_dataloader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method", type=str, choices=["lgv", "mh", "random", "randomwalk", "grad"]
)
parser.add_argument("--model", type=str, choices=["oe", "prood", "vit_hf_md"])
parser.add_argument("--manifold", type=str, choices=["affineV0", "pgstylegan2_z64"])
parser.add_argument("--idx", type=int, choices=[0, 1, 2, 3, 4])
parser.add_argument("--device", default=0)
args = parser.parse_args()

if args.device == "cpu":
    device = f"cpu"
elif args.device == "auto":
    gpu_allocation = AutoGPUAllocation()
    device = gpu_allocation.device
else:
    device = f"cuda:{args.device}"


if args.method == "lgv":
    cfg = OmegaConf.load(
        f"../../results/CIFAR10/{args.model}/svhn_{args.manifold}_lgv/benchmark_{args.idx}/cifar_{args.model}_svhn_{args.manifold}_lgv.yml"
    )
    cfg["advdist"]["sampler"]["n_step"] = 200
    cfg["advdist"]["sampler"]["noise_std"] = 0.01
    cfg["advdist"]["sampler"]["stepsize"] = 0.1
    cfg["advdist"]["sampler"]["mh"] = False
elif args.method == "grad":
    cfg = OmegaConf.load(
        f"../../results/CIFAR10/{args.model}/svhn_{args.manifold}_lgv/benchmark_{args.idx}/cifar_{args.model}_svhn_{args.manifold}_lgv.yml"
    )
    cfg["advdist"]["sampler"]["n_step"] = 200
    cfg["advdist"]["sampler"]["noise_std"] = 0.0
    cfg["advdist"]["sampler"]["stepsize"] = 0.1
    cfg["advdist"]["sampler"]["mh"] = False

elif args.method == "random":
    cfg = OmegaConf.load(
        f"../../results/CIFAR10/{args.model}/svhn_{args.manifold}_random/benchmark_{args.idx}/cifar_{args.model}_svhn_{args.manifold}_random.yml"
    )
    cfg["advdist"]["sampler"]["n_step"] = 200
elif args.method == "randomwalk":
    cfg = OmegaConf.load(
        f"../../results/CIFAR10/{args.model}/svhn_{args.manifold}_mh/benchmark_{args.idx}/cifar_{args.model}_svhn_{args.manifold}_mh.yml"
    )
    cfg["advdist"]["sampler"]["n_step"] = 200
    cfg["advdist"]["sampler"]["stepsize"] = 0.01
    cfg["advdist"]["sampler"]["mh"] = False
elif args.method == "mh":
    cfg = OmegaConf.load(
        f"../../results/CIFAR10/{args.model}/svhn_{args.manifold}_mh/benchmark_{args.idx}/cifar_{args.model}_svhn_{args.manifold}_mh.yml"
    )
    cfg["advdist"]["sampler"]["n_step"] = 200
    cfg["advdist"]["sampler"]["stepsize"] = 0.01


if args.manifold == "pgstylegan2_z64":
    cfg["advdist"]["stylegan2_g"]["root"] = "../../pretrained"

cfg["device"] = device

advdist = get_advdist(cfg)
detector = get_detector(**cfg, normalize=True, root="../../")
detector.to(device)

advdist.detector = detector


d = cfg["data"]["out_eval"]
d["path"] = "../../datasets"

dl = get_dataloader(d, subset=range(0, 100))
xx, _ = next(iter(dl))

z = torch.load(
    f"../../results/CIFAR10/{args.model}/svhn_{args.manifold}_mh/benchmark_{args.idx}/advsample_z_0.pkl"
)

d_sample = advdist.sample(img=xx.to(device), z0=z.to(device))
torch.save(
    d_sample, f"result/{args.method}_{args.model}_{args.manifold}_{args.idx}.pkl"
)

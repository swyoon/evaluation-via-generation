"""
ensemble_attack.py
=========
Aggregate multiple attack results and performs ensemble.
Operates after attack result aggregation
"""
import argparse
import os
import os.path
from itertools import chain

import numpy as np
import torch
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
from tqdm import tqdm

from attacks import Detector, EnsembleDetector, get_advdist, get_detector, get_ensemble
from augmentations import get_composed_augmentations
from gpu_utils import AutoGPUAllocation
from loader import get_dataloader
from models import load_pretrained
from utils import (
    batch_run,
    mkdir_p,
    parse_nested_args,
    parse_unknown_args,
    roc_btw_arr,
    save_yaml,
)

parser = argparse.ArgumentParser()
parser.add_argument("--detector", type=str, help="detector config path", required=True)
parser.add_argument(
    "--ensemble",
    type=str,
    help="select pre-set ensemble mode",
    choices=[
        "V1",
    ],
    required=True,
)
parser.add_argument(
    "--outlier",
    type=str,
    help="select outlier manifold",
    choices=["svhn_affineV1", "celeba_colorV1"],
    required=True,
)
parser.add_argument("--device", default=0)
parser.add_argument("--logdir", default="results/")
parser.add_argument("--n_sample", type=int, help="number of samples. None means all")
parser.add_argument(
    "--strict",
    help="raise exception when there is no attack result",
    action="store_true",
    default=False,
)
args, unknown = parser.parse_known_args()


"""parse unknown argument"""
# d_cmd_cfg = parse_unknown_args(unknown)
# d_cmd_cfg = parse_nested_args(d_cmd_cfg)
# print(d_cmd_cfg)

detector_cfg = OmegaConf.load(args.detector)

if args.device == "cpu":
    device = f"cpu"
elif args.device == "auto":
    gpu_allocation = AutoGPUAllocation()
    device = gpu_allocation.device
else:
    device = f"cuda:{args.device}"

# cfg = OmegaConf.merge(cfg, d_cmd_cfg)
# cfg["device"] = device
# print(OmegaConf.to_yaml(cfg))


"""prepare ensemble"""
if args.ensemble == "V1":
    print("ensemble mode: V1")
    l_attack = ["mh", "random", "grad", "lgv"]
    l_manifold = [None]  # no manifold ensemble
    print("outlier manifold: {}".format(args.outlier))
    print(l_attack)
    print(l_manifold)
else:
    raise NotImplementedError

attack_basename = f"{args.outlier}_ensemble{args.ensemble}"


"""prepare result directory"""
result_dir = os.path.join(
    args.logdir,
    detector_cfg.get("indist_dataset", "CIFAR10"),
    "pairwise",
    detector_cfg["alias"],
    detector_cfg["alias"],
)
ensemble_dir = os.path.join(result_dir, attack_basename)
mkdir_p(ensemble_dir)
print("Result directory: {}".format(ensemble_dir))


"""load pretrained model"""
# detector = get_detector(**cfg, normalize=True)


"""do ensemble"""
l_rank = []
l_sample = []
l_sample_x = []
l_sample_z = []
l_score = []
for manifold in l_manifold:
    for attack in l_attack:
        print(f"Ensemble {manifold} {attack}")
        if manifold is not None:
            each_result_dir = os.path.join(
                result_dir, f"{args.outlier}_{manifold}_{attack}"
            )
        else:
            each_result_dir = os.path.join(result_dir, f"{args.outlier}_{attack}")
        if args.strict:
            assert os.path.exists(each_result_dir), f"{each_result_dir} does not exist"
        else:
            if not os.path.exists(each_result_dir):
                print(f"{each_result_dir} does not exist. Proceed with next attack")
                continue

        rank = torch.tensor(torch.load(os.path.join(each_result_dir, "rank.pkl")))
        sample = torch.load(os.path.join(each_result_dir, "sample.pkl"))
        sample_x = torch.load(os.path.join(each_result_dir, "sample_x.pkl"))
        sample_z = torch.load(os.path.join(each_result_dir, "sample_z.pkl"))
        score = torch.load(os.path.join(each_result_dir, "score.pkl"))

        assert len(rank) == len(sample) == len(sample_x) == len(sample_z) == len(score)

        l_rank.append(rank)
        l_sample.append(sample)
        l_sample_x.append(sample_x)
        l_sample_z.append(sample_z)
        l_score.append(score)

rank = torch.stack(l_rank, dim=0)
sample = torch.stack(l_sample, dim=0)
sample_x = torch.stack(l_sample_x, dim=0)
sample_z = torch.stack(l_sample_z, dim=0)
score = torch.stack(l_score, dim=0)  # this is unnormalized score

min_score, min_idx = torch.min(score, dim=0)

min_rank = rank[min_idx, torch.arange(len(rank[0]))]
min_sample = sample[min_idx, torch.arange(len(sample[0]))]
min_sample_x = sample_x[min_idx, torch.arange(len(sample_x[0]))]
min_sample_z = sample_z[min_idx, torch.arange(len(sample_z[0]))]


"""save result"""
torch.save(min_rank, os.path.join(ensemble_dir, "rank.pkl"))
torch.save(min_sample, os.path.join(ensemble_dir, "sample.pkl"))
torch.save(min_sample_x, os.path.join(ensemble_dir, "sample_x.pkl"))
torch.save(min_sample_z, os.path.join(ensemble_dir, "sample_z.pkl"))
torch.save(min_score, os.path.join(ensemble_dir, "score.pkl"))
print("Saved all results")


"""save intermediate results as well"""
torch.save(rank, os.path.join(ensemble_dir, "rank_all.pkl"))
torch.save(sample, os.path.join(ensemble_dir, "sample_all.pkl"))
torch.save(sample_x, os.path.join(ensemble_dir, "sample_x_all.pkl"))
torch.save(sample_z, os.path.join(ensemble_dir, "sample_z_all.pkl"))
torch.save(score, os.path.join(ensemble_dir, "score_all.pkl"))
print("Saved all intermediate results")


"""print auc """
# load in-distribution score
in_test_score = torch.load(
    os.path.join(
        "results", detector_cfg["indist_dataset"], detector_cfg["alias"], "IN_score.pkl"
    )
)
auc = roc_btw_arr(min_score, in_test_score)
print("AUC:", auc)
with open(os.path.join(ensemble_dir, f"auc_{args.idx}.txt"), "w") as f:
    f.write(str(auc))

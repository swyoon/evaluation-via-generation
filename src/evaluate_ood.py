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
from loader import get_dataloader
from models import get_model, load_pretrained
from utils import batch_run, mkdir_p, parse_nested_args, parse_unknown_args, roc_btw_arr

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="path to detector config")
parser.add_argument("--ood", type=str, help="list of OOD datasets, separated by comma")
parser.add_argument("--device", type=str, help="device")
parser.add_argument(
    "--dataset",
    type=str,
    choices=[
        "MNIST_OOD",
        "CIFAR10_OOD",
        "CIFAR100_OOD",
    ],
    help="inlier dataset dataset",
)

parser.add_argument(
    "--split", type=str, help="test dataset split", choices=["test", "val"]
)
args, unknown = parser.parse_known_args()
d_cmd_cfg = parse_unknown_args(unknown)
d_cmd_cfg = parse_nested_args(d_cmd_cfg)


"""load model"""
device = f"cuda:{args.device}"
cfg_detector = OmegaConf.load(args.config)
model = get_detector(**cfg_detector, device=device, normalize=False)

"""output directory setting"""
result_dir = os.path.join("results", args.dataset.split("_")[0], cfg_detector.alias)
mkdir_p(result_dir)


def evaluate(m, in_dl, out_dl, device):
    """computes OOD detection score"""
    in_pred = batch_run(m, in_dl, device, method="predict")
    out_pred = batch_run(m, out_dl, device, method="predict")
    auc = roc_btw_arr(out_pred, in_pred)
    return auc


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

# choose split
if args.split == "val":
    data_dict["split"] = "validation"
else:
    data_dict["split"] = "evaluation"


data_dict_ = copy.copy(data_dict)
data_dict_["dataset"] = args.dataset
in_dl = get_dataloader(data_dict_)

l_ood = [s.strip() for s in args.ood.split(",")]
l_ood_dl = []
for ood_name in l_ood:
    data_dict_ = copy.copy(data_dict)
    data_dict_["dataset"] = ood_name
    dl = get_dataloader(data_dict_)
    dl.name = ood_name
    l_ood_dl.append(dl)


"""Compute AUROC and save results"""
time_s = time()
in_pred = batch_run(model, in_dl, device=device, no_grad=False)
print(f"{time() - time_s:.3f} sec for inlier inference")
in_score_file = os.path.join(result_dir, f"IN_score.pkl")
torch.save(in_pred, in_score_file)

l_ood_pred = []
for dl in l_ood_dl:
    out_pred = batch_run(model, dl, device=device, no_grad=False)
    l_ood_pred.append(out_pred)

    out_score_file = os.path.join(result_dir, f"OOD_score_{dl.name}.pkl")
    torch.save(out_pred, out_score_file)

    """Compute Minimal and Average Rank of smaple"""
    out_rank_list = []
    in_test_score_list = in_pred.tolist()
    out_score_list = out_pred.tolist()
    sorted_list = np.sort(in_test_score_list)
    out_rank_list = np.searchsorted(sorted_list, out_score_list)
    print(f"MinRank: {out_rank_list.min()}")
    torch.save(out_rank_list, os.path.join(result_dir, f"OOD_rank_{dl.name}.pkl"))

    # idx = np.arange(len(out_rank_list))
    # rng = np.random.default_rng(1)
    # rng.shuffle(idx)
    # out_rank_list = out_rank_list[idx]
    # n_split = 5
    # split_size = len(out_rank_list) // n_split
    # l_min = []
    # for i in range(n_split):
    #     l_min.append((out_rank_list[split_size * i: split_size * (i+1)]).min())

    # avg_minrank = np.mean(l_min)
    # sem_minrank = scipy.stats.sem(l_min, ddof=0)
    # print(f'[{dl.name}] AvgMinRank: {avg_minrank}  SemMinRank {sem_minrank}')


l_ood_auc = []
for pred in l_ood_pred:
    l_ood_auc.append(roc_btw_arr(pred, in_pred))
print(l_ood_auc)

for ood_name, auc in zip(l_ood, l_ood_auc):
    with open(os.path.join(result_dir, f"{ood_name}.txt"), "w") as f:
        f.write(str(auc))

# print('out_rank',out_rank_list[:10])
# if args.testdataset is not None:
#     ood_name = 'auc_'
#     for test in l_ood:
#         ood_name += test + '_'
#     with open(os.path.join(result_dir, f'{ood_name}_rank.txt'), 'w') as f:
#         f.write(str(min_rank)  + '\n')
#         f.write(str(avg_rank)  + '\n')
# else:
#     for ood_name, auc in zip(l_ood, l_ood_auc):
#         with open(os.path.join(model_dir, f'{ood_name}_rank.txt'), 'w') as f:
#             f.write(str(min_rank)  + '\n')
#             f.write(str(avg_rank)  + '\n')

print("")
"""
evaluate OOD detection performance
using samples from adversarial/detector distribution through AUROC score

Example:
    python evaluate_adv_samples.py
            --resultdir results_attack/cifar_mh_md/z32nh8 \
            --config cifar_mh_md.yml \
            --device 1
"""
import argparse
import copy
import glob
import os

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from attacks import Detector, EnsembleDetector, get_detector
from augmentations import get_composed_augmentations
from loader import get_dataloader
from models import get_model, load_pretrained
from utils import batch_run, mkdir_p, parse_nested_args, parse_unknown_args, roc_btw_arr

parser = argparse.ArgumentParser()
parser.add_argument("--detector", type=str, help="detector config file path")
parser.add_argument(
    "--target",
    type=str,
    help="path to the target config file. advsamples \
                                                should be present in the same directory",
)
parser.add_argument(
    "--logdir",
    type=str,
    help="directory where evaluation result to be stored \
                                                log dir. results/... or pretrained/...",
)
parser.add_argument("--device", default=0, type=str, help="device")

args, unknown = parser.parse_known_args()
d_cmd_cfg = parse_unknown_args(unknown)
d_cmd_cfg = parse_nested_args(d_cmd_cfg)


"""parse unknown argument"""
d_cmd_cfg = parse_unknown_args(unknown)
d_cmd_cfg = parse_nested_args(d_cmd_cfg)
print(d_cmd_cfg)
if args.device == "cpu":
    device = f"cpu"
else:
    device = f"cuda:{args.device}"

# load config file
cfg_detector = OmegaConf.load(args.detector)
cfg_target = OmegaConf.load(args.target)
target_dir = os.path.dirname(args.target)

"""load pretrained model"""
detector = get_detector(**cfg_detector, device=device, normalize=True)
no_grad = cfg_detector.get("detector_no_grad", False)

# train_dl = get_dataloader(cfg_target['data']['indist_val'])
# do_ensemble = any([k.startswith('ensemble') for k in cfg_detector['detector'].keys()])
#
# if do_ensemble:
#     detector = get_ensemble(device=device, **cfg_detector['detector'])
#     detector.to(device)
#     detector.learn_normalization(dataloader=train_dl, device=device).detach().cpu().numpy()
#     no_grad = detector.no_grad
# else:
#     detector, _ = load_pretrained(**cfg_detector['detector'], device=device)
#
#     if 'detector_aug' in cfg_detector:
#         aug = get_composed_augmentations(cfg_detector['detector_aug'])
#     else:
#         aug = None
#     no_grad = cfg_detector.get('detector_no_grad', False)
#     detector = Detector(detector, bound=-1, transform=aug, no_grad=no_grad, use_rank=False)
#     detector.to(device)

"""load inlier dataset"""
# data_cfg = {'dataset': 'CIFAR10_OOD',
#             'path': 'datasets',
#             'batch_size': 128,
#             'n_workers': 4,
#             'split': 'evaluation'}
#
# test_dl = get_dataloader(data_cfg)
print("In-distribution: ", "CIFAR10_OOD")

# sample_model_name_list = ['ae', 'csi', 'glow', 'md', 'nae', 'oe', 'pixelcnn']
# sample_ae_cfg_list = ['z32', 'z64']
# for sample_model_name in sample_model_name_list:
#     for sample_ae_cfg in sample_ae_cfg_list:

# sample_model_name = args.samplemodel
# sample_ae_cfg = args.latentcfg
# detector_dir = 'cifar_mh_' + sample_model_name
# sub_dir = 'train_std_norm_block=1/iteration=0'


"""load ood samples from adversarial/detector distribution"""
file_list = sorted(glob.glob(os.path.join(target_dir, "advsample_x_*.pkl")))
assert len(file_list) > 0, "No advsamples detected"
l_sample = []
for file_path in file_list:
    l_sample.append(torch.load(file_path))
x_saved_samples = torch.cat(l_sample)
print("x sample shape", x_saved_samples.shape)

"""concat z samples"""
file_list = sorted(glob.glob(os.path.join(target_dir, "advsample_z_*.pkl")))
assert len(file_list) > 0, "No advsamples detected"
l_sample = []
for file_path in file_list:
    l_sample.append(torch.load(file_path))
z_saved_samples = torch.cat(l_sample)
print("z sample shape", z_saved_samples.shape)


"""Compute AUC score"""
sample_batch_size = 128
in_test_score = torch.load(
    os.path.join("results", "CIFAR10", cfg_detector["alias"], "IN_score.pkl")
)
# in_test_score = batch_run(detector, test_dl, device=device, no_grad=no_grad, normalize=False)
out_dl = DataLoader(TensorDataset(x_saved_samples), batch_size=sample_batch_size)
out_score = batch_run(detector, out_dl, device=device, no_grad=no_grad, normalize=False)
auc = roc_btw_arr(out_score, in_test_score)
print(f"OOD samples: AUC:{auc}")


# detector_name = os.path.basename(args.detector).split(".")[0]
# target_name = os.path.basename(args.target).split(".")[0]
result_dir = args.logdir
mkdir_p(result_dir)

auc_save_path = os.path.join(result_dir, f"auc.txt")
with open(auc_save_path, "w") as f:
    f.write(str(auc))
    print("save auc score", auc_save_path)

"""Save samples OOD score"""
out_score_path = os.path.join(result_dir, f"score.pkl")
torch.save(out_score, out_score_path)

out_x_path = os.path.join(result_dir, f"sample_x.pkl")
torch.save(x_saved_samples, out_x_path)

out_x_path = os.path.join(result_dir, f"sample.pkl")
torch.save(x_saved_samples, out_x_path)

out_z_path = os.path.join(result_dir, f"sample_z.pkl")
torch.save(z_saved_samples, out_z_path)

sorted_in_score = np.sort(in_test_score)
out_rank = np.searchsorted(sorted_in_score, out_score)
torch.save(out_rank, os.path.join(result_dir, f"rank.pkl"))
print("save OOD score of all samples", out_score_path)

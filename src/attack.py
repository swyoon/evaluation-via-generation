"""
attack.py
=========
Performs MCMC-based adversarial attack
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
parser.add_argument("--detector", type=str)
parser.add_argument("--attack", type=str)
parser.add_argument("--config", type=str)
parser.add_argument("--device", default=0)
parser.add_argument("--logdir", default="results/")
parser.add_argument("--run", default=None, help="unique run id of the experiment")
parser.add_argument("--n_sample", type=int, help="number of samples. None means all")
parser.add_argument(
    "--save_intermediate", action="store_true", help="save intermediate results"
)
parser.add_argument(
    "--idx",
    default=0,
    type=int,
    help="split index. the output will be named as advdist_{idx}.npy. default: 0",
)
parser.add_argument(
    "--split",
    default=1,
    type=int,
    help="n_sample is splitted into {split} for parallel computation.",
)
args, unknown = parser.parse_known_args()


"""parse unknown argument"""
d_cmd_cfg = parse_unknown_args(unknown)
d_cmd_cfg = parse_nested_args(d_cmd_cfg)
print(d_cmd_cfg)

detector_cfg = OmegaConf.load(args.detector)
attack_cfg = OmegaConf.load(args.attack)
cfg = OmegaConf.merge(detector_cfg, attack_cfg)
detector_basename = os.path.basename(args.detector).split(".")[0]
attack_basename = os.path.basename(args.attack).split(".")[0]
config_basename = f"{detector_basename}_{attack_basename}"

if args.device == "cpu":
    device = f"cpu"
elif args.device == "auto":
    gpu_allocation = AutoGPUAllocation()
    device = gpu_allocation.device
else:
    device = f"cuda:{args.device}"

cfg = OmegaConf.merge(cfg, d_cmd_cfg)
cfg["device"] = device
print(OmegaConf.to_yaml(cfg))


"""prepare result directory"""
run_id = args.run
result_dir = os.path.join(
    args.logdir,
    detector_cfg.get("indist_dataset", "CIFAR10"),
    detector_cfg["alias"],
    attack_basename,
)
logdir = os.path.join(result_dir, str(run_id))
mkdir_p(logdir)
tensorboard_dir = os.path.join(result_dir + "_tensorboard", str(run_id))
writer_logdir = os.path.join(tensorboard_dir, f"split_{args.idx}")
writer = SummaryWriter(logdir=writer_logdir)
print("Result directory: {}".format(logdir))

"""copy config file"""
copied_yml = os.path.join(logdir, config_basename + ".yml")
save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
print(f"config saved as {copied_yml}")


"""initialize adversarial distribution"""
advdist = get_advdist(cfg)
advdist.sampler.writer = writer
# advdist.to(device)

"""load pretrained model"""
detector = get_detector(**cfg, normalize=True)
no_grad = cfg.get("detector_no_grad", False)
whitebox = cfg.get("whitebox", False)
advdist.detector = detector


"""preprocessing"""

# classifier score checking
if hasattr(advdist, "classifier") and advdist.clasifier is not None:
    pred = batch_run(advdist, train_dl, device=device, method="classifier_predict")
    ratio_in_support = (advdist.classifier_thres_logit < pred).sum() / len(pred)

    print("Ratio of inliers inside classifier boundary", ratio_in_support.item())
    print("minimum logit of inliers", pred.min().item())
    print("minimum prob of inliers", torch.sigmoid(pred.min()).item())


"""output filenames"""
if args.idx is not None:
    out_file_x = os.path.join(logdir, f"advsample_x_{args.idx}.pkl")
    out_file_z = os.path.join(logdir, f"advsample_z_{args.idx}.pkl")
    out_file_E = os.path.join(logdir, f"advsample_E_{args.idx}.pkl")
    out_file_score = os.path.join(logdir, f"advsample_score_{args.idx}.pkl")
    out_file_xlast = os.path.join(logdir, f"advsample_xlast_{args.idx}.pkl")
    out_file_l_E = os.path.join(logdir, f"advsample_l_E_{args.idx}.pkl")
    out_file_l_z = os.path.join(logdir, f"advsample_l_z_{args.idx}.pkl")
else:
    out_file_x = os.path.join(logdir, "advsample_x.pkl")
    out_file_z = os.path.join(logdir, "advsample_z.pkl")


"""load test OOD dataset"""
d_eval_data = cfg["data"]["out_eval"]
if args.n_sample is not None:
    n_testset = args.n_sample
else:
    if d_eval_data["dataset"] == "SVHN_OOD":
        n_testset = 26032
    elif d_eval_data["dataset"] == "CelebA_OOD":
        n_testset = 19962
    else:
        raise ValueError(f"invalid dataset")
    print(f'Using the whole test set of {d_eval_data["dataset"]}: {n_testset}')

batch_size = d_eval_data["batch_size"]
n_batch = np.ceil(n_testset / batch_size)
batch_per_split = n_batch // args.split

start = int((args.idx) * batch_per_split * batch_size)
if args.idx == args.split - 1:
    end = n_testset
else:
    end = int((args.idx + 1) * batch_per_split * batch_size)


print(f"Split index [{args.idx} / {args.split}]: running from {start} to {end}")
print(f"Running {end - start} MCMC chains for total.")
outval_dl = get_dataloader(d_eval_data, subset=range(start, end))


"""main MCMC loop"""
l_sample_x = []
l_sample_z = []
l_sample_E = []
l_sample_xlast = []
l_sample_l_E = []  # history of E
l_sample_l_z = []  # history of z
for i_iter, (x, y) in enumerate(tqdm(outval_dl)):
    x = x.to(device)
    d_sample = advdist.sample(img=x)
    l_sample_x.append(d_sample["min_img"].detach().cpu())
    l_sample_z.append(d_sample["min_x"].detach().cpu())
    l_sample_E.append(d_sample["min_E"].detach().cpu())
    l_sample_xlast.append(d_sample["last_img"].detach().cpu())

    if args.save_intermediate:
        l_sample_l_E.append(d_sample["l_E"].detach().cpu())
        l_sample_l_z.append(d_sample["l_x"].detach().cpu())

    # save every new batch, to become fault-tolerant
    adv_samples_x = torch.cat(l_sample_x)
    adv_samples_z = torch.cat(l_sample_z)
    adv_samples_E = torch.cat(l_sample_E)
    adv_samples_xlast = torch.cat(l_sample_xlast)
    torch.save(adv_samples_x, out_file_x)
    torch.save(adv_samples_z, out_file_z)
    torch.save(adv_samples_E, out_file_E)
    torch.save(adv_samples_xlast, out_file_xlast)

    if args.save_intermediate:
        adv_samples_l_E = torch.cat(l_sample_l_E)
        adv_samples_l_z = torch.cat(l_sample_l_z)
        torch.save(adv_samples_l_E, out_file_l_E)
        torch.save(adv_samples_l_z, out_file_l_z)

    result_img = make_grid(d_sample["min_img"].detach().cpu(), nrow=8, range=(0, 1))
    writer.add_image(f"min_img", result_img, i_iter)
    result_img = make_grid(d_sample["last_img"].detach().cpu(), nrow=8, range=(0, 1))
    writer.add_image(f"last_img", result_img, i_iter)
print(f"x samples saved in {out_file_x}")
print(f"z samples saved in {out_file_z}")
print(f"E samples saved in {out_file_E}")
print(f"xlast samples saved in {out_file_xlast}")
print("Adversarial samples with shape", adv_samples_x.shape)

"""print auc """
test_dl = get_dataloader(cfg["data"]["indist_test"])
in_test_score = batch_run(
    detector, test_dl, device=device, normalize=False, no_grad=no_grad
)

out_dl = DataLoader(TensorDataset(adv_samples_x), batch_size=batch_size)
out_score = batch_run(detector, out_dl, device=device, normalize=False, no_grad=no_grad)

auc = roc_btw_arr(out_score, in_test_score)
print("AUC:", auc)
with open(os.path.join(logdir, f"auc_{args.idx}.txt"), "w") as f:
    f.write(str(auc))
torch.save(out_score, out_file_score)

sorted_in_score = np.sort(in_test_score)
out_rank = np.searchsorted(sorted_in_score, out_score)
print(f"{out_rank.min()}")

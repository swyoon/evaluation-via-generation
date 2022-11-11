"""
loop across result directories and see if each result is there
"""

import argparse
import glob
import itertools
import os

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="CIFAR10")

args = parser.parse_args()


root = "results"
if args.dataset == "CIFAR10":
    l_models = [
        "glow",
        "pixelcnn",
        "ae",
        "nae",
        "good",
        "acet",
        "ceda",
        "ssd",
        "md",
        "sngp",
        "atom",
        "oe",
        "rowl",
        "csi",
        "prood",
        "vit_hf_md",
    ]
    l_ood = ["svhn", "celeba"]
    l_attack = [
        "affineV1_random",
        "affineV1_mh",
        "colorV1_random",
        "colorV1_mh",
        "pgstylegan2_z16_random",
        "pgstylegan2_z16_mh",
    ]
elif args.dataset == "RImgNet":
    l_models = ["msp", "oe", "prood", "vit_hf_md"]
    l_ood = ["fgvc", "flowers", "eurosat"]
    l_attack = [
        "affineV2_random",
        "affineV2_mh",
        "colorV1_random",
        "colorV1_mh",
        "pgstylegan2_z16_random",
        "pgstylegan2_z16_mh",
    ]
else:
    raise ValueError("dataset not supported")


print("Dataset: {}".format(args.dataset))
# loop across models
for model in l_models:
    print("Model: {}".format(model))
    # check if result directory exists
    model_dir = os.path.join(root, args.dataset, model)
    assert os.path.exists(model_dir), f"{model} result directory does not exist"
    l_result_dir = [
        s
        for s in os.listdir(model_dir)
        if not s.endswith("tensorboard")
        and not s.endswith(".txt")
        and not s.endswith(".pkl")
    ]

    for ood, attack in itertools.product(l_ood, l_attack):
        result_dir = os.path.join(model_dir, f"{ood}_{attack}", "run")
        file_list = sorted(glob.glob(os.path.join(result_dir, "advsample_x_*.pkl")))
        l_sample = []
        for file_path in file_list:
            l_sample.append(torch.load(file_path))

        if len(l_sample) == 0:
            n_samples = 0
        else:
            x_saved_samples = torch.cat(l_sample)
            n_samples = len(x_saved_samples)

        status = f"sample: {n_samples}"

        print(f"    {ood}_{attack}: {status}")

print()
print()
print("###############   Merged Results")
print()
# check if merged result exists
for model in l_models:
    print("Model: {}".format(model))
    model_dir = os.path.join(root, args.dataset, "pairwise", model, model)
    assert os.path.exists(model_dir), f"{model_dir} result directory does not exist"
    l_result_dir = [
        s
        for s in os.listdir(model_dir)
        if not s.endswith("tensorboard")
        and not s.endswith(".txt")
        and not s.endswith(".pkl")
    ]

    for ood, attack in itertools.product(l_ood, l_attack):
        result_dir = os.path.join(model_dir, f"{ood}_{attack}")
        if not os.path.exists(result_dir):
            n_samples = 0
        else:
            samples = torch.load(os.path.join(result_dir, "sample_x.pkl"))

            n_samples = len(samples)

        status = f"\t{n_samples}"
        print(f"    {ood}_{attack}: {status}")

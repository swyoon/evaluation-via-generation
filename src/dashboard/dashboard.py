"""
streamlit dashboard

streamlit run dashboard.py --server.port 8890 --server.address 0.0.0.0
"""

import os

import pandas as pd
import streamlit as st
import torch

l_model = ["atom", "csi", "prood", "vit_hf_md"]
l_variation = [
    "svhn_stylegan2ada_z16_mh",
    "celeba_stylegan2ada_z16_mh",
    # "svhn_stylegan2ada_z32_mh",
    # "svhn_stylegan2ada_z512_mh",
]

result_dir = "../results/CIFAR10/pairwise/"

df = {"model": []}
for variation in l_variation:
    df[variation] = []


for m in l_model:
    df["model"].append(m)

    for variation in l_variation:
        leaf_dir = os.path.join(result_dir, m, m, variation)
        try:
            rank = torch.load(os.path.join(leaf_dir, "rank.pkl"))
            score = torch.load(os.path.join(leaf_dir, "score.pkl"))
            sample = torch.load(os.path.join(leaf_dir, "sample.pkl"))
            with open(os.path.join(leaf_dir, "auc.txt")) as f:
                auc = float(f.read().strip())

            df[variation].append(rank.min().item())
        except FileNotFoundError:
            df[variation].append(-1)

df = pd.DataFrame(df)
df

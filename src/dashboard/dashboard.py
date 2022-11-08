"""
streamlit dashboard

streamlit run dashboard.py --server.port 8890 --server.address 0.0.0.0
"""

import os

import pandas as pd
import streamlit as st
import torch

st.set_page_config(layout="wide")

st.header("CIFAR10")

l_model = [
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
l_variation = [
    "svhn_clean",
    "svhn_affineV1_mh",
    "svhn_colorV2_mh",
    "svhn_stylegan2ada_z16_mh",
    "celeba_clean",
    "celeba_affineV1_mh",
    "celeba_colorV2_mh",
    "celeba_stylegan2ada_z16_mh",
    # "svhn_stylegan2ada_z32_mh",
    # "svhn_stylegan2ada_z512_mh",
]

result_dir = "../results/CIFAR10/pairwise/"
clean_dir = "../results/CIFAR10/"

df = {"model": []}
for variation in l_variation:
    df[variation + "_auc"] = []
    df[variation + "_rank"] = []


for m in l_model:
    df["model"].append(m)

    for variation in l_variation:
        if variation == "svhn_clean":
            result_file = os.path.join(clean_dir, m, f"SVHN_OOD.txt")
            with open(result_file, "r") as f:
                auc = float(f.readline().strip())

            rank = (
                torch.load(os.path.join(clean_dir, m, f"OOD_rank_SVHN_OOD.pkl"))
                .min()
                .item()
            )
            df[variation + "_auc"].append(auc)
            df[variation + "_rank"].append(rank)
        elif variation == "celeba_clean":
            result_file = os.path.join(clean_dir, m, f"CelebA_OOD.txt")
            with open(result_file, "r") as f:
                auc = float(f.readline().strip())

            rank = (
                torch.load(os.path.join(clean_dir, m, f"OOD_rank_CelebA_OOD.pkl"))
                .min()
                .item()
            )
            df[variation + "_auc"].append(auc)
            df[variation + "_rank"].append(rank)
        else:
            leaf_dir = os.path.join(result_dir, m, m, variation)
            try:
                rank = torch.load(os.path.join(leaf_dir, "rank.pkl"))
                score = torch.load(os.path.join(leaf_dir, "score.pkl"))
                sample = torch.load(os.path.join(leaf_dir, "sample.pkl"))
                with open(os.path.join(leaf_dir, "auc.txt")) as f:
                    auc = float(f.read().strip())

                df[variation + "_rank"].append(rank.min().item())
                df[variation + "_auc"].append(auc)
            except FileNotFoundError:
                df[variation + "_rank"].append(-1)
                df[variation + "_auc"].append(-1)

df = pd.DataFrame(df)
l_col = [
    "model",
    "svhn_clean_auc",
    "svhn_affineV1_mh_auc",
    "svhn_colorV2_mh_auc",
    "celeba_clean_auc",
    "celeba_affineV1_mh_auc",
    "celeba_colorV2_mh_auc",
    "svhn_clean_rank",
    "svhn_stylegan2ada_z16_mh_rank",
    "celeba_clean_rank",
    "celeba_stylegan2ada_z16_mh_rank",
]
st.table(df[l_col])


st.header("RImgNet")
l_model = ["prood", "vit_hf_md"]
l_variation = [
    "cars_clean",
    "cars_affine",
    "cars_colorV1",
    "cars_stylegan2ada_z16_mh",
    "fgvc_clean",
    "fgvc_affine",
    "fgvc_colorV1",
    "fgvc_stylegan2ada_z16_mh",
    "flowers_clean",
    "flowers_affine",
    "flowers_colorV1",
    "flowers_stylegan2ada_z16_mh",
]

result_dir = "../results/RImgNet/pairwise/"
clean_dir = "../results/RImgNet/"

df = {"model": []}
for variation in l_variation:
    df[variation + "_auc"] = []
    df[variation + "_rank"] = []

for m in l_model:
    df["model"].append(m)

    for variation in l_variation:
        if variation.endswith("_clean"):
            dataset_name = variation.split("_")[0]
            if dataset_name == "cars":
                dataset_name = "Cars"
            elif dataset_name == "fgvc":
                dataset_name = "FGVC"
            elif dataset_name == "flowers":
                dataset_name = "Flowers"
            else:
                raise ValueError
            result_file = os.path.join(clean_dir, m, f"{dataset_name}.txt")
            with open(result_file, "r") as f:
                auc = float(f.readline().strip())

            rank = (
                torch.load(os.path.join(clean_dir, m, f"OOD_rank_{dataset_name}.pkl"))
                .min()
                .item()
            )
            df[variation + "_auc"].append(auc)
            df[variation + "_rank"].append(rank)
        else:
            leaf_dir = os.path.join(result_dir, m, m, variation)
            try:
                rank = torch.load(os.path.join(leaf_dir, "rank.pkl"))
                score = torch.load(os.path.join(leaf_dir, "score.pkl"))
                sample = torch.load(os.path.join(leaf_dir, "sample.pkl"))
                with open(os.path.join(leaf_dir, "auc.txt")) as f:
                    auc = float(f.read().strip())

                df[variation + "_rank"].append(rank.min().item())
                df[variation + "_auc"].append(auc)
            except FileNotFoundError:
                df[variation + "_rank"].append(-1)
                df[variation + "_auc"].append(-1)

df = pd.DataFrame(df)
l_col = [
    "model",
    "cars_clean_auc",
    "cars_affine_auc",
    "cars_colorV1_auc",
    "fgvc_clean_auc",
    "fgvc_affine_auc",
    "fgvc_colorV1_auc",
    "flowers_clean_auc",
    "flowers_affine_auc",
    "flowers_colorV1_auc",
    "cars_clean_rank",
    "cars_stylegan2ada_z16_mh_rank",
    "fgvc_clean_rank",
    "fgvc_stylegan2ada_z16_mh_rank",
    "flowers_clean_rank",
    "flowers_stylegan2ada_z16_mh_rank",
]
st.table(df[l_col])

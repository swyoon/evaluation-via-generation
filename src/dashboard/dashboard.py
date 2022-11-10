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
    # "svhn_linf",
    # "celeba_linf"
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
    # "svhn_linf_auc",
    "svhn_affineV1_mh_auc",
    "svhn_colorV2_mh_auc",
    "celeba_clean_auc",
    # "celeba_linf_auc",
    "celeba_affineV1_mh_auc",
    "celeba_colorV2_mh_auc",
    "svhn_clean_rank",
    "svhn_stylegan2ada_z16_mh_rank",
    "celeba_clean_rank",
    "celeba_stylegan2ada_z16_mh_rank",
]
st.table(df[l_col])

"""generate latex code for table"""
s = ""
for i, row in df[l_col].iterrows():
    if row["model"] == "vit_hf_md":
        s += f"ViT"
    else:
        s += f'{row["model"].upper()}'
    for j, col in enumerate(l_col[1:]):
        if col.endswith("auc"):
            if row[col] == -1:
                s += f" & "
            elif row[col] > 0.999:
                s += "  & " + f"{row[col]:0.3f}"
            else:
                s += "  & " + f"{row[col]:0.3f}"[1:]
        else:
            s += f"  & {row[col]}"

    if i == 2:
        s += " \\\\ \n    \\midrule   \\multicolumn{3}{l}{\\textbf{Strong Detectors}} \\\\ \n"
    else:
        s += " \\\\ \n"


st.code(s, language="python")


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
    # "cars_clean_auc",
    # "cars_affine_auc",
    # "cars_colorV1_auc",
    "fgvc_clean_auc",
    "fgvc_affineV2_auc",
    "fgvc_colorV1_auc",
    "flowers_clean_auc",
    "flowers_affineV2_auc",
    "flowers_colorV1_auc",
    "eurosat_clean_auc",
    "eurosat_affineV2_auc",
    "eurosat_colorV1_auc",
    # "cars_clean_rank",
    # "cars_stylegan2ada_z16_mh_rank",
    "fgvc_clean_rank",
    "fgvc_pgstylegan2_z16_mh_rank",
    "flowers_clean_rank",
    "flowers_pgstylegan2_z16_mh_rank",
    "eurosat_clean_rank",
    "eurosat_pgstylegan2_z16_mh_rank",
]
st.table(df[l_col])

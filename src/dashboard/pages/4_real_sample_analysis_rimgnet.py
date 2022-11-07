import sys

sys.path.append("..")
import os

import streamlit as st
import torch
from torchvision.datasets import SVHN

from loader import get_dataloader

st.title("Real sample visualization")

# load image data
dataset = st.selectbox(
    "Select dataset",
    [
        "RImgNet",
        "Cars",
        "FGVC",
        "Flowers",
        "SVHN_OOD",
        "CelebA_OOD",
        "MNIST_OOD",
        "FashionMNIST_OOD",
        "dtd",
        "OpenImages-O",
        "EuroSAT",
    ],
)
split = st.selectbox("Select split", ["evaluation"])


@st.cache
def load_data():
    global split
    size = 224
    channel = 3
    data_dict = {
        "path": "../datasets",
        "size": size,
        "channel": channel,
        "batch_size": 64,
        "n_workers": 0,
        "split": split,
    }
    data_dict["dataset"] = f"{dataset}"
    ds = get_dataloader(data_dict).dataset
    return ds


ds = load_data()
n_image_show = st.number_input(
    "Number of images to show", min_value=1, max_value=100, value=10
)

l_selected_detector = st.multiselect(
    "Select detector",
    ["prood", "vit_hf_md"],
    default=["prood"],
)


st.subheader("Worst Inlier Visualization")

for detector in l_selected_detector:
    # load inlier score
    inlier_dir = os.path.join("../results/RImgNet/", detector)
    if dataset == "RImgNet":
        score = torch.load(os.path.join(inlier_dir, "IN_score.pkl"))
    else:
        score = torch.load(os.path.join(inlier_dir, f"OOD_rank_{dataset}.pkl"))
    in_sorted_score, in_sorted_idx = torch.sort(torch.tensor(score), descending=True)

    st.text(detector)
    st.caption("most outlier-like")
    imgs = [
        ds[in_sorted_idx[i]][0].permute(1, 2, 0).numpy() for i in range(n_image_show)
    ]
    st.image(
        imgs,
        use_column_width="auto",
        caption=[f"{in_sorted_score[i]}" for i in range(n_image_show)],
    )
    assert ds[in_sorted_idx[0]][0].shape[1] == 224

    st.caption("most inlier-like")
    in_sorted_score, in_sorted_idx = torch.sort(torch.tensor(score), descending=False)
    imgs = [
        ds[in_sorted_idx[i]][0].permute(1, 2, 0).numpy() for i in range(n_image_show)
    ]
    st.image(
        imgs,
        use_column_width="auto",
        caption=[f"{in_sorted_score[i]}" for i in range(n_image_show)],
    )

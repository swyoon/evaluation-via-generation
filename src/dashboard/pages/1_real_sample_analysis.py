import sys

sys.path.append("..")
import os

import streamlit as st
import torch
from torchvision.datasets import SVHN

from loader import get_dataloader

st.title("Real sample visualization")

# load image data
dataset = st.selectbox("Select dataset", ["SVHN", "CIFAR10", "CelebA"])


@st.cache
def load_data():
    size = 32
    channel = 3
    data_dict = {
        "path": "../datasets",
        "size": size,
        "channel": channel,
        "batch_size": 64,
        "n_workers": 0,
        "split": "evaluation",
    }
    data_dict["dataset"] = f"{dataset}_OOD"
    ds = get_dataloader(data_dict).dataset
    return ds


ds = load_data()
n_image_show = st.number_input(
    "Number of images to show", min_value=1, max_value=100, value=10
)

l_selected_detector = st.multiselect(
    "Select detector",
    [
        "ae",
        "pixelcnn",
        "md",
        "oe",
        "nae",
        "glow",
        "ssd",
        "good",
        "acet",
        "ceda",
        "rowl",
        "due",
        "atom",
        "csi",
        "prood",
    ],
    default=["atom", "md", "csi", "prood"],
)


st.subheader("Worst Inlier Visualization")

for detector in l_selected_detector:
    # load inlier score
    inlier_dir = os.path.join("../results/CIFAR10/", detector)
    if dataset == "CIFAR10":
        score = torch.load(os.path.join(inlier_dir, "IN_score.pkl"))
    else:
        score = torch.load(os.path.join(inlier_dir, f"OOD_rank_{dataset}_OOD.pkl"))
    in_sorted_score, in_sorted_idx = torch.sort(torch.tensor(score), descending=True)

    st.text(detector)
    st.caption("most unconfident inlier")
    imgs = [
        ds[in_sorted_idx[i]][0].permute(1, 2, 0).numpy() for i in range(n_image_show)
    ]
    st.image(
        imgs,
        use_column_width="auto",
        caption=[
            f"{in_sorted_score[i]:.2f}"
            if dataset == "CIFAR10"
            else f"{in_sorted_score[i]}"
            for i in range(n_image_show)
        ],
    )

    st.caption("most confident inlier")
    in_sorted_score, in_sorted_idx = torch.sort(torch.tensor(score), descending=False)
    imgs = [
        ds[in_sorted_idx[i]][0].permute(1, 2, 0).numpy() for i in range(n_image_show)
    ]
    st.image(
        imgs,
        use_column_width="auto",
        caption=[
            f"{in_sorted_score[i]:.2f}"
            if dataset == "CIFAR10"
            else f"{in_sorted_score[i]}"
            for i in range(n_image_show)
        ],
    )

import os

import numpy as np
import streamlit as st
import torch

dataset = st.selectbox("Select dataset", ["CIFAR10", "RImgNet"])

detector = st.selectbox(
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
        "vit_hf_md",
    ],
)

n_image_show = st.number_input(
    "Number of images to show", min_value=1, max_value=100, value=10
)


detector_dir = os.path.join(
    f"../results/{dataset}/",
    detector,
)
l_result_dirs = sorted(os.listdir(detector_dir))
l_result_dirs = [
    s
    for s in l_result_dirs
    if not s.endswith(".txt")
    and not s.endswith(".pkl")
    and not s.endswith("_tensorboard")
]

result_folder = st.selectbox("Select result", l_result_dirs)
result_dir = os.path.join(detector_dir, result_folder)

l_run_dirs = sorted(os.listdir(result_dir))
run_folder = st.selectbox("Select run", l_run_dirs)
run_dir = os.path.join(result_dir, run_folder)

# rank = torch.load(os.path.join(run_dir, "rank.pkl"))

run_splits = [s for s in os.listdir(run_dir) if s.startswith("advsample_x_")]
split_idx = st.selectbox("Select run split", list(range(len(run_splits))))

sample = torch.load(os.path.join(run_dir, f"advsample_x_{split_idx}.pkl"))
score = torch.load(os.path.join(run_dir, f"advsample_score_{split_idx}.pkl"))

sorted_score, sorted_idx = torch.sort(torch.tensor(score), descending=False)


# load inlier score
inlier_score = torch.load(os.path.join(detector_dir, "IN_score.pkl"))
sorted_in_score = np.sort(inlier_score)
print(sorted_in_score)
out_rank = np.searchsorted(sorted_in_score, sorted_score)


st.text(detector)
st.caption("most unconfident inlier")
imgs = [sample[sorted_idx[i]].permute(1, 2, 0).numpy() for i in range(n_image_show)]
st.image(
    imgs,
    use_column_width="auto",
    caption=[f"{sorted_score[i]:.2f} ({out_rank[i]})" for i in range(n_image_show)],
    clamp=True,
)

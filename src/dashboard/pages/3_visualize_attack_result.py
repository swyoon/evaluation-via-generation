import os

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

result_folder = st.selectbox("Select result", l_result_dirs)
result_dir = os.path.join(detector_dir, result_folder)

l_run_dirs = sorted(os.listdir(result_dir))
run_folder = st.selectbox("Select run", l_run_dirs)
run_dir = os.path.join(result_dir, run_folder)

# rank = torch.load(os.path.join(run_dir, "rank.pkl"))
sample = torch.load(os.path.join(run_dir, "advsample_x_0.pkl"))
score = torch.load(os.path.join(run_dir, "advsample_score_0.pkl"))

sorted_score, sorted_idx = torch.sort(torch.tensor(score), descending=False)

st.text(detector)
st.caption("most unconfident inlier")
imgs = [sample[sorted_idx[i]].permute(1, 2, 0).numpy() for i in range(n_image_show)]
st.image(
    imgs,
    use_column_width="auto",
    caption=[f"{sorted_score[i]:.2f}" for i in range(n_image_show)],
)

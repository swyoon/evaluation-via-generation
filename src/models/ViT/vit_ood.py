import glob
import os
import pickle
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
from absl import logging

from .vision_transformer.vit_jax import checkpoint, models

# from .vision_transformer.vit_jax import train
from .vision_transformer.vit_jax.configs import augreg as augreg_config
from .vision_transformer.vit_jax.configs import models as models_config

# Import files from repository.


# if "./vision_transformer" not in sys.path:
#     sys.path.append("./vision_transformer")

# %load_ext autoreload
# %autoreload 2


tf.config.experimental.set_visible_devices([], "GPU")
# import tensorflow_datasets as tfds
from matplotlib import pyplot as plt


def get_model():

    model = VIT_Maha()
    return model


class ViT_Maha:
    def __init__(
        self,
        model_type="L_16",
        vit_checkpoint=None,
        mahalanobis_statistic=None,
        relative_maha=True,
    ):
        # filename = vit_checkpoint
        # filename = "L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0--cifar100-steps_2k-lr_0.01-res_384"

        # tfds_name = filename.split("--")[1].split("-")[0]  # cifar100
        model_config = models_config.AUGREG_CONFIGS[model_type]
        # resolution = int(filename.split("_")[-1])  # 384
        resolution = 384
        num_classes = 100
        model_config2 = {k: v for k, v in model_config.items() if k != "model_name"}
        # ds, ds_info = tfds.load(tfds_name, with_info=True, batch_size=128)

        # Get a clean model
        model = models.VisionTransformer(num_classes=num_classes, **model_config2)

        model_prelogits = models.VisionTransformer_prelogits(
            num_classes=num_classes, **model_config2
        )
        self.model = model
        self.model_prelogits = model_prelogits
        self.resolution = resolution

        # path = "/opt/home3/swyoon/exploring_the_limits_of_OOD_detection/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0--cifar100-steps_2k-lr_0.01-res_384.npz"
        if vit_checkpoint is not None:
            self.params = checkpoint.load(vit_checkpoint)

        # load pre-computed mahalanobis statistic
        if mahalanobis_statistic is not None:
            maha_intermediate_dict_full = pickle.load(open(mahalanobis_statistic, "rb"))
            self.class_cov_invs = maha_intermediate_dict_full["class_cov_invs"]
            self.class_means = maha_intermediate_dict_full["class_means"]
            self.cov_inv = maha_intermediate_dict_full["cov_inv"]
            self.mean = maha_intermediate_dict_full["mean"]

        self.indist_classes = 100
        self.relative_maha = relative_maha

    def forward_logits(self, batch_x):
        batch_x = self.preprocess(batch_x)
        logits = self.model.apply({"params": self.params}, batch_x, train=False)
        return logits

    def forward_prelogits(self, batch_x):
        batch_x = self.preprocess(batch_x)
        prelogits = self.model_prelogits.apply(
            {"params": self.params}, batch_x, train=False
        )
        return prelogits

    def preprocess(self, batch_x):
        """
        Preprocesses the image to be fed into the model.
        batch_x: batch of images. shape: (batch_size, 32, 32, 3)
        """
        assert batch_x.shape[1:] == (32, 32, 3)
        sz = self.resolution
        # we will assume inputs to be [0, 1]
        # batch_x = tf.cast(batch_x, float) / 255.0
        batch_x = tf.image.resize(batch_x, [sz, sz])
        return batch_x

    def predict(self, batch_x):
        norm_name = "L2"
        embeds = self.forward_prelogits(batch_x)
        out_totrainclasses = [
            maha_distance(
                embeds, self.class_cov_invs[c], self.class_means[c], norm_name
            )
            for c in range(self.indist_classes)
        ]

        out_scores = np.min(np.stack(out_totrainclasses, axis=0), axis=0)

        if self.relative_maha:  # True: relative mahalanobis distance
            out_totrain = maha_distance(embeds, self.cov_inv, self.mean, norm_name)
            out_scores = out_scores - out_totrain

        return out_scores


class ViT_Maha_torch:
    """wrapper of ViT_Maha for using PyTorch"""

    def __init__(self, **kwargs):
        self.model = ViT_Maha(**kwargs)

    def predict(self, batch_x):
        assert isinstance(batch_x, torch.Tensor)
        batch_x = batch_x.permute(0, 2, 3, 1)
        out = self.model.predict(batch_x.detach().numpy())
        return torch.tensor(np.array(out))


from sklearn.metrics import roc_auc_score


def maha_distance(xs, cov_inv_in, mean_in, norm_type=None):
    diffs = xs - mean_in.reshape([1, -1])

    second_powers = np.matmul(diffs, cov_inv_in) * diffs

    if norm_type in [None, "L2"]:
        return np.sum(second_powers, axis=1)
    elif norm_type in ["L1"]:
        return np.sum(np.sqrt(np.abs(second_powers)), axis=1)
    elif norm_type in ["Linfty"]:
        return np.max(second_powers, axis=1)


def get_scores(
    indist_train_embeds_in,
    indist_train_labels_in,
    indist_test_embeds_in,
    outdist_test_embeds_in,
    subtract_mean=True,
    normalize_to_unity=True,
    subtract_train_distance=True,
    indist_classes=100,
    norm_name="L2",
):

    # storing the replication results
    maha_intermediate_dict = dict()

    description = ""

    all_train_mean = np.mean(indist_train_embeds_in, axis=0, keepdims=True)

    indist_train_embeds_in_touse = indist_train_embeds_in
    indist_test_embeds_in_touse = indist_test_embeds_in
    outdist_test_embeds_in_touse = outdist_test_embeds_in

    if subtract_mean:
        indist_train_embeds_in_touse -= all_train_mean
        indist_test_embeds_in_touse -= all_train_mean
        outdist_test_embeds_in_touse -= all_train_mean
        description = description + " subtract mean,"

    if normalize_to_unity:
        indist_train_embeds_in_touse = indist_train_embeds_in_touse / np.linalg.norm(
            indist_train_embeds_in_touse, axis=1, keepdims=True
        )
        indist_test_embeds_in_touse = indist_test_embeds_in_touse / np.linalg.norm(
            indist_test_embeds_in_touse, axis=1, keepdims=True
        )
        outdist_test_embeds_in_touse = outdist_test_embeds_in_touse / np.linalg.norm(
            outdist_test_embeds_in_touse, axis=1, keepdims=True
        )
        description = description + " unit norm,"

    # full train single fit
    mean = np.mean(indist_train_embeds_in_touse, axis=0)
    cov = np.cov((indist_train_embeds_in_touse - (mean.reshape([1, -1]))).T)

    eps = 1e-8
    cov_inv = np.linalg.inv(cov)

    # getting per class means and covariances
    class_means = []
    class_cov_invs = []
    class_covs = []
    for c in range(indist_classes):

        mean_now = np.mean(
            indist_train_embeds_in_touse[indist_train_labels_in == c], axis=0
        )

        cov_now = np.cov(
            (
                indist_train_embeds_in_touse[indist_train_labels_in == c]
                - (mean_now.reshape([1, -1]))
            ).T
        )
        class_covs.append(cov_now)
        # print(c)

        eps = 1e-8
        cov_inv_now = np.linalg.inv(cov_now)

        class_cov_invs.append(cov_inv_now)
        class_means.append(mean_now)

    # the average covariance for class specific
    class_cov_invs = [
        np.linalg.inv(np.mean(np.stack(class_covs, axis=0), axis=0))
    ] * len(class_covs)

    maha_intermediate_dict["class_cov_invs"] = class_cov_invs
    maha_intermediate_dict["class_means"] = class_means
    maha_intermediate_dict["cov_inv"] = cov_inv
    maha_intermediate_dict["mean"] = mean

    out_totrain = maha_distance(outdist_test_embeds_in_touse, cov_inv, mean, norm_name)
    in_totrain = maha_distance(indist_test_embeds_in_touse, cov_inv, mean, norm_name)

    out_totrainclasses = [
        maha_distance(
            outdist_test_embeds_in_touse, class_cov_invs[c], class_means[c], norm_name
        )
        for c in range(indist_classes)
    ]
    in_totrainclasses = [
        maha_distance(
            indist_test_embeds_in_touse, class_cov_invs[c], class_means[c], norm_name
        )
        for c in range(indist_classes)
    ]

    out_scores = np.min(np.stack(out_totrainclasses, axis=0), axis=0)
    in_scores = np.min(np.stack(in_totrainclasses, axis=0), axis=0)

    if subtract_train_distance:
        out_scores = out_scores - out_totrain
        in_scores = in_scores - in_totrain

    onehots = np.array([1] * len(out_scores) + [0] * len(in_scores))
    scores = np.concatenate([out_scores, in_scores], axis=0)

    return onehots, scores, description, maha_intermediate_dict


if __name__ == "__main__":

    model = get_model()

    batch_x = np.random.randn(100, 32, 32, 3)
    pred = model.predict(batch_x)

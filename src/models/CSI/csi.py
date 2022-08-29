import sys
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models.CSI.models.classifier as C
import models.CSI.models.transform_layers as TL
from models.CSI.common.common import parse_args
from models.CSI.datasets import get_dataset, get_subclass_dataset, get_superclass_list
from models.CSI.utils.utils import normalize, set_random_seed

hflip = TL.HorizontalFlipLayer()


class CSI_detector(nn.Module):
    def __init__(self, P, model, **kwargs):
        super().__init__()
        self.P = P
        self.model = model
        self.kwargs = kwargs

    def get_features(self, loader, device, layers=("simclr", "shift")):

        simclr_aug = self.kwargs["simclr_aug"].to(device)
        sample_num = self.kwargs["sample_num"]

        if not isinstance(layers, (list, tuple)):
            layers = [layers]

        feats_dict = dict()
        left = [layer for layer in layers if layer not in feats_dict.keys()]
        if len(left) > 0:
            _feats_dict = self._get_features(loader, device, layers=left)

            for layer, feats in _feats_dict.items():
                feats_dict[layer] = feats  # update value

        return feats_dict

    def _get_features(
        self, loader, device, interp=False, imagenet=False, layers=("simclr", "shift")
    ):

        simclr_aug = self.kwargs["simclr_aug"].to(device)
        sample_num = self.kwargs["sample_num"]

        if not isinstance(layers, (list, tuple)):
            layers = [layers]

        # check if arguments are valid
        assert simclr_aug is not None

        # compute features in full dataset
        self.model.eval()
        feats_all = {layer: [] for layer in layers}  # initialize: empty list
        for i, x in enumerate(loader):
            x = x.to(device)  # gpu tensor

            # compute features in one batch
            feats_batch = {layer: [] for layer in layers}  # initialize: empty list
            for seed in range(sample_num):
                set_random_seed(seed)

                if self.P.K_shift > 1:
                    x_t = torch.cat(
                        [self.P.shift_trans(hflip(x), k) for k in range(self.P.K_shift)]
                    )
                else:
                    x_t = x  # No shifting: SimCLR

                x_t = simclr_aug(x_t)

                # compute augmented features
                with torch.no_grad():
                    kwargs = {
                        layer: True for layer in layers
                    }  # only forward selected layers
                    _, output_aux = self.model(x_t, **kwargs)

                # add features in one batch
                for layer in layers:
                    feats = output_aux[layer].cpu()
                    if imagenet is False:
                        feats_batch[layer] += feats.chunk(self.P.K_shift)
                    else:
                        feats_batch[layer] += [feats]  # (B, d) cpu tensor

            # concatenate features in one batch
            for key, val in feats_batch.items():
                if imagenet:
                    feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
                else:
                    feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)

            # add features in full dataset
            for layer in layers:
                feats_all[layer] += [feats_batch[layer]]

        # concatenate features in full dataset
        for key, val in feats_all.items():
            feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

        # reshape order
        if imagenet is False:
            # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
            for key, val in feats_all.items():
                N, T, d = val.size()  # T = K * T'
                val = val.view(N, -1, self.P.K_shift, d)  # (N, T', K, d)
                val = val.transpose(2, 1)  # (N, 4, T', d)
                val = val.reshape(N, T, d)  # (N, T, d)
                feats_all[key] = val

        return feats_all

    def get_scores(self, feats_dict, device, ood_score="CSI"):
        # convert to gpu tensor
        feats_sim = feats_dict["simclr"].to(device)
        feats_shi = feats_dict["shift"].to(device)
        N = feats_sim.size(0)

        # compute scores
        scores = []
        for f_sim, f_shi in zip(feats_sim, feats_shi):
            f_sim = [
                f.mean(dim=0, keepdim=True) for f in f_sim.chunk(self.P.K_shift)
            ]  # list of (1, d)
            f_shi = [
                f.mean(dim=0, keepdim=True) for f in f_shi.chunk(self.P.K_shift)
            ]  # list of (1, 4)
            score = 0
            for shi in range(self.P.K_shift):
                score += (f_sim[shi] * self.P.axis[shi].to(device)).sum(
                    dim=1
                ).max().item() * self.P.weight_sim[shi]
                score += f_shi[shi][:, shi].item() * self.P.weight_shi[shi]
            score = score / self.P.K_shift
            scores.append(score)
        scores = torch.tensor(scores)

        assert scores.dim() == 1 and scores.size(0) == N  # (N)
        return scores

    def predict(self, data):
        device = data.device
        data_loader = DataLoader(data, shuffle=False, batch_size=len(data))
        feats_id = self.get_features(data_loader, device)  # (N, T, d)
        return -self.get_scores(feats_id, device).to(device)

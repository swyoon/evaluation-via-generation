from __future__ import print_function

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy import misc
from torch.autograd import Variable


def get_curve(dir_name, stypes=["MSP", "ODIN"]):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()
    for stype in stypes:
        known = np.loadtxt(
            "{}/confidence_{}_In.txt".format(dir_name, stype), delimiter="\n"
        )
        novel = np.loadtxt(
            "{}/confidence_{}_Out.txt".format(dir_name, stype), delimiter="\n"
        )
        known.sort()
        novel.sort()

        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]

        threshold = known[round(0.05 * num_k)]

        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1 :] = tp[stype][l]
                fp[stype][l + 1 :] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1 :] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1 :] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]

        fpr_at_tpr95[stype] = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def metric(dir_name, stypes=["MSP", "ODIN"], verbose=False):
    tp, fp, fpr_at_tpr95 = get_curve(dir_name, stypes)
    results = dict()
    mtypes = ["FPR", "AUROC", "DTERR", "AUIN", "AUOUT"]
    if verbose:
        print("      ", end="")
        for mtype in mtypes:
            print(" {mtype:6s}".format(mtype=mtype), end="")
        print("")

    for stype in stypes:
        if verbose:
            print("{stype:5s} ".format(stype=stype), end="")
        results[stype] = dict()

        # FPR
        mtype = "FPR"
        results[stype][mtype] = fpr_at_tpr95[stype]
        if verbose:
            print(" {val:6.3f}".format(val=100.0 * results[stype][mtype]), end="")

        # AUROC
        mtype = "AUROC"
        tpr = np.concatenate([[1.0], tp[stype] / tp[stype][0], [0.0]])
        fpr = np.concatenate([[1.0], fp[stype] / fp[stype][0], [0.0]])
        results[stype][mtype] = -np.trapz(1.0 - fpr, tpr)
        if verbose:
            print(" {val:6.3f}".format(val=100.0 * results[stype][mtype]), end="")

        # DTERR
        mtype = "DTERR"
        results[stype][mtype] = (
            (tp[stype][0] - tp[stype] + fp[stype]) / (tp[stype][0] + fp[stype][0])
        ).min()
        if verbose:
            print(" {val:6.3f}".format(val=100.0 * results[stype][mtype]), end="")

        # AUIN
        mtype = "AUIN"
        denom = tp[stype] + fp[stype]
        denom[denom == 0.0] = -1.0
        pin_ind = np.concatenate([[True], denom > 0.0, [True]])
        pin = np.concatenate([[0.5], tp[stype] / denom, [0.0]])
        results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
        if verbose:
            print(" {val:6.3f}".format(val=100.0 * results[stype][mtype]), end="")

        # AUOUT
        mtype = "AUOUT"
        denom = tp[stype][0] - tp[stype] + fp[stype][0] - fp[stype]
        denom[denom == 0.0] = -1.0
        pout_ind = np.concatenate([[True], denom > 0.0, [True]])
        pout = np.concatenate([[0.0], (fp[stype][0] - fp[stype]) / denom, [0.5]])
        results[stype][mtype] = np.trapz(pout[pout_ind], 1.0 - fpr[pout_ind])
        if verbose:
            print(" {val:6.3f}".format(val=100.0 * results[stype][mtype]), end="")
            print("")

    return results

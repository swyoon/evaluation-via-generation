import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn

from models.pixelcnn_utils import *


class nin(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        # self.lin_a = wn(nn.Linear(dim_in, dim_out))
        # self.dim_out = dim_out
        self.conv_1x1 = wn(nn.Conv2d(dim_in, dim_out, kernel_size=1))

    def forward(self, x):
        # og_x = x
        # # assumes pytorch ordering
        # """ a network in network layer (1x1 CONV) """
        # # TODO : try with original ordering
        # x = x.permute(0, 2, 3, 1)
        # shp = [int(y) for y in x.size()]
        # out = self.lin_a(x.contiguous().view(shp[0] * shp[1] * shp[2], shp[3]))
        # shp[-1] = self.dim_out
        # out = out.view(shp)
        # return out.permute(0, 3, 1, 2)

        return self.conv_1x1(x)


class down_shifted_conv2d(nn.Module):
    def __init__(
        self,
        num_filters_in,
        num_filters_out,
        filter_size=(2, 3),
        stride=(1, 1),
        padding=None,
        shift_output_down=False,
        norm="weight_norm",
    ):
        super(down_shifted_conv2d, self).__init__()

        assert norm in [None, "batch_norm", "weight_norm"]
        # self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        # self.pad  = nn.ZeroPad2d((int((filter_size[1] - 1) / 2), # pad left
        #                           int((filter_size[1] - 1) / 2), # pad right
        #                           filter_size[0] - 1,            # pad top
        #                           0) )                           # pad down

        if padding is None:
            padding = (filter_size[0] - 1, int((filter_size[1] - 1) / 2))

        self.conv = nn.Conv2d(
            num_filters_in, num_filters_out, filter_size, stride=stride, padding=padding
        )

        if norm == "weight_norm":
            self.conv = wn(self.conv)
        elif norm == "batch_norm":
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down:
            self.down_shift = lambda x: down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        # x = self.pad(x)
        # x = self.conv(x)
        shp = x.shape
        # self.conv.weight[:, :, 1:2, 1:3] = 0
        x = self.conv(x)[
            :,
            :,
            : int(shp[2] / self.conv.stride[0]),
            : int(shp[3] / self.conv.stride[1]),
        ]
        x = self.bn(x) if self.norm == "batch_norm" else x
        return self.down_shift(x) if self.shift_output_down else x


class down_shifted_deconv2d(nn.Module):
    def __init__(
        self, num_filters_in, num_filters_out, filter_size=(2, 3), stride=(1, 1)
    ):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(
            nn.ConvTranspose2d(
                num_filters_in, num_filters_out, filter_size, stride, output_padding=1
            )
        )
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[
            :,
            :,
            : (xs[2] - self.filter_size[0] + 1),
            int((self.filter_size[1] - 1) / 2) : (
                xs[3] - int((self.filter_size[1] - 1) / 2)
            ),
        ]


class down_right_shifted_conv2d(nn.Module):
    def __init__(
        self,
        num_filters_in,
        num_filters_out,
        filter_size=(2, 2),
        stride=(1, 1),
        padding=None,
        shift_output_right=False,
        norm="weight_norm",
    ):
        super(down_right_shifted_conv2d, self).__init__()

        assert norm in [None, "batch_norm", "weight_norm"]
        # self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        # self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if padding is None:
            padding = (filter_size[0] - 1, filter_size[1] - 1)

        self.conv = nn.Conv2d(
            num_filters_in, num_filters_out, filter_size, stride=stride, padding=padding
        )

        if norm == "weight_norm":
            self.conv = wn(self.conv)
        elif norm == "batch_norm":
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right:
            self.right_shift = lambda x: right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        # x = self.pad(x)
        # x = self.conv(x)
        shp = x.shape
        # self.conv.weight[:, :, 1:2, 1:2] = 0
        x = self.conv(x)[
            :,
            :,
            : int(shp[2] / self.conv.stride[0]),
            : int(shp[3] / self.conv.stride[1]),
        ]
        x = self.bn(x) if self.norm == "batch_norm" else x
        return self.right_shift(x) if self.shift_output_right else x


class down_right_shifted_deconv2d(nn.Module):
    def __init__(
        self,
        num_filters_in,
        num_filters_out,
        filter_size=(2, 2),
        stride=(1, 1),
        shift_output_right=False,
    ):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(
            nn.ConvTranspose2d(
                num_filters_in, num_filters_out, filter_size, stride, output_padding=1
            )
        )
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[
            :,
            :,
            : (xs[2] - self.filter_size[0] + 1) :,
            : (xs[3] - self.filter_size[1] + 1),
        ]
        return x


"""
skip connection parameter : 0 = no skip connection
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
"""


class gated_resnet(nn.Module):
    def __init__(
        self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0
    ):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters)  # cuz of concat elu

        if skip_connection != 0:
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)

    def forward(self, og_x, a=None):
        # x = self.conv_input(self.nonlinearity(og_x))
        x = self.nonlinearity(og_x)
        x = self.conv_input(x)
        if a is not None:
            # x += self.nin_skip(self.nonlinearity(a))
            a = self.nonlinearity(a)
            x += self.nin_skip(a)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * torch.sigmoid(b)
        return og_x + c3


class gated_BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, num_filters, conv_op, filter_size=3, padding=None, skip_connection=0
    ):
        super(gated_BasicBlock, self).__init__()

        self.skip_connection = skip_connection
        self.conv_input = conv_op(
            num_filters, num_filters, filter_size=filter_size, padding=padding
        )
        self.bn_input = nn.BatchNorm2d(num_filters)

        if skip_connection != 0:
            self.nin_skip = nin(skip_connection * num_filters, num_filters)

        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(
            num_filters, num_filters, filter_size=filter_size, padding=padding
        )
        self.bn_out = nn.BatchNorm2d(num_filters)

        self.elu = nn.ReLU(inplace=True)

    def forward(self, og_x, a=None):
        out = self.conv_input(og_x)
        # out = self.bn_input(out)

        if a is not None:
            out += self.nin_skip(a)

        out = self.elu(out)

        # out = self.dropout(out)
        out = self.conv_out(out)
        # out = self.bn_out(out)
        # out, a = torch.chunk(out, 2, dim=1)
        # out = out * F.sigmoid(a)

        out += og_x
        out = self.elu(out)

        return out

##picelcnn_cifar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

def assert_nan(x):
    assert not torch.isnan(x).any(), "Tensor contains NaNs!"

def log_sum_exp(logits):
    dim = logits.dim() - 1
    max_logits, _ = torch.max(logits, dim, keepdim=True)
    return torch.log(torch.sum(torch.exp(logits - max_logits), dim)) + max_logits.squeeze()

def log_prob_from_logits(logits):
    dim = logits.dim() - 1
    max_logits, _ = torch.max(logits, dim, keepdim=True)
    return logits - max_logits - torch.log(torch.sum(torch.exp(logits - max_logits), dim, keepdim=True))

class WN_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 train_scale=False, init_stdv=1.0):
        super(WN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))

        self.train_scale = train_scale
        self.init_stdv = init_stdv
        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input):
        weight_scale = self.weight_scale.view(-1, 1, 1, 1)
        norm = torch.sqrt((self.weight ** 2).sum(dim=(1, 2, 3), keepdim=True) + 1e-6)
        norm_weight = self.weight * (weight_scale / norm)
        activation = F.conv2d(input, norm_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)
        if self.bias is not None:
            activation = activation + self.bias.view(1, -1, 1, 1)
        return activation

class GatedResNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p, nonlinearity=F.elu):
        super(GatedResNet, self).__init__()
        self.nonlinearity = nonlinearity
        self.conv1 = WN_Conv2d(in_channels, out_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = WN_Conv2d(out_channels, out_channels * 2, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        residual = x
        out = self.nonlinearity(x)
        out = self.conv1(out)
        a, b = out.chunk(2, dim=1)
        out = a * torch.sigmoid(b)
        out = self.nonlinearity(out)
        out = self.conv2(out)
        a, b = out.chunk(2, dim=1)
        out = a * torch.sigmoid(b)
        out = self.dropout(out)
        return residual + out

class PixelCNN(nn.Module):
    def  __init__(self, nr_filters=160, nr_resnet=5, nr_logistic_mix=10, disable_third=False, dropout_p=0.5, n_channel=3,
                 image_wh=32):
        super(PixelCNN, self).__init__()
        self.nr_filters = nr_filters
        self.nr_resnet = nr_resnet
        self.nr_logistic_mix = nr_logistic_mix
        self.disable_third = disable_third
        self.dropout_p = dropout_p
        self.n_channel = n_channel
        self.image_wh = image_wh

        self.conv_down = WN_Conv2d(n_channel, nr_filters, kernel_size=7, stride=1, padding=3)
        self.relu = nn.ReLU(True)

        self.resnets = nn.ModuleList()
        for _ in range(nr_resnet):
            self.resnets.append(GatedResNet(nr_filters, nr_filters, self.dropout_p, nonlinearity=F.elu))

        self.resnets_down = nn.ModuleList()
        for _ in range(nr_resnet):
            self.resnets_down.append(GatedResNet(nr_filters, nr_filters, self.dropout_p, nonlinearity=F.elu))

        self.conv_final = WN_Conv2d(nr_filters, n_channel * nr_logistic_mix * 3, kernel_size=1)

    def forward(self, input):
        assert_nan(input)
        x = self.conv_down(input)
        x = self.relu(x)

        for resnet in self.resnets:
            x = resnet(x)

        for resnet in self.resnets_down:
            x = resnet(x)

        x = self.conv_final(x)
        return x

def discretized_mix_logistic_loss(x, l, nr_mix=10, sum_all=True, return_per_image=False):
    B, C, H, W = x.size()
    l = l.view(B, C, nr_mix * 3, H, W)

    logit_probs = l[:, :, :nr_mix, :, :]
    means = l[:, :, nr_mix:2 * nr_mix, :, :]
    log_scales = torch.clamp(l[:, :, 2 * nr_mix:3 * nr_mix, :, :], min=-7.)

    x = x.unsqueeze(2) * 255.0
    means = means * 255.0

    inv_stdv = torch.exp(-log_scales)
    centered_x = x - means
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = torch.sigmoid(min_in)
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.log(torch.clamp(cdf_delta, min=1e-12))
    log_probs = log_probs + log_prob_from_logits(logit_probs)
    log_probs = log_sum_exp(log_probs)

    # log_probs shape: B x C x H x W
    if return_per_image:
        # 返回每张图像的NLL
        nll_per_image = -torch.sum(log_probs, dim=(1,2,3)) / (C * H * W)
        return nll_per_image
    else:
        if sum_all:
            return -torch.sum(log_probs) / (B * C * H * W)
        else:
            return -torch.mean(log_probs)

"""Modified from
https://github.com/xmed-lab/EPL_SemiDG/blob/master/network/deeplabv3p.py.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import Bottleneck, ResNet

Norm = Type[nn.modules.batchnorm._NormBase]


class ASPP(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            dilation_rates=(12, 24, 36),
            hidden_channels=256,
    ):
        nn.Module.__init__(self)

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels,
                      hidden_channels,
                      3,
                      bias=False,
                      dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels,
                      hidden_channels,
                      3,
                      bias=False,
                      dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels,
                      hidden_channels,
                      3,
                      bias=False,
                      dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = nn.BatchNorm2d(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels,
                                             hidden_channels,
                                             1,
                                             bias=False)
        self.global_pooling_bn = nn.BatchNorm2d(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4,
                                  out_channels,
                                  1,
                                  bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels,
                                       out_channels,
                                       1,
                                       bias=False)
        self.red_bn = nn.BatchNorm2d(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)  # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
        pool = pool.view(x.size(0), x.size(1), 1, 1)
        return pool


class Head(nn.Module):

    def __init__(self, bn_momentum: float = 0.1):
        nn.Module.__init__(self)

        self.aspp = ASPP(2048, 256, [6, 12, 18])

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
        )

    def forward(self, feats):
        f = feats[-1]
        f = self.aspp(f)

        low_level_features = feats[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)

        f = F.interpolate(f,
                          size=(low_h, low_w),
                          mode="bilinear",
                          align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)  # skip connection
        f = self.last_conv(f)
        return f


class DeepLabV3Plus(nn.Module):

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 2,
    ):
        nn.Module.__init__(self)
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3],
                               n_channels,
                               deep_stem=True,
                               stem_width=64)

        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(DeepLabV3Plus._nostride_dilate,
                            dilate=self.dilate))
            self.dilate *= 2

        self.head = Head()
        self.classifier = nn.Conv2d(256, n_classes, kernel_size=1, bias=True)

        # Initialize weights
        for init in [self.head, self.classifier]:
            for m in init.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_in",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        h, w = input.shape[-2:]

        feats = self.backbone(input)
        f = self.head(feats)
        pred = self.classifier(f)

        pred = F.interpolate(pred,
                             size=(h, w),
                             mode="bilinear",
                             align_corners=True)
        return pred

    @staticmethod
    def _nostride_dilate(m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    @classmethod
    def from_pretrained(cls, n_channels: int = 3, n_classes: int = 2):
        model_path = Path(__file__).parent / "resnet50_v1c.pth"
        model = cls(n_channels, n_classes)

        model_dict = model.backbone.state_dict()
        state_dict = torch.load(model_path)

        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].size() == v.size():
                model_dict[k] = v

        model.backbone.load_state_dict(model_dict)
        return model

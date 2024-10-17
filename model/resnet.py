from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ResNetBlock(nn.Module):
    expansion = -1

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(ResNetBlock):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        bn_eps=1e-5,
        bn_momentum=0.1,
        downsample=None,
        inplace=True,
    ):
        nn.Module.__init__(self)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual

        out = self.relu_inplace(out)

        return out


class Bottleneck(ResNetBlock):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        bn_eps=1e-5,
        bn_momentum=0.1,
        downsample=None,
        inplace=True,
    ):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  eps=bn_eps,
                                  momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: type[ResNetBlock],
        layers: list[int],
        in_channels: int = 3,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        deep_stem: bool = False,
        stem_width: int = 32,
        relu_inplace: bool = True,
    ):
        self.inplanes = stem_width * 2 if deep_stem else 64
        nn.Module.__init__(self)
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels,
                          stem_width,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=relu_inplace),
                nn.Conv2d(stem_width,
                          stem_width,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=relu_inplace),
                nn.Conv2d(stem_width,
                          stem_width * 2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels,
                                   64,
                                   kernel_size=7,
                                   stride=2,
                                   padding=3,
                                   bias=False)

        self.bn1 = nn.BatchNorm2d(stem_width * 2 if deep_stem else 64,
                                  eps=bn_eps,
                                  momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,
                                       64,
                                       layers[0],
                                       relu_inplace,
                                       bn_eps=bn_eps,
                                       bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       relu_inplace,
                                       stride=2,
                                       bn_eps=bn_eps,
                                       bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       relu_inplace,
                                       stride=2,
                                       bn_eps=bn_eps,
                                       bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       relu_inplace,
                                       stride=2,
                                       bn_eps=bn_eps,
                                       bn_momentum=bn_momentum)

    def _make_layer(
        self,
        block: type[ResNetBlock],
        planes: int,
        blocks: int,
        inplace: bool = True,
        stride: int = 1,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion,
                               eps=bn_eps,
                               momentum=bn_momentum),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, bn_eps, bn_momentum,
                  downsample, inplace))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      bn_eps=bn_eps,
                      bn_momentum=bn_momentum,
                      inplace=inplace))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)

        return blocks

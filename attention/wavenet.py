# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv1d(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3,
                 stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(BasicConv1d, self).__init__()
        self.add_module('conv', nn.Conv1d(in_planes, out_planes, kernel_size,
                  stride, padding, groups=groups, bias=False)),
        self.add_module('norm', nn.BatchNorm1d(out_planes)),
        self.add_module('relu', nn.ReLU6(inplace=True))


class CascadeConv1d(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(CascadeConv1d, self).__init__()
        self.add_module('convbnrelu_0', BasicConv1d(in_planes, out_planes, kernel_size=1))
        num_layers = (kernel_size - 1) // 2
        for i in range(num_layers):
            self.add_module('convbnrelu_%d'%(i+1), BasicConv1d(out_planes, out_planes, kernel_size=3))


class InceptionX(nn.Module):

    def __init__(self, num_scales, in_planes, out_planes, conv_block=None):
        super(InceptionX, self).__init__()
        if conv_block is None:
            conv_block = BasicConv1d
        self.layers = nn.ModuleList()
        for i in range(num_scales):
            kernel_size = 2 * i + 1
            layer = conv_block(in_planes, out_planes, kernel_size)
            self.layers.append(layer)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x1 = layer(x)
            outputs.append(x1)
        return torch.cat(outputs, 1)


class Downsample(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(Downsample, self).__init__()
        self.add_module('conv', nn.Conv1d(in_planes, out_planes,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))


class GroupSparseLayer(nn.Module):
    def __init__(self, in_features, out_features, groups, lambda_reg=0.01, bias=True):
        super(GroupSparseLayer, self).__init__()
        self.use_bias = bias
        self.groups = groups
        self.lambda_reg = lambda_reg

        # Putting in_features first is easy to construct groups
        self.w = nn.Parameter(torch.zeros(in_features, out_features))
        if self.use_bias:
            self.b = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('b', None)

    def forward(self, x):
        loss_reg = self.lambda_reg * self.w.reshape((self.groups, -1)).norm(2, dim=1).sum()
        return F.linear(x, self.w.T, self.b), loss_reg


class MultiScaleNet(nn.Module):
    def __init__(self, num_scales=4, in_planes=3, init_planes=16, input_dim=32, num_classes=4):
        super(MultiScaleNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_planes, num_scales * init_planes, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_scales * init_planes),
            nn.ReLU(inplace=True),
        )
        # self.features = InceptionX(num_scales, in_planes, init_planes, conv_block=CascadeConv1d)
        self.downsample = Downsample(num_scales * init_planes, num_scales * init_planes)
        feature_dim = input_dim # // 2
        self.selector = nn.Linear(num_scales * init_planes * feature_dim, 4096)
        # self.selector = GroupSparseLayer(num_scales * init_planes * feature_dim, 4096, groups=num_scales)
        self.classifier = nn.Sequential(
            nn.Linear(num_scales * init_planes * feature_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.downsample(x)
        x = x.view(x.size(0), -1)
        # x, loss_reg = self.selector(x), torch.tensor(0)
        x, loss_reg = self.classifier(x), torch.tensor(0)
        return x, loss_reg


if __name__ == '__main__':
    x = torch.randn((100, 1, 35))
    # model = InceptionX(4, 1, 16, conv_block=CascadeConv1d)
    model = MultiScaleNet(4, 1, 16, input_dim=35, num_classes=4)
    y, _ = model(x)
    print(y.size())
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, in_planes, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm1d(in_planes)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv1d(in_planes, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_planes, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_planes + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(in_planes))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(in_planes, out_planes,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))


class DenseNet_1d(nn.Module):

    def __init__(self, in_planes=3, feature_dim=32, growth_rate=32, block_config=(6, 12, 24, 16),
                 init_planes=64, bn_size=4, drop_rate=0, num_classes=4):

        super(DenseNet_1d, self).__init__()

        # First convolution
        """
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(in_planes, init_planes, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm1d(init_planes)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
        ]))
        feature_dim = feature_dim // 4
        """
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(in_planes, init_planes, kernel_size=1, stride=1, bias=False)),
            ('norm0', nn.BatchNorm1d(init_planes)),
            ('relu0', nn.ReLU(inplace=True))
        ]))

        # Each denseblock
        num_planes = init_planes
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, in_planes=num_planes,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_planes = num_planes + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(in_planes=num_planes, out_planes=num_planes // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_planes = num_planes // 2
                feature_dim = feature_dim // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_planes))

        # Linear layer
        self.classifier = nn.Linear(num_planes*feature_dim, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = out.view(features.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    model = DenseNet_1d(init_planes=16, growth_rate=4, block_config=(4, 8))
    print(model)
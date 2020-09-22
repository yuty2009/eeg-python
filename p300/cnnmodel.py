# -*- coding: utf-8 -*-

import torch.nn as nn

INPUT_SHAPE = [78, 64]
MAP_NUM_LAYER1 = 10
MAP_NUM_LAYER2 = 50
BOTTLENECK_SIZE = 100


class CNNModel(nn.Module):

    def __init__(self, dropout=0.5):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, MAP_NUM_LAYER1, kernel_size=(1, 64)),
            nn.ReLU(inplace=True),
            nn.Conv2d(MAP_NUM_LAYER1, MAP_NUM_LAYER2, kernel_size=(13, 1), stride=(13, 1)),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(MAP_NUM_LAYER2*6, BOTTLENECK_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(BOTTLENECK_SIZE, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



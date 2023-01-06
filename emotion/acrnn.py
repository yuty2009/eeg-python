# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from common.torchutils import Expression, safe_log, square, log_cov
from common.torchutils import Conv2dNormWeight


class ACRNN(nn.Module):
    """
    Attention-based convolutional recurrent neural network
    Reference: 
    Wei Tao, Chang Li, Rencheng Song, Juan Cheng, Yu Liu, Feng Wan and Xun Chen,
    "EEG-based Emotion Recognition via Channel-wise Attention and Self Attention,"
    in IEEE Transactions on Affective Computing, doi: 10.1109/TAFFC.2020.3025777.
    """
    def __init__(
        self, n_timepoints, n_channels, n_classes,
        n_filters_t = 8, filter_size_t = 20,
        n_filters_s = 4, filter_size_s = -1, dropout = 0.5,
        pool_size_1 = 50, pool_stride_1 = 25,
    ):
        super().__init__()
        assert filter_size_t <= n_timepoints, "Temporal filter size error"
        if filter_size_s <= 0: filter_size_s = n_channels

        ## channel-wise attention
        attn_dim = n_channels // 4
        self.channel_attention = nn.Sequential(
            nn.AvgPool2d((n_timepoints, 1)),
            nn.Flatten(1),
            nn.Linear(n_channels, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, n_channels),
            nn.Softmax(dim=1)
        )
        ## feature extraction
        self.features = nn.Sequential(
            # temporal filtering
            nn.Conv2d(1, n_filters_t, (filter_size_t, 1), padding=(filter_size_t//2, 0), bias=False),
            nn.BatchNorm2d(n_filters_t),
            # spatial filtering
            nn.Conv2d(
                n_filters_t, n_filters_t*n_filters_s, (1, filter_size_s), 
                groups=n_filters_t, bias=False
            ),
            nn.BatchNorm2d(n_filters_t*n_filters_s),
            Expression(square),
            nn.AvgPool2d((pool_size_1, 1), stride=(pool_stride_1, 1)),
            Expression(safe_log),
            nn.Dropout(dropout), 
        )
        n_features = (n_timepoints - pool_size_1)//pool_stride_1 + 1
        ## classfier
        self.classifier = nn.Sequential(
            Conv2dNormWeight(n_filters_t * n_filters_s, n_classes, (n_features, 1), max_norm=0.5),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        chan_attn = self.channel_attention(x)
        chan_attn_mat = chan_attn.unsqueeze(1).unsqueeze(1).expand_as(x)
        x = x * chan_attn_mat
        x = self.features(x)
        x = self.classifier(x)
        x = x[:, :, 0, 0]
        return x


if __name__ == '__main__':

    model = ACRNN(600, 62, 3)
    
    x = torch.randn((10, 1, 600, 62))
    print(model)
    y = model(x)
    print(y.shape)
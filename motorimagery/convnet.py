# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from common.torchutils import Expression, safe_log, square, log_cov
from common.torchutils import DepthwiseConv2d, SeparableConv2d, Conv2dNormWeight, Swish


class CSPNet(nn.Module):
    """
    ConvNet model mimics CSP
    """
    def __init__(
        self, n_timepoints, n_channels, n_classes, dropout = 0.5,
        n_filters_t = 20, filter_size_t = 25,
        n_filters_s = 2, filter_size_s = -1,
        pool_size_1 = 75, pool_stride_1 = 15,
        n_filters_f = 16, filter_size_f = 16,
        pool_size_2 = 8, pool_stride_2 = 8,
    ):
        super().__init__()
        assert filter_size_t <= n_timepoints, "Temporal filter size error"
        if filter_size_s <= 0: filter_size_s = n_channels

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

        n_features = (n_timepoints - pool_size_1) // pool_stride_1 + 1
        n_filters_out = n_filters_t * n_filters_s
        self.classifier = nn.Sequential(
            Conv2dNormWeight(n_filters_out, n_classes, (n_features, 1), max_norm=0.5),
            nn.LogSoftmax(dim=1)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x[:, :, 0, 0]
        return x


class EEGNet(nn.Module):
    """ 
    Pytorch Implementation of EEGNet from [1]
    code: https://github.com/vlawhern/arl-eegmodels
         
    References
    ----------

    .. [1] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung,
           and B. J. Lance, "EEGNet: a compact convolutional neural network for 
           EEG-based brain-computer interfaces," J Neural Eng, vol. 15, no. 5, 
           p. 056013, Oct 2018.
           Online: http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Inputs:
    filter_size_time_1: length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
    """
    def __init__(
        self, n_timepoints, n_channels, n_classes, dropout = 0.5,
        n_filters_1 = 8, filter_size_time_1 = 125, 
        pool_size_time_1 = 4, pool_stride_time_1 = 4,
        n_filters_2 = 16, filter_size_time_2 = 22,
        pool_size_time_2 = 8, pool_stride_time_2 = 8,
    ):
        super().__init__()
        assert filter_size_time_1 <= n_timepoints, "Temporal filter size error"

        self.features = nn.Sequential(
            # temporal filtering
            nn.Conv2d(1, n_filters_1, (filter_size_time_1, 1), padding=(filter_size_time_1//2, 0), bias=False),
            nn.BatchNorm2d(n_filters_1),
            # spatial filtering
            # DepthwiseConv2d(n_filters_1, 2, (1, n_channels), bias=False),
            Conv2dNormWeight(
                n_filters_1, n_filters_1*2, (1, n_channels), 
                max_norm=1, groups=n_filters_1, bias=False
                ),
            nn.BatchNorm2d(n_filters_1*2),
            nn.ELU(),
            nn.AvgPool2d((pool_size_time_1, 1), stride=(pool_stride_time_1, 1)),
            nn.Dropout(dropout),
            # SeparableConv2d
            SeparableConv2d(
                n_filters_1*2, n_filters_2, (filter_size_time_2, 1), 
                padding=(filter_size_time_2//2, 0), bias=False
                ),
            nn.BatchNorm2d(n_filters_2),
            nn.ELU(),
            nn.AvgPool2d((pool_size_time_2, 1), stride=(pool_stride_time_2, 1)),
            nn.Dropout(dropout),
        )

        n_features_1 = (n_timepoints - pool_size_time_1)//pool_stride_time_1 + 1
        n_features_2 = (n_features_1 - pool_size_time_2)//pool_stride_time_2 + 1
        n_filters_out = n_filters_2
        n_features_out = n_filters_out * n_features_2
        self.classifier = nn.Sequential(
            Conv2dNormWeight(n_filters_2, n_classes, (n_features_2, 1), max_norm=0.5),
            # nn.Linear(n_features_out, n_classes),
            nn.LogSoftmax(dim=1)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x[:, :, 0, 0]
        return x


class ShallowConvNet(nn.Module):
    """
    Shallow ConvNet model from [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., 
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping, Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """
    def __init__(
        self, n_timepoints, n_channels, n_classes, dropout = 0.5,
        n_filters = 40, filter_size_time = 25,
        pool_size_time = 75, pool_stride_time = 15,
    ):
        super().__init__()
        assert filter_size_time <= n_timepoints, "Temporal filter size error"

        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_size_time, 1)),               # temporal filtering
            nn.Conv2d(n_filters, n_filters, (1, n_channels), bias=False), # spatial filtering
            nn.BatchNorm2d(n_filters),
            Expression(square),
            nn.AvgPool2d((pool_size_time, 1), stride=(pool_stride_time, 1)),
            Expression(safe_log),
            nn.Dropout(dropout),
        )

        outlen_time = (n_timepoints - filter_size_time) + 1
        outlen_time = (outlen_time - pool_size_time)//pool_stride_time + 1
        self.classifier = nn.Sequential(
            Conv2dNormWeight(n_filters, n_classes, (outlen_time, 1), max_norm=0.5, bias=True),
            nn.LogSoftmax(dim=1)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x[:, :, 0, 0]
        return x


class DeepConvNet(nn.Module):
    """
    Deep ConvNet model from [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., 
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping, Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """
    def __init__(
        self, n_timepoints, n_channels, n_classes, dropout = 0.5,
        n_filters = 25, filter_size_time = 10,
        pool_size_time = 3, pool_stride_time = 3,
        n_conv_blocks = 4
    ):
        super().__init__()
        assert filter_size_time <= n_timepoints, "Temporal filter size error"

        conv1 = nn.Sequential(
            Conv2dNormWeight(1, n_filters, (filter_size_time, 1), max_norm=2),               # temporal filtering
            Conv2dNormWeight(n_filters, n_filters, (1, n_channels), max_norm=2, bias=False), # spatial filtering
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.MaxPool2d((pool_size_time, 1), stride=(pool_stride_time, 1)),
        )

        n_filters_prev = n_filters
        outlen_time = (n_timepoints - filter_size_time) + 1
        outlen_time = (outlen_time - pool_size_time)//pool_stride_time + 1
        conv_blocks = nn.ModuleList()
        for i in range(n_conv_blocks-1):
            n_filters_now = 2 * n_filters_prev
            conv_blocks.append(self._make_block(
                n_filters_prev, n_filters_now, filter_size_time,pool_size_time, pool_stride_time, dropout
            ))
            n_filters_prev = n_filters_now
            outlen_time = (outlen_time - filter_size_time) + 1
            outlen_time = (outlen_time - pool_size_time)//pool_stride_time + 1

        self.features = nn.Sequential(conv1, *conv_blocks)
        
        self.classifier = nn.Sequential(
            Conv2dNormWeight(n_filters_now, n_classes, (outlen_time, 1), max_norm=0.5, bias=True),
            nn.LogSoftmax(dim=1)
        )

        self._reset_parameters()

    def _make_block(self, in_planes, out_planes, filter_size,
                    pool_size, pool_stride, dropout=0.5):
        return nn.Sequential(
            nn.Dropout(dropout),
            Conv2dNormWeight(in_planes, out_planes, (filter_size, 1), max_norm=2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ELU(),
            nn.MaxPool2d((pool_size, 1), stride=(pool_stride, 1)),
        )

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x[:, :, 0, 0]
        return x


if __name__ == '__main__':

    # model = ShallowConvNet(534, 44, 4)
    # model = DeepConvNet(534, 44, 4)
    # model = EEGNet(534, 44, 4)
    model = CSPNet(534, 44, 4)
    
    x = torch.randn((10, 1, 534, 44))
    print(model)
    y = model(x)
    print(y.shape)
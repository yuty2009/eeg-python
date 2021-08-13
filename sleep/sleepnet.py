# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


class Conv2dBnReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if stride[0] > 1 or stride[1] > 1:
            ConvLayer = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, 
                padding=(kernel_size[0]//2, kernel_size[1]//2), bias=False, **kwargs
                )
        else:
            ConvLayer = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding='same', bias=False, **kwargs
                )
        super(Conv2dBnReLU, self).__init__(
            ConvLayer,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class LSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers,
        dropout=0.5, bidirectional=False, return_last=True
        ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_last = return_last
        self.D = 1 
        if bidirectional is True: self.D = 2
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=bidirectional
            )
    
    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(self.num_layers*self.D, x.size(0), self.hidden_size).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers*self.D, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # RNN output (batch_size, seq_length, hidden_size)
        if self.return_last: out = out[:, -1, :]
        return out


class DeepSleepNet(nn.Module):
    """
    Reference:
    A. Supratak, H. Dong, C. Wu, and Y. Guo, "DeepSleepNet: A Model for Automatic
    Sleep Stage Scoring Based on Raw Single-Channel EEG," IEEE Trans Neural Syst 
    Rehabil Eng, vol. 25, no. 11, pp. 1998-2008, 2017.
    https://github.com/akaraspt/deepsleepnet
    """
    def __init__(
        self, n_timepoints, n_seqlen, n_classes, dropout = 0.5,
        # Conv layers
        n_filters_1 = 64, filter_size_1 = 50, filter_stride_1 = 6,
        n_filters_2 = 64, filter_size_2 = 400, filter_stride_2 = 50,
        pool_size_11 = 8, pool_stride_11 = 8, 
        pool_size_21 = 4, pool_stride_21 = 4,
        n_filters_1x3 = 128, filter_size_1x3 = 8,
        n_filters_2x3 = 128, filter_size_2x3 = 6,
        pool_size_12 = 4, pool_stride_12 = 4, 
        pool_size_22 = 2, pool_stride_22 = 2,
        # LSTM layers
        n_rnn_layers = 2,
        n_hidden_rnn = 512, n_hidden_fc = 1024,
    ):
        super().__init__()
        self.n_seqlen = n_seqlen
        self.conv1 = nn.Sequential(
            Conv2dBnReLU(1, n_filters_1, (filter_size_1, 1), (filter_stride_1, 1)),
            nn.MaxPool2d((pool_size_11, 1), (pool_stride_11, 1)),
            nn.Dropout(dropout),
            Conv2dBnReLU(n_filters_1,   n_filters_1x3, (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3, (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3, (filter_size_1x3, 1), stride=1),
            nn.MaxPool2d((pool_size_12, 1), (pool_stride_12, 1)),
        )
        self.conv2 = nn.Sequential(
            Conv2dBnReLU(1, n_filters_2, (filter_size_2, 1), (filter_stride_2, 1)),
            nn.MaxPool2d((pool_size_21, 1), (pool_stride_21, 1)),
            nn.Dropout(dropout),
            Conv2dBnReLU(n_filters_2,   n_filters_2x3, (filter_size_2x3, 1), stride=1),
            Conv2dBnReLU(n_filters_2x3, n_filters_2x3, (filter_size_2x3, 1), stride=1),
            Conv2dBnReLU(n_filters_2x3, n_filters_2x3, (filter_size_2x3, 1), stride=1),
            nn.MaxPool2d((pool_size_22, 1), (pool_stride_22, 1)),
        )
        self.drop1 = nn.Dropout(dropout)

        outlen_conv1 = n_timepoints // filter_stride_1 // pool_stride_11 // pool_stride_12
        outlen_conv2 = n_timepoints // filter_stride_2 // pool_stride_21 // pool_stride_22
        outlen_conv = outlen_conv1*n_filters_1x3 + outlen_conv2*n_filters_2x3

        self.classifier_pretrain = nn.Sequential(
            nn.Linear(outlen_conv, n_classes),
            nn.LogSoftmax(dim=1)
        )

        self.res_branch = nn.Sequential(
            nn.Linear(outlen_conv, n_hidden_fc),
            nn.ReLU(inplace=True)
        )
        self.rnn_branch = LSTM(outlen_conv, n_hidden_rnn, n_rnn_layers, bidirectional=True)
        self.drop2 = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden_fc, n_classes),
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
        x = x.reshape((-1,)+ x.shape[2:])
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = x1.view(x1.size(0), -1) # flatten (b, c, t, 1) -> (b, c*t)
        x2 = x2.view(x2.size(0), -1) # flatten (b, c, t, 1) -> (b, c*t)
        x = torch.cat((x1, x2), dim=1) # concat in feature dimention
        x = self.drop1(x)
        x_seq = x.view(-1, self.n_seqlen, x.size(-1))
        x_res = self.res_branch(x_seq[:, -1, :])
        x_rnn = self.rnn_branch(x_seq)
        x = x_rnn + x_res
        x = self.drop2(x)
        x = self.classifier(x)
        return x


class TinySleepNet(nn.Module):
    """
    Reference:
    A. Supratak and Y. Guo, "TinySleepNet: An Efficient Deep Learning Model
    for Sleep Stage Scoring based on Raw Single-Channel EEG," Annu Int Conf
    IEEE Eng Med Biol Soc, vol. 2020, pp. 641-644, Jul 2020.
    https://github.com/akaraspt/tinysleepnet
    """
    def __init__(
        self, n_timepoints, n_seqlen, n_classes, dropout = 0.5,
        # Conv layers
        n_filters_1 = 128, filter_size_1 = 50, filter_stride_1 = 6,
        pool_size_1 = 8, pool_stride_1 = 8, 
        n_filters_1x3 = 128, filter_size_1x3 = 8,
        pool_size_2 = 4, pool_stride_2 = 4, 
        # LSTM layers
        n_rnn_layers = 2,
        n_hidden_rnn = 128,
    ):
        super().__init__()
        self.n_seqlen = n_seqlen
        self.conv1 = nn.Sequential(
            Conv2dBnReLU(1, n_filters_1, (filter_size_1, 1), (filter_stride_1, 1)),
            nn.MaxPool2d((pool_size_1, 1), (pool_stride_1, 1)),
            nn.Dropout(dropout),
            Conv2dBnReLU(n_filters_1,   n_filters_1x3, (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3, (filter_size_1x3, 1), stride=1),
            Conv2dBnReLU(n_filters_1x3, n_filters_1x3, (filter_size_1x3, 1), stride=1),
            nn.MaxPool2d((pool_size_2, 1), (pool_stride_2, 1)),
            nn.Dropout(dropout)
        )

        outlen_conv1 = n_timepoints // filter_stride_1 // pool_stride_1 // pool_stride_2
        outlen_conv = outlen_conv1*n_filters_1x3

        self.classifier_pretrain = nn.Sequential(
            nn.Linear(outlen_conv, n_classes),
            nn.LogSoftmax(dim=1)
        )

        self.rnn_branch = LSTM(outlen_conv, n_hidden_rnn, n_rnn_layers)
        self.drop2 = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden_rnn, n_classes),
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
        x = x.reshape((-1,)+ x.shape[2:])
        x = self.conv1(x)
        x = x.view(x.size(0), -1) # flatten (b, c, t, 1) -> (b, c*t)
        x_seq = x.view(-1, self.n_seqlen, x.size(-1)) # (batch, seqlen, hidden)
        x = self.rnn_branch(x_seq)
        x = self.drop2(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':

    x = torch.randn((20, 10, 1, 3000, 1))
    # model = DeepSleepNet(3000, 10, 5)
    model = TinySleepNet(3000, 10, 5)
    print(model)
    y = model(x)
    print(y.shape)
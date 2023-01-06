# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from common.torchutils import Conv2dNormWeight


class Transformer(nn.Module):
    """Transformer class copy-paste from torch.nn.Transformer
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", layer_norm_eps: float = 1e-5, 
                 batch_first: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first,
                                                **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self._reset_parameters()

    def forward(self, src):
        output = self.encoder(src)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TemporalTransformer(nn.Module):
    """
    Transformer applied to an (n_timepoints by n_channels) EEG epoch
    and output a temporally self-attended epoch with the same dimension.
    Multi-head self-attention is applied on each single EEG channel (electrode).
    """
    def __init__(
        self, n_timepoints, n_channels,
        patch_size_t = 50, nhead = 4, num_layers = 3,
        pool = 'cls', dropout = 0.1,
    ):
        super().__init__()
        assert patch_size_t <= n_timepoints, "Temporal patch size error"
        assert n_timepoints % patch_size_t == 0, \
               'Temporal dimension must be divisible by the patch size.'
        self.pool = pool
        d_model = n_timepoints
        n_patches = n_timepoints // patch_size_t
        # patch embedding
        self.patch_embedding = nn.Conv2d(1, d_model, (patch_size_t, 1), stride=(patch_size_t, 1))
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)
        # transformer
        self.transformer = Transformer(
            d_model = d_model,
            nhead = nhead,
            num_layers = num_layers,
            dim_feedforward = 3 * d_model,
            batch_first = True,
        )

    def forward(self, x): # x is (bs, 1, n_timepoints, n_channels)
        x = self.patch_embedding(x) # (bs, d_model, n_patches, n_channels)
        bs, n_channels = x.size(0), x.size(3)
        input_shape = (bs*n_channels, x.size(2), x.size(1))
        x = x.permute(0, 3, 2, 1).reshape(input_shape) # (bs*n_channels, n_patches, d_model)
        # prepend class token
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat([cls_token, x], dim=1) # (bs*n_channels, num_patches+1, d_model)
        x = self.pos_drop(x + self.pos_embedding)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = x.reshape(bs, n_channels, -1).permute(0, 2, 1).unsqueeze(1) # (bs, 1, d_model, n_channels)
        return x


class EEGTransformer(nn.Module):
    """
    Transformer for EEG classification
    """
    def __init__(
        self, n_timepoints, n_channels, n_classes, dropout = 0.1,
        patch_size_t = 50, nhead = 8, num_layers = 3, pool = 'cls',
        n_filters_s = 6, filter_size_s = -1,
        pool_size_1 = 50, pool_stride_1 = 50,
    ):
        super().__init__()
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
        ## transformer encoder layer
        self.transformer_t = TemporalTransformer(
            n_timepoints, n_channels,
            patch_size_t, nhead, num_layers,
            pool, dropout
        )
        ## feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters_s, (1, n_channels), bias=False),
            nn.BatchNorm2d(n_filters_s),
            nn.ELU(),
            # pooling
            nn.AvgPool2d((pool_size_1, 1), stride=(pool_stride_1, 1)),
            nn.Dropout(dropout),
        )
        n_features = (n_timepoints - pool_size_1)//pool_stride_1 + 1
        ## classfier
        self.classifier = nn.Sequential(
            Conv2dNormWeight(n_filters_s, n_classes, (n_features, 1), max_norm=0.5),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        chan_attn = self.channel_attention(x)
        chan_attn_mat = chan_attn.unsqueeze(1).unsqueeze(1).expand_as(x)
        x = x * chan_attn_mat # (bs, 1, n_timepoints, n_channels)
        x = self.transformer_t(x)
        x = self.features(x)
        x = self.classifier(x)
        x = x[:, :, 0, 0]
        return x


if __name__ == '__main__':

    model = EEGTransformer(600, 62, 3)
    
    x = torch.randn((10, 1, 600, 62))
    print(model)
    y = model(x)
    print(y.shape)
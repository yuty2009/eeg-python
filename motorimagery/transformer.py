
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.attention import MultiheadAttention
from common.torchutils import Expression, safe_log, square, log_cov
from common.torchutils import DepthwiseConv2d, SeparableConv2d, Conv2dNormWeight


class EEGTransformer(nn.Module):
    def __init__(self, n_timepoints, n_channels, n_classes, dropout = 0.1,
                 n_filters_t = 20, filter_size_t = 25,
                 n_filters_s = 2, filter_size_s = -1,
                 pool_size_1 = 100, pool_stride_1 = 25,
                 nhead = 8, num_layers = 2, ffn_dim = 64):
        super().__init__()
        assert filter_size_t <= n_timepoints, "Temporal filter size error"
        if filter_size_s <= 0: filter_size_s = n_channels
        d_model = n_timepoints
        # temporal filtering
        self.feature_pre = nn.Sequential(
            nn.Conv2d(1, n_filters_t, (filter_size_t, 1), padding=(filter_size_t//2, 0), bias=False),
            nn.BatchNorm2d(n_filters_t),
        )
        n_patches = n_filters_t * n_channels
        # positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, d_model))
        self.pos_drop = nn.Dropout(dropout)
        # transformer
        self.transformer = Transformer(
            d_model = d_model,
            nhead = nhead,
            num_layers = num_layers,
            dim_feedforward = ffn_dim,
            batch_first = True,
        )
        # spatial filtering
        self.feature_post = nn.Sequential(
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
        n_features = n_filters_t * n_filters_s
        # classfier
        self.classifier = nn.Sequential(
            nn.Conv2d(n_features, n_classes, kernel_size=1),
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

    def forward(self, eeg_epoch):
        x = self.feature_pre(eeg_epoch) # (bs, filters, n_timepoints, n_channels)
        x = x.permute(0, 1, 3, 2) # (bs, filters, n_channels, n_timepoints)
        raw_shape = x.shape
        x = x.reshape(x.size(0), -1, x.size(-1)) # (bs, n_patches, n_timepoints)
        x = self.pos_drop(x + self.pos_embedding)
        x = self.transformer(x)
        x = x.view(raw_shape) # (bs, filters, n_channels, n_timepoints)
        x = x.permute(0, 1, 3, 2) # (bs, filters, n_timepoints, n_channels)
        x = self.feature_post(x)
        x = self.classifier(x)
        out = x[:, :, 0, 0]
        return out


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

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
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


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer copy-paste from torch.nn.TransformerEncoderLayer
        with modifications:
        * layer norm before add residual
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, normalize_before=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, 
        #                                      batch_first=batch_first, **factory_kwargs)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward_post(self, src, src_mask = None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask = None):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask)
        return self.forward_post(src, src_mask)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


if __name__ == '__main__':

    model = EEGTransformer(
        n_timepoints = 534,
        n_channels = 44,
        n_classes = 4,
    )

    x = torch.randn((10, 1, 534, 44))
    print(model)
    y = model(x)
    print(y.shape)

"""
Reference:
    https://github.com/jadore801120/attention-is-all-you-need-pytorch/
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        d_model = q.size(-1)
        q = q / math.sqrt(d_model)
        attn = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        if self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout)
        output = torch.matmul(attn, v)

        return output, attn


class MultiheadAttention(nn.Module):
    ''' Multi-Head Attention Module '''

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()

        # assert d_model % n_head == 0, 
        #   f'd_model {d_model} not divisible by n_head {n_head}'
        # we do not require d_model divisible by n_head here
        d_head = math.ceil(d_model / n_head)
        self.n_head = n_head
        self.d_head = d_head

        self.w_qs = nn.Linear(d_model, n_head * d_head)
        self.w_ks = nn.Linear(d_model, n_head * d_head)
        self.w_vs = nn.Linear(d_model, n_head * d_head)
        self.w_out = nn.Linear(n_head * d_head, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, q, k, v, attn_mask=None):

        sz_b = q.size(0)

        # Pass through the pre-attention projection: (b, lq, n*dm)
        # Separate different heads: (b, lq, n, dm)
        q = self.w_qs(q).view(sz_b, -1, self.n_head, self.d_head)
        k = self.w_ks(k).view(sz_b, -1, self.n_head, self.d_head)
        v = self.w_vs(v).view(sz_b, -1, self.n_head, self.d_head)
        # Transpose for attention dot product: (b, n, lq, dm)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)   # For head axis broadcasting.
        out, attn = self.attention(q, k, v, mask=attn_mask) # out: (b, n, lq, dm)

        # Transpose to move the head dimension back: (b, lq, n, dm)
        # Concatenate all the heads together: (b, lq, n*dm)
        out = out.transpose(1, 2).contiguous() \
            .view(sz_b, -1, self.n_head * self.d_head)
        out = self.w_out(out)

        return out, attn


if __name__ == '__main__':

    input_src = torch.randn((64, 20, 256)) # (batch x seqlen x d_model)
    input_tgt = torch.randn((64, 10, 256))
    model = MultiheadAttention(256, 8)
    output, attn = model(input_tgt, input_src, input_src)
    print(output.shape)
    print(attn.shape)
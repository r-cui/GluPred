import math
import torch
import torch.nn as nn


def positional_encoding(length, d_model):
    """ Generate the positional encoding in the raw paper.
    Returns:
        (length, d_model)
    """
    pe = torch.zeros(length, d_model)
    pos = torch.arange(length).unsqueeze(1)

    pe[:, 0::2] = torch.sin(pos / torch.pow(10000, torch.arange(0, d_model, step=2, dtype=torch.float32) / d_model))
    pe[:, 1::2] = torch.cos(pos / torch.pow(10000, torch.arange(1, d_model, step=2, dtype=torch.float32) / d_model))

    return pe


def get_past_mask(l):
    """ Generate the past mask.
    Args:
        l: int
    Returns:
        (l, l)
            Row is query, col is key. True is not accessible.
            Ex: l=5.
                [[False,  True,  True,  True,  True],
                 [False, False,  True,  True,  True],
                 [False, False, False,  True,  True],
                 [False, False, False, False,  True],
                 [False, False, False, False, False]]
    """
    return torch.triu(torch.ones((l, l)), diagonal=1).bool()


# def get_kernel_mask(l, kernel_size, dilation_factor):
#     """ This mask has two properties.
#     1. causal: each element can only see itself and elements before itself.
#     2. kernelised: up to kernel_size elements are accessible by each element.
#     Args:
#         l: int
#         kernel_size: int
#             the receptive field of each query
#         dilation_factor: int
#             the index difference of adjacent accessible elements
#     Returns:
#         (length, length)
#             Row is query, col is key. True is not accessible.
#             Ex: l=10, kernel_size=3, dilation_factor=2
#                       keys
#             queries [[False,  True,  True,  True,  True,  True,  True,  True,  True,  True],
#                      [ True, False,  True,  True,  True,  True,  True,  True,  True,  True],
#                      [False,  True, False,  True,  True,  True,  True,  True,  True,  True],
#                      [ True, False,  True, False,  True,  True,  True,  True,  True,  True],
#                      [False,  True, False,  True, False,  True,  True,  True,  True,  True],
#                      [ True, False,  True, False,  True, False,  True,  True,  True,  True],
#                      [ True,  True, False,  True, False,  True, False,  True,  True,  True],
#                      [ True,  True,  True, False,  True, False,  True, False,  True,  True],
#                      [ True,  True,  True,  True, False,  True, False,  True, False,  True],
#                      [ True,  True,  True,  True,  True, False,  True, False,  True, False]])
#     """
#     res = torch.ones((l, l))
#     for i in range(l):
#         idx = torch.arange(i, max(i - ((kernel_size - 1) * dilation_factor + 1), -1), step=-dilation_factor)
#         res[i, idx] = False
#
#     return res.bool()


class MultiHeadedAttention(nn.Module):
    """ The multihead attention module, including dot-product attention.
    
    Adapted from https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/multi_headed_attn.py
    """
    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0
        self.dim_per_head = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.linear_keys = nn.Linear(d_model, n_heads * self.dim_per_head)
        self.linear_values = nn.Linear(d_model, n_heads * self.dim_per_head)
        self.linear_query = nn.Linear(d_model, n_heads * self.dim_per_head)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(d_model, d_model)

    def forward(self, key, value, query, mask=None):
        """
        Args:
            key: (N, l_key, d_model)
            value: (N, l_key, d_model)
            query: (N, l_query, d_model)
            mask: (N, l_query, l_key)
                binary mask 1/0
        """
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        n_heads = self.n_heads
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * dim_per_head)

        key = self.linear_keys(key)  # (N, l_key, d_model)
        value = self.linear_values(value)  # (N, l_key, d_model)
        query = self.linear_query(query)  # (N, l_query, l_key)
        key = shape(key)  # (N, n_heads, l_key, dim_per_head)
        value = shape(value)  # (N, n_heads, l_key, dim_per_head)
        query = shape(query)  # # (N, n_heads, l_query, dim_per_head)

        query = query / math.sqrt(dim_per_head)
        query_key = torch.matmul(query, key.transpose(2, 3))
        scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e18)

        attn = self.softmax(scores).to(query.dtype)  # (N, n_heads, l_query, l_key)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)  # (N, n_heads, l_query, dim_per_head)
        context = unshape(context_original)  # (N, l_query, d_model)
        output = self.final_linear(context)  # (N, l_query, d_model)
        attns = attn.view(batch_size, n_heads, query_len, key_len)

        return output, attns


class PositionwiseFeedForward(nn.Module):
    """ A two layer FF with residual layer norm.
    Args:
        d_model: int
        d_ff: int
        dropout: float
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (N, input_len, d_model)
        Returns:
            (N, input_len, d_model)
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, l, d_model)
            mask: (N, l, l)
        Returns:
            (N, l, d_model)
        """
        input_norm = self.layer_norm(x)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + x
        return self.feed_forward(out)


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, attention_dropout):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_ff, dropout, attention_dropout) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, l, d_model)
            mask: (N, l, l)
        Returns:
            (N, l, d_model)
        """
        x += positional_encoding(x.shape[1], x.shape[2]).to(x.device)
        out = x
        for layer in self.layers:
            out = layer(out, mask)
        out = self.layer_norm(out)

        return out.contiguous()
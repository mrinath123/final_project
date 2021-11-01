import numpy as np

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, d_model=48, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        d_xq = d_xk = d_xv = d_model
        # Embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0
        self.d_k = d_model // self.num_heads
        # These are still of dimension d_model. To split into number of heads
        self.Wq = nn.Linear(d_xq, d_model)
        self.Wk = nn.Linear(d_xk, d_model)
        self.Wv = nn.Linear(d_xv, d_model)
        # Outputs of all sub-layers need to be of dimension d_model
        self.Wh = nn.Linear(d_model, d_model)

    def dot_product_attn(self, Q, K, V):
        batch_size = Q.size(0)
        k_length = K.size(-2)

        # Scaling by d_k so that the soft(arg)max doesnt saturate
        Q = Q / np.sqrt(self.d_k)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(Q, K.transpose(2, 3))  # (bs, n_heads, q_length, k_length)

        A = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)

        # Get the weighted average of the values
        H = torch.matmul(A, V)  # (bs, n_heads, q_length, dim_per_head)

        return H, A

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

    def forward(self, Xq, Xk, Xv):
        batch_size, seq_length, dim = Xq.size()

        Q = self.split_heads(self.Wq(Xq), batch_size)
        K = self.split_heads(self.Wk(Xk), batch_size)
        V = self.split_heads(self.Wv(Xv), batch_size)

        H_cat, A = self.dot_product_attn(Q, K, V)

        H_cat = self.group_heads(H_cat, batch_size)

        H = self.Wh(H_cat)  # (bs, seq_length, dim)
        return H, A


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(48, 16)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(16, 48)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        return x


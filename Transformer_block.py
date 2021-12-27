import torch
import torch.nn as nn
import torch.nn.functional as F
from CrossAttention import CrossAttention, MLP

class T_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = CrossAttention()

        self.ln1 = nn.LayerNorm(normalized_shape=48, eps=1e-6)
        self.ln2 = nn.LayerNorm(normalized_shape=48, eps=1e-6)

        self.mlp = MLP()

    def forward(self, Xq, Xk, Xv):
        attn_op, _ = self.cross_attn(Xq, Xk, Xv)
        attn_op = Xq + attn_op
        op = self.ln1(attn_op)
        mlp_op = self.mlp(op)
        mlp_op = self.ln2(mlp_op + op)

        return mlp_op
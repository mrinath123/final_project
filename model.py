import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


from CrossAttention import CrossAttention, MLP
from Backbone import Backbone

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.cross_attn = CrossAttention()

        self.ln1 = nn.LayerNorm(normalized_shape=48, eps=1e-6)
        self.ln2 = nn.LayerNorm(normalized_shape=48, eps=1e-6)

        self.mlp = MLP()

    def forward(self, image, patch):
        Xq, Xk, Xv = self.backbone(image, patch)
        # flattening
        Xq = Xq.view(Xq.shape[0], Xq.shape[1], -1)
        Xv = Xv.view(Xv.shape[0], Xv.shape[1], -1)
        Xk = Xk.view(Xk.shape[0], Xk.shape[1], -1)
        # shape -> (n_batch , dim , sq_len)

        # transposing
        Xq = Xq.permute(0, 2, 1)
        Xk = Xk.permute(0, 2, 1)
        Xv = Xv.permute(0, 2, 1)
        # shape -> (n_batch, seq,len , dim)

        attn_op, _ = self.cross_attn(Xq, Xk, Xv)
        attn_op = Xq + attn_op
        op = self.ln1(attn_op)
        mlp_op = self.mlp(op)
        mlp_op = self.ln2(mlp_op + op)

        return mlp_op


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(48, 64, 2, stride=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((20, 20)),
            nn.ConvTranspose2d(64, 16, 2, stride=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((50, 50)))
        self.up = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.view(x.shape[0], -1, 8, 8)
        x = self.layer(x)
        x = F.sigmoid(self.up(x))

        return x


class TrackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, image, patch):
        enc_op = self.enc(image, patch)
        decoder_op = self.dec(enc_op)

        return decoder_op

if __name__ == "__main__":
    i1 = torch.randn(1, 3, 224, 224)
    i2 = torch.randn(1, 3, 64, 64)

    b = TrackNet()
    op = b(i1, i2)

    print(op.shape)
    

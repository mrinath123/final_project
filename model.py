import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from Backbone import Backbone
from Transformer_block import T_block


class Encoder(nn.Module):
    def __init__(self,name , pretrained=False,num_layers = 5):
        super().__init__()
        self.n = name
        self.p = pretrained
        self.nl = num_layers
        self.backbone = Backbone(self.n , pretrained = self.p)
        
        self.layers = nn.ModuleList(
            [
                T_block() for _ in range(self.nl)
            ]
        )

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

        for layer in self.layers:
            Xq = layer(Xq, Xk, Xv)

        return Xq


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnv1 = nn.Conv2d(48,48,1)
        self.cnv2 = nn.Conv2d(48,32,1)
        self.cnv3 = nn.Conv2d(32,16,1)
        self.cnv4 = nn.Conv2d(16,1,1)
        self.cnv5 = nn.Conv2d(1,1,1)
        self.pool = nn.AdaptiveMaxPool2d((100, 100))

    def forward(self, x):
      
        x = x.permute(0, 2, 1)
        x = x.view(x.shape[0], -1, 8, 8)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.cnv1(x))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.cnv2(x))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.cnv3(x))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.cnv4(x))
        x = self.pool(x)
        x = F.sigmoid(self.cnv5(x))

        return x


class TrackNet(nn.Module):
    def __init__(self,name , pretrained=False):
        super().__init__()
        self.n = name
        self.p = pretrained
        self.enc = Encoder(self.n , pretrained = self.p)
        self.dec = Decoder()

    def forward(self, image, patch):
        enc_op = self.enc(image, patch)
        decoder_op = self.dec(enc_op)

        return decoder_op

if __name__ == "__main__":
    i1 = torch.randn(1, 3, 224, 224)
    i2 = torch.randn(1, 3, 64, 64)

    b = TrackNet("mobilnet")
    op = b(i1, i2)

    print(op.shape)
    

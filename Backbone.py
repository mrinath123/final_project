import torch
import torch.nn as nn
import timm


class CNN(nn.Module):
    def __init__(self, in_channels, op_shape):
        super().__init__()
        self.inc = in_channels
        self.op = op_shape
        self.cnv = nn.Conv2d(self.inc, self.inc, 5, stride=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((self.op, self.op))

    def forward(self, x):
        x = self.cnv(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.pool(x)

        return x


class Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.back = timm.create_model('efficientnet_b3', pretrained=pretrained, features_only=True)
        self.cnn = CNN(48, 8)

    def forward(self, image, patch):
        image = self.back(image)
        image = image[2]  # n_batch,48,28,28

        patch = self.back(patch)
        patch = patch[2]  # n_batch,48,8,8

        image = self.cnn(image)  # n_batch,48,8,8

        Xk = image.clone()
        Xv = image.clone()
        Xq = patch.clone()

        return Xq, Xk, Xv


if __name__ == "__main__":
    i1 = torch.randn(1, 3, 224, 224)
    i2 = torch.randn(1, 3, 64, 64)

    b = Backbone()
    q, k, v = b(i1, i2)

    print(q.shape, k.shape, v.shape)

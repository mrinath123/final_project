import torch
import torch.nn as nn
import timm

class CNN(nn.Module):
    def __init__(self, in_channels, op_shape):
        super().__init__()
        self.inc = in_channels
        self.op = op_shape
        self.cnv1 = nn.Conv2d(self.inc, 48, 1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((self.op, self.op))
        self.cnv2 = nn.Conv2d(48, 48, 1)

    def forward(self, x):
        x = self.cnv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnv2(x)
        return x


class Backbone(nn.Module):
    def __init__(self, name , pretrained=False):
        super().__init__()

        if(name == "effnet"):
            self.back = timm.create_model('efficientnet_b3', pretrained=pretrained, features_only=True)
            self.cnn = CNN(48, 8)
        if(name == "mobilnet"):
            self.back = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
            self.cnn = CNN(32, 8)
        if(name == "densenet"):
            self.back = timm.create_model('densenet121', pretrained=pretrained, features_only=True)
            self.cnn = CNN(512, 8)
      

    def forward(self, image, patch):
        image = self.back(image)
        image = image[2]  # n_batch,nc,28,28

        patch = self.back(patch)
        patch = patch[2]  # n_batch,nc,8,8

        image = self.cnn(image)  # n_batch,48,8,8
        patch = self.cnn(patch)  # n_batch,48,8,8

        Xk = image.clone()
        Xv = image.clone()
        Xq = patch.clone()

        return Xq, Xk, Xv


if __name__ == "__main__":
    i1 = torch.randn(1, 3, 224, 224)
    i2 = torch.randn(1, 3, 64, 64)

    b = Backbone("mobilnet")
    q, k, v = b(i1, i2)

    print(q.shape, k.shape, v.shape)

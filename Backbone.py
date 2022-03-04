import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

#for image densenet
class CNN1(nn.Module):
    def __init__(self, in_channels = 1792, op_shape = 8):
        super().__init__()
        self.inc = in_channels
        self.op = op_shape
        self.cnv1 = nn.Conv2d(self.inc, 48, 2)
        self.relu = nn.PReLU()
        self.pool = nn.AdaptiveAvgPool2d((self.op, self.op))
        self.cnv2 = nn.Conv2d(48, 48, 1)
        self.norm = nn.BatchNorm2d(48)

    def forward(self, x):
        x = F.relu(self.cnv1(x))
        x = self.pool(x)
        x = self.cnv2(x)
        x = self.relu(x)
        x = self.norm(x)
        return x

#for patch effnet
class CNN2(nn.Module):
    def __init__(self, in_channels = 48, op_shape = 8 ):
        super().__init__()
        self.inc = in_channels
        self.op = op_shape
        self.cnv1 = nn.Conv2d(self.inc, 48, 2)
        self.relu = nn.PReLU()
        self.pool = nn.AdaptiveAvgPool2d((self.op, self.op))
        self.cnv2 = nn.Conv2d(48, 48, 1)
        self.norm = nn.BatchNorm2d(48)

    def forward(self, x):
        x = F.relu(self.cnv1(x))
        x = self.pool(x)
        x = self.cnv2(x)
        x = self.relu(x)
        x = self.norm(x)
        return x


class Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

    
        self.back1 = timm.create_model('densenet201', pretrained=pretrained, features_only=True)
        self.cnn1 = CNN1()

        self.back2 = timm.create_model('efficientnet_b3', pretrained=pretrained, features_only=True)
        self.cnn2 = CNN2()

    def forward(self, image, patch):

        image = self.back1(image) # b x 1792 x 14 x 14 
        image = self.cnn1(image[3])

        patch = self.back2(patch) # b x 48 x 8 x 8 
        patch = self.cnn2(patch[2])

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

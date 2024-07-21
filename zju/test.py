import torch
from torch import nn

if __name__ == '__main__':
    x = torch.randn(4, 3, 10, 10)
    conv = nn.Conv2d(3, 10, 5, padding=0)
    print(conv.weight.shape)
    y = conv(x)
    print(y.shape)

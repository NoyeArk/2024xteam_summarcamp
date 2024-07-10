import torch
from torch import nn
import torch.nn.functional as F


class AFT_Full(nn.Module):
    def __init__(self, max_len=10000, dim=128, **kwargs):
        super().__init__()
        self.dim = dim
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w = nn.Parameter(torch.Tensor(max_len, max_len))
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        w_bias = self.w[:H, :W].unsqueeze(0)

        num = torch.exp(w_bias) @ (torch.exp(k) * v)
        den = torch.exp(w_bias) @ torch.exp(k)
        y = F.sigmoid(q) * num / den

        return self.out(y)


if __name__ == '__main__':
    model = AFT_Full(100, 64)
    test_x = torch.randn(256, 56, 56, 64)
    test_y = model(test_x)
    print(test_y)


import torch
from torch import nn


class AFT_Simple(nn.Module):
    def __init__(self, dim, hidden_dim=64, **kwargs):
        super().__init__()
        self.w_q = nn.Linear(dim, hidden_dim)
        self.w_k = nn.Linear(dim, hidden_dim)
        self.w_v = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        max_k = k.max(dim=0, keepdims=True)[0]
        exp_k = torch.exp(k - max_k)

        num = exp_k * v
        den = exp_k

        y = torch.sigmoid(q) * num / den
        return self.out(y)


if __name__ == '__main__':
    model = AFT_Simple(3)
    test_x = torch.randn(1, 1, 1, 3)
    test_y = model(test_x)
    print(test_y)

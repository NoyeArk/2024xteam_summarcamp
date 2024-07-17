import torch
from torch import nn


class AFT_Full(nn.Module):
    def __init__(self, max_len, dim, hidden_dim=64, **kwargs):
        super().__init__()
        self.dim = dim
        self.w_q = nn.Linear(dim, hidden_dim)
        self.w_k = nn.Linear(dim, hidden_dim)
        self.w_v = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.w = nn.Parameter(torch.Tensor(max_len, max_len))

    def forward(self, x):
        B, H, W, C = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        w_bias = self.w[:H, :W].unsqueeze(0)

        max_k = k.max(dim=0, keepdims=True)[0]
        max_w_bias = w_bias.max(dim=0, keepdims=True)[0]

        exp_k = torch.exp(k - max_k)
        exp_w_bias = torch.exp(w_bias - max_w_bias)
        print('exp_w_bias.shape:', exp_w_bias.shape)
        print('(exp_k * v).shape:', (exp_k * v).shape)

        num = exp_w_bias @ (exp_k * v)
        print('num.shape:', num.shape)
        den = exp_w_bias @ exp_k
        y = torch.sigmoid(q) * num / den

        return self.out(y)


if __name__ == '__main__':
    model = AFT_Full(100, 64)
    test_x = torch.randn(256, 56, 56, 64)
    test_y = model(test_x)
    print(test_y.shape)


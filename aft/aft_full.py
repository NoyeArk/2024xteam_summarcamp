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
        # w的因式分解形式
        self.u = nn.Parameter(torch.Tensor(max_len, hidden_dim), requires_grad=True)
        self.v = nn.Parameter(torch.Tensor(max_len, hidden_dim), requires_grad=True)
        # 初始化u和v
        nn.init.normal_(self.u, 0, 0.01)
        nn.init.normal_(self.v, 0, 0.01)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        w = torch.matmul(self.u.transpose(0, 1), self.v)
        w_bias = w[:H * W, :H * W].unsqueeze(0)

        max_k = k.max(dim=0, keepdims=True)[0]
        max_w_bias = w_bias.max(dim=0, keepdims=True)[0]

        exp_k = torch.exp(k - max_k)
        exp_w_bias = torch.exp(w_bias - max_w_bias)

        num = exp_w_bias @ (exp_k * v)
        den = exp_w_bias @ exp_k
        y = torch.sigmoid(q) * num / den

        return self.out(y).view(B, H, W, C)


if __name__ == '__main__':
    dim, hidden_dim = 5, 128
    model = AFT_Full(100, dim, hidden_dim)
    test_x = torch.randn(2, 3, 3, dim)
    test_y = model(test_x)
    print(test_y.shape)

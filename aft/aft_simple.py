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
        x = x.reshape(B, -1, C)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        y = torch.sigmoid(q) * (torch.softmax(k, dim=1) * v).sum(dim=1, keepdim=True)
        return self.out(y).view(B, H, W, C)


if __name__ == '__main__':
    dim, hidden_dim = 1, 2
    model = AFT_Simple(dim, hidden_dim)
    test_x = torch.randn(3, 1, 2, dim)
    test_y = model(test_x)
    print(test_y.shape)

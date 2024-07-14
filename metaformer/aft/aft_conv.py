import torch
from torch import nn


class AFT_Conv(nn.Module):
    def __init__(self, dim, hidden_dim=64, head_num=4, kernel_size=7, **kwargs):
        super().__init__()
        self.head_num = head_num
        self.dim = dim // head_num
        self.hidden_dim = hidden_dim

        self.w_q = nn.Linear(self.dim, hidden_dim)
        self.w_v = nn.Linear(self.dim, hidden_dim)
        self.w_k = nn.Linear(self.dim, hidden_dim)
        # self.w = nn.Parameter(torch.zeros(head_num, kernel_size), requires_grad=True)
        self.conv2d = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=3)
        # self.conv1d.weight.data = torch.exp(self.w) - 1  # 为什么要减1
        self.out = nn.Linear(hidden_dim, self.dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, self.head_num, H, W, -1)
        print('x.shape:', x.shape)

        q = self.w_q(x)
        v = self.w_v(x)
        k = self.w_k(x)

        print('q.shape:', q.shape)
        print('v.shape:', v.shape)
        print('k.shape:', k.shape)

        conv_result = self.conv2d((torch.exp(k) * v).view(-1, self.hidden_dim, H, W))
        print('conv_result.shape:', conv_result.shape)
        conv_result = conv_result.view(B, self.head_num, H, W, -1)

        num = conv_result + torch.mul(torch.exp(k), v)
        den = conv_result + torch.exp(k)
        y = torch.sigmoid(q) * num / den
        t = self.out(y)
        print('t.shape:', t.shape)
        t = t.view(B, H, W, -1)
        print('t.shape:', t.shape)
        return self.out(y).view(B, H, W, -1)


if __name__ == '__main__':
    # w = 2p + 1
    dim = 48
    model = AFT_Conv(dim)
    test_x = torch.randn(20, 32, 32, dim)
    test_y = model(test_x)
    print(test_y.shape)

# 32 - 4 + 4 + 1 = 29
#

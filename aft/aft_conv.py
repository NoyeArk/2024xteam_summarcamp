import torch
from torch import nn


def attention(q, k, v, conv):
    max_k = k.max(dim=0, keepdims=True)[0]
    exp_k = torch.exp(k - max_k)

    num = conv(exp_k * v) + exp_k.sum(dim=1, keepdim=True) * v
    den = conv(exp_k) + exp_k.sum(dim=1, keepdim=True)

    y = torch.sigmoid(q) * num / den
    return y


class AFT_Conv(nn.Module):
    def __init__(self, dim, hidden_dim=64, head_num=20, kernel_size=7, **kwargs):
        super().__init__()
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // head_num

        self.w_q = nn.Linear(dim, hidden_dim)
        self.w_v = nn.Linear(dim, hidden_dim)
        self.w_k = nn.Linear(dim, hidden_dim)
        self.kernels = [
            nn.Parameter(torch.Tensor(self.head_dim, self.head_dim), requires_grad=True)
            for _ in range(head_num)
        ]
        self.conv2ds = nn.ModuleList([
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size, padding=3)
            for _ in range(head_num)
        ])
        for i in range(head_num):
            self.conv2ds[i].weight.data = torch.exp(self.kernels[i]) - 1
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        assert C % self.head_num == 0

        q = self.w_q(x).view(B, self.head_num, -1, H, W)
        v = self.w_v(x).view(B, self.head_num, -1, H, W)
        k = self.w_k(x).view(B, self.head_num, -1, H, W)

        q_s = [q[:, i, :self.hidden_dim // self.head_num, :, :].contiguous() for i in range(self.head_num)]
        k_s = [k[:, i, :self.hidden_dim // self.head_num, :, :].contiguous() for i in range(self.head_num)]
        v_s = [v[:, i, :self.hidden_dim // self.head_num, :, :].contiguous() for i in range(self.head_num)]

        attentions = [attention(q_, k_, v_, conv) for conv, q_, k_, v_ in zip(self.conv2ds, q_s, k_s, v_s)]

        y = torch.cat(attentions, dim=1).view(B, H, W, -1)
        return self.out(y)


if __name__ == '__main__':
    # w = 2p + 1
    dim = 48
    model = AFT_Conv(dim)
    test_x = torch.randn(20, 32, 32, dim)
    test_y = model(test_x)
    print(test_y.shape)

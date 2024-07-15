import torch
from torch import nn


class AFT_Conv(nn.Module):
    def __init__(self, dim, hidden_dim=64, head_num=4, kernel_size=3, **kwargs):
        super().__init__()
        self.head_num = head_num
        self.hidden_dim = hidden_dim

        self.w_q = nn.Linear(dim, hidden_dim)
        self.w_v = nn.Linear(dim, hidden_dim)
        self.w_k = nn.Linear(dim, hidden_dim)
        self.conv2ds = nn.ModuleList([
            nn.Conv2d(hidden_dim // head_num, hidden_dim // head_num, kernel_size, padding=1)
            for _ in range(head_num)
        ])
        self.out = nn.Linear(hidden_dim, dim)

    def attention(self, q, k, v, conv):
        max_k = k.max(dim=0, keepdims=True)[0]
        exp_k = torch.exp(k - max_k)

        num = conv(exp_k * v) + exp_k * v
        den = conv(exp_k) + exp_k

        y = torch.sigmoid(q) * num / den
        return y

    def forward(self, x):
        B, H, W, C = x.shape
        assert C % self.head_num == 0

        q = self.w_q(x).view(B, self.head_num, -1, H, W)
        v = self.w_v(x).view(B, self.head_num, -1, H, W)
        k = self.w_k(x).view(B, self.head_num, -1, H, W)

        q_s = [q[:, i, :self.hidden_dim // self.head_num, :, :].contiguous() for i in range(self.head_num)]
        k_s = [k[:, i, :self.hidden_dim // self.head_num, :, :].contiguous() for i in range(self.head_num)]
        v_s = [v[:, i, :self.hidden_dim // self.head_num, :, :].contiguous() for i in range(self.head_num)]

        attentions = [self.attention(q_, k_, v_, conv) for conv, q_, k_, v_ in zip(self.conv2ds, q_s, k_s, v_s)]

        y = torch.cat(attentions, dim=1).view(B, H, W, -1)
        return self.out(y)


if __name__ == '__main__':
    # w = 2p + 1
    dim = 48
    model = AFT_Conv(dim)
    test_x = torch.randn(20, 32, 32, dim)
    test_y = model(test_x)
    print(test_y.shape)

# 32 - 4 + 4 + 1 = 29
#

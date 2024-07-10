import torch
from torch import nn
import torch.nn.functional as F


class AFT_Conv(nn.Module):
    def __init__(self, d_model, head_num, window_size):
        super().__init__()
        self.head_num = head_num
        self.dim = d_model // head_num
        self.window_size = window_size

        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, head_num)
        self.w = nn.Parameter(torch.zeros(head_num, head_num, window_size), requires_grad=True)
        self.conv1d = nn.Conv1d(head_num, head_num, window_size)
        self.conv1d.weight.data = torch.exp(self.w) - 1  # 为什么要减1
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.head_num, -1)
        v = self.w_v(x).view(batch_size, seq_len, self.head_num, -1)
        k = self.w_k(x).unsqueeze(-1)

        print('q.shape:', q.shape)
        print('v.shape:', v.shape)
        print('k.shape:', k.shape)

        conv_result = self.conv1d((torch.exp(k) * v).view(-1, self.head_num, self.dim))
        conv_result = conv_result.view(batch_size, seq_len, self.head_num, -1)

        num = conv_result + torch.mul(torch.exp(k), v)
        den = conv_result + torch.exp(k)
        y = F.sigmoid(q) * num / den
        y = y.view(batch_size, seq_len, -1)

        return self.out(y)


if __name__ == '__main__':
    model = AFT_Conv(128, 2, 1)
    test_x = torch.randn(20, 100, 128)
    test_y = model(test_x)
    print(test_y)
    print(test_y.shape)

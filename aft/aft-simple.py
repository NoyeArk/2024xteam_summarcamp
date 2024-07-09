import torch
from torch import nn
import torch.nn.functional as F


class AFT_Simple(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        t = F.softmax(k, dim=1) * v
        y = F.sigmoid(q) * (F.softmax(k, dim=1) * v)

        return self.out(y)


if __name__ == '__main__':
    model = AFT_Simple(1)
    x = torch.tensor([[[23.], [49.]]])
    y = model(x)
    print(y)


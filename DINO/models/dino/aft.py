import torch
from torch import nn
import torch.nn.functional as F


class AFT_Simple(nn.Module):
    def __init__(self, dim, hidden_dim=64, **kwargs):
        super().__init__()
        self.dim = dim
        self.w_q = nn.Linear(dim, hidden_dim)
        self.w_k = nn.Linear(dim, hidden_dim)
        self.w_v = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = x.tensors
        print(x.shape)
        B, H, W, C = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        num = torch.exp(k) * v
        den = torch.exp(k)

        y = F.sigmoid(q) * num / den
        return self.out(y)


def build_aft_simple(model_name):
    assert model_name in ['aft_full', 'aft_simple']
    model_para_dict = {
        'aft_full': dict(
            dim=817
        )
    }
    model = AFT_Simple(model_para_dict['aft_full']['dim'])
    print('aft_simple初始化成功')
    return model

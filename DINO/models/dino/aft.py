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
        print('x.shape:', x.shape)
        print('x:', x)
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        t = F.softmax(k, dim=1) * v
        y = F.sigmoid(q) * (F.softmax(k, dim=1) * v)

        return self.out(y)


def build_aft_simple(model_name):
    assert model_name in ['aft_full', 'aft_simple']
    model_para_dict = {
        'aft_full': dict(
            d_model=32
        )
    }
    model = AFT_Simple(model_para_dict['aft_full']['d_model'])
    print('aft_simple初始化成功')
    return model

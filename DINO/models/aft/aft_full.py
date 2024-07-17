import torch
from torch import nn
import torch.nn.functional as F
from DINO.util.misc import NestedTensor


class AFT_Full(nn.Module):
    def __init__(self, max_len=1000, dim=3, hidden_dim=64, **kwargs):
        super().__init__()
        self.dim = dim
        self.w_q = nn.Linear(dim, hidden_dim)
        self.w_k = nn.Linear(dim, hidden_dim)
        self.w_v = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.w = nn.Parameter(torch.Tensor(max_len, max_len))

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, H, W, C)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        w_bias = self.w[:H, :W].unsqueeze(0)

        max_k = k.max(dim=0, keepdims=True)[0]
        max_w_bias = w_bias.max(dim=0, keepdims=True)[0]

        exp_k = torch.exp(k - max_k)
        exp_w_bias = torch.exp(w_bias - max_w_bias)

        num = exp_w_bias @ (exp_k * v)
        den = exp_w_bias @ exp_k
        y = torch.sigmoid(q) * num / den

        return self.out(y).view(B, C, H, W)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        outs = self.forward_features(x)

        outs_dict = {}
        for idx, out_i in enumerate(outs):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(out_i, mask)

        return outs_dict


def build_aft_full(model_name):
    model = AFT_Full()
    print('aft_full初始化成功')
    return model


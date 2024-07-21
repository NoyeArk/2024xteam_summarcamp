import torch
from torch import nn
import torch.nn.functional as F
from DINO.util.misc import NestedTensor


class AFT_Simple(nn.Module):
    def __init__(self, dim=3, hidden_dim=64, **kwargs):
        super().__init__()
        self.dim = dim
        self.w_q = nn.Linear(dim, hidden_dim)
        self.w_k = nn.Linear(dim, hidden_dim)
        self.w_v = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, -1, C)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        y = torch.sigmoid(q) * (torch.softmax(k, dim=1) * v).sum(dim=1, keepdim=True)
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


def build_aft_simple(model_name):
    model = AFT_Simple()
    print('aft_simple初始化成功')
    return model

import torch
from torch import nn
import torch.nn.functional as F
from DINO.util.misc import NestedTensor


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class AFT_Simple(nn.Module):
    def __init__(self, dim=3, hidden_dim=64, **kwargs):
        super().__init__()
        self.dim = dim
        self.w_q = nn.Linear(dim, hidden_dim)
        self.w_k = nn.Linear(dim, hidden_dim)
        self.w_v = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)

        self.downsample_layers = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=4, stride=4),
            LayerNorm(3, eps=1e-6, data_format="channels_first")
        )

    def forward_features(self, x):
        outs = []
        B, C, H, W = x.shape
        x = x.reshape(B, -1, C)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        y = torch.sigmoid(q) * (torch.softmax(k, dim=1) * v).sum(dim=1, keepdim=True)
        y = self.out(y).view(B, C, H, W)

        outs.append(self.downsample_layers(y))
        return tuple(outs)

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

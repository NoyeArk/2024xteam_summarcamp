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


class AFT_Full(nn.Module):
    def __init__(self, dim=3, hidden_dim=128, max_len=10000, **kwargs):
        super().__init__()
        self.dim = dim
        self.w_q = nn.Linear(dim, hidden_dim)
        self.w_k = nn.Linear(dim, hidden_dim)
        self.w_v = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        # w的因式分解形式
        self.u = nn.Parameter(torch.Tensor(max_len, hidden_dim), requires_grad=True)
        self.v = nn.Parameter(torch.Tensor(max_len, hidden_dim), requires_grad=True)
        # 初始化u和v
        nn.init.normal_(self.u, 0, 0.01)
        nn.init.normal_(self.v, 0, 0.01)

        self.downsample_layers = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=4, stride=4),
            LayerNorm(3, eps=1e-6, data_format="channels_first")
        )

    def forward_features(self, x):
        outs = []
        B, C, H, W = x.shape
        x = x.reshape(B, H, W, C)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        w = torch.matmul(self.u, self.v.transpose(0, 1))
        w_bias = w[:H * W, :H * W].unsqueeze(0)

        max_k = k.max(dim=0, keepdims=True)[0]
        max_w_bias = w_bias.max(dim=0, keepdims=True)[0]

        exp_k = torch.exp(k - max_k)
        exp_w_bias = torch.exp(w_bias - max_w_bias)

        num = exp_w_bias @ (exp_k * v)
        den = exp_w_bias @ exp_k
        y = torch.sigmoid(q) * num / den
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


def build_aft_full(model_name):
    model = AFT_Full()
    print('aft_full初始化成功')
    return model


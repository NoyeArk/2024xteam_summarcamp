import torch
from torch import nn
import torch.nn.functional as F
from DINO.util.misc import NestedTensor


class AFT_Conv(nn.Module):
    def __init__(self, dim=3, hidden_dim=64, head_num=1, kernel_size=7, **kwargs):
        super().__init__()
        self.head_num = head_num
        self.dim = dim // head_num
        self.hidden_dim = hidden_dim

        self.w_q = nn.Linear(self.dim, hidden_dim)
        self.w_v = nn.Linear(self.dim, hidden_dim)
        self.w_k = nn.Linear(self.dim, hidden_dim)
        self.conv2d = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=3)
        self.out = nn.Linear(hidden_dim, self.dim)

    def forward_features(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, self.head_num, H, W, -1)

        q = self.w_q(x)
        v = self.w_v(x)
        k = self.w_k(x)

        max_k = k.max(dim=0, keepdims=True)[0]
        exp_k = torch.exp(k - max_k)

        conv_result = self.conv2d((exp_k * v).view(-1, self.hidden_dim, H, W))
        conv_result = conv_result.view(B, self.head_num, H, W, -1)

        num = conv_result + torch.mul(exp_k, v)
        den = conv_result + exp_k
        y = torch.sigmoid(q) * num / den

        return self.out(y).view(B, H, W, -1)

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


def build_aft_conv(model_name):
    model = AFT_Conv()
    print('aft_conv初始化成功')
    return model


if __name__ == '__main__':
    # w = 2p + 1
    dim = 48
    model = AFT_Conv(dim)
    test_x = torch.randn(20, 32, 32, dim)
    test_y = model(test_x)
    print(test_y.shape)

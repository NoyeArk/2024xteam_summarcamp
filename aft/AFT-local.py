import torch
from torch import nn
from typing import Optional
from labml_helpers.module import Module


class AFTLocal(Module):
    def __init__(self, d_model, seq_len, local_window_size, bias):
        super().__init__()
        self.local_window_size = local_window_size
        # Q K V
        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)
        # 成对位置偏差
        self.pos_bias = nn.Parameter(torch.zeros(seq_len, seq_len), requires_grad=True)
        # 掩码
        self.w_mask = nn.Parameter(self.create_local_mask(seq_len, local_window_size), requires_grad=False)
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(d_model, d_model)

    @staticmethod
    def create_local_mask(seq_len, local_window_size):
        """
        创建局部掩码
        :param seq_len:
        :param local_window_size:
        :return:
        """
        # 初始化为1
        local_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        # 将 [t+s,+∞] 设置为0
        local_mask = torch.tril(local_mask, local_window_size - 1)
        # 将 [-∞,t + s] 设置为0
        local_mask = torch.tril(local_mask, -(local_window_size - 1))
        return local_mask

    def forward(self,
                query: torch.Tensor,
                key:   torch.Tensor,
                value: torch.Tensor,
                mask:  Optional[torch.Tensor] = None):
        seq_len, _, _ = query.shape

        if mask is not None:
            assert mask.shape[0] == 1 or mask.shape[0] == query.shape[0]
            assert mask.shape[1] == key.shape[0]
            assert mask.shape[2] == 1 or mask.shape[2] == query.shape[1]

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        pos_bias = self.pos_bias[:seq_len, :seq_len] * self.local_mask[:seq_len, :seq_len]
        pos_bias = pos_bias.unsqueeze(-1)
        pos_bias.masked_fill_(~mask, float('-inf'))

        max_key = key.max(dim=0, keepdims=True)[0]
        max_pos_bias = pos_bias.max(dim=1, keepdims=True)[0]

        exp_key = torch.exp(key - max_key)
        exp_pos_bias = torch.exp(pos_bias - max_pos_bias)

        num = torch.einsum('ijb,jbd->ibd', exp_pos_bias, exp_key * value)
        den = torch.einsum('ijb,jbd->ibd', exp_pos_bias, exp_key)

        y = self.activation(query) * num / den

        return self.output(y)


def _test_local_mask():
    from labml.logger import inspect
    inspect(AFTLocal.create_local_mask(10, 4))


if __name__ == '__main__':
    _test_local_mask()

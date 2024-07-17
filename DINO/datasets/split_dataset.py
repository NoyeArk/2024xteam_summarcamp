import torch.nn as nn
import torch

num_channels = 256
num_groups = 32
group_norm = nn.GroupNorm(num_groups, num_channels)

# 使用示例
input_tensor = torch.randn(1, num_channels, 736, 981)  # 假设的输入张量
output_tensor = group_norm(input_tensor)
print(output_tensor.shape)

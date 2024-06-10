import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个形状为 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加一个维度，使得形状变为 (1, max_len, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加到输入的嵌入上
        x = x + self.pe[:x.size(0), :]
        return x

# 测试代码
d_model = 512  # 嵌入维度
max_len = 60  # 序列最大长度
batch_size = 32  # 批次大小
seq_len = 50  # 当前序列长度

# 创建一个随机的嵌入输入 (seq_len, batch_size, d_model)
input_embeddings = torch.randn(seq_len, batch_size, d_model)

# 创建位置编码模块
pos_encoding = PositionalEncoding(d_model, max_len)

# 添加位置编码
output = pos_encoding(input_embeddings)
print(output.shape)  # 输出的形状应为 (seq_len, batch_size, d_model)

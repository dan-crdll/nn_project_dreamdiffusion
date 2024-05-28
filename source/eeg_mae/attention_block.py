"""
Multihead self attention block
"""
import torch
from torch import nn
import torch.nn.functional as F


class HSA(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

        self.q_linear = nn.Linear(head_dim, head_dim)
        self.k_linear = nn.Linear(head_dim, head_dim)
        self.v_linear = nn.Linear(head_dim, head_dim)

        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)
        return attention_output


class MHSA(nn.Module):
    def __init__(self, head_num, embed_dim):
        super().__init__()
        assert embed_dim % head_num == 0, f'Cannot divide {embed_dim} into {head_num} heads'
        self.head_dim = embed_dim // head_num
        self.head_num = head_num

        self.heads = nn.ModuleList(HSA(self.head_dim) for _ in range(head_num))
        self.linear = nn.Linear(embed_dim, embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()

        x = x.view(batch_size, seq_length, self.head_num, self.head_dim).transpose(1, 2)
        x = torch.cat([head(x[:, i]) for i, head in enumerate(self.heads)], dim=-1)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)

        out = self.linear(x)
        return out


class Block(nn.Module):
    def __init__(self, embed_dim, head_num):
        super().__init__()

        self.norm0 = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.mhsa = MHSA(head_num, embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        if isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)

    def forward(self, x):
        x = self.norm0(x)
        x = x + self.mhsa(x)
        x = self.norm1(x)
        x = x + F.relu(self.linear(x))
        out = self.norm2(x)
        return out

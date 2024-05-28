import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, embed_dim, seed=0):
        super(MLP, self).__init__()
        torch.manual_seed(seed)
        
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        _x = self.linear(x)
        out=  self.relu(_x) + x 
        
        out = self.norm(out)
        return out
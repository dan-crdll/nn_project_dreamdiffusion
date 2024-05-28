import torch
from torch import nn
import torch.nn.functional as F

class MHABlock(nn.Module):
    def __init__(self, embed_dim, n_heads, seed=0):
        super(MHABlock, self).__init__()
        
        torch.manual_seed(seed)
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True)
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        :param x: [bsz, n_tokens, embed_dim]
        """
        if isinstance(x, torch.Tensor):
            x1 = x 
            x2 = x
        else: 
            x1, x2 = x
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        
        q = self.q_linear(x1)
        k = self.k_linear(x1)
        v = self.v_linear(x2)
        
        out, _ = self.mha(q, k, v)
        
        out = self.norm(out + x2)
        return out
        
    
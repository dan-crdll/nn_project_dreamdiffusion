from torch import nn
from torch import flatten

class Projector(nn.Module):
    def __init__(self, embed_dimension, destination_size, token_num=31):
        super(Projector, self).__init__()
        
        if isinstance(destination_size, tuple):
            linear_units = 1 
            for d in destination_size:
                linear_units *= d
        else: 
            linear_units = destination_size
            destination_size = (destination_size, )
        
        self.linear_projection = nn.Linear(embed_dimension * token_num, linear_units, bias=False)
        self.reshaper = nn.Unflatten(-1, destination_size)
                
    def forward(self, x):
        x = flatten(x, start_dim=1)
        x = self.linear_projection(x)
        x = self.reshaper(x)
        return x
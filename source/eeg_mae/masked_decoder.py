"""
This autoencoder is an adaptation of the MAE for images proposed in 'Masked Autoencoders Are Scalable Vision Learners'
by Kaiming et al [1].
"""
import torch
from torch import nn
from source.eeg_mae.attention_block import Block



class MaskedDecoder(nn.Module):
    def __init__(self, time_dim, token_num, channels,
                 embed_dim, decoder_depth, decoder_heads, seed=0):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.time_dim = time_dim    # eeg input signal length
        self.token_num = token_num  # number of tokens to divide the eeg input signal into
        self.token_dim = int(time_dim / token_num)  # temporal length of each token
        self.embed_dim = embed_dim  # embedding dimension
        self.channels = channels

        # positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, token_num, embed_dim), requires_grad=False)
        # masked token that needs to be predicted
        self.masked_token = nn.Parameter(torch.zeros((1, 1, embed_dim)), requires_grad=True)
        # decoder transformer
        self.decoder_blocks = nn.ModuleList([
                Block(embed_dim, decoder_heads)
                for _ in range(decoder_depth)
            ])
        # from embeddings to token size
        self.decoder_embed = nn.Linear(embed_dim, self.token_dim * channels)
        self.unflattener = nn.Unflatten(-1, (self.token_dim, self.channels))

        self.initialize_weights()

    def initialize_weights(self):
        self._get_sin_pos()

        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def _get_sin_pos(self):
        for i in range(self.token_num):
            for j in range(self.embed_dim):
                self.pos_embed[:, i, j] = torch.sin_(torch.Tensor([i * j]))

    def _detokenize(self, x):
        """
        Detokenize token tensor to retrieve initial eeg shape tensor
        :param x: token tensor
        :return: detokenized tensor with initial dimensions
        """
        unflattened = []

        for token in range(self.token_num):
            unflat = self.unflattener(x[:, token, :].squeeze(dim=1))
            unflattened.append(unflat)

        x = torch.cat(unflattened, dim=1)
        return x

    

    def forward(self, x_, restore_indexes):        
        masked_tokens = self.masked_token.repeat(x_.shape[0], restore_indexes.shape[1] + 1 - x_.shape[1], 1)

        x_reconstructed = torch.cat([x_, masked_tokens], dim=1)
        x_reconstructed = torch.gather(x_reconstructed, dim=1, index=restore_indexes.unsqueeze(-1).repeat(1, 1, x_.shape[2]))
        x_ = x_reconstructed + self.pos_embed

        for db in self.decoder_blocks:
            x_ = db(x_)

        x_ = self.decoder_embed(x_)
        x_ = self._detokenize(x_)
        return x_







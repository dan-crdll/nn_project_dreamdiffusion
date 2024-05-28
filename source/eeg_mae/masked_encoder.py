import torch
from torch import nn
from transformers import PreTrainedModel
from source.eeg_mae.encoder_config import EncoderConfig
from source.eeg_mae.attention_block import Block
from source.eeg_mae.attention_block import Block

"""
Masked encoder class for EEG signals.
"""

class MaskedEncoder(PreTrainedModel):
    config_class = EncoderConfig    # defining the config class (needed for huggingfacehub)

    def __init__(self, config: EncoderConfig, seed=0):
        super().__init__(config)
        
        torch.manual_seed(seed)
        
        self.config = config
        assert config.time_dim % config.token_num == 0, f'Cannot divide {config.time_dim} in {config.token_num} tokens'
        
        self.time_dim = config.time_dim
        self.token_num = config.token_num
        self.token_dim = int(config.time_dim / config.token_num)
        self.embed_dim = config.embed_dim

        # convolution as described in DreamDiffusion paper
        self.lin_embed = nn.Conv1d(
                in_channels=config.channels, 
                out_channels=self.embed_dim,
                kernel_size=self.token_dim, stride=self.token_dim
            )
        
        # positional embedding needed for transformer architecture
        self.pos_embed = nn.Parameter(torch.zeros(1, config.token_num, config.embed_dim), requires_grad=False)

        self.mask_perc = config.mask_perc

        # MHSA Blocks
        self.encoder_blocks = nn.ModuleList(
            [
                Block(self.embed_dim, config.encoder_heads)
                for _ in range(config.encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def initialize_weights(self):
        self._get_sin_pos()

        # calls _init_weights() on each layer of the network
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            # weights initialized from a random uniform distribution
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    # positional embeddings through different values of sin
    def _get_sin_pos(self):
        for i in range(self.token_num):
            for j in range(self.embed_dim):
                self.pos_embed[:, i, j] = torch.sin_(torch.Tensor([i * j]))
         
    # tokenize eeg embeddings       
    def tokenize(self, x):
        """
        Tokenize the eeg signal on time dimension.
        :param x: batch (bsz, time_dim, channels)
        :return: tokenized batch
        """
        # tokenization happens on time (1) dimension
        tokens = []
        
        for i in range(self.token_num):
            token = x[:, i * self.token_dim: (i + 1) * self.token_dim]  # bsz, token_dim, embed_dim
            tokens.append(token.unsqueeze(dim=1))
        tokens = torch.cat(tokens, dim=1)
        return tokens

    # mask embeddings and shuffle them
    def _mask(self, x):
        """
        Shuffle and mask a certain percentage of token tensor
        :param x: sequence of tokens to mask
        :return: shuffled masked sequence, indexes to restore positions and mask
        """
        BSZ, N_TOKENS, EMBED_DIM = x.shape
        to_keep = int((1 - self.mask_perc) * N_TOKENS)

        # The idea is to shuffle the tokens, then keep only the first to_keep
        # I need to maintain index of precedent positions for decoder to use
        # to shuffle I implemented a version of random noise argsort shuffle
        # refer to : https://github.com/facebookresearch/mae/blob/main/models_mae.py#L74
        noise = torch.randn((BSZ, N_TOKENS), device=x.device)
        shuffle_indexes = torch.argsort(noise, dim=1)
        keep_indexes = shuffle_indexes[:, :to_keep]
        
        x_masked = torch.gather(
                x, 
                index=keep_indexes.unsqueeze(-1).repeat(1, 1, EMBED_DIM),
                dim=1
            )

        restore_indexes = torch.argsort(shuffle_indexes, dim=1)

        mask = torch.ones([BSZ, N_TOKENS], device=x.device)
        mask[:, :to_keep] = 0
        mask = torch.gather(mask, dim=1, index=restore_indexes)

        return x_masked, restore_indexes, mask

    def forward(self, x):
        x_ = torch.permute(x, (0, 2, 1))
        x_ = self.lin_embed(x_)
        
        embeddings = torch.nn.functional.leaky_relu(x_)
        embeddings = embeddings.permute(0, 2, 1) + self.pos_embed # BSZ, N_TOKENS, EMBED_DIM

        x_masked, restore_indexes, mask = self._mask(embeddings)
        
        # Pass through the transformer blocks (MHSA)
        for b in self.encoder_blocks:
            x_masked = b(x_masked)
        x_masked = self.norm(x_masked)
        
        return x_masked, restore_indexes, mask
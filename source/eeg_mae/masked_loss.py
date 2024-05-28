from torch import nn
import torch

class MaskedLoss(nn.Module):
    def __init__(self, tokenizer):
        super(MaskedLoss, self).__init__()
        self.tokenizer = tokenizer
        
    def _compute_loss(self, pred, true, mask):
        """
        :param pred: predicted sequence
        :param true: true sequence
        :param mask: 1 - 0 mask
        :return: computed loss only on unmasked tokens as stated in [1]
        """
        difference = (pred - true) ** 2 # compute the difference signal
        tokens_difference = self.tokenizer(difference) # tokenize the difference tensor
        tokens_difference = torch.sum(tokens_difference, dim=-1) # Sum error on different channels
        loss = torch.mean(tokens_difference, dim=-1) # mean the error in the token temporal dimension

        loss = (loss * mask).sum() / mask.sum()  # compute the mean loss only for masked tokens
        return loss 
    
    def forward(self, pred, true, mask):
        loss = self._compute_loss(pred, true, mask)
        return loss
    
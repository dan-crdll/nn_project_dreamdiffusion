import sys
sys.path.append('../')

import torch
from source.eeg_mae.masked_decoder import MaskedDecoder
from source.eeg_mae.masked_encoder import MaskedEncoder
from source.eeg_mae.masked_loss import MaskedLoss
from torch import optim
import lightning as L


class EegAutoEncoder(L.LightningModule):
    def __init__(self, encoder:MaskedEncoder, decoder:MaskedDecoder, learning_rate):
        super().__init__()
        torch.random.manual_seed(0)
        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = learning_rate
        self.loss_fn = MaskedLoss(self.encoder.tokenize)
        
    def training_step(self, batch):
        z, restore_indexes, mask = self.encoder(batch)
        pred = self.decoder(z, restore_indexes)
        loss = self.loss_fn(pred, batch, mask)  
        
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch):
        z, restore_indexes, mask = self.encoder(batch)
        pred = self.decoder(z, restore_indexes)
        loss = self.loss_fn(pred, batch, mask)
        self.log("test_loss", loss)
        return loss
    
    def validation_step(self, batch):
        z, restore_indexes, mask = self.encoder(batch)
        pred = self.decoder(z, restore_indexes)
        loss = self.loss_fn(pred, batch, mask)
        self.log("epoch_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=(self.learning_rate or self.lr), betas=(0.9, 0.95))
        return optimizer
    
    def forward(self, batch):
        x, idx, _ = self.encoder(batch)
        pred = self.decoder(x, idx)
        return pred
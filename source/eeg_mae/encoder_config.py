from transformers import PretrainedConfig

"""
This class is used to contain the pretrained config for the EEG masked encoder
in order to be able to upload it on to hugging face hub.
"""

class EncoderConfig(PretrainedConfig):
    def __init__(
            self, 
            model_type = 'eeg encoder', 
            time_dim=500, 
            token_num=125, 
            channels=128,
            embed_dim=256, 
            mask_perc=0.75, 
            encoder_depth=5,
            encoder_heads=8, 
            **kwargs
        ):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.time_dim = time_dim
        self.token_num = token_num
        self.channels = channels
        self.embed_dim = embed_dim
        self.mask_perc = mask_perc
        self.encoder_depth = encoder_depth
        self.encoder_heads = encoder_heads
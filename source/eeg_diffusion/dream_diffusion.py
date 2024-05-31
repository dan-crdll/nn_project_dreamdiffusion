import lightning as L
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms as T
from tqdm.auto import tqdm


class DreamDiffusion(L.LightningModule):
    def __init__(self, unet, vae, scheduler, clip_model, encoder, tau, h, learning_rate=1e-5):
        super(DreamDiffusion, self).__init__()
        self.scheduler = scheduler
        self.learning_rate = learning_rate
        
        vae.requires_grad_(False)
        clip_model.requires_grad_(False)
        h.requires_grad_(False)
        
        self.list_cpu = [
            vae,
            clip_model,
            h,
            unet,
            tau,
            encoder
        ]
        self.crop = T.CenterCrop((224, 224))
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(list(self.list_cpu[3].parameters()) + list(self.list_cpu[4].parameters()) + list(self.list_cpu[5].parameters()),
                                lr=(self.learning_rate or self.lr), betas=(0.9, 0.95))
        return optimizer
        
    def training_step(self, batch):
        vae = self.list_cpu[0]
        clip_model = self.list_cpu[1]
        h = self.list_cpu[2]
        unet = self.list_cpu[3]
        tau = self.list_cpu[4]
        encoder = self.list_cpu[5]
        
        eeg, image = batch

        encoder.to(self.device)
        print('Allocated encoder')
        # Encode EEG signals and project to hidden states
        embeddings, l, m = encoder(eeg)
        
        encoder.to('cpu')
        print('Deallocated encoder')
        del l 
        del m 
        torch.cuda.empty_cache()
        
        tau.to(self.device)
        print('Allocated Tau')
        hidden_states = tau(embeddings)
        tau.to('cpu')
        print('Deallocated Tau')
        torch.cuda.empty_cache()
        
        # VAE latent space encoding with scaling
        vae.to(self.device)
        print('Allocated Vae')
        latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
        vae.to('cpu')
        print('Deallocated Vae')
        torch.cuda.empty_cache()
        
        # Generate noise and timesteps
        noise = torch.randn_like(latents, device=self.device)
        bsz = latents.size(0)
        timesteps = torch.randint(0, 1000, (bsz,), device=self.device, dtype=torch.long)

        # Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents.float(), noise.float(), timesteps)

        # UNet forward pass
        unet.to(self.device)
        print('Allocated Unet')
        pred = unet(noisy_latents, timesteps, hidden_states, return_dict=False)[0]
        unet.to('cpu')
        print('Deallocated Unet')
        torch.cuda.empty_cache()
        
        loss_unet = F.mse_loss(pred, noise, reduction='mean')

        # Image processing and CLIP encoding
        image = self.crop(image)
        
        clip_model.to(self.device)
        print('Allocated Clip')
        image_encodings = clip_model.encode_image(image)
        clip_model.to('cpu')
        print('Deallocated clip')
        torch.cuda.empty_cache()

        # Project hidden states and calculate CLIP loss
        h.to(self.device)
        print('Allocated h')
        projection = h(hidden_states)
        h.to('cpu')
        print('Deallocated h')
        torch.cuda.empty_cache()
        
        loss_clip = 1 - F.cosine_similarity(image_encodings, projection).mean()

        # Combined loss
        loss = loss_unet + loss_clip

        # Logging losses
        self.log_dict({
            'unet_loss_training': loss_unet,
            'clip_alignment_loss': loss_clip,
            'training_loss': loss
        })
        
        del loss_clip
        del loss_unet
        torch.cuda.empty_cache()

        return loss
    
    def forward(self, eeg, image_num):
        pass
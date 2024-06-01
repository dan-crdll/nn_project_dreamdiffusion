 ⚠️ Still a work in progress ️⚠️

### Re-implementation of the method proposed in ''DreamDiffusion: Generating High-Quality Images from Brain EEG Signals'' by Y. Bai, X. Wang et al.
*By Daniele Santino Cardullo | 2127806 | cardullo.2127806@studenti.uniroma1.it*

*original work: [DreamDiffusion (arXiv)](https://arxiv.org/abs/2306.16934)*

This work is part of the Neural Network Course Exam for academic year 2023 / 2024,
all the credits for the original work 
and publication go to the original authors.

#### Abstract
DreamDiffusion is a method for generating images directly from electroencephalogram signals. This is achieved by combinating different methodologies such as: self-supervised learning to learn meaningful and efficient latent representations for signals; latent diffusion generative model to generate high quality images; large language model to align signals embeddings with image-text ones.

#### Run The Code
To run the code create a virtual environment and install requirements, then take a look at `solution_description.ipynb`.

#### Directory Tree
<pre>
📦 nn_project_dreamdiffusion
├─ .gitignore
├─ README.md
├─ default_config.yaml
├─ requirements.txt
├─ solution_description.ipynb
├─ datasets
│  ├─ finetune_images/
│  ├─ finetune_dataset.pth
│  └─ pretrain_dataset.pth
├─ pretrained_models
│  ├─ pretrained_mae.ckpt
│  ├─ finetuned_eeg_encoder.pth
│  ├─ finetuned_unet.pth
│  ├─ finetuned_projector_tau.pth
│  └─ train_loss_mae.csv
└─ source
   ├─ datasets
   │  ├─ finetuning_dataset
   │  └─ pretraining_dataset.py
   ├─ eeg_diffusion
   │  ├─ dream_diffusion.
   │  └─ projector.py
   └─ eeg_mae
      ├─ attention_block.py
      ├─ eeg_autoencoder.py
      ├─ encoder_config.py
      ├─ masked_decoder.py
      ├─ masked_encoder.py
      └─ masked_loss.py
</pre>

#### Links
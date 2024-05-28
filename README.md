 ⚠️ Still a work in progress ️⚠️

### Re-implementation of the method proposed in ''DreamDiffusion: Generating High-Quality Images from Brain EEG Signals'' by Y. Bai, X. Wang et al.
*By Daniele Santino Cardullo | 2127806 | cardullo.2127806@studenti.uniroma1.it*

*original work: [DreamDiffusion (arXiv)](https://arxiv.org/abs/2306.16934)*

This work is part of the Neural Network Course Exam for academic year 2023 / 2024,
all the credits for the original work 
and publication go to the original authors.

###### Preliminary Steps
In order to be able to run the code with no issues some dependencies
are required. First of all is suggested to create a pip virtual
environment, after that it is possible to install all
packages required by running: `pip install -r requirements.txt`.

EEG - ImageNet dataset is also needed and can be downloaded from this
[link](https://github.com/perceivelab/eeg_visual_classification), 
the required file is `eeg_5_95_std.pth`, and place it in the root 
directory as shown in section ''Directory Tree''. We also need to download 
the set of ImageNet images used in the experiment, it can be 
downloaded by this [link](https://drive.google.com/file/d/1y7I9bG1zKYqBM94odcox_eQjnP9HGo9-/view?usp=drive_link),
then place it in the root directory and rename it to `images`.
###### Directory Tree
```
nn_project/
├── images/
├── models/
│   ├── dream_diffusion_pipeline.py
│   ├── eeg_masked_autoencoder.py
│   └── utils/
│       ├── eeg_image_dataset.py
│       ├── embedding_projector.py
│       ├── multi_head_self_attention.py
│       └── pretraining_dataset.py
├── eeg_5_95_std.pth
├── pretrain_eeg_mae.py
├── eeg_to_image.py
├── finetune_stablediff.py
├── README.md
├── requirements.txt
└── solution_description.ipynb
```

###### Instructions to run the code
1. ***EEG MAE PRETRAINING*** to pretrain the autoencoder is possible to run `python ./pretrain_eeg_mae.py <epoch number> <batch size>` a new file will be saved, named `pretrained_model.pth`. _You need to login to wandb_.
2. ***STABLEDIFFUSION FINETUNING AND CLIP ALIGNMENT*** to align the embeddings with clip and finetune stablediffusion on our dataset you need to run: `python ./finetune_stablediff.py <dataset path> <batch size> <pretrained embedder path>`
3. ***GENERATE IMAGES*** to generate images from an eeg file you have to run `python ./eeg_to_image <eeg path> <image number> <saving directory path>`

###### DreamDiffusion Pipeline
It is possible to use and import this implementation (along with the pretrained weights) of DreamDiffusion
using the hugging face diffusion pipeline using the repository: [osusume/finetuned-stable-dream-diffusion](https://huggingface.co/osusume/finetuned-stable-dream-diffusion/tree/main).
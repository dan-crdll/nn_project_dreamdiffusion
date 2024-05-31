from torch.utils.data import Dataset
import torch
from tqdm.auto import tqdm 
import cv2 as cv
import os


class EegImageDataset(Dataset):
    ds = None
    train_set = None 
    test_set = None
    
    def __init__(self, ds_folder='datasets/', subject=4, time_dim=500, train_perc=0.75, split='train'):
        super(EegImageDataset, self).__init__()
        if EegImageDataset.ds is None:
            ds = torch.load(f"{ds_folder}/finetune_dataset.pth")['dataset']
            
            subject_ds = []
            
            # Selecting only one subject
            for d in tqdm(ds):
                if d['subject'] == subject:
                    subject_ds.append(d)
            
            # Padding the signals to reach time dimension
            for d in tqdm(subject_ds):
                if d['eeg'].shape[1] >= time_dim:
                    d['eeg'] = d['eeg'][:, :time_dim]
                else:
                    res = time_dim - d['eeg'].shape[1]
                    pad = torch.zeros((128, res))
                    d['eeg'] = torch.hstack([d['eeg'], pad])
            
            # Getting label folders and pics indexes
            labels = os.listdir(f'{ds_folder}/finetune_images')
            pics = torch.zeros((len(labels), ))
            
            y = []
            x = []
            
            # creating x and y pairs
            for d in tqdm(subject_ds):
                x.append(d['eeg'].permute(1, 0))
                
                images = os.listdir(f"{ds_folder}/finetune_images/{labels[d['label']]}")
                im_ind = int(pics[d['label']].item())
 
                try:
                    y.append(f"{ds_folder}/finetune_images/{labels[d['label']]}/{images[im_ind]}")
                    pics[d['label']] += 1
                except IndexError:
                    y.append(f"{ds_folder}/finetune_images/{labels[d['label']]}/{images[im_ind - 1]}")
                
            ds_complete = []
            
            for i in tqdm(range(len(y))):
                im = cv.imread(y[i], cv.IMREAD_COLOR)
                if im is None or im.shape[-1] != 3:
                  print(f'Invalid Image: {y[i]}')
                  continue
                im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
                im = cv.resize(im, (512, 512))
                image = torch.Tensor(im) / 255.0
                image = torch.permute(image, (2, 0, 1))
                
                ds_complete.append(
                    {
                        'eeg': x[i],
                        'image': image
                    }
                )
            
            EegImageDataset.ds = ds_complete
            # splitting the dataset
            self.split(train_perc, ds_complete)
            
        self.ds = EegImageDataset.train_set if split=='train' else EegImageDataset.test_set
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, index):
        eeg = self.ds[index]['eeg']
        image = self.ds[index]['image']

        return eeg, image
        
    def split(self, train_perc, ds):
        train_len = int(len(ds) * train_perc)
        test_len = len(ds) - train_len
        
        noise = torch.randn((len(ds)))
        indexes = torch.argsort(noise)
        
        train_idx = indexes[:train_len]
        test_idx = indexes[test_len:]
        
        train_set = []
        test_set = []
        
        print('Generating Splits')
        for id in train_idx:
            train_set.append(ds[id])
        for id in test_idx:
            test_set.append(ds[id])
        
        EegImageDataset.train_set = train_set
        EegImageDataset.test_set = test_set
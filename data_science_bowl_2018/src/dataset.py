import pandas as pd

from skimage.io import imread

from utils import get_mask

import torch.nn as nn

class dataSet(nn.Module):
    def __init__(self, df, fold=0, train=True, transforms=None):
        self.transforms = transforms

        if train:
            self.df = df[df.kfold != fold].reset_index(drop=True)
        else:
            self.df = df[df.kfold == fold].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "img_"]
        mask_path = self.df.loc[idx, "mask_"]

        img = imread(img_path).astype('float32')
        img = img[:,:,:3]
        mask = get_mask(mask_path, img.shape[0], img.shape[1]).astype('float32')

        augmented = self.transforms(image = img, mask = mask)
        image = augmented['image']
        mask = augmented['mask']
        mask = mask[0].permute(2,0,1)

        return image,mask
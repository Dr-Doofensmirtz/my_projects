import pandas as pd
import os

from skimage.io import imread

from utils import get_mask, show_dataset
import config

import torch.nn as nn
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensor


class dataSet(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.img_list = os.listdir(self.path)[:200]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_folder = os.path.join(self.path, self.img_list[idx], "images/")
        mask_folder = os.path.join(self.path, self.img_list[idx], 'masks/')
        img_path = os.path.join(img_folder, os.listdir(img_folder)[0])

        img = imread(img_path)[:, :, :3].astype('float32')
        mask = get_mask(
            mask_folder, img.shape[0], img.shape[1]).astype('float32')

        augmented = self.transforms(image=img, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        mask = mask[0].permute(2, 0, 1)

        return image, mask


class transform:
    train_transform = A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        ToTensor()
    ])


# if __name__ == "__main__":
#     dataset_ = dataSet(config.DATA_PATH, transform.train_transform)
#     show_dataset(dataset_, 5)
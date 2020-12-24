import os
import glob
import zipfile

import numpy as np
import pandas as pd

from skimage.io import imread, imshow
from skimage.transform import resize

from torch.utils.data import random_split,DataLoader

from sklearn import model_selection
import matplotlib.pyplot as plt

import torch


def get_mask(mask_dir, IMG_HEIGHT, IMG_WIDTH): 
    '''
    mask_dir = glob.glob('./train_data/*/masks/')
    this will return a complete mask by concatenating all the small masks.
    '''  
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    
    for mask_f in os.listdir(mask_dir):
        mask_ = imread(os.path.join(mask_dir, mask_f))
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT,IMG_WIDTH)), axis=-1)
        mask = np.maximum(mask, mask_)
    return mask

def format_image(img):
    '''
    this func will convert the image into its original form, so it can be visualized.
    '''
    img = np.array(np.transpose(img, (1,2,0)))
    mean = np.array((0.485, 0.456, 0.406))
    std = np.array((0.229, 0.224, 0.225))
    img  = std * img + mean
    img = img*255
    img = img.astype(np.uint8)
    return img

def format_mask(mask):
    '''
    this func will convert the mask into its original form, so it can be visualized.
    '''
    mask = np.squeeze(np.transpose(mask, (1,2,0)))
    return mask

def split(dataset):
    split_ratio = 0.20
    train_size=int(np.round(dataset.__len__()*(1 - split_ratio),0))
    valid_size=int(np.round(dataset.__len__()*split_ratio,0))

    train_data, valid_data = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)

    val_loader = DataLoader(dataset=valid_data, batch_size=4)

    print("Length of train and valid datas: {}, {}".format(len(train_data), len(valid_data)))
    return train_loader, val_loader


def show_dataset(dataset, n=5):
    '''
    this will display n no of (img, mask) pair from the dataset.
    '''
    _ , ax = plt.subplots(n, 2,figsize=(n*3,8))

    for i in range(n):
        x ,y = dataset.__getitem__(np.random.randint(0,30))
        x = format_image(x)
        y = format_mask(y)
        ax[i, 0].imshow(x)
        ax[i, 1].imshow(y, interpolation="nearest", cmap="gray")
        # ax[i, 0].set_title("Ground Truth Image")
        # ax[i, 1].set_title("Mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()

def visualize_predict(model, data, output, target, valid_loader ,n_images):
  _, ax = plt.subplots(nrows=n_images, ncols=3, figsize=(15, 18))
  for img_no in range(0, n_images):
    tm=output[img_no][0].data.cpu().numpy()
    img = data[img_no].data.cpu()
    msk = target[img_no].data.cpu()
    img = format_image(img)
    msk = format_mask(msk)
    ax[img_no, 0].imshow(img)
    ax[img_no, 1].imshow(msk, interpolation="nearest", cmap="gray")
    ax[img_no, 2].imshow(tm, interpolation="nearest", cmap="gray")
    ax[img_no, 0].set_title("Ground Truth Image")
    ax[img_no, 1].set_title("Ground Truth Mask")
    ax[img_no, 2].set_title("Predicted Mask")
    ax[img_no, 0].set_axis_off()
    ax[img_no, 1].set_axis_off()
    ax[img_no, 2].set_axis_off()
  plt.tight_layout()
  plt.show()
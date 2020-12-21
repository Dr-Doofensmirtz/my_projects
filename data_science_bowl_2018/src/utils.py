import os
import glob
import zipfile

import numpy as np
import pandas as pd

from skimage.io import imread, imshow
from skimage.transform import resize

from sklearn import model_selection
import matplotlib.pyplot as plt

def get_mask(mask_dir, IMG_HEIGHT, IMG_WIDTH): 
    '''
    mask_dir = glob.glob('./train_data/*/masks/')
    this will return a complete mask by concatenating all the small masks.
    '''  
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    
    for mask_f in os.listdir(mask_dir):
        mask_ = imread(os.path.join(mask_dir, mask_f))
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT,IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    return mask

def get_df(img_dir, mask_dir, num_folds=5):
    '''
    this func will create a df containing n folds of the given dataset.
    '''
    data = {"img_": glob.glob(img_dir), "mask_": glob.glob(mask_dir)}
    df = pd.DataFrame(data=data)
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.KFold(n_splits=num_folds)

    for fold_, (_, x) in enumerate(kf.split(df)):
        for xs in x:
            df.loc[xs, "kfold"] = fold_

    return df

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

def unzip_data(path_to_zip_file, DATA_PATH):
    '''
    unzip the data files.
    '''
    with zipfile.ZipFile(path_to_zip_file, "r") as data:
        data.extractall(DATA_PATH)
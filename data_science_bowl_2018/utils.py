import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
from sklearn import model_selection

from skimage.io import imread, imshow, imsave
from skimage.transform import resize

import albumentations as A
from albumentations.pytorch import ToTensor

def get_mask(mask_dir, IMG_HEIGHT, IMG_WIDTH):   
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    
    for mask_f in next(os.walk(mask_dir))[2]:
        mask_ = imread(mask_dir)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT,IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    
    os.mkdir(s_path)
    imsave(s_path+ img +".jpg", mask)
    shutil.rmtree(m_path)

def get_folds(df):

  df['kfold'] = -1
  df = df.sample(frac=1).reset_index(drop=True)
  kf = model_selection.KFold(n_splits=5)

  for fold_, (_, x) in enumerate(kf.split(df)):
      for xs in x:
          df.loc[xs, "kfold"] = fold_

  return df

def get_train_transform():
   return A.Compose(
       [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        ToTensor()
        ])



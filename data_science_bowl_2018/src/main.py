import os
import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import dataSet
from transforms import transform
from models import UNet
from loss import DiceLoss
from utils import *
from train import train_one_epoch

def argparser():
    parser = argparse.ArgumentParser(description= 'nuceli pipeline')
    parser.add_argument('epoch', type = int)
    parser.add_argument('extract', type=bool)
    parser.add_argument('fold', type=int)
    parser.add_argument('img_dir', type=str)
    parser.add_argument('mask_dir', type=str)
    parser.add_argument('zip_path', type=str)
    parser.add_argument('unzip_path', type=str)

    return parser.parse_args()

if __name__ == "__main__":
    arg = argparser()
    
    #unzip data if not done already
    if arg.extract:
        unzip_data(arg.zip_path, arg.extract_path)

    #make a df with n no of folds
    df = get_df(arg.img_dir, arg.mask_dir, arg.fold)

    #get dataset
    train_dataset = dataSet(df, fold=0, train=True, transforms=transform.train_transform)
    valid_dataset = dataSet(df, fold=0, train=False, transforms=transform.valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=10, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=8, num_workers=0)

    #defining model and its parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.to(device)
    optimizer = Adam(model.parameters(), lr = 1e-3)
    criterion = DiceLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)

    #training
    for epoch in range(arg.epoch):
        train_one_epoch(epoch, model, train_loader, valid_loader, optimizer, criterion, scheduler, train_on_gpu=True)
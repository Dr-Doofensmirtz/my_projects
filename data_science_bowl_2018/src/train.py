import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
from engine import *
from dataset import dataSet, transform
from model import UNet
from utils import *


def run():
    #get dataset
    dataset_ = dataSet(config.DATA_PATH, transform.train_transform)
    train_loader, valid_loader = split(dataset_)

    #defining model and its parameters
    model = UNet().to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr = 1e-3)
    criterion = DiceLoss()

    accuracy = 0
    #training
    for epoch in range(config.EPOCHS):
        train_fn(epoch, model, train_loader, optimizer, criterion, True)
        val_loss = eval_fn(epoch, model, valid_loader, criterion, True)

        if (1-val_loss)*100 > accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)

        accuracy = (1-val_loss)*100

if __name__ == "__main__":
    run()
import torch
import torch.nn as nn

from tqdm.auto import tqdm as tq
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(self, input, target):
        return self.soft_dice_loss(input, target, per_image=self.per_image)

    def soft_dice_loss(self, outputs, targets, per_image=False):
        batch_size = outputs.size()[0]
        eps = 1e-5
        if not per_image:
            batch_size = 1
        dice_target = targets.contiguous().view(batch_size, -1).float()
        dice_output = outputs.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
        loss = (1 - (2 * intersection + eps) / union).mean()

        return loss

def train_fn(epoch, model, train_loader, optimizer, criterion, train_on_gpu=False):
    model.train()
    train_loss = 0.0
    bar = tq(train_loader, postfix={"train_loss":0.0})
    for data, target in bar:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        bar.set_postfix(ordered_dict={'train_loss':loss.item()})

    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {}  Training Loss: {:.6f}'.format(epoch, train_loss))

def eval_fn(epoch, model, valid_loader, criterion, train_on_gpu=False):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        bar = tq(valid_loader, postfix={"valid_loss":0.0})
        for data, target in bar:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            bar.set_postfix(ordered_dict={'train_loss':loss.item()})
            val_loss += loss.item()*data.size(0)

    val_loss = val_loss/len(valid_loader.dataset)
    print('Epoch: {}  Validation Loss: {:.6f}'.format(epoch, val_loss))
    return val_loss
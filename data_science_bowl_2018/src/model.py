#add various segmentation model

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding =1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, 2,2)
        self.up_conv_1 = double_conv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(512, 256, 2,2)
        self.up_conv_2 = double_conv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(256, 128, 2,2)
        self.up_conv_3 = double_conv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(128, 64, 2,2)
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, image):
        #encoder
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        
        #decoder
        x = self.up_trans_1(x9)
        y = self.up_conv_1(torch.cat([x7, x], 1))
        x = self.up_trans_2(y)
        y = self.up_conv_2(torch.cat([x5, x], 1))
        x = self.up_trans_3(y)
        y = self.up_conv_3(torch.cat([x3, x], 1))
        x = self.up_trans_4(y)
        y = self.up_conv_4(torch.cat([x1, x], 1))

        x = self.out(y)
        x = torch.sigmoid(x)

        return x

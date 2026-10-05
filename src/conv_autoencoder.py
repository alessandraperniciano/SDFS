
### Se define un autoencoder convolucional siguiendo la arquitectura UNet

import torch.nn as nn



class Block(nn.Module):
    def __init__(self, inputs, middles, outs):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inputs, middles, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(middles, outs, kernel_size = 3, stride = 1, padding = 1)
        self.silu = nn.SiLU()
        self.bn = nn.BatchNorm2d(outs)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.silu(self.conv1(x))
        x = self.silu(self.bn(self.conv2(x)))
        return self.pool(x), x

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, base_channels=16):
        super(UNet, self).__init__()
        self.en1 = Block(input_channels, base_channels, base_channels)
        self.en2 = Block(base_channels, base_channels*2, base_channels*2)
        self.en3 = Block(base_channels*2, base_channels*4, base_channels*4)
        
        self.upsample2 = nn.ConvTranspose2d(base_channels*4, base_channels*4, 2, stride=2)
        self.de2 = Block(base_channels*4, base_channels*4, base_channels*2)
        
        self.upsample1 = nn.ConvTranspose2d(base_channels*2, base_channels*2, 2, stride=2)
        self.de1 = Block(base_channels*2, base_channels*2, base_channels)
        
        self.conv_last = nn.Conv2d(base_channels, output_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x, e1 = self.en1(x) #1x28x28 --> 16x14x14
        x, e2 = self.en2(x) #16x14x14 --> 32x7x7
        _, x = self.en3(x) #32x7x7 --> 64x7x7
        
        x = self.upsample2(x) #64x7x7 --> 64x14x14
        _, x = self.de2(x) #64x14x14 --> 32x14x14
        x = x + e2  #32x14x14 + 32x14x14 --> 32x14x14
        
        x = self.upsample1(x) #32x14x14 --> 32x28x28
        _, x = self.de1(x) #32x28x28 --> 16x28x28
        x = x + e1  #16x28x28 + 16x28x28 --> 16x28x28
        
        x = self.conv_last(x) #16x28x28 --> 1x28x28
        return x

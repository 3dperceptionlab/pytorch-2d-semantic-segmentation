"""UNetModel

Inspired by zijundeng/pytorch-semantic-segmentation implementation.
See https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/models/u_net.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetNetwork(nn.Module):

    class _EncoderBlock(nn.Module):

        def __init__(self, inChannels, outChannels, dropout=False):

            super().__init__()

            layers_ = [
                    nn.Conv2d(inChannels, outChannels, kernel_size=3),
                    nn.BatchNorm2d(outChannels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(outChannels, outChannels, kernel_size=3),
                    nn.BatchNorm2d(outChannels),
                    nn.ReLU(inplace=True),]

            if dropout:
              layers_.append(nn.Dropout())

            layers_.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            self.encode = nn.Sequential(*layers_)

        def forward(self, x):

            return self.encode(x)

    class _DecoderBlock(nn.Module):

        def __init__(self, inChannels, midChannels, outChannels):

            super().__init__()
            
            layers_ = [
                    nn.Conv2d(inChannels, midChannels, kernel_size=3),
                    nn.BatchNorm2d(midChannels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(midChannels, midChannels, kernel_size=3),
                    nn.BatchNorm2d(midChannels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(midChannels, outChannels, kernel_size=2, stride=2),]

            self.decode = nn.Sequential(*layers_)

        def forward(self, x):

            return self.decode(x)

    def __init__(self, num_classes):

        super(UNetNetwork, self).__init__()

        self.encoder1 = self._EncoderBlock(3, 64)
        self.encoder2 = self._EncoderBlock(64, 128)
        self.encoder3 = self._EncoderBlock(128, 256)
        self.encoder4 = self._EncoderBlock(256, 512, dropout=True)
        self.center = self._DecoderBlock(512, 1024, 512)
        self.decoder4 = self._DecoderBlock(1024, 512, 256)
        self.decoder3 = self._DecoderBlock(512, 256, 128)
        self.decoder2 = self._DecoderBlock(256, 128, 64)
        self.decoder1 = nn.Sequential(
                            nn.Conv2d(128, 64, kernel_size=3),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, kernel_size=3),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def __repr__(self):

        return "U-Net model..."

    def forward(self, x):

        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        center = self.center(encoder4)
        decoder4 = self.decoder4(torch.cat([
                                    center,
                                    F.upsample(encoder4, center.size()[2:], mode='bilinear')], 1))
        decoder3 = self.decoder3(torch.cat([
                                    decoder4,
                                    F.upsample(encoder3, decoder4.size()[2:], mode='bilinear')], 1))
        decoder2 = self.decoder2(torch.cat([
                                    decoder3,
                                    F.upsample(encoder2, decoder3.size()[2:], mode='bilinear')], 1))
        decoder1 = self.decoder1(torch.cat([
                                    decoder2,
                                    F.upsample(encoder1, decoder2.size()[2:], mode='bilinear')], 1))
        final = self.final(decoder1)

        return F.upsample(final, x.size()[2:], mode='bilinear')

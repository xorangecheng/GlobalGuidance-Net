import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .danet import DGNLB
class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm,mm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        elif backbone=='daf_ds':
            low_level_inplanes = 256
        elif backbone=='dense':
            low_level_inplanes=256



        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(256, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn2 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.mm=mm
        
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        if self.mm=='dgnlb':
            self.dg=DGNLB(256,256,WSE=False)
   
        self._init_weight()


    def forward(self, x, low_level_feat,layer1):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        layer1 = self.conv2(layer1)
        layer1 = self.bn2(layer1)
        layer1 = self.relu(layer1)


        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        if self.mm is not None:
            x=self.dg(x,low_level_feat)
        x=torch.cat((x,layer1),dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm,mm):
    return Decoder(num_classes, backbone, BatchNorm,mm)
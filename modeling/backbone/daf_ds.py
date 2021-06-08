import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.backbone.resnext import ResNeXt101
class DAF(nn.Module):
    def __init__(self):
        super(DAF, self).__init__()
        self.resnext = ResNeXt101()

        self.down0 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )

        

        self.predict0 = nn.Conv2d(128, 1, kernel_size=1)
        self.predict3 = nn.Conv2d(128, 1, kernel_size=1)
        self.predict2 = nn.Conv2d(128, 1, kernel_size=1)
        self.predict1 = nn.Conv2d(128, 1, kernel_size=1)
        self.sigmoid=nn.Sigmoid()
        self.pool=nn.MaxPool2d(3,stride=1,padding=1)


    def forward(self, x):
        
        layer0 = self.resnext.layer0(x)
        layer1 = self.resnext.layer1(layer0)
        layer2 = self.resnext.layer2(layer1)
        layer3 = self.resnext.layer3(layer2)


        down0 = F.interpolate(self.down0(layer0), size=layer1.size()[2:], mode='bilinear',align_corners=True)
        down3 = F.interpolate(self.down3(layer3), size=layer1.size()[2:], mode='bilinear',align_corners=True)
        down2 = F.interpolate(self.down2(layer2), size=layer1.size()[2:], mode='bilinear',align_corners=True)
        down1 = self.down1(layer1)

        predict0 = self.predict0(down0)
        predict3 = self.predict3(down3)
        predict2 = self.predict2(down2)
        predict1 = self.predict1(down1)

        fuse1 = self.fuse1(torch.cat((down0, down3, down2, down1), 1))


        o1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear',align_corners=True)
        o2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear',align_corners=True)
        o3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear',align_corners=True)
        o0 = F.interpolate(predict0, size=x.size()[2:], mode='bilinear',align_corners=True)

        p4_edge=o0
        smoothed_4=self.pool(p4_edge)
        edge4=torch.abs(p4_edge-smoothed_4)
        o0_e=o0+edge4

        p3_edge=o3
        smoothed_3=self.pool(p3_edge)
        edge3=torch.abs(p3_edge-smoothed_3)
        o3_e=o3+edge3


        p2_edge=o2
        smoothed_2=self.pool(p2_edge)
        edge2=torch.abs(p2_edge-smoothed_2)
        o2_e=o2+edge2


        p1_edge=o1
        smoothed_1=self.pool(p1_edge)
        edge1=torch.abs(p1_edge-smoothed_1)
        o1_e=o1+edge1


        return layer3, layer1, fuse1,o1_e,o2_e,o3_e,o0_e,edge1,edge2,edge3,edge4



def daf_ds():
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DAF()
    return model

if __name__ == "__main__":
    import torch
    model = daf_ds()
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat, predict1,predict2,predict3,predict4= model(input)
    print(output.size())
    print(low_level_feat.size())
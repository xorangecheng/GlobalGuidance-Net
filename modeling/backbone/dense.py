import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.backbone.resnext import ResNeXt101

class Densenet169(nn.Module):
    def __init__(self):
        super(Densenet169, self).__init__()
        net =  models.densenet169(pretrained=True)
        # net.load_state_dict(torch.load(resnext_101_32_path))

        net = list(net.children())[0]
        self.layer0 = nn.Sequential(*net[:4])
        self.layer1 = nn.Sequential(*net[4: 6])
        self.layer2 = net[6:8]
        self.layer3 = net[8:10]
        self.layer4 = net[10:12]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4

class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.densenet = Densenet169()

        self.down4 = nn.Sequential(
            nn.Conv2d(1664, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.predict4 = nn.Conv2d(128, 1, kernel_size=1)
        self.predict3 = nn.Conv2d(128, 1, kernel_size=1)
        self.predict2 = nn.Conv2d(128, 1, kernel_size=1)
        self.predict1 = nn.Conv2d(128, 1, kernel_size=1)

        # self.attention4 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
        # )
        # self.attention3 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
        # )
        # self.attention2 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
        # )
        # self.attention1 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
        # )

        # self.refine4 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        # )
        # self.refine3 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        # )
        # self.refine2 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        # )
        # self.refine1 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        # )

        # self.predict4 = nn.Conv2d(64, 1, kernel_size=1)
        # self.predict3 = nn.Conv2d(64, 1, kernel_size=1)
        # self.predict2 = nn.Conv2d(64, 1, kernel_size=1)
        # self.predict1 = nn.Conv2d(64, 1, kernel_size=1)

        # self.predict4_2 = nn.Conv2d(64, 1, kernel_size=1)
        # self.predict3_2 = nn.Conv2d(64, 1, kernel_size=1)
        # self.predict2_2 = nn.Conv2d(64, 1, kernel_size=1)
        # self.predict1_2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        layer0 = self.densenet.layer0(x)
        layer1 = self.densenet.layer1(layer0)
        layer2 = self.densenet.layer2(layer1)
        layer3 = self.densenet.layer3(layer2)
        layer4 = self.densenet.layer4(layer3)
        # print(layer1.size())

        down4 = F.interpolate(self.down4(layer4), size=layer1.size()[2:], mode='bilinear',align_corners=True)
        down3 = F.interpolate(self.down3(layer3), size=layer1.size()[2:], mode='bilinear',align_corners=True)
        down2 = F.interpolate(self.down2(layer2), size=layer1.size()[2:], mode='bilinear',align_corners=True)
        down1 = self.down1(layer1)

        predict4 = self.predict4(down4)
        predict3 = self.predict3(down3)
        predict2 = self.predict2(down2)
        predict1 = self.predict1(down1)

  

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear',align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear',align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear',align_corners=True)
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear',align_corners=True)

        # attention4 = self.attention4(torch.cat((down4, fuse1), 1))
        # attention3 = self.attention3(torch.cat((down3, fuse1), 1))
        # attention2 = self.attention2(torch.cat((down2, fuse1), 1))
        # attention1 = self.attention1(torch.cat((down1, fuse1), 1))

        # refine4 = self.refine4(torch.cat((down4, attention4 * fuse1), 1))
        # refine3 = self.refine3(torch.cat((down3, attention3 * fuse1), 1))
        # refine2 = self.refine2(torch.cat((down2, attention2 * fuse1), 1))
        # refine1 = self.refine1(torch.cat((down1, attention1 * fuse1), 1))

        # predict4_2 = self.predict4_2(refine4)
        # predict3_2 = self.predict3_2(refine3)
        # predict2_2 = self.predict2_2(refine2)
        # predict1_2 = self.predict1_2(refine1)

        # predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')
        # predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
        # predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
        # predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')

        # predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='bilinear')
        # predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='bilinear')
        # predict3_2 = F.upsample(predict3_2, size=x.size()[2:], mode='bilinear')
        # predict4_2 = F.upsample(predict4_2, size=x.size()[2:], mode='bilinear')

        return layer4, layer1, fuse1,predict1,predict2,predict3,predict4

        # if self.training:
        #     return predict1, predict2, predict3, predict4, predict1_2, predict2_2, predict3_2, predict4_2
        # else:
        #     return F.sigmoid((predict1_2 + predict2_2 + predict3_2 + predict4_2) / 4)


def Dense_net():
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Dense()
    return model

if __name__ == "__main__":
    import torch
    model = Dense_net()
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat, predict1,predict2,predict3,predict4= model(input)
    print(output.size())
    print(low_level_feat.size())
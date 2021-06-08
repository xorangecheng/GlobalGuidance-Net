import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,mm=None,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm,mm)
        self.predict5 = nn.Conv2d(256, 1, kernel_size=1)
        self.pool=nn.MaxPool2d(3,stride=1,padding=1)


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, layer1,low_level_feat,pre1,pre2,pre3,pre4,e1,e2,e3,e4 = self.backbone(input)
        
        x = self.aspp(x)
        pre5=self.predict5(F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True))
        x = self.decoder(x, low_level_feat,layer1)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        smoothed_4=self.pool(x)
        edge4=torch.abs(x-smoothed_4)
        o0_e=x+edge4

        return o0_e,pre1,pre2,pre3,pre4,pre5,edge4,e1,e2,e3,e4

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output,pre1,pre2,pre3,pre4,pre5,p_out,c_out = model(input)
    print(output.size())



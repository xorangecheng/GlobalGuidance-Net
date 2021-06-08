import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']

class SELayer(Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x*y.expand_as(x)
        

class G_PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim,in_dim_guide):
        super(G_PAM_Module, self).__init__()
        # self.chanel_in = in_dim
        
        self.query_guide=Conv2d(in_channels=in_dim_guide, out_channels=in_dim_guide, kernel_size=1)
        self.key_guide=Conv2d(in_channels=in_dim_guide, out_channels=in_dim_guide, kernel_size=1)
        # self.value_guide=Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x,g):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        gm_batchsize, gC, gheight, gwidth = g.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        proj_query_guide=self.query_guide(g).view(gm_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key_guide=self.key_guide(g).view(gm_batchsize, -1, width*height)
        energy_guide=torch.bmm(proj_query_guide, proj_key_guide)
        attention = self.softmax(energy)
        attention_guide=self.softmax(energy_guide)
        guide_energy=torch.bmm(attention,attention_guide)
        guide_attention=self.softmax(guide_energy)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, guide_attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class G_CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_channels, in_channels_guide,WSE=True):
        super(G_CAM_Module, self).__init__()
        self.z=Conv2d(in_channels, in_channels, kernel_size=1)
        self.WSE=WSE
        self.se=SELayer(in_channels)
        self.softmax = Softmax(dim=-1)
        self.seconv=Conv2d(in_channels_guide, in_channels, kernel_size=1)
        self.W = Conv2d(in_channels, in_channels,
                             kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.gamma = Parameter(torch.zeros(1))
    def forward(self,x,g):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        gm_batchsize, gC, gheight, gwidth = g.size()

        proj_query = x.view(m_batchsize, C, -1)

        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy_new = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy_new)

        if self.WSE:
            g=self.se(g)

        proj_query_guide=g.view(m_batchsize, C, -1)
        proj_key_guide=g.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy_guide_new=torch.bmm(proj_query_guide,proj_key_guide)
        attention_guide=self.softmax(energy_guide_new)
        guide_energy=torch.bmm(attention,attention_guide)
        guide_energy_new=torch.max(guide_energy,-1,keepdim=True)[0].expand_as(guide_energy)-guide_energy
        guide_attention=self.softmax(guide_energy_new)

        
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(guide_attention,proj_value).contiguous().view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out




class DGNLB(Module):
    def __init__(self,in_channels, in_channels_guide,WSE=True):
        super(DGNLB,self).__init__()

        self.pconv=nn.Sequential(
            nn.Conv2d(in_channels,in_channels, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels), nn.PReLU(),
            nn.Conv2d(in_channels,in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.PReLU()
            )
        self.cconv=nn.Sequential(
            nn.Conv2d(in_channels,in_channels, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels), nn.PReLU(),
            nn.Conv2d(in_channels,in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.PReLU()
            )
        self.fconv=nn.Sequential(
            nn.Conv2d(in_channels,in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.PReLU()
        )
        self.PAM=G_PAM_Module(in_channels,in_channels_guide)
        self.CAM=G_CAM_Module(in_channels,in_channels_guide,WSE=WSE)
        self.cov1=nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, 1, 1))
        self.cov2=nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, 1, 1))
        self.predict=nn.Conv2d(in_channels,1,kernel_size=1)
    def forward(self,x,g,WSE=True):
        pam=self.PAM(x,g)
        pam2=self.pconv(pam)
        pam2_out=self.cov1(pam2)
        cam=self.CAM(pam2,g)
        cam2=self.cconv(cam)
        cam2_out=self.cov2(cam2)
        final=self.fconv(cam2)
        return final




class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


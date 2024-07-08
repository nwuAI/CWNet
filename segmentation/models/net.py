# 创建时间: 2024-07-01 9:57
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torchvision.transforms as trans
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from .vit_seg_configs import get_b16_config
from .vit_seg_modeling import VisionTransformer
from .gabor_filter import build_bn_filters


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = residual + out
        out = self.relu(out)
        return out

class Attention(nn.Module):
    def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temprature
        # assert in_planes > ratio         # 注意改回去***************
        if in_planes == 1:
            hidden_planes = in_planes
        # hidden_planes = in_planes // ratio
        else:
            hidden_planes = in_planes // ratio
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
        )

        if init_weight:
            self._initialize_weights()

    def update_temprature(self):
        if self.temprature > 1:
            self.temprature -= 1

    def _initialize_weights(self):  # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)
        att = self.net(att).view(x.shape[0], -1)
        return F.softmax(att / self.temprature, -1)


class DynamicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, grounps=1, bias=True, K=8,
                 temprature=30, ratio=2, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, ratio=ratio, K=K, temprature=temprature,
                                   init_weight=init_weight)
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if self.init_weight:
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape  # ********  in_planels=c
        softmax_att = self.attention(x)  #      直接改为可学习系参数
        x = x.view(1, -1, h, w)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        filters = build_bn_filters()
        filters = torch.unsqueeze(filters, dim=1)
        filters = torch.unsqueeze(filters, dim=1)
        filters = filters.to(device)
        weight = torch.mul(self.weight, filters)
        weight = weight.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size, self.kernel_size)
        #  矩阵相乘得到
        if self.bias is not None:
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # b,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
            # [1,b*c,h,w]
            # groups的作用
            # 变成b个[1,c,h,w]与[b*out_planes,in_planes,3,3] 做卷积操作然后cancat

        output = output.view(bs, self.out_planes, h, w)
        output = self.relu(self.bn(output))
        return output


def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


# 拼接最大和平均  空间注意力
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

# 双分支融合模块  CNN与Transformer
class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block   通道注意力
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l     空间注意力
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # g: CNN     x: Transformer
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class CWF(nn.Module):
    def __init__(self, in_channel, M=2, r=16, stride=1, L=32):
        super(CWF, self).__init__()
        d = max(int(in_channel / r), L)
        self.M = M
        self.in_channel = in_channel
        self.fc = nn.Linear(in_channel, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, in_channel)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        x = x.unsqueeze(dim=1)
        y = y.unsqueeze(dim=1)              # [b,c,h,w]--->[b,1,c,h,w]
        feas = torch.cat([x, y], dim=1)     # [b,1,c,h,w]-->[b,2,c,h,w]
        fea_U = torch.sum(feas, dim=1)      # [b,2,c,h,w]-->[b,c,h,w]
        fea_s = fea_U.mean(-1).mean(-1)     # [b,c,h,w]---->[b,c]
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        # print("attention_vectors", attention_vectors.shape)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        # print("attention_vectors", attention_vectors.shape)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class SFA(nn.Module):
    def __init__(self, in_ch):
        super(SFA, self).__init__()
        self.dilation_1 = nn.Conv2d(in_ch, in_ch, 3, 1, padding=1, dilation=1, bias=False)
        self.dilation_3 = nn.Conv2d(in_ch, in_ch, 3, 1, padding=3, dilation=3, bias=False)
        self.dilation_5 = nn.Conv2d(in_ch, in_ch, 3, 1, padding=5, dilation=5, bias=False)

        self.fusion_12 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 2, 1, 1)
        )

        self.fusion_23 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 2, 1, 1)
        )

        self.att_fusion = nn.Conv2d(in_ch, in_ch, 1, 1)


    def forward(self, x):
        f1 = self.dilation_1(x)
        f2 = self.dilation_3(x)
        f3 = self.dilation_5(x)

        f12 = torch.cat([f1, f2], dim=1)
        f23 = torch.cat([f2, f3], dim=1)

        fusion_12 = self.fusion_12(f12)
        fusion_23 = self.fusion_23(f23)

        att_12 = torch.softmax(fusion_12, dim=1)
        w_alpha1, w_beta1 = torch.split(att_12, 1, dim=1)

        att_23 = torch.softmax(fusion_23, dim=1)
        w_alpha2, w_beta2 = torch.split(att_23, 1, dim=1)

        att_1 = w_alpha1*f1 + w_beta1*f2
        att_2 = w_alpha2*f2 + w_beta2*f3
        out = att_1 + att_2 + x
        out = self.att_fusion(out)
        return out


class EF(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super(EF, self).__init__()
        in_ch1 = in_ch*2
        hidden_ch = (in_ch*2) // reduction
        self.se = nn.Sequential(
            nn.Conv2d(in_ch1, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, in_ch1, 1),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(in_ch1, in_ch, 1)

    def forward(self, x1, x2):

        x12 = torch.cat([x1, x2], dim=1)
        se = self.se(x12)
        se = self.conv1x1(se)
        se = F.adaptive_avg_pool2d(se, 1)
        se = torch.sigmoid(se)
        w1 = se * x1
        out = w1 + x2
        return out


class up_conv8(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv8, self).__init__()
        self.upsam = nn.Upsample(scale_factor=2)
        self.upconv1 = conv_block(ch_in, 256)
        self.upconv2 = conv_block(256, 128)
        self.upconv3 = conv_block(128, ch_out)

    def forward(self, x):
        x = self.upsam(x)
        x = self.upconv1(x)
        x = self.upsam(x)
        x = self.upconv2(x)
        x = self.upsam(x)
        x = self.upconv3(x)
        return x


class up_conv4(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv4, self).__init__()
        self.upsam = nn.Upsample(scale_factor=2)
        self.upconv1 = conv_block(ch_in, 128)
        self.upconv2 = conv_block(128, ch_out)

    def forward(self, x):
        x = self.upsam(x)
        x = self.upconv1(x)
        x = self.upsam(x)
        x = self.upconv2(x)
        return x


class MyNet(nn.Module):
    def __init__(self, in_c=1, n_classes=1):
        super(MyNet, self).__init__()
        self.n_classes = n_classes
        self.down = downsample()

        self.Conv1 = conv_block(ch_in=in_c, ch_out=64)
        self.Conv4 = DynamicConv(in_planes=in_c, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)

        self.Trans = VisionTransformer(get_b16_config())   # transformer

        self.CWF = CWF(in_channel=64)

        # self.SFA = SFA(256)

        # self.SAFM2 = SAFM(in_channel=128)
        # self.SAFM3 = SAFM(in_channel=256)
        # self.SAFM4 = SAFM(in_channel=256)  # 瓶颈层的 Transformer前面

        self.Up4 = up_conv(256, 256)
        # self.Up_conv4 = Decoder(512, 256)    # DSA用的地方
        self.EF1 = EF(256)

        self.Up3 = up_conv(256, 128)
        # self.Up_conv3 = Decoder(256, 128)
        self.EF2 = EF(128)

        self.Up2 = up_conv(128, 64)
        # self.Up_conv2 = Decoder(128, 64)
        self.EF3 = EF(64)

        self.fconv = nn.Conv2d(64, self.n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        xl1 = self.Conv1(x)
        xr1 = self.Conv4(x)

        x1 = self.CWF(xl1, xr1)
        x2 = self.down(x1)
        x3 = self.Conv2(x2)
        x4 = self.down(x3)
        x5 = self.Conv3(x4)
        x6 = self.down(x5)

#        d7 = self.Trans(x6)

        x7 = self.Up4(x6)
        d5 = self.EF1(x5, x7)
        # d5 = torch.cat((m3, d5), dim=1)
        # d5 = self.Up_conv4(d5)

        # feature_map_dict['d5'] = d5

        d3 = self.Up3(d5)
        d3 = self.EF2(x3, d3)

        d1 = self.Up2(d3)
        d1 = self.EF3(x1, d1)

        out = self.fconv(d1)

        return out

if __name__ == '__main__':
    img = cv2.imread('C:/Users/Lenovo/Desktop/c.png')
    img = cv2.resize(img, (584, 568))
    img = torch.tensor(img).float()
    img = img.permute(2, 1, 0)
    img = torch.unsqueeze(img, 0)
    model = MyNet(3, 1)
    out = model(img)
    print('模型输出', out.shape)


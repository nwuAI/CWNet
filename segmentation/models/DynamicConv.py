# 创建时间: 2023-10-13 19:32
import torch
from torch import nn
from torch.nn import functional as F
from .gabor_filter import build_bn_filters


class Attention(nn.Module):
    def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # [b,c,h,w]-->[b,c,1,1]
        self.temprature = temprature
        assert in_planes > ratio
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
        att = self.avgpool(x)  # b,dim,1,1   [b,c,h,w]--->[b,c,1,1]
        att = self.net(att).view(x.shape[0], -1)  # b,K  [b,c,1,1]-->[b,k,1,1]  ---->[b,k]
        return F.softmax(att / self.temprature, -1)  # 对每个batch的K个数进行softmax


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
        softmax_att = self.attention(x)  # [b,K]
        x = x.view(1, -1, h, w)  # [1,b*c,h,w]

        gabor_filter = build_bn_filters()
        gabor_filter = torch.unsqueeze(gabor_filter, dim=1)
        gabor_filter = torch.unsqueeze(gabor_filter, dim=1)
        # print(gabor_filter.shape)  # [8,1,1,3,3]
        weight = torch.mul(self.weight, gabor_filter)
        weight = weight.view(self.K, -1)  # [K, out_planes, in_planes//1, 3,3]---->[K,out_planes*in_planes*3*3]
        # 在这加入滤波？？？？？？
        # 将K个构建好的 3x3的Gabor滤波器与weight最后两个维度相乘？？？？？？
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size, self.kernel_size)
        #  矩阵相乘得到 [b,out_planes*in_planes*3*3]----->[b*out_planes,in_planes,3,3]
        if self.bias is not None:
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # b,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
            # [1,b*c,h,w]    [b*out_planes,in_planes,3,3]    --->[1,b*out_planes,h,w]
            # groups的作用
            # 变成b个[1,c,h,w]与[b*out_planes,in_planes,3,3] 做卷积操作然后cancat  [1,b*out_planes,h,w]

        output = output.view(bs, self.out_planes, h, w)  # [1,b*out_planes,h,w]--->[b,out_planes,h,w]
        return output


if __name__ == '__main__':
    input = torch.randn(2, 3, 512, 512)
    m = DynamicConv(in_planes=3, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False)
    out = m(input)
    print(out.shape)
    m = DynamicConv(in_planes=64, out_planes=128, kernel_size=3, stride=1, padding=1, bias=False)
    out = m(out)
    print(out.shape)

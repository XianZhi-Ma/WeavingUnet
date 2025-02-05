"""
xz & 2024/3/23 14:41
"""
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial
from mmcv.cnn import ConvModule, build_norm_layer
from typing import Optional
from networks.DSConv import DSConv
from timm.models.layers import SqueezeExcite
import os

nonlinearity = partial(F.relu, inplace=True)

class WeavingUnet(nn.Module):  # strip-unet
    def __init__(self, num_classes=1):
        super(WeavingUnet, self).__init__()
        filters = [48, 64, 160, 1280]
        model = models.efficientnet_v2_s(pretrained=True)
        # 1024*3
        self.downsampling1 = model.features[0] # 512 512 24
        self.downsampling2 = model.features[1] # 512 512 24

        self.encoder1 = model.features[2] # 256 256 48

        self.encoder2 = model.features[3] # 128 128 64

        self.encoder31 = model.features[4] # 64 64 128
        self.encoder32 = model.features[5] # 64 64 160

        self.encoder41 = model.features[6] # 32 32 256
        self.encoder42 = model.features[7] # 32 32 1280

        self.conv11 = nn.Conv2d(24, filters[0], 1)
        self.conv12 = nn.Conv2d(filters[0], filters[1], 1)
        self.conv1313 = ConvModule(filters[0], filters[0], (13, 13), 2,
                                  (6, 6), groups=filters[0],
                                  norm_cfg=None, act_cfg=None)
        self.conv77 = ConvModule(filters[1], filters[1], (7, 7), 2,
                                  (3, 3), groups=filters[1],
                                  norm_cfg=None, act_cfg=None)
        self.att1 = SWA(1, filters[0])
        self.att2 = SWA(2, filters[1])

        self.conv13 = nn.Conv2d(filters[1], filters[2], 1)
        self.conv14 = nn.Conv2d(filters[2], filters[3], 1)
        self.conv55 = ConvModule(filters[2], filters[2], (5, 5), 2,
                                   (2, 2), groups=filters[2],
                                   norm_cfg=None, act_cfg=None)
        self.conv33 = ConvModule(filters[3], filters[3], (3, 3), 2,
                                 (1, 1), groups=filters[3],
                                 norm_cfg=None, act_cfg=None)
        self.ciwm1 = CIWM(1, filters[2])
        self.ciwm2 = CIWM(2, filters[3])

        self.giem = GIEM()

        self.decoder4 = MSWD(4, filters[3], filters[2])  # 32*1280=>64*160
        self.decoder3 = MSWD(3, filters[2], filters[1])  # 64*160=>128*64
        self.decoder2 = MSWD(2, filters[1], filters[0])  # 128*64=>256*48
        self.decoder1 = MSWD(1, filters[0], 24)  # 256*48=>512*24

        self.finaldeconv1 = nn.ConvTranspose2d(24, 12, 4, 2, 1)  # 1024*12
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(12, 12, 3, padding=1)  # 1024*12
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(12, num_classes, 3, padding=1)  # 1024*1

    def forward(self, x):
        # Encoder
        x = self.downsampling1(x) # 512 512 24
        x = self.downsampling2(x) # 512 512 24

        e1 = self.encoder1(x) # 256 256 48

        e2 = self.encoder2(e1) # 128 128 64

        e30 = self.encoder31(e2) # 64 64 128
        e3 = self.encoder32(e30)# 64 64 160

        e40 = self.encoder41(e3) # 32 32 256
        e4 = self.encoder42(e40) # 32 32 1280

        #注意力机制
        e11 = self.conv11(x) # 512 512 48
        e21 = self.conv12(e1) # 256 256 64
        e1att = self.conv1313(e11) # 256 256 48
        e2att = self.conv77(e21) # 128 128 64
        e1 = e1att + e1 # 256 256 48
        e2 = e2att + e2 # 128 128 64
        att1 = self.att1(e1) # 256 256 48
        att2 = self.att2(e2) # 128 128 64

        #CIWM
        e31 = self.conv13(att2)
        e3ciwm = self.conv55(e31)
        e3 = self.ciwm1(e3 + e3ciwm)
        e41 = self.conv14(e3)
        e4ciwm = self.conv33(e41)
        e44 = self.ciwm2(e4 + e4ciwm)

        #GIPM  (e4+e44)
        giem = self.giem(e4, e44)

        #center
        d4 = e44 + giem

        # Decoder
        d3 = self.decoder4(d4) + e3
        d2 = self.decoder3(d3) + att2
        d1 = self.decoder2(d2) + att1
        d0 = self.decoder1(d1)

        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class SWA(nn.Module):  #Snake Weaving Attention
    def __init__(self,
                 num: int,
                 channels: int,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='SiLU'),
                 ):
        super(SWA, self).__init__()
        self.num = num
        self.convfirst = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=19 - 6 * num, padding= 9 - 3 * num, bias=False)
        self.avg_pool = nn.AvgPool2d(19 - 6 * num, 1, 9 - 3 * num)

        self.h_conv1 = ConvModule(channels, channels, (1, 21 - 6 * num), 1,
                                  (0, 10 - 3 * num), groups=channels,
                                  norm_cfg=None, act_cfg=None)
        self.h_conv2 = ConvModule(channels, channels, (1, 19 - 6 * num), 1,
                                  (0, 9 - 3 * num), groups=channels,
                                  norm_cfg=None, act_cfg=None)
        self.h_conv3 = ConvModule(channels, channels, (1, 17 - 6 * num), 1,
                                  (0, 8 - 3 * num), groups=channels,
                                  norm_cfg=None, act_cfg=None)
        self.h_snake_conv = DSConv(channels, channels, kernel_size=19 - 6 * num, extend_scope=1, morph=0, if_offset=True, device="cuda")

        self.v_conv1 = ConvModule(channels, channels, (21 - 6 * num, 1), 1,
                                 (10 - 3 * num, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv2 = ConvModule(channels, channels, (19 - 6 * num, 1), 1,
                                 (9 - 3 * num, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv3 = ConvModule(channels, channels, (17 - 6 * num, 1), 1,
                                 (8 - 3 * num, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_snake_conv = DSConv(channels, channels, kernel_size=19 - 6 * num, extend_scope=1, morph=1, if_offset=True, device="cuda")

        self.convlast = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.convfirst(x)
        x = self.avg_pool(x)

        h = self.h_snake_conv(self.h_conv3(self.h_conv2(self.h_conv1(x))))

        v = self.v_snake_conv(self.v_conv3(self.v_conv2(self.v_conv1(x))))

        x = h + v + x
        x = self.convlast(x)

        return x


class CIWM(nn.Module): #contextual information weaving module
    def __init__(self, num, channels):
        super(CIWM, self).__init__()
        self.num = num
        self.convputong = nn.Conv2d(channels, channels, kernel_size= 7 - 2 * num, padding=3 - num)
        self.dconv1 = nn.Conv2d(channels, channels, kernel_size=9 - 2 * num, dilation=2, padding=8 - 2 * num)
        self.dconv2 = nn.Conv2d(channels, channels, kernel_size=11 - 2 * num, dilation=3, padding=15 - 3 * num)
        self.dwconv1 = ConvModule(channels, channels, (13 - 2 * num, 13 - 2 * num), 1,
                                  (6 - num, 6 - num), groups=channels,
                                  norm_cfg=None, act_cfg=None)
        self.dwconv2 = ConvModule(channels, channels, (15 - 2 * num, 15 - 2 * num), 1,
                                  (7 - num, 7 - num), groups=channels,
                                  norm_cfg=None, act_cfg=None)
        self.conv11 = nn.Conv2d(3 * channels, channels, kernel_size=1)
        self.conv12 = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)


    def forward(self, x):
        x1 = self.convputong(x)

        x20 = self.dconv1(x)
        x2 = self.dconv2(x20)

        x30 = self.dwconv1(x)
        x3 = self.dwconv2(x30)

        x0 = torch.cat((x1, x2, x3), 1)
        x0 = self.conv11(x0)
        x0 = nonlinearity(self.norm(x0))
        x0 = torch.mul(x, x0)
        x = x0 + x

        xh = F.adaptive_avg_pool2d(x, ((3-self.num)*32, 1))
        xv = F.adaptive_avg_pool2d(x, (1, (3-self.num)*32))

        xh = torch.mul(x, xh)
        xv = torch.mul(x, xv)

        x = x + nonlinearity(self.norm(self.conv12(xh +xv)))

        return x


class GIEM(nn.Module): #global information extraction module
    def __init__(self):
        super(GIEM, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.block = RepViTBlock(1280, 2560, 1280, 3, 1, True, True)
        self.conv11 = nn.Conv2d(5120, 1280, kernel_size=1)
        self.norm = nn.BatchNorm2d(1280)

    def forward(self, x1, x2):
        xgap1 = self.gap(x1)
        xgap1 = torch.mul(x1, xgap1)
        xplus1 = x1 + xgap1
        xrep1 = self.block(xplus1)

        xgap2 = self.gap(x2)
        xgap2 = torch.mul(x2, xgap2)
        xplus2 = x2 + xgap2
        xrep2 = self.block(xplus2)
        x = torch.cat((xrep1, xrep2, x1, x2), 1)
        x = self.conv11(x)
        x = nonlinearity(self.norm(x))

        return x


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class MSWD(nn.Module):
    def __init__(self, num, in_channels, out_channels):
        super(MSWD, self).__init__()

        self.conv11 = nn.Conv2d(in_channels, out_channels, 1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

        self.h_conv1 = ConvModule(out_channels, out_channels, (1, 4 * (2 ** (4 - num) - 1) + 3 + 2 ** (5 - num)), 1,
                                  (0, 2 * (2 ** (4 - num) - 1) + 1 + 2 ** (4 - num)), groups=out_channels,
                                  norm_cfg=None, act_cfg=None)
        self.h_conv2 = ConvModule(out_channels, out_channels, (1, 4 * (2 ** (4 - num) - 1) + 3), 1,
                                  (0, 2 * (2 ** (4 - num) - 1) + 1), groups=out_channels,
                                  norm_cfg=None, act_cfg=None)
        self.v_conv1 = ConvModule(out_channels, out_channels, (4 * (2 ** (4 - num) - 1) + 3 + 2 ** (5 - num), 1), 1,
                                 (2 * (2 ** (4 - num) - 1) + 1 + 2 ** (4 - num), 0), groups=out_channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv2 = ConvModule(out_channels, out_channels, (4 * (2 ** (4 - num) - 1) + 3, 1), 1,
                                 (2 * (2 ** (4 - num) - 1) + 1, 0), groups=out_channels,
                                 norm_cfg=None, act_cfg=None)

        self.conv12 = nn.Conv2d(3 * out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv11(x)
        x = self.deconv(x)
        x = nonlinearity(self.norm(x))

        x1 = self.h_conv2(self.h_conv1(x))
        x2 = self.v_conv2(self.v_conv1(x))

        x = torch.cat((x1, x2, x), 1)

        x = self.conv12(x)
        x = nonlinearity(self.norm(x))

        return x



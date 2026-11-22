import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from .dct import dct, idct
import numpy as np

class FrequencyAttention(nn.Module):
    """频域注意力模块"""
    def __init__(self, in_c):
        super().__init__()
        self.dct_layer = nn.Linear(256, 256)  # 提取低频特征
        self.attn = nn.Sequential(
            nn.Conv2d(in_c*2, in_c//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_c//8, in_c, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 空域分支
        spatial = x
        
        # 频域分支
        batch, c, h, w = x.shape
        dct_x = dct(x.view(batch, c, -1).permute(0,2,1))
        dct_x = self.dct_layer(dct_x).permute(0,2,1).view(batch, -1, h, w)
        
        # 注意力融合
        attn = self.attn(torch.cat([spatial, dct_x], dim=1))
        return spatial * attn + dct_x * (1 - attn)

class WatermarkExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 多尺度特征提取
        self.encoder = nn.Sequential(
            ResConvBlock(3, 64, dilation=1),
            ResConvBlock(64, 128, dilation=2),
            ResConvBlock(128, 256, dilation=3),
        )
        
        # 可变形卷积层
        self.deform_conv = DeformConv2d(256, 256, kernel_size=3, padding=1)
        self.offset_conv = nn.Conv2d(
            256, 
            2 * 3 * 3,  # 每个位置生成2*K*K的offset（K=3）
            kernel_size=3,
            padding=1
        )
        
        # 频域注意力
        self.freq_attn = FrequencyAttention(256)
        
        # 结构解码器
        self.decoder = nn.Sequential(
            DenseBlock(256, 512, growth_rate=32),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 64)
        )

    def forward(self, x):
        x = self.encoder(x)          # (4,256,64,64)
        offset = self.offset_conv(x)
        x = self.deform_conv(x,offset)      # 几何失真补偿
        # x = self.freq_attn(x)        # 频域特征增强
        # print(x.shape)  # (4,256,64,64)
        return torch.sigmoid(self.decoder(x))  # (4,64)

class ResConvBlock(nn.Module):
    """残差卷积块"""
    def __init__(self, in_c, out_c, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        
    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))

class DenseBlock(nn.Module):
    """密集连接块"""
    def __init__(self, in_c, out_c, growth_rate=32):
        super().__init__()
        layers = []
        current_c = in_c
        while current_c < out_c:
            layers.append(nn.Sequential(
                nn.Conv2d(current_c, growth_rate, 3, padding=1),
                nn.BatchNorm2d(growth_rate),
                nn.ReLU()
            ))
            current_c += growth_rate
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        features = [x]
        for layer in self.net:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features[-3:], dim=1)  # 最后三层特征拼接
    


class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, msg_dim=1, num_heads=4):
        super().__init__()
        # 主卷积路径
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        
        # 残差调整模块
        self.msg_adjust = nn.Sequential(
            nn.Conv2d(msg_dim, out_ch, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=out_ch,
            num_heads=num_heads,
            batch_first=True
        ) if out_ch % num_heads == 0 else None

    def forward(self, x, msg):
        # 主卷积路径
        residual = x
        x = self.conv(x)
        
        # 残差连接MSG
        adjusted_msg = self.msg_adjust(msg.unsqueeze(2).unsqueeze(3))
        adjusted_msg = F.interpolate(adjusted_msg, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + adjusted_msg
        
        # # 多头自注意力
        # if self.attention:
        #     n, c, h, w = x.shape
        #     x_attn = x.view(n, c, -1).permute(0, 2, 1)  # [N, H*W, C]
        #     attn_output, _ = self.attention(x_attn, x_attn, x_attn)
        #     x = x + attn_output.permute(0, 2, 1).view(n, c, h, w)
            
        return x

class WatermarkEmbedder(nn.Module):
    def __init__(self, msg_dim=1):
        super().__init__()
        # 网络层级配置
        channels = [3, 64, 128, 256, 512, 512, 256, 128, 64]
        
        # 构建残差块序列
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(in_ch, out_ch, msg_dim)
            for in_ch, out_ch in zip(channels[:-1], channels[1:])
        ])
        
        # 最终输出层
        self.final_conv = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, msg):
        x0= x
        # 逐层处理
        for block in self.blocks:
            x = block(x, msg)
        return x0+self.final_conv(x)
    


def GroupNorm32(channels):
    return nn.GroupNorm(32, channels)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type="batch"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            normalization(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            normalization(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="batch"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_type=norm_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, norm_type="batch"):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_type=norm_type)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_type=norm_type)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # print(x1.shape, x2.shape)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=3, bilinear=False, norm_type="batch"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, norm_type=norm_type))
        self.down1 = (Down(64, 128, norm_type=norm_type))
        self.down2 = (Down(128, 256, norm_type=norm_type))
        self.down3 = (Down(256, 512, norm_type=norm_type))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, norm_type=norm_type))
        self.up1 = (Up(1024, 512 // factor, bilinear, norm_type=norm_type))
        self.up2 = (Up(512, 256 // factor, bilinear, norm_type=norm_type))
        self.up3 = (Up(256, 128 // factor, bilinear, norm_type=norm_type))
        self.up4 = (Up(128, 64, bilinear, norm_type=norm_type))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
       
        x3 = self.down2(x2)
       
        x4 = self.down3(x3)
       
        x5 = self.down4(x4)
       
        # id=id.unsqueeze(2).unsqueeze(3)
        # x5=x5+id.repeat(1,2,16,16)
        x = self.up1(x5, x4)
       
        # print(x.shape,id.shape)
        # x=x+id.repeat(1,1,32,32)
        # print(x.shape)
        x = self.up2(x, x3)
      
        # x=x+id.repeat(1,1,32,32)
        # print(x.shape)
        x = self.up3(x, x2)
      
        # print(x.shape)
        x = self.up4(x, x1)
     
        # print(x.shape)
        x = self.outc(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

from .layers import *

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        # self.up=nn.Sequential(
        #     ConvNorm(ch_in, ch_out*4, kernel_size=3, stride=1, norm='BN'),
        #         nn.PixelShuffle(2))

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t), Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.Conv1x1 = nn.Conv2d(F_int, 1, kernel_size=1, bias=False)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        x2 = self.W(x)
        q = self.Conv1x1(x2)
        q = self.norm(q)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * (psi + q)

class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, message_dim: int = 100):
        super(AttU_Net, self).__init__()
        self.name = "AttentionUnet"
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # Message embedding layers for each level (parameterized by message_dim)
        self.msg_embed1 = nn.Linear(message_dim, 64 * 16 * 16)  # For level 1 (64x64)
        self.msg_embed2 = nn.Linear(message_dim, 128 * 8 * 8)   # For level 2 (32x32)
        self.msg_embed3 = nn.Linear(message_dim, 256 * 4 * 4)   # For level 3 (16x16)
        self.msg_embed4 = nn.Linear(message_dim, 512 * 2 * 2)   # For level 4 (8x8)
        self.msg_embed5 = nn.Linear(message_dim, 1024 * 1 * 1)  # For level 5 (4x4)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        # self.final_conv=nn.Sequential(
        #     single_conv(ch_in=64, ch_out=64),
        #     single_conv(ch_in=64, ch_out=64),
        #     single_conv(ch_in=64, ch_out=64),
        #     nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        # )

    def forward(self, x, message=None, sigmoid=True):
        batch_size = x.size(0)
        
        # encoding path
        x1 = self.Conv1(x)
        if message is not None:
            msg1 = self.msg_embed1(message).view(batch_size, 64, 16, 16)
            msg1 = F.interpolate(msg1, size=x1.shape[2:], mode='bilinear', align_corners=False)
            x1 = x1 +  msg1  # Deep embedding with scaling factor

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        if message is not None:
            msg2 = self.msg_embed2(message).view(batch_size, 128, 8, 8)
            msg2 = F.interpolate(msg2, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x2 = x2 +  msg2

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        if message is not None:
            msg3 = self.msg_embed3(message).view(batch_size, 256, 4, 4)
            msg3 = F.interpolate(msg3, size=x3.shape[2:], mode='bilinear', align_corners=False)
            x3 = x3 +  msg3

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        if message is not None:
            msg4 = self.msg_embed4(message).view(batch_size, 512, 2, 2)
            msg4 = F.interpolate(msg4, size=x4.shape[2:], mode='bilinear', align_corners=False)
            x4 = x4 +  msg4

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        if message is not None:
            msg5 = self.msg_embed5(message).view(batch_size, 1024, 1, 1)
            msg5 = F.interpolate(msg5, size=x5.shape[2:], mode='bilinear', align_corners=False)
            x5 = x5 +  msg5

        # decoding + concat path
        d5 = self.Up5(x5)
        # print(x5.shape, d5.shape)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        # d1 = self.final_conv(d2)
        # if sigmoid:
        #     d1 = nn.Sigmoid()(d1)

        return d1


class EmbedUNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=3, bilinear=False, norm_type="batch"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, norm_type=norm_type))
        self.down1 = (Down(64, 128, norm_type=norm_type))
        self.down2 = (Down(132, 256, norm_type=norm_type))
        self.down3 = (Down(256+4, 512, norm_type=norm_type))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512+4, 1024 // factor, norm_type=norm_type))
        self.up1 = (Up(1024, 512 // factor, bilinear, norm_type=norm_type))
        self.up2 = (Up(512, 256 // factor, bilinear, norm_type=norm_type))
        self.up3 = (Up(256, 128 // factor, bilinear, norm_type=norm_type))
        self.up4 = (Up(128, 64, bilinear, norm_type=norm_type))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x, scale_msg):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # print(x2.shape, scale_msg[0].shape)
        x21 = torch.cat([x2, scale_msg[0]], dim=1)  # Concatenate the scale message
        x3 = self.down2(x21)
        # print(x3.shape, scale_msg[1].shape)
        x31 = torch.cat([x3, scale_msg[1]], dim=1)  # Concatenate the scale message
        x4 = self.down3(x31)
        x41 = torch.cat([x4, scale_msg[2]], dim=1)  # Concatenate the scale message
        x5 = self.down4(x41)
        # print(x5.shape, scale_msg[3].shape)
        x51 = torch.cat([x5, scale_msg[3]], dim=1)  # Concatenate the scale message
        # id=id.unsqueeze(2).unsqueeze(3)
        # x5=x5+id.repeat(1,2,16,16)
        # print(x5.shape, scale_msg[3].shape)
        x = self.up1(x5, x4)
        # x = torch.cat([x, scale_msg[3]], dim=1)  # Concatenate the scale message
        # print(x.shape,id.shape)
        # x=x+id.repeat(1,1,32,32)
        # print(x.shape)
        x = self.up2(x, x3)
        # x = torch.cat([x, scale_msg[2]], dim=1)  # Concatenate the scale message
        # x=x+id.repeat(1,1,32,32)
        # print(x.shape)
        x = self.up3(x, x2)
        # x = torch.cat([x, scale_msg[1]], dim=1)  # Concatenate the scale message
        # print(x.shape)
        x = self.up4(x, x1)
        # x = torch.cat([x, scale_msg[0]], dim=1)  # Concatenate the scale message
        # print(x.shape)
        x = self.outc(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None):
        super(Interpolate, self).__init__()
        self.function = F.interpolate
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x):
        return self.function(x, size=self.size, scale_factor=self.scale_factor)


class ConvTBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=2):
        super(ConvTBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=stride, padding=0),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class ExpandNet(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(ExpandNet, self).__init__()

        layers = [ConvTBNRelu(in_channels, out_channels)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = ConvTBNRelu(out_channels, out_channels)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EncoderUnetNew(nn.Module):
    def __init__(self, image_size=64, message_length=16, in_channels=4, channels=64, norm_type="batch", final_skip=False):
        super(EncoderUnetNew, self).__init__()

        message_convT_blocks = int(np.log2(image_size // int(np.sqrt(message_length))))
        interpolate_scale = 1
        max_blocks_num = 6
        if message_convT_blocks > max_blocks_num:
            interpolate_scale = 2 ** (message_convT_blocks - max_blocks_num)
            message_convT_blocks = max_blocks_num

        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError

        self.message_pre_layer = nn.Sequential(
            nn.Conv2d(1, channels, 3, 1, padding=1),
            normalization(channels),
            nn.ReLU(inplace=True),
            ExpandNet(channels, channels, blocks=message_convT_blocks),
        )
        if interpolate_scale != 1:
            self.message_pre_layer.add_module(
                name="interpolate_layer",
                module=Interpolate(scale_factor=interpolate_scale)
            )

        self.final_skip = final_skip

        if final_skip:
            self.unet = UNet(n_channels=in_channels + channels, n_classes=channels, norm_type=norm_type)
            self.final_layer = nn.Conv2d(channels + in_channels, in_channels, kernel_size=1)
        else:
            self.unet = UNet(n_channels=in_channels + channels, n_classes=in_channels, norm_type=norm_type)

    def forward(self, image, message, id):
        # Message Processor
        size = int(np.sqrt(message.shape[1]))
        message_image = message.view(-1, 1, size, size)
        message_pre = self.message_pre_layer(message_image)
        # concatenate
        # print(image.shape,message_pre.shape)
        concat = torch.cat([image, message_pre], dim=1)
        output = self.unet(concat,id)

        if self.final_skip:
            # final skip connection
            concat_final = torch.cat([output, image], dim=1)
            # last Conv part of Encoder
            output = self.final_layer(concat_final)
        output=torch.clip(output,0,1)
        return output



class EncoderUnet(nn.Module):
    def __init__(self, image_size=64, message_length=100, in_channels=16, channels=64, norm_type="batch", final_skip=False):
        super(EncoderUnet, self).__init__()
        # print(image_size, message_length)
        message_convT_blocks = int(np.log2(image_size // int(np.sqrt(message_length))))-1
        interpolate_scale = 1
        max_blocks_num = 6
        if message_convT_blocks > max_blocks_num:
            interpolate_scale = 2 ** (message_convT_blocks - max_blocks_num)
            message_convT_blocks = max_blocks_num
        # print("message_convT_blocks", message_convT_blocks)
        # print("interpolate_scale", interpolate_scale)

        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError
        if image_size == 256:
            blocksize=4
        elif image_size == 128:
            blocksize=3
        self.message_pre_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1,    # 输入通道数
                out_channels=3,   # 输出通道数
                kernel_size=4,    # 卷积核尺寸
                stride=2,         # 步长
                padding=3         # 填充
            ),
            nn.Conv2d(3, channels, 3, 1, padding=1),
            normalization(channels),
            nn.ReLU(inplace=True),
            ExpandNet(channels, channels, blocks=blocksize),
        )
        # Project arbitrary-length message to a fixed 10x10 grid for the conv path
        self.project_to_grid = nn.Sequential(
            nn.Linear(message_length, 10 * 10),
            nn.ReLU(inplace=True)
        )
        if interpolate_scale != 1:
            self.message_pre_layer.add_module(
                name="interpolate_layer",
                module=Interpolate(scale_factor=interpolate_scale)
            )

        self.final_skip = final_skip

        if final_skip:
            # print('inchannels + channels', in_channels + channels)
            self.unet = AttU_Net(in_channels + channels, channels, message_dim=message_length)
            self.final_layer = nn.Conv2d(channels + in_channels, in_channels, kernel_size=1)
        else:
            # print('inchannels + channels', in_channels + channels)
            self.unet = AttU_Net(in_channels + channels, in_channels, message_dim=message_length)

    def forward(self, image, message):
        # Message Processor: project variable-length message to 10x10 map for conv branch
        message_image = self.project_to_grid(message).view(-1, 1, 10, 10)
        message_pre = self.message_pre_layer(message_image)
        # concatenate
        # print(image.shape,message_pre.shape)
        concat = torch.cat([image, message_pre], dim=1)
        output = self.unet(concat, message)

        if self.final_skip:
            # final skip connection
            concat_final = torch.cat([output, image], dim=1)
            # last Conv part of Encoder
            output = self.final_layer(concat_final)
        return output  



class EncoderHideUnet(nn.Module):
    def __init__(self, image_size=64, message_length=16, in_channels=4, channels=64, norm_type="batch", final_skip=False):
        super(EncoderHideUnet, self).__init__()

        self.final_skip = final_skip

        if final_skip:
            self.unet = AttU_Net(in_channels + in_channels,in_channels + in_channels)
            self.final_layer = nn.Conv2d(in_channels + in_channels+ in_channels, in_channels, kernel_size=1)
        else:
            self.unet = AttU_Net(in_channels + channels, in_channels)

    def forward(self, image, message):
        # Message Processor
        concat = torch.cat([image, message], dim=1)
        output = self.unet(concat)

        if self.final_skip:
            # final skip connection
            concat_final = torch.cat([output, image], dim=1)
            # last Conv part of Encoder
            output = self.final_layer(concat_final)
        return  output


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, channel_num, dilation=1, group=1, norm_type="batch"):
        super(ResidualBlock, self).__init__()

        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(in_channel, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)
        self.norm1 = normalization(channel_num)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)
        self.norm2 = normalization(channel_num)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class DecoderResnet(nn.Module):
    def __init__(self, image_size=64, message_length=16, in_channels=3, norm_type="batch"):
        super(DecoderResnet, self).__init__()
        self.target_message_length = message_length
        stride_blocks = int(np.log2(image_size // int(np.sqrt(message_length))))-1
        interpolate_scale = 1
        max_blocks_num = 6
        if stride_blocks > max_blocks_num:
            interpolate_scale = 2 ** (max_blocks_num - stride_blocks)
            stride_blocks = max_blocks_num

        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError
        # stride_blocks=1
        self.first_layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1, bias=False),
            normalization(256),
            nn.ReLU(inplace=True),
        )

        self.down_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(256 * (2 ** i), 256 * (2 ** (i+1)), 3, 2, 1, bias=False),
            normalization(256 * (2 ** (i+1))),
            nn.ReLU(inplace=True),
        ) for i in range(stride_blocks)])

        self.down_layers.append(nn.Sequential(
            nn.Conv2d(256 * (2 ** stride_blocks), 256, 3, 1, 1, bias=False),
            normalization(256),
            nn.ReLU(inplace=True)
        ))
        if interpolate_scale != 1:
            self.down_layers.append(Interpolate(scale_factor=interpolate_scale))

        self.keep_layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,    # 输入通道数
                out_channels=256,    # 输出通道数（可调整）
                kernel_size=3,     # 卷积核尺寸
                stride=1,          # 步长
                padding=0          # 填充
            ),
            ResidualBlock(256, 256, dilation=2, norm_type=norm_type),
            ResidualBlock(256, 256, dilation=2, norm_type=norm_type),
            ResidualBlock(256, 256, dilation=2, norm_type=norm_type),
            ResidualBlock(256, 256, dilation=2, norm_type=norm_type),
            ResidualBlock(256, 256, dilation=4, norm_type=norm_type),
            ResidualBlock(256, 256, dilation=4, norm_type=norm_type),
            ResidualBlock(256, 256, dilation=4, norm_type=norm_type),
            ResidualBlock(256, 256, dilation=4, norm_type=norm_type),
            ResidualBlock(256, 256, dilation=1, norm_type=norm_type),
        )

        self.final_layers = nn.Sequential(
            nn.Conv2d(256, 64, 3, 1, 1, bias=False),
            normalization(64) if norm_type != "group" else nn.GroupNorm(1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1, bias=False),
        )
        # Adapt to a fixed grid then map to exact message length
        side = int(np.ceil(np.sqrt(message_length)))
        self.adapt_pool = nn.AdaptiveAvgPool2d((side, side))
        self.fc_out = nn.Linear(side * side, message_length)

    def forward(self, x):
        y = self.first_layers(x)
        for down_layer in self.down_layers:
            y = down_layer(y)
        y = self.keep_layers(y)
        # print("y shape before final layers:", y.shape)
        y = self.final_layers(y)
        # Pool to fixed grid, flatten and map to exact message length
        y = self.adapt_pool(y)
        y = y.view(y.shape[0], -1)
        y = self.fc_out(y)
        # y = nn.Sigmoid()(y)
        return y

class DecoderHideResnet(nn.Module):
    def __init__(self, image_size=64, message_length=64, in_channels=3, norm_type="batch"):
        super(DecoderHideResnet, self).__init__()
        stride_blocks = int(np.log2(image_size // int(np.sqrt(message_length))))-1
        interpolate_scale = 1
        max_blocks_num = 6
        if stride_blocks > max_blocks_num:
            interpolate_scale = 2 ** (max_blocks_num - stride_blocks)
            stride_blocks = max_blocks_num

        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError

        self.first_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, 1, 1, bias=False),
            normalization(128),
            nn.ReLU(inplace=True),
        )

        # self.down_layers = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(128 * (2 ** i), 128 * (2 ** (i+1)), 3, 2, 1, bias=False),
        #     normalization(128 * (2 ** (i+1))),
        #     nn.ReLU(inplace=True),
        # ) for i in range(stride_blocks)])

        # self.down_layers.append(nn.Sequential(
        #     nn.Conv2d(128 * (2 ** stride_blocks), 128, 3, 1, 1, bias=False),
        #     normalization(128),
        #     nn.ReLU(inplace=True)
        # ))
        # if interpolate_scale != 1:
        #     self.down_layers.append(Interpolate(scale_factor=interpolate_scale))

        self.keep_layers = nn.Sequential(
            ResidualBlock(128, 128, dilation=2, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=2, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=2, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=2, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=4, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=4, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=4, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=4, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=1, norm_type=norm_type),
        )

        self.final_layers = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            normalization(64) if norm_type != "group" else nn.GroupNorm(1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        y = self.first_layers(x)
        # for down_layer in self.down_layers:
        #     y = down_layer(y)
        y = self.keep_layers(y)
        y = self.final_layers(y)
        return y

class DecoderLinear(nn.Module):
    def __init__(self, image_size=64, message_length=16, in_channels=3, norm_type="batch"):
        super(DecoderLinear, self).__init__()
        self.fc1=nn.Linear(512,1024)
        self.bn1=nn.BatchNorm1d(1024)
        self.fc_layers1 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc_layers2 = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc4=nn.Linear(512,message_length)

    def forward(self, x):
        x=nn.ReLU()(self.bn1(self.fc1(x)))
        x=self.fc_layers1(x)
        x=self.fc_layers2(x)
        x=nn.ReLU()(self.fc4(x))
        return x

class IDWatermarkModel(nn.Module):
    def __init__(
        self,
        wm_enc_config,
        wm_dec_config,
        device="cuda",
        weight_dtype=torch.float32
    ):
        super(IDWatermarkModel, self).__init__()

        self.idencoder = IdentityMarkEncoder(**wm_enc_config)
        self.iddecoder = DecoderLinear(**wm_dec_config)
        self.encoder = EncoderUnetNew(**wm_enc_config)
        # self.idencoder = IdentityMarkEncoder(**wm_enc_config)
        self.decoder = DecoderResnet(**wm_dec_config)
        self.wm_enc_config = wm_enc_config
        self.wm_dec_config = wm_dec_config
        self.weight_dtype = weight_dtype
        self.device = device

class WatermarkModel(nn.Module):
    def __init__(
        self,
        noiser,
        wm_enc_config,
        wm_dec_config,
        device="cuda",
        weight_dtype=torch.float16
    ):
        super(WatermarkModel, self).__init__()

        self.encoder = EncoderUnet(**wm_enc_config)
        # self.idencoder = IdentityMarkEncoder(**wm_enc_config)
        self.decoder = DecoderResnet(**wm_dec_config)
        # self.decoder2 = DecoderResnet(**wm_dec_config2)
        # self.decoder = DecoderLinear(**wm_dec_config)
        self.noiser = noiser
        self.wm_enc_config = wm_enc_config
        self.wm_dec_config = wm_dec_config
        self.weight_dtype = weight_dtype
        self.device = device

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        # noised_and_cover = self.noiser([encoded_image, image])
        # noised_image = noised_and_cover[0]
        # noised_image=F.interpolate(noised_image,(256,256))
        # print(noised_image.shape)
        decoded_message = self.decoder(encoded_image)
        return encoded_image, decoded_message


class RecWatermarkModel(nn.Module):
    def __init__(
        self,
        noiser,
        wm_enc_config,
        wm_dec_config,
        device="cuda",
        weight_dtype=torch.float16
    ):
        super(RecWatermarkModel, self).__init__()

        self.encoder = AttU_Net(32, 16)
        # self.idencoder = IdentityMarkEncoder(**wm_enc_config)
        self.decoder = AttU_Net(16, 16)
        # self.decoder = DecoderLinear(**wm_dec_config)
        self.noiser=noiser
        self.wm_enc_config = wm_enc_config
        self.wm_dec_config = wm_dec_config
        self.weight_dtype = weight_dtype
        self.device = device

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        decoded_message = self.decoder(encoded_image)
        return encoded_image, decoded_message

class RecWatermarkModel(nn.Module):
    def __init__(
        self,
        noiser,
        wm_enc_config,
        wm_dec_config,
        device="cuda",
        weight_dtype=torch.float16
    ):
        super(RecWatermarkModel, self).__init__()

        self.encoder = AttU_Net(32, 16)
        # self.idencoder = IdentityMarkEncoder(**wm_enc_config)
        self.decoder = AttU_Net(16, 16)
        # self.decoder = DecoderLinear(**wm_dec_config)
        self.noiser=noiser
        self.wm_enc_config = wm_enc_config
        self.wm_dec_config = wm_dec_config
        self.weight_dtype = weight_dtype
        self.device = device

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        # noised_and_cover = self.noiser([encoded_image, image])
        # noised_image = noised_and_cover[0]
        # noised_image=F.interpolate(noised_image,(256,256))
        # print(noised_image.shape)
        decoded_message = self.decoder(encoded_image)
        return encoded_image, decoded_message
    
class WatermarkHideModel(nn.Module):
    def __init__(
        self,
        noiser,
        wm_enc_config,
        wm_dec_config,
        device="cuda",
        weight_dtype=torch.float16
    ):
        super(WatermarkHideModel, self).__init__()

        self.encoder = EncoderHideUnet(**wm_enc_config)
        # self.idencoder = IdentityMarkEncoder(**wm_enc_config)
        self.decoder = DecoderHideResnet(**wm_dec_config)
        # self.decoder = DecoderLinear(**wm_dec_config)
        self.noiser=noiser
        self.wm_enc_config = wm_enc_config
        self.wm_dec_config = wm_dec_config
        self.weight_dtype = weight_dtype
        self.device = device

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        # noised_and_cover = self.noiser([encoded_image, image])
        # noised_image = noised_and_cover[0]
        # noised_image=F.interpolate(noised_image,(256,256))
        # print(noised_image.shape)
        decoded_message = self.decoder(encoded_image)
        return encoded_image, decoded_message

class Discriminator(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator, self).__init__()

        kw = 4
        padw = 1
        self.down1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kw, stride=2, padding=padw),
            norm_layer(128), nn.LeakyReLU(0.2, True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=kw, stride=2, padding=padw),
            norm_layer(256), nn.LeakyReLU(0.2, True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=kw, stride=2, padding=padw),
            norm_layer(512), nn.LeakyReLU(0.2, True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kw, stride=1, padding=padw),
            norm_layer(512),
            nn.LeakyReLU(0.2, True)
        )

        if use_sigmoid:
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)
            )

    def forward(self, input):
        out = []
        x = self.down1(input)
        out.append(x)
        x = self.down2(x)
        out.append(x)
        x = self.down3(x)
        out.append(x)
        x = self.down4(x)
        out.append(x)
        x = self.conv1(x)
        out.append(x)
        x = self.conv2(x)
        out.append(x)
        
        return out

class IdentityMarkEncoder(nn.Module):
    def __init__(self,  image_size=64, message_length=16, in_channels=4, channels=64, norm_type="batch", final_skip=False):
        super(IdentityMarkEncoder, self).__init__()
        self.fc1=nn.Linear(512,1024)
        self.bn1=nn.BatchNorm1d(1024)
        self.fc2=nn.Linear(2048,1024)
        self.bn2=nn.BatchNorm1d(1024)
        self.fc3=nn.Linear(1024,512)
        self.bn3=nn.BatchNorm1d(512)
        self.fc11=nn.Linear(message_length,1024)
        self.fc4=nn.Linear(512,512)

    def forward(self, x1, x2):
        x=torch.concatenate((self.bn1(self.fc1(x1)),self.fc11(x2)),dim=1)
        # x=nn.ReLU()(self.bn1(self.fc1(x)))
        x=nn.ReLU()(self.bn2(self.fc2(x)))
        x=nn.ReLU()(self.bn3(self.fc3(x)))
        x=self.fc4(x)
        x=nn.Tanh()(x)
        x=x1+0.1*x
        return x

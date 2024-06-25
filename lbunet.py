import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    

class Down(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.conv(self.bn(x))

class Down2(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, groups=in_channels)
    
    def forward(self, x):
        return self.conv(self.bn(x))

class ConvLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim, padding_mode='reflect') # depthwise conv
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, 4 * dim, kernel_size=1, padding=0, stride=1)
        self.act1 = nn.GELU()
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(4 * dim, dim, kernel_size=1, padding=0, stride=1)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.conv3(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x
    
    
class Boundary_Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        boundary = torch.sigmoid(self.conv(x))
        x = x + x * boundary
        return x, boundary

class Image_Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        gt_pre = self.conv(x)
        x = x + x * torch.sigmoid(gt_pre)
        return x, gt_pre

class Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        boundary = torch.sigmoid(self.conv1(x))
        gt_pre = self.conv2(x)
        return (x + x * boundary + x * torch.sigmoid(gt_pre)), gt_pre, boundary


class Group_shuffle_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        c_dim = dim_in // 4

        self.share_space1 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )
        self.share_space2 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )
        self.share_space3 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )
        self.share_space4 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        x1 = x1 * self.conv1(F.interpolate(self.share_space1, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        x2 = x2 * self.conv2(F.interpolate(self.share_space2, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        x3 = x3 * self.conv3(F.interpolate(self.share_space3, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        x4 = x4 * self.conv4(F.interpolate(self.share_space4, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        x = torch.cat([x2,x4,x1,x3], dim=1)
        x = self.norm2(x)
        x = self.ldw(x)
        return x
    
class Merge(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

    def forward(self, x1, x2, gt_pre, w):
        x = x1 + x2 + torch.sigmoid(gt_pre) * x2 * w
        return x

class Merge2(nn.Module):
    def __init__(self, dim_in):
        super().__init__()      

    def forward(self, x1, x2, gt_pre, boundary_pre, w1, w2):
        x = x1 + x2 + torch.sigmoid(gt_pre) * x2 * w1 + boundary_pre * x2 * w2
        return x


class LBUNet(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64]):
        super().__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
            ConvLayer(c_list[2]),
        )
        self.encoder4 = nn.Sequential(
            Group_shuffle_block(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            Group_shuffle_block(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            Group_shuffle_block(c_list[4], c_list[5]),
        )


        self.Down1 = Down(c_list[0])
        self.Down2 = Down(c_list[1])
        self.Down3 = Down(c_list[2])

        self.merge1 = Merge2(c_list[0])
        self.merge2 = Merge2(c_list[1])
        self.merge3 = Merge2(c_list[2])
        self.merge4 = Merge(c_list[3])
        self.merge5 = Merge(c_list[4])
        
        self.decoder1 = nn.Sequential(
            Group_shuffle_block(c_list[5], c_list[4]),
        ) 
        self.decoder2 = nn.Sequential(
            Group_shuffle_block(c_list[4], c_list[3]),
        ) 
        self.decoder3 = nn.Sequential(
            Group_shuffle_block(c_list[3], c_list[2]),
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  

        self.pred1 = Image_Prediction_Generator(c_list[4])
        self.pred2 = Image_Prediction_Generator(c_list[3])
        self.gate1 = Prediction_Generator(c_list[2])
        self.gate2 = Prediction_Generator(c_list[1])
        self.gate3 = Prediction_Generator(c_list[0])

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Sequential(
            nn.Conv2d(c_list[0], num_classes, kernel_size=1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        out = self.encoder1(x)
        out = F.gelu(self.Down1(self.ebn1(out)))
        t1 = out # b, 8, 128, 128

        out = self.encoder2(out)
        out = F.gelu(self.Down2(self.ebn2(out)))
        t2 = out # b, 16, 64, 64

        out = self.encoder3(out)
        out = F.gelu(self.Down3(self.ebn3(out)))
        t3 = out # b, 24, 32, 32
        
        out = self.encoder4(out)
        out = F.gelu(F.max_pool2d(self.ebn4(out), 2))
        t4 = out # b, 32, 16, 16
        
        out = self.encoder5(out)
        out = F.gelu(F.max_pool2d(self.ebn5(out), 2))
        t5 = out # b, 48, 8, 8
        
        out = self.encoder6(out)
        out = F.gelu(out) # b, 64, 8, 8
        
        out = self.decoder1(out)
        out = F.gelu(self.dbn1(out)) # b, 48, 8, 8

        out, gt_pre5 = self.pred1(out)
        out = self.merge5(out, t5, gt_pre5, 0.1) # b, 48, 8, 8
        gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode ='bilinear', align_corners=True)

        
        out = self.decoder2(out)
        out = F.gelu(F.interpolate(self.dbn2(out),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, 32, 16, 16
        out, gt_pre4 = self.pred2(out)
        out = self.merge4(out, t4, gt_pre4, 0.2) # b, 32, 16, 16
        gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
        
        out = self.decoder3(out)
        out = F.gelu(F.interpolate(self.dbn3(out),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, 24, 32, 32
        out, gt_pre3, weight1 = self.gate1(out)
        out = self.merge3(out, t3, gt_pre3, weight1, 0.3, 0.1) # b, 24, 32, 32
        weight1 = F.interpolate(weight1, scale_factor=8, mode ='bilinear', align_corners=True)
        gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
        
        out = self.decoder4(out)
        out = F.gelu(F.interpolate(self.dbn4(out),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, 16, 64, 64
        out, gt_pre2, weight2 = self.gate2(out)
        out = self.merge2(out, t2, gt_pre2, weight2, 0.4, 0.2) # b, 16, 64, 64 
        weight2 = F.interpolate(weight2, scale_factor=4, mode ='bilinear', align_corners=True)
        gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
        
        out = self.decoder5(out)
        out = F.gelu(F.interpolate(self.dbn5(out),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, 8, 128, 128
        out, gt_pre1, weight3 = self.gate3(out)
        out = self.merge1(out, t1, gt_pre1, weight3, 0.5, 0.3) # b, 3, 128, 128
        weight3 = F.interpolate(weight3, scale_factor=2, mode ='bilinear', align_corners=True)
        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)
        
        out = self.final(out)
        out = F.interpolate(out,scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W

        gt_pre1 = torch.sigmoid(gt_pre1)
        gt_pre2 = torch.sigmoid(gt_pre2)
        gt_pre3 = torch.sigmoid(gt_pre3)
        gt_pre4 = torch.sigmoid(gt_pre4)
        gt_pre5 = torch.sigmoid(gt_pre5)

        return (gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1), (weight1, weight2, weight3), torch.sigmoid(out)
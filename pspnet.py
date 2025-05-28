import os,sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
###########
import torch
from torch import nn
import torch.nn.functional as F
from model.semseg.basenet import BaseNet

# 金字塔模块,将从前面卷积结构提取的特征分别进行不同的池化操作,得到不同感受野以及全局语境信息(或者叫做不同层级的信息)
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]                                         #list,初始值tensor:x
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)                          # channel=in_dim+len(bins)*reduction_dim



class PSPNet(BaseNet):
    def __init__(self, backbone, nclass):
        super(PSPNet, self).__init__(backbone,nclass)
        bins = (1, 2, 3, 6)
        dropout = 0.1
        zoom_factor =1
        use_ppm = True

        assert 2048 % len(bins) == 0
        assert nclass > 1
        assert zoom_factor in [1, 2, 4, 8]

        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, nclass, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, nclass, kernel_size=1)
            )

    def forward(self, x, y=None):
        x_size = x.size()

        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        _,_,c3,c4 = self.backbone.base_forward(x)

        if self.use_ppm:
            x = self.ppm(c4)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:                                                                                               # 训练时采用辅助损失函数：为了加快网络训练，用于梯度回归的损失为：loss=main_loss+aux_weight*aux_loss
            aux = self.aux(c3)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            x = F.interpolate(x,x_size[2::],mode='bilinear',align_corners=True)                                         # resnet50出来的特征图尺寸是原图的1/8,所以还需要做一次上采样恢复到原图
            aux = F.interpolate(aux,x_size[2::],mode='bilinear',align_corners=True )
            return x,aux

        else:
            x = F.interpolate(x, x_size[2::], mode='bilinear',align_corners=True)   # resnet50出来的特征图尺寸是原图的1/8,所以还需要做一次上采样恢复到原图
            return x
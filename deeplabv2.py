import torch
from model.semseg.basenet import BaseNet
from torch import nn
import torch.nn.functional as F
from torchsummaryX import summary

class DeepLabV2(BaseNet):
    def __init__(self, backbone, nclass):
        super(DeepLabV2, self).__init__(backbone,nclass)
        #ASPP-L模块
        self.classifier = nn.ModuleList()
        for dilation in [6, 12, 18, 24]:
            self.classifier.append(
                nn.Conv2d(2048, nclass, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=True))
        for m in self.classifier:
            m.weight.data.normal_(0, 0.01)

    def base_forward(self, x):
        h, w = x.shape[-2:]
        x = self.backbone.base_forward(x)[-1]
        out = self.classifier[0](x)
        for i in range(len(self.classifier) - 1):
            out += self.classifier[i+1](x)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        return out

if __name__ == '__main__':
    input  = torch.randn(4, 3, 224, 224)
    Net    = DeepLabV2(backbone='resnet50',nclass=3)
    output = Net(input)
    print(output.shape)
    # 计算模型的参数量
    # summary(Net, torch.zeros(1, 3, 224, 224))
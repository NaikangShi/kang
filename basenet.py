from model.backbone.resnet import resnet50, resnet101,resnext50_32x4d
from torch import nn
import torch.nn.functional as F
import torch
from model.block.utils import Conv2dBnRelu
from torchinfo import summary
import torchvision

#resnet-fcn
class BaseNet(nn.Module):
    def __init__(self, backbone,nclass):
        super(BaseNet, self).__init__()
        backbone_zoo = {'resnet50': resnet50, 'resnet101': resnet101,'resnext50_32x4d':resnext50_32x4d}
        # backbone_zoo = {'resnet50': torchvision.models.resnet50}
        self.backbone = backbone_zoo[backbone](pretrained=True)

        self.feadim = self.backbone.channels[-1]
        # self.feadim = 2048
        # encoder=self.backbone
        # self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # 1/4
        # self.pool = encoder.maxpool
        # self.conv2_x = encoder.layer1  # 1/4
        # self.conv3_x = encoder.layer2  # 1/8
        # self.conv4_x = encoder.layer3  # 1/16
        # self.conv5_x = encoder.layer4  # 1/32


        self.fcnhead=nn.Sequential(
            Conv2dBnRelu(self.feadim, self.feadim//4, kernel_size=3, padding=1, bias=False),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(self.feadim//4,nclass, kernel_size=1)
        )

    def base_forward(self, x):
        h, w = x.shape[-2:]

        # x = self.conv1(x)
        # x1 = self.pool(x)
        # x2 = self.conv2_x(x1)
        # x3 = self.conv3_x(x2)
        # x4 = self.conv4_x(x3)
        # x = self.conv5_x(x4)

        x = self.backbone.base_forward(x)[-1] #resnet_layer4输出特征图 2048*28*28
        x = self.fcnhead(x)                   #nclass*28*28
        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=True)
        return x

    def forward(self, x):
        return self.base_forward(x)

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # input  = torch.randn(1,3, 224, 224)
    Net    = BaseNet(backbone='resnet50',nclass=3)
    # output = Net(input)
    # print(output.shape)
    # 计算模型的参数量
    # summary(Net,(1,3,224,224),)

    from torchstat import stat
    stat(Net, (3, 224, 224))
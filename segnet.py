import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.block.utils import Conv2dBnRelu

model_urls="https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"

#encoder vgg16
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.encode1 = nn.Sequential(
            Conv2dBnRelu(in_channels, 64, kernel_size=3, padding=1, bias=True),
            Conv2dBnRelu(64, 64, kernel_size=3, padding=1, bias=True)
        )
        self.encode2 = nn.Sequential(
            Conv2dBnRelu(64, 128, kernel_size=3, padding=1, bias=True),
            Conv2dBnRelu(128, 128, kernel_size=3, padding=1, bias=True)
        )
        self.encode3 = nn.Sequential(
            Conv2dBnRelu(128, 256, kernel_size=3, padding=1, bias=True),
            Conv2dBnRelu(256, 256, kernel_size=3, padding=1, bias=True),
            Conv2dBnRelu(256, 256, kernel_size=3, padding=1, bias=True)
        )
        self.encode4 = nn.Sequential(
            Conv2dBnRelu(256, 512, kernel_size=3, padding=1, bias=True),
            Conv2dBnRelu(512, 512, kernel_size=3, padding=1, bias=True),
            Conv2dBnRelu(512, 512, kernel_size=3, padding=1, bias=True)
        )
        self.encode5 = nn.Sequential(
            Conv2dBnRelu(512, 512, kernel_size=3, padding=1, bias=True),
            Conv2dBnRelu(512, 512, kernel_size=3, padding=1, bias=True),
            Conv2dBnRelu(512, 512, kernel_size=3, padding=1, bias=True)
        )
        # Initialize the encoder parameters with vgg16
        # self.init_vgg16_params()

    def init_vgg16_params(self):
        # Initializes the encoder layer with vgg16 parameters
        vgg16 = models.vgg16(pretrained=True)
        vgg_layers = []
        for layer in list(vgg16.features.children()):
            if isinstance(layer, nn.Conv2d):
                vgg_layers.append(layer)

        segnet_layers = []
        for encoder in self.children():
            for conv_layer in encoder.children():
                for inner_layer in conv_layer.children():
                    for layer in list(inner_layer.children()):
                        if isinstance(layer, nn.Conv2d):
                            segnet_layers.append(layer)
        assert len(vgg_layers) == len(segnet_layers)

        for first, second in zip(vgg_layers, segnet_layers):
            if isinstance(first, nn.Conv2d) and isinstance(second, nn.Conv2d):
                assert first.weight.size() == second.weight.size()
                assert first.bias.size() == second.bias.size()
                second.weight.data = first.weight.data
                second.bias.data = first.bias.data

    def forward(self, x):
        idx = []
        # (3, 224, 224) -> (64, 112, 112)
        x = self.encode1(x)
        x, id1 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id1)
        # (64, 112, 112) -> (128, 56, 56)
        x = self.encode2(x)
        x, id2 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id2)
        # (128, 56,56) -> (256, 28, 28)
        x = self.encode3(x)
        x, id3 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id3)
        # (256, 28, 28) -> (512,14, 14)
        x = self.encode4(x)
        x, id4 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id4)
        # (512, 14, 14) -> (512, 7, 7)
        x = self.encode5(x)
        x, id5 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id5)
        return x, idx

#倒vgg16
class Deocder(nn.Module):
    def __init__(self,outchannel):
        super(Deocder, self).__init__()
        self.decode1 = nn.Sequential(
            Conv2dBnRelu(512, 512, kernel_size=3, padding=1, bias=False),
            Conv2dBnRelu(512, 512, kernel_size=3, padding=1, bias=False),
            Conv2dBnRelu(512, 512, kernel_size=3, padding=1, bias=False)
        )
        self.decode2 = nn.Sequential(
            Conv2dBnRelu(512, 512, kernel_size=3, padding=1, bias=False),
            Conv2dBnRelu(512, 512, kernel_size=3, padding=1, bias=False),
            Conv2dBnRelu(512, 256, kernel_size=3, padding=1, bias=False)
        )
        self.decode3 = nn.Sequential(
            Conv2dBnRelu(256, 256, kernel_size=3, padding=1, bias=False),
            Conv2dBnRelu(256, 256, kernel_size=3, padding=1, bias=False),
            Conv2dBnRelu(256, 128, kernel_size=3, padding=1, bias=False)
        )

        self.decode4 = nn.Sequential(
            Conv2dBnRelu(128, 128, kernel_size=3, padding=1, bias=False),
            Conv2dBnRelu(128, 64, kernel_size=3, padding=1, bias=False)
        )
        self.decode5 = nn.Sequential(
            Conv2dBnRelu(64, 64, kernel_size=3, padding=1, bias=False),
            Conv2dBnRelu(64, outchannel, kernel_size=3, padding=1)
        )
        # self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, idx):
        """
        :param x: 经过卷积操作后的特征图
        :param idx: decode中每次最大池化时最大值的位置索引
        """
        # (512, 7, 7) -> (512, 14, 14) -> (512, 14, 14)
        x = F.max_unpool2d(x, idx[4], kernel_size=2, stride=2)
        x = self.decode1(x)
        # (512, 14, 14) -> (512, 28, 28) -> (256, 28, 28)
        x = F.max_unpool2d(x, idx[3], kernel_size=2, stride=2)
        x = self.decode2(x)
        # (256, 28, 28) -> (256, 56, 56) -> (128, 56, 56)
        x = F.max_unpool2d(x, idx[2], kernel_size=2, stride=2)
        x = self.decode3(x)
        # (128, 56, 56) -> (128, 122, 122) -> (64, 122, 122)
        x = F.max_unpool2d(x, idx[1], kernel_size=2, stride=2)
        x = self.decode4(x)
        # (64, 122, 122) -> (64, 244, 244) -> (64, 244, 244)
        x = F.max_unpool2d(x, idx[0], kernel_size=2, stride=2)
        x = self.decode5(x)
        return x


class SegNet(nn.Module):
    def __init__(self, nclass):
        super(SegNet, self).__init__()
        self.encoder = Encoder(in_channels=3)
        self.decoder = Deocder(outchannel=nclass)
        # Conv layer for classification
        # self.classifier = nn.Conv2d(64,nclass, 3, 1, 1)

    def forward(self, x):
        x, idx = self.encoder(x)
        x = self.decoder(x, idx)
        return x #self.classifier(x)

if __name__ == '__main__':
    inputs = torch.randn(1, 3, 512, 512)
    model = SegNet(nclass=21)
    output = model(inputs)
    print(output.shape)
from typing import List, Any
import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F


class InitialBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concatenate branches
        out = torch.cat((main, ext), 1)

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_activation(out)


class RegularBottleneck(nn.Module):
    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)




class DownsamplingBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=return_indices)

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out), max_indices

class UpsamplingBottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())

        # Transposed convolution
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        # 1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(
            main, max_indices, output_size=output_size)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)

class ENet(nn.Module):
    def __init__(self, n_classes=3, encoder_relu=False, decoder_relu=True):
        super().__init__()

        # Stage 1 - Encoder
        self.down1=DownsamplingBottleneck(3, 64,internal_ratio=1,return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        encoder1:List[nn.Module]=[RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)]
        encoder1.append( RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu))
        encoder1.append(RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu))
        encoder1.append( RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu))
        self.encoder1=nn.Sequential(*encoder1)
        # Stage2 -  Encoder
        self.down2=DownsamplingBottleneck(64, 128,return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        encoder2:List[nn.Module]=[RegularBottleneck(128, padding=1, dropout_prob=0.01, relu=encoder_relu)]
        encoder2.append( RegularBottleneck(128, padding=1, dropout_prob=0.01, relu=encoder_relu))
        encoder2.append(RegularBottleneck(128, padding=1, dropout_prob=0.01, relu=encoder_relu))
        encoder2.append( RegularBottleneck(128, padding=1, dropout_prob=0.01, relu=encoder_relu))
        self.encoder2=nn.Sequential(*encoder2)
        # Stage 3 - Encoder
        self.down3=DownsamplingBottleneck(128,256, return_indices=True,dropout_prob=0.1,relu=encoder_relu)
        encoder3: List[nn.Module] = [RegularBottleneck(256, padding=1, dropout_prob=0.1, relu=encoder_relu)]
        encoder3.append( RegularBottleneck(256, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu))
        encoder3.append(RegularBottleneck(256,kernel_size=5,padding=2,asymmetric=True,dropout_prob=0.1,relu=encoder_relu))
        encoder3.append(RegularBottleneck(256, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu))
        encoder3.append( RegularBottleneck(256, padding=1, dropout_prob=0.1, relu=encoder_relu))
        encoder3.append( RegularBottleneck( 256, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu))
        encoder3.append(RegularBottleneck(256,kernel_size=5,asymmetric=True,padding=2,dropout_prob=0.1,relu=encoder_relu))
        encoder3.append(RegularBottleneck( 256, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu))
        self.encoder3=nn.Sequential(*encoder3)
        # Stage 4 - Encoder
        encoder4: List[nn.Module]=[RegularBottleneck(256, padding=1, dropout_prob=0.1, relu=encoder_relu)]
        encoder4.append(RegularBottleneck(256, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu))
        encoder4.append(RegularBottleneck(256,kernel_size=5,padding=2,asymmetric=True,dropout_prob=0.1,relu=encoder_relu))
        encoder4.append(RegularBottleneck(256, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu))
        encoder4.append(RegularBottleneck(256, padding=1, dropout_prob=0.1, relu=encoder_relu))
        encoder4.append(RegularBottleneck(256, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu))
        encoder4.append(RegularBottleneck(256,kernel_size=5,asymmetric=True,padding=2,dropout_prob=0.1,relu=encoder_relu))
        encoder4.append(RegularBottleneck(256, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu))
        self.encoder4 = nn.Sequential(*encoder4)

        #cls decoder
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim,256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            #nn.Conv2d(128, n_classes, kernel_size=1)
        )
        self.cls2 = nn.Sequential(
            nn.Conv2d(n_classes,n_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_classes),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

        # if self.training:
        #     self.aux1 = nn.Sequential(
        #         nn.Conv2d(n_classes, n_classes, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(n_classes),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout2d(p=0.1),
        #         )
        #
        #     self.aux2 = nn.Sequential(
        #         nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(32),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout2d(p=0.1),
        #         )
        #
        #     self.aux3 = nn.Sequential(
        #         nn.Conv2d(128,64, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout2d(p=0.1),
        #
        #         )
        #

        # upsample
        self.upsample3 = UpsamplingBottleneck(256, 128, dropout_prob=0.1, relu=decoder_relu)
        self.regular3_1 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular3_2 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.upsample2 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular2_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular2_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.upsample1 = UpsamplingBottleneck(64, n_classes, dropout_prob=0.1, relu=decoder_relu)

        # self.upsample0_3 = UpsamplingBottleneck(256, 128, dropout_prob=0.1, relu=decoder_relu)
        # self.upsample0_2 = UpsamplingBottleneck(192, 64, dropout_prob=0.1, relu=decoder_relu)
        # self.upsample0_1 = UpsamplingBottleneck(96,n_classes, dropout_prob=0.1, relu=decoder_relu)
    def forward(self, x):

        # Stage 1 - Encoder
        stage1_input_size = x.size()
        x, max_indices1_0 = self.down1(x)
        x1 = self.encoder1(x)

        # Stage 2 - Encoder
        x=x1
        stage2_input_size = x.size()
        x, max_indices2_0 = self.down2(x)
        x2 = self.encoder2(x)


        # Stage 3&4 - Encoder
        x=x2
        stage3_input_size = x.size()
        x, max_indices3_0 = self.down3(x)
        x3 = self.encoder3(x)
        x = self.encoder4(x3)

        #PPM
        x = self.ppm(x)
        x = self.cls(x)

        # if self.training:
        #     x = self.upsample3(x, max_indices3_0, output_size=stage3_input_size)
        #     x = self.upsample2(x, max_indices2_0, output_size=stage2_input_size)
        #     x = self.upsample1(x, max_indices1_0, output_size=stage1_input_size)
        #
        #
        #     ex3 = self.upsample0_3(x3, max_indices3_0, output_size=stage3_input_size)
        #     aux3 = self.aux3(ex3)
        #     x2=torch.cat([aux3,x2],1)
        #     ex2 = self.upsample0_2(x2, max_indices2_0, output_size=stage2_input_size)
        #     aux2 = self.aux2(ex2)
        #     x1=torch.cat([aux2,x1],1)
        #     ex1 = self.upsample0_1(x1, max_indices1_0, output_size=stage1_input_size)
        #     aux= self.aux1(ex1)
        #     return x,aux
        # else:
        # upsample
        x = self.upsample3(x, max_indices3_0, output_size=stage3_input_size)
        x = self.regular3_1(x)
        x = self.regular3_2(x)
        x = self.upsample2(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular2_1(x)
        x = self.regular2_2(x)
        x = self.upsample1(x, max_indices1_0, output_size=stage1_input_size)
        x = self.cls2(x)
        return x



if __name__ == '__main__':
    # 随机生成输入数据
    inputs = torch.randn((16, 3, 224, 224))
    # 定义网络
    model = ENet(n_classes=3)
    # 前向传播
    out = model(inputs)
    # 打印输出大小
    print('-----' * 5)
    print(out.size())
    print('-----' * 5)

    model= model.cuda()
    batch = torch.cuda.FloatTensor(16, 3, 224, 224)

    print('Output shape: {}'.format(list(out.shape)))
    total_paramters = sum(p.numel() for p in model.parameters())
    print('Total paramters: {}'.format(total_paramters))



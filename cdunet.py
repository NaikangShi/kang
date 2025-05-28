import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from thop import profile
from torchsummary import summary
# from etc.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number


# conv
class Conv2dBn(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.conv(x)


class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class RRB(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(RRB, self).__init__()
        baseWidth = 26
        scale = 6
        width = int(math.floor(outplanes * (baseWidth / 128.0)))
        convs = []

        self.conv1 = Conv2dBnRelu(inplanes, width * scale, kernel_size=1, stride=stride)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1

        convs.append(Conv2dBnRelu(width, width, kernel_size=3, stride=1, padding=1))
        convs.append(Conv2dBnRelu(width, width, kernel_size=3, stride=1, padding=1))
        convs.append(Conv2dBnRelu(width, width, kernel_size=3, stride=1, padding=1))
        convs.append(nn.Sequential(
            Conv2dBnRelu(width, width, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            Conv2dBnRelu(width, width, kernel_size=(5, 1), stride=1, padding=(2, 0))
        ))
        convs.append(nn.Sequential(
            Conv2dBnRelu(width, width, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            Conv2dBnRelu(width, width, kernel_size=(1, 5), stride=1, padding=(0, 2))
        ))
        convs.append(nn.Sequential(
            Conv2dBnRelu(width, width, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            Conv2dBnRelu(width, width, kernel_size=(7, 1), stride=1, padding=(3, 0))
        ))
        convs.append(nn.Sequential(
            Conv2dBnRelu(width, width, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            Conv2dBnRelu(width, width, kernel_size=(1, 7), stride=1, padding=(0, 3))
        ))
        self.convs = nn.ModuleList(convs)

        self.conv3 = nn.Conv2d(width * scale, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)

        self.scale = scale
        self.width = width

    def forward(self, x):
        resident = x
        inx = self.conv1(x)

        spx = list(torch.split(inx, self.width, 1))
        y = []

        for i in range(self.scale):
            y.append(torch.zeros_like(spx[i]))

        for i in range(self.nums):  # nums=3,i:0-->2
            if i == 0:
                spx[i] = self.convs[i](spx[i])
                y[i] = spx[i]
            elif i == 1 or 2:
                y[i] = self.convs[i](spx[i]+y[i-1])
            else:
                x1 = self.convs[i](spx[i]+y[i-1])
                x2 = self.convs[i+1](spx[i]+y[i-1])
                y[i] = x1 + x2

        out = torch.cat([y[i] for i in range(self.nums)], 1)
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)  # 经过1x1的conv比中间层提升4倍的通道数,最终输出为[1,128x4,56,56]-->[1,512,56,56]
        out = self.bn3(out)
        return out + resident

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, reduction=4):
        super(MHSA, self).__init__()
        self.heads = heads
        self.reduction = reduction
        self.query = Conv2dBn(n_dims, n_dims//reduction, kernel_size=1)
        self.key = Conv2dBn(n_dims, n_dims//reduction, kernel_size=1)
        self.value = Conv2dBn(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims//reduction // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims//reduction // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        resident = x
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C//self.reduction // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C//self.reduction // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.reduction // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out+resident


def rescale(x, scale_factor):
    '''
    scale x to the same size as y
    '''
    x_scaled = torch.nn.functional.interpolate(
            x, scale_factor=scale_factor, mode='bilinear',
            align_corners=True, recompute_scale_factor=True)
    return x_scaled


# High Frequency Enhancer Block
class HFE(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(HFE, self).__init__()
        self.conv1x1 = Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1)
        # self.pool = nn.AdaptiveAvgPool2d(14)
        self.gamma = nn.Parameter(torch.ones(1),requires_grad=True)

    def forward(self, x, y):
        y = self.conv1x1(y)
        # y = self.pool(y)
        # y = nn.Upsample(size=x.size()[2:],mode='bilinear',align_corners=True)(y)
        y = rescale(y, 2)
        hf = x-y

        return x + self.gamma * hf


# Channel Spatial Attention Block
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        # torch.max(x,1)返回值[0]和索引[1]，这里只要值不要索引，最终返回[n,2,h,w]


class SCAB(nn.Module):
    def __init__(self, out_ch):
        super(SCAB, self).__init__()

        # Spatial Attention Block
        self.compress = ChannelPool()  # [n,c,h,w] --> [n,2,h,w]
        self.spatial = nn.Sequential(
            Conv2dBn(2, 2, (1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(inplace=True),
            Conv2dBn(2, 2, (7, 1), stride=1, padding=(3, 0))
        )
        # Channel Attention Block
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_ch, out_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch, 2*out_ch, bias=False)
        )
        self.up = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

    # x: low level features
    # y: high level features
    def forward(self, x, y):
        n, c, h, w = x.size()
        x_down = rescale(x, 0.5)
        attn = x_down + y

        attn_c = attn
        attn_s = attn

        # CAB
        attn_c = self.avg_pool(attn_c).view(n, c)
        attn_c = self.fc(attn_c)
        attn_c = torch.unsqueeze(attn_c, 1).view(-1, 2, c, 1, 1)
        attn_c = F.softmax(attn_c, 1)
        x_c = x * attn_c[:, 0, :, :, :].squeeze(1)
        y_c = y * attn_c[:, 1, :, :, :].squeeze(1)

        # SAB
        attn_s = self.compress(attn_s)
        attn_s = self.spatial(attn_s)
        attn_s = F.softmax(attn_s, 1)
        attn_s1, attn_s2 = torch.split(attn_s, 1, dim=1)
        x_s = x * rescale(attn_s1, 2)
        y_s = y * attn_s2

        x = x_c + x_s
        y = y_c + y_s

        y = self.up(y)
        out = x + y

        return out


class FuseLayer(nn.Module):
    def __init__(self, in_ch, out_ch, transformer=False, resolution=None):
        super(FuseLayer, self).__init__()

        if transformer:
            self.conv1 = nn.Sequential(
                RRB(out_ch, out_ch),
                RRB(out_ch, out_ch))
            self.conv2 = nn.Sequential(
                MHSA(in_ch, width=int(resolution[0]), height=int(resolution[0])),
                MHSA(in_ch, width=int(resolution[0]), height=int(resolution[0]))
            )
        else:
            self.conv1 = nn.Sequential(
                RRB(out_ch, out_ch),
                RRB(out_ch, out_ch))
            self.conv2 = nn.Sequential(
                RRB(in_ch, in_ch),
                RRB(in_ch, in_ch))
        self.down = Conv2dBn(out_ch, in_ch, kernel_size=3, stride=2, padding=1)
        self.up = Conv2dBn(in_ch, out_ch, kernel_size=1, stride=1)
        # y_ch match x_ch
        self.match_ch = Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1)

    # x: low level feature
    # y: high level feature
    def forward(self, x, y):
        h, w = x.size()[2], x.size()[3]

        # fuse
        x = self.conv1(x)+x
        y = self.conv2(y)+y

        y_b = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(self.up(y))
        x_b = self.down(x)

        x = F.relu(x+y_b)
        y = F.relu(y+x_b)

        y = self.match_ch(y)

        return x, y


class ZDS(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, n_class=3, dropout_rate=0.1):
        '''
        :param backbone: Bcakbone network
        '''
        super(ZDS, self).__init__()
        if backbone.lower() == 'resnet34':
            encoder = torchvision.models.resnet34(pretrained)
            bottom_ch = 512
        elif backbone.lower() == 'resnet50':
            encoder = torchvision.models.resnet50(pretrained)
            bottom_ch = 2048
        elif backbone.lower() == 'resnet101':
            encoder = torchvision.models.resnet101(pretrained)
            bottom_ch = 2048
        elif backbone.lower() == 'resnet152':
            encoder = torchvision.models.resnet152(pretrained)
            bottom_ch = 2048
        else:
            raise NotImplementedError('{} Backbone not implement'.format(backbone))

        self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # 1/4
        self.pool = encoder.maxpool
        self.conv2_x = encoder.layer1  # 1/4
        self.conv3_x = encoder.layer2  # 1/8
        self.conv4_x = encoder.layer3  # 1/16
        self.conv5_x = encoder.layer4  # 1/32

        self.hfe4 = HFE(bottom_ch // 2, bottom_ch // 4)
        self.hfe3 = HFE(bottom_ch // 4, bottom_ch // 8)
        self.hfe2 = HFE(bottom_ch // 8, bottom_ch // 16)
        self.hfe1 = HFE(bottom_ch // 16, bottom_ch // 32)

        self.conv1x1_5 = Conv2dBnRelu(bottom_ch, bottom_ch // 2, kernel_size=1, stride=1)        # 2048-->1024
        self.conv1x1_4 = Conv2dBnRelu(bottom_ch // 2, bottom_ch // 4, kernel_size=1, stride=1)   # 1024-->512
        self.conv1x1_3 = Conv2dBnRelu(bottom_ch // 4, bottom_ch // 8, kernel_size=1, stride=1)   # 512-->256
        self.conv1x1_2 = Conv2dBnRelu(bottom_ch // 8, bottom_ch // 16, kernel_size=1, stride=1)  # 256-->128

        self.fuse4 = FuseLayer(bottom_ch // 2, bottom_ch // 4, transformer=True, resolution=(7, 7))
        self.fuse3 = FuseLayer(bottom_ch // 4, bottom_ch // 8, transformer=True, resolution=(14, 14))
        self.fuse2 = FuseLayer(bottom_ch // 8, bottom_ch // 16, transformer=True, resolution=(28, 28))
        self.fuse1 = FuseLayer(bottom_ch // 16, bottom_ch // 32)

        self.scab4 = SCAB(bottom_ch // 4)
        self.scab3 = SCAB(bottom_ch // 8)
        self.scab2 = SCAB(bottom_ch // 16)
        self.scab1 = SCAB(bottom_ch // 32)

        self.final = nn.Sequential(
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(bottom_ch // 32, n_class, kernel_size=1, stride=1)
        )

        auxs = []
        aux_in_ch = [bottom_ch // 16, bottom_ch // 8, bottom_ch // 4]
        for i in range(len(aux_in_ch)):
            auxs.append(
                nn.Sequential(
                    Conv2dBnRelu(aux_in_ch[i],  bottom_ch // 32, kernel_size=3, stride=1, padding=1),  # 64
                    nn.Dropout2d(p=dropout_rate),
                    nn.Conv2d(bottom_ch // 32, n_class, kernel_size=1, stride=1)))
        self.auxs = nn.ModuleList(auxs)

    def forward(self, x, vs_feature=False):
        h, w = x.size(2), x.size(3)

        # stage 1-5
        x = self.conv1(x)
        x1 = self.pool(x)
        x2 = self.conv2_x(x1)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)
        # channel decrease
        x5 = self.conv1x1_5(x5)
        x4 = self.conv1x1_4(x4)
        x3 = self.conv1x1_3(x3)
        x2 = self.conv1x1_2(x2)
        # fuse stage
        x4_out, y4_out = self.fuse4(self.hfe4(x4, x5), x5)
        x4_out = self.scab4(x4_out, y4_out)
        x3_out, y3_out = self.fuse3(self.hfe3(x3, x4), x4_out)
        x3_out = self.scab3(x3_out, y3_out)
        x2_out, y2_out = self.fuse2(self.hfe2(x2, x3), x3_out)
        x2_out = self.scab2(x2_out, y2_out)
        x1_out, y1_out = self.fuse1(self.hfe1(x, x2), x2_out)
        x_ = self.scab1(x1_out, y1_out)
        # output
        x = self.final(x_)
        out = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)

        if self.training:
            # aux out
            aux1 = self.auxs[0](x2_out)
            aux1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(aux1)
            aux2 = self.auxs[1](x3_out)
            aux2 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(aux2)
            aux3 = self.auxs[2](x4_out)
            aux3 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(aux3)
            return out, aux1, aux2, aux3
        else:
            # visualize features
            if vs_feature:
                features = []
                features.extend([x_, x1_out])
                return out, features
            else:
                return out


if __name__ == '__main__':
    model = ZDS(backbone='resnet50', pretrained=True, n_class=3)
    # model.load_state_dict(torch.load(r'F:\code\code_practice\multi_frequency_net\summary\ours_resnet50_hfe_rrb_transformer_reduction4_scabs\miou_0.935171_199.pth'))
    # print(model)
    model.train()
    input = torch.randn(1, 3, 224, 224)
    output = model(input)[0]
    print(output.shape)
    # summary(model.cuda(), input_size=(3, 224, 224))


    from torchstat import stat
    stat(model, (3, 224, 224))

    # batch = torch.cuda.FloatTensor(1, 3, 224, 224)
    # model_eval = add_flops_counting_methods(model)
    # model_eval.eval().start_flops_count()
    # out = model_eval(batch)  # ,only_encode=True)
    # print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    # print('Params: ' + get_model_parameters_number(model))
    # print('Output shape: {}'.format(list(out.shape)))
    # total_paramters = sum(p.numel() for p in model.parameters())
    # print('Total paramters: {}'.format(total_paramters))
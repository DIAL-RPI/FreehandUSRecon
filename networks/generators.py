import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import time

__all__ = ['ResNeXt', 'resnet50', 'resnet101']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=16,
                 num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=2,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
            bias=False)
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=(2, 2, 2),
            padding=1,
            bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 64, layers[1], shortcut_type, cardinality, stride=1)
        self.layer3 = self._make_layer(
            block, 32, layers[1], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 128, layers[1], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)

        self.conv3d_8 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.conv3d_9 = nn.Conv3d(in_channels=128, out_channels=16, kernel_size=5, stride=(1, 1, 1), padding=0)

        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        self.fc_layer1 = nn.Linear(36864, 128)
        self.fc_layer2 = nn.Linear(128, 6)

        self.fc1 = nn.Linear(76800, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 6)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):

        diary = False
        # diary = True

        if diary:
            print('network input {}'.format(x.shape))
            x = self.conv1(x)
            print('conv1 {}'.format(x.shape))
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            print('maxpool {}'.format(x.shape))
            x = self.conv2(x)
            print('conv2 {}'.format(x.shape))
            x_0 = self.conv3(x)
            print('conv3 {}'.format(x_0.shape))
            # time.sleep(30)

            x = self.layer1(x)
            print('layer1 {}'.format(x.shape))
            x = self.layer2(x)
            print('layer2 {}'.format(x.shape))
            x_1 = self.layer3(x)
            print('layer3 {}'.format(x_1.shape))

            x_01 = torch.cat((x_0, x_1), 1)
            x = self.bn2(x_01)
            print('10_bn {}'.format(x.shape))
            print('x_0 {}, x_1 {}'.format(x_0.shape, x_1.shape))

            x = self.conv3d_8(x)
            x = self.relu(x)
            print('11_3d {}'.format(x.shape))

            x = self.conv3d_9(x)
            # x = self.relu(x)
            print('12_3d {}'.format(x.shape))

            x = x.view(x.size()[0], -1)
            x = self.relu(x)
            print('13_fl {}'.format(x.shape))

            x = self.fc1(x)
            x = self.relu(x)
            print('15_fc {}'.format(x.shape))

            x = self.fc2(x)
            x = self.relu(x)
            print('17_fc {}'.format(x.shape))

            x = self.fc3(x)
            print('19_fc {}'.format(x.shape))
            time.sleep(30)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.conv2(x)
            x_0 = self.conv3(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x_1 = self.layer3(x)

            x_01 = torch.cat((x_0, x_1), 1)
            x = self.bn2(x_01)

            x = self.conv3d_8(x)
            x = self.relu(x)

            x = self.conv3d_9(x)

            x = x.view(x.size()[0], -1)
            x = self.relu(x)

            x = self.fc1(x)
            x = self.relu(x)

            x = self.fc2(x)
            x = self.relu(x)

            x = self.fc3(x)
            # x = self.conv1(x)
            # x = self.bn1(x)
            # x = self.relu(x)
            # x = self.maxpool(x)
            #
            # x = self.layer1(x)
            # x = self.layer2(x)
            # x = self.layer3(x)
            # x = self.layer4(x)
            # x = self.avgpool(x)
            #
            # x = x.view(x.size(0), -1)
            # x = self.fc_layer1(x)
            # x = self.relu(x)
            # x = self.fc_layer2(x)

        # print('output size {}'.format(x.shape))
        # time.sleep(30)
        # print('we are in resnext101')
        return x


class PrevostNet(nn.Module):

    def __init__(self):
        self.inplanes = 64
        super(PrevostNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(576, 6)

    def forward(self, x):

        show_size = False
        # show_size = True
        if show_size:
            print('input shape {}'.format(x.shape))
            x = self.conv1(x)
            print('conv1 shape {}'.format(x.shape))
            x = self.relu(x)
            print('relu shape {}'.format(x.shape))
            x = self.conv2(x)
            print('conv2 shape {}'.format(x.shape))
            x = self.relu(x)
            print('relu shape {}'.format(x.shape))
            x = self.maxpool(x)
            print('maxpool shape {}'.format(x.shape))

            x = self.conv3(x)
            print('conv3 shape {}'.format(x.shape))
            x = self.relu(x)
            print('relu shape {}'.format(x.shape))
            x = self.conv4(x)
            print('conv4 shape {}'.format(x.shape))
            x = self.relu(x)
            print('relu shape {}'.format(x.shape))
            x = self.maxpool(x)
            print('maxpool shape {}'.format(x.shape))

            x = x.view(x.size(0), -1)
            print('view before fc {}'.format(x.shape))
            x = self.fc(x)
            print('fc {}'.format(x.shape))
            time.sleep(30)

        else:
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = x.view(x.size(0), -1)

            x = self.fc(x)

        return x, x

class My3DNet(nn.Module):

    def __init__(self):
        self.inplanes = 64
        super(My3DNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                               stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3,
                      stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3),
                               stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(128)
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3,
                      stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1, 3, 3),
                               stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3,
                      stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv4 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(1, 3, 3),
                               stride=(1, 2, 2), padding=(0, 1, 1), bias=False)

        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 7, 7), stride=(1, 1, 1), padding=0)

        self.relu = nn.ReLU(inplace=True)


        self.fc = nn.Linear(64, 6)

    def forward(self, x):

        show_size = False
        # show_size = True
        if show_size:
            print('input shape {}'.format(x.shape))

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            print('conv1 shape {}'.format(x.shape))

            at1 = self.layer1(x)
            print('at1 shape {}'.format(at1.shape))
            x = x * at1
            print('x shape {}'.format(x.shape))

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            print('conv2 shape {}'.format(x.shape))

            at2 = self.layer2(x)
            print('at2 shape {}'.format(at2.shape))
            x = x * at2
            print('x shape {}'.format(x.shape))

            x = self.conv3(x)
            x = self.bn1(x)
            x = self.relu(x)
            print('conv3 shape {}'.format(x.shape))

            at2 = self.layer3(x)
            print('at2 shape {}'.format(at2.shape))
            x = x * at2
            print('x shape {}'.format(x.shape))

            x = self.maxpool1(x)
            print('maxpool1 shape {}'.format(x.shape))

            x = self.conv4(x)
            print('conv4 shape {}'.format(x.shape))

            x = self.maxpool2(x)
            print('maxpool2 shape {}'.format(x.shape))

            x = x.view(x.size(0), -1)
            print('view before fc {}'.format(x.shape))
            x = self.fc(x)
            print('fc {}'.format(x.shape))
            time.sleep(30)

        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            at1 = self.layer1(x)
            x = x * at1

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            at2 = self.layer2(x)
            x = x * at2

            x = self.conv3(x)
            x = self.bn1(x)
            x = self.relu(x)
            at2 = self.layer3(x)
            x = x * at2

            x = self.maxpool1(x)
            x = self.conv4(x)
            x = self.maxpool2(x)

            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)

        self.conv3d_1 = nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.conv3d_2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=(1, 2, 2), padding=1)

        self.conv3d_3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=(1, 1, 1), padding=1)

        self.conv3d_4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=(2, 2, 2), padding=1)

        self.conv3d_5 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.conv3d_6 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=(1, 1, 1), padding=1)

        self.conv3d_7 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=(1, 1, 1), padding=1)

        self.conv3d_8 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.conv3d_9 = nn.Conv3d(in_channels=128, out_channels=16, kernel_size=5, stride=(1, 1, 1), padding=0)
        # self.conv3d_10 = nn.Conv3d(in_channels=16, out_channels=8, kernel_size=3, stride=(1, 1, 1), padding=1)

        self.drop3d_1 = nn.Dropout3d(0.25)

        self.drop2d_1 = nn.Dropout2d(0.25)
        self.drop2d_2 = nn.Dropout2d(0.1)

        # self.fc1 = nn.Linear(38400, 128)
        self.fc1 = nn.Linear(76800, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 6)


    def forward(self, x):

        # diary = True
        diary = False

        if diary == True:
            print('give {}'.format(x.shape))

            x = self.conv3d_1(x)
            x = self.relu(x)
            print('1_3d {}'.format(x.shape))

            x = self.conv3d_2(x)
            x = self.relu(x)
            print('2_3d {}'.format(x.shape))

            x = self.bn1(x)
            print('3_bn {}'.format(x.shape))

            x = self.conv3d_3(x)
            x = self.relu(x)
            print('4_3d {}'.format(x.shape))

            x = self.conv3d_4(x)
            x_0 = self.relu(x)
            print('5_3d {}'.format(x_0.shape))

            x = self.conv3d_5(x_0)
            x = self.relu(x)
            print('6_3d {}'.format(x.shape))

            x = self.conv3d_6(x)
            x = self.relu(x)
            print('7_3d {}'.format(x.shape))

            x = self.conv3d_7(x)
            x_1 = self.relu(x)
            print('8_3d {}'.format(x_1.shape))

            x_01 = torch.cat((x_0, x_1), 1)
            print('9_cn {}'.format(x_01.shape))

            x = self.bn2(x_01)
            print('10_bn {}'.format(x.shape))

            x = self.conv3d_8(x)
            x = self.relu(x)
            print('11_3d {}'.format(x.shape))

            x = self.conv3d_9(x)
            # x = self.relu(x)
            print('12_3d {}'.format(x.shape))

            x = x.view(x.size()[0], -1)
            x = self.relu(x)
            x = self.drop3d_1(x)
            print('13_fl {}'.format(x.shape))

            x = self.fc1(x)
            x = self.relu(x)
            # x = self.drop2d_1(x)
            print('15_fc {}'.format(x.shape))

            x = self.fc2(x)
            x = self.relu(x)
            # x = self.drop2d_2(x)
            print('17_fc {}'.format(x.shape))

            x = self.fc3(x)
            print('19_fc {}'.format(x.shape))
            time.sleep(30)
        else:
            x = self.conv3d_1(x)
            x = self.relu(x)

            x = self.conv3d_2(x)
            x = self.relu(x)

            x = self.bn1(x)

            x = self.conv3d_3(x)
            x = self.relu(x)

            x = self.conv3d_4(x)
            x_0 = self.relu(x)

            x = self.conv3d_5(x_0)
            x = self.relu(x)

            x = self.conv3d_6(x)
            x = self.relu(x)

            x = self.conv3d_7(x)
            x_1 = self.relu(x)

            x_01 = torch.cat((x_0, x_1), 1)

            x = self.bn2(x_01)

            x = self.conv3d_8(x)
            x = self.relu(x)

            x = self.conv3d_9(x)

            x = x.view(x.size()[0], -1)
            x = self.relu(x)
            x = self.drop3d_1(x)

            x = self.fc1(x)
            x = self.relu(x)
            x = self.drop2d_1(x)

            x = self.fc2(x)
            x = self.relu(x)
            x = self.drop2d_2(x)

            x = self.fc3(x)
            # time.sleep(30)

        return x
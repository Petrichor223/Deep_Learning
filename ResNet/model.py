import torch.nn as nn
import torch


class BasicBlock(nn.Module):                   #定义18层、34层对应的残差结构
    expansion = 1                              #expansion对应残差结构中主分支采用的卷积核个数的变化

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):    #定义初始函数及残差结构所需要使用的一系列层结构，其中下采样参数downsample对应虚线的残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,         #stride等于1时对应实线残差结构，因为当步长为1时卷积不会改变特征矩阵的高和宽
                               kernel_size=3, stride=stride, padding=1, bias=False)      #output=(input-kernel_size+2*padding)/stride+1=(input-3+2*1)/1+1=input(向下取整)
        self.bn1 = nn.BatchNorm2d(out_channel)                                           #stride等于2时对应虚线残差结构，要将特征矩阵的高和宽缩减为原来的一半
        self.relu = nn.ReLU()                                                            #使用BN时不需要使用偏置
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):                           #定义正向传播过程，输入特征矩阵x
        identity = x                                #将x赋值给identity
        if self.downsample is not None:             #如果没有输入下采样函数，那么对应实线的残差结构，就跳过这里
            identity = self.downsample(x)           #如果输入下采样函数不等于None，就将输入特征矩阵x输入到下采样函数中得到捷径分支的输出

        out = self.conv1(x)                         #主分支的输出
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity                            #将主分支与捷径分支的输出相加
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4                                            #在50层、101层、152层的残差结构中的第三层卷积层的卷积核个数时第一层、第二层卷积核个数的四倍，所以这里为4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,                     #定义初始函数及残差结构所需要使用的一系列层结构
                 groups=1, width_per_group=64):                                                #resnext多传入groups和width_per_group
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups                            #计算resnet和rennext网络第一个卷积层和第二个卷积层采用的卷积核个数

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,                     #对于resnet，不传入groups和width_per_group两个参数，out_channels=width=out_channels
                               kernel_size=1, stride=1, bias=False)  # squeeze channels        #对于resnext，传入这两个参数，width等于两倍的resnet的out_channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)            #步长为2，因此这里步长根据传入的stride调整
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,     #卷积核个数为四倍的前一层卷积核个数
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):                      #定义正向传播过程，原理同18层正向传播过程
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):                              #定义ResNet网络

    def __init__(self,
                 block,                               #block对应的为定义的残差结构BasicBlock、Bottleneck
                 blocks_num,                          #列表，对应的为该层所使用的残差结构的数目，如34层的conv2_x中包含了3个、conv3_x中包含了4个
                 num_classes=1000,                    #训练集分类个数
                 include_top=True,                    #方便以后在Resnet基础上搭建更加复杂的网络
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top                #将include_top传入类变量中
        self.in_channel = 64                          #输入特征矩阵的深度(通过maxpool之后的特征矩阵)

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,        #7x7卷积层
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)            #最大池化
        self.layer1 = self._make_layer(block, 64, blocks_num[0])                   #对应表格Conv2_x的残差结构，通过_make_layer()函数生成
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)        #对应表格Conv3_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)        #对应表格Conv4_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)        #对应表格Conv5_x
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):           #定义_make_layer，这里的channel为残差结构中第一个卷积层所使用卷积核的个数
        downsample = None                                                 #对应18、34层
        if stride != 1 or self.in_channel != channel * block.expansion:   #如果步长不等于1或者输入通道不等于channel * block.expansion，即50层以上的
            downsample = nn.Sequential(                                   #则生成下采样函数
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []                                                       #定义列表
        layers.append(block(self.in_channel,                              #将第一层残差结构添加进去
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):                                     #通过循环将从第二层开始的网络结构组合进去
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)                                     #将列表转换成非关键参数传入到nn.Sequential()函数中

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):                   #
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth     #权重下载地址
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],                                #调用ResNet类，与之前ResNet相同，但多了两个参数
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,                                           #多传入了这两个参数
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):                      #ch为输入特征矩阵的channel，要将ch调整为divisor的整数倍，min_ch最小通道数
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:           #如果min_ch是None就将divisor赋给min_ch
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)   #将输入ch调整到离divisor最近的整数倍
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:           #确保向下取整不会减少超过10%
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):                    #定义Conv+BN+ReLU6的组合层，继承nn.Sequential副类(根据Pytorch官方样例，要使用官方预训练权重)
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):    #初始化函数，在pytorch中dw卷积也是调用的nn.Conv2d类，如果传入groups=1则为普通卷积；如果groups=in_channel则为dw卷积
        padding = (kernel_size - 1) // 2            #卷积过程中需要设置的填充参数
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),    #卷积，groups默认为1
            nn.BatchNorm2d(out_channel),          #BN层
            nn.ReLU6(inplace=True)                #ReLU6
        )


class InvertedResidual(nn.Module):                  #定义倒残差结构
    def __init__(self, in_channel, out_channel, stride, expand_ratio):   #expand_ratio为扩展因子t
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio                       #hidden_channel为第一层卷积核的个数
        self.use_shortcut = stride == 1 and in_channel == out_channel    #判断在正向传播中是否使用捷径分支

        layers = []                                                      #定义层列表
        if expand_ratio != 1:                                            #如果扩展因子不等1，就有第一个1x1卷积层，等于1就没有
            # 1x1 pointwise conv                                         #在第一个bottleneck中t=1，不对输入特征矩阵深度扩充，不需要1x1卷积层
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)                              #通过nn.Sequential()类将layers以未知参数的形式传入，将一系列层结构打包组合

    def forward(self, x):                  #正向传播，输入特征矩阵x
        if self.use_shortcut:              #判断是否使用捷径分支
            return x + self.conv(x)        #捷径分支输出+主分支输出
        else:
            return self.conv(x)            #直接使用主分支输出


class MobileNetV2(nn.Module):                                               #定义MobileNetV2网络结构
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):       #初始化函数：类别个数、超参数α、
        super(MobileNetV2, self).__init__()
        block = InvertedResidual                                            #将倒残差结构传给block
        input_channel = _make_divisible(32 * alpha, round_nearest)          #第一层卷积层所以使用的卷积核个数，_make_divisible()会将输出的通道个数调整为输入的整数倍，可能为了更好的调用硬件设备
        last_channel = _make_divisible(1280 * alpha, round_nearest)         #last_channel代表模型参数中conv2d 1x1卷积核

        inverted_residual_setting = [                                       #根据模型参数创建list列表
            # t, c, n, s                                                    #对应7个block
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []                                                       #定义空列表
        # conv1 layer                                                       #第一个卷积层
        features.append(ConvBNReLU(3, input_channel, stride=2))             #3为输入彩色图，输入特征矩阵卷积核个数，模型参数s=2
        # building inverted residual residual blockes                       #定义一系列block结构
        for t, c, n, s in inverted_residual_setting:                        #遍历inverted_residual_setting参数列表
            output_channel = _make_divisible(c * alpha, round_nearest)      #将输出channel个数进行调整
            for i in range(n):                                              #搭建每个block中的倒残差结构
                stride = s if i == 0 else 1                                 #模型参数中的s对应第一层的步长，判断是否是第一层
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))   #在列表中添加一系列倒残差结构
                input_channel = output_channel                              #将output_channel传入input_channel作为下一层的输入特征矩阵的深度
        # building last several layers                                      #定义模型参数中bottlenck下的1x1卷积层，1为卷积核的大小
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)                            #将以上部分打包成特征提取部分

        # building classifier                                               #定义分类器部分
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                         #平均池化下采样
        self.classifier = nn.Sequential(                                    #将dropout和去全连接层组合
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization                                             #初始化权重流程
        for m in self.modules():                                            #如果子模块是卷积层就会其权重进行初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:                                      #如果存在偏置就将偏置设置为0
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):                             #如果子模块是BN层就将方差设置为1，偏执设置为0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):                                  #如果子模块是全连接层，就将其权重初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):                    #定义正向传播
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

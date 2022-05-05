import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def _make_divisible(ch, divisor=8, min_ch=None):   #将传入的channel个数调整到离它最近的8的整数倍，对硬件更加友好
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNActivation(nn.Sequential):   #卷积+BN+激活函数
    def __init__(self,
                 in_planes: int,         #输入特征矩阵channel
                 out_planes: int,        #输出特征矩阵channel
                 kernel_size: int = 3,   #卷积核大小
                 stride: int = 1,        #步距
                 groups: int = 1,        #g=1使用dw卷积,g=2使用普通卷积
                 norm_layer: Optional[Callable[..., nn.Module]] = None,   #BN结构
                 activation_layer: Optional[Callable[..., nn.Module]] = None):  #BN结构后的激活函数
        padding = (kernel_size - 1) // 2    #根据kernel_size计算padding
        if norm_layer is None:              #如果没有传入norm_layer则使用BN
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:        #如果没有传入activation_layer则使用SiLU激活函数(和swish激活函数一样)
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)
        #传入所需要构建的一系列层结构
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,   #传入卷积层所需要参数
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),             #使用BN结构则不使用bias
                                               norm_layer(out_planes),            #BN结构，传入的参数为上一层的输出channel
                                               activation_layer())                #不传入参数默认使用SiLU激活函数


class SqueezeExcitation(nn.Module):      #SE模块
    def __init__(self,
                 input_c: int,   # block input channel   对应MBConv输入特征矩阵的channel
                 expand_c: int,  # block expand channel  对应第一个1x1卷积层升维后的channel(dw卷积不改变channel)
                 squeeze_factor: int = 4):              #对应第一个全连接层的节点个数，论文中默认等于4
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor           #squeeze_c的计算公式
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)    #构建第一个全连接层
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)    #构建第二个全连接层
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:             #前向传播过程
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))   #对输入进行平均池化
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class InvertedResidualConfig:     #残差结构配置，与mobilenetv3类似
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,          # 3 or 5
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,  # 1 or 6
                 stride: int,          # 1 or 2
                 use_se: bool,         # True
                 drop_rate: float,
                 index: str,           # 1a, 2a, 2b, ...
                 width_coefficient: float):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):   #MBConv模块
    def __init__(self,
                 cnf: InvertedResidualConfig,    #残差结构配置
                 norm_layer: Callable[..., nn.Module]):   #BN结构
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:   #判断dw卷积的步长是否在1和2中
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)   #根据步长判断是否使用shortcut连接

        layers = OrderedDict()         #定义有序字典来搭建MBConv结构
        activation_layer = nn.SiLU  # alias Swish

        # expand               搭建第一个1x1卷积层   注意当n=1时不需要第一个1x1卷积层
        if cnf.expanded_c != cnf.input_c:     #当expanded_c = input_c时n=1，跳过
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,   #调用ConvBNActivation并传入相关参数
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise       搭建DW卷积
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c, #dw卷积输入和输出特征矩阵的channel不发生变化
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        if cnf.use_se:      #判断是否使用SE模块
            layers.update({"se": SqueezeExcitation(cnf.input_c,  #这里注意，输入特征矩阵为输入MBConv模块的特征矩阵的channel
                                                   cnf.expanded_c)})

        # project     搭建最后的1x1卷积层
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})   #在最后1x1卷积层后没有激活函数，因此不做任何处理，传入nn.Identity

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


class EfficientNet(nn.Module):          #实现EfficientNet
    def __init__(self,
                 width_coefficient: float,        #网络宽度的倍率因子
                 depth_coefficient: float,        #网络深度的倍率因子
                 num_classes: int = 1000,         #分类的类别个数
                 dropout_rate: float = 0.2,       #MB模块中的dropout的随机失活比例
                 drop_connect_rate: float = 0.2,  #最后一个全连接层的dropout的随机失活比例
                 block: Optional[Callable[..., nn.Module]] = None,   #MBConv模块
                 norm_layer: Optional[Callable[..., nn.Module]] = None   #BN结构
                 ):
        super(EfficientNet, self).__init__()

        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],   #以B0构建的默认配置表,stage2~stage8
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):   #depth_coefficient * repeats向上取整
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:     #如果block为空就默认等于MBConv
            block = InvertedResidual

        if norm_layer is None:  #如果norm_layer为空就默认为BN结构
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,  #对传入的channel乘倍率因子再调整到离它最近的8的整数倍
                                  width_coefficient=width_coefficient)

        # build inverted_residual_setting
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0   #统计搭建MBblock的次数
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))  #遍历default_cnf列表获取重复次数
        inverted_residual_setting = []   #定义空列表存储MBConv的配置文件
        for stage, args in enumerate(default_cnf):   #遍历default_cnf列表
            cnf = copy.copy(args)    #后面会对数据进行修改，为了不影响原数据
            for i in range(round_repeats(cnf.pop(-1))):   #遍历每个stage中的MBConv模块
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv
        layers.update({"stem_conv": ConvBNActivation(in_planes=3,
                                                     out_planes=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # build top
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)

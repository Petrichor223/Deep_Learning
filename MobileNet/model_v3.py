from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial


def _make_divisible(ch, divisor=8, min_ch=None):
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


class ConvBNActivation(nn.Sequential):         #定义结构:Conv+BN+激活函数
    def __init__(self,
                 in_planes: int,               #输入特征矩阵channel
                 out_planes: int,              #输出特征矩阵channel
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,           #卷积后接的BN层
                 activation_layer: Optional[Callable[..., nn.Module]] = None):    #激活函数
        padding = (kernel_size - 1) // 2
        if norm_layer is None:                  #如果没有传入norm_layer就默认设置为BN
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:            #如果没有传入激活函数寄默认使用ReLU6
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))


class SqueezeExcitation(nn.Module):                                      #定义注意力机制模块，此处注意对比上文讲到的注意力机制
    def __init__(self, input_c: int, squeeze_factor: int = 4):           #squeeze_factor为第一个全连接层的节点个数为输入特征矩阵channel的1/4，所以int=4
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)        #计算第一个全连接层的节点个数：输出特征矩阵channel/4，计算完成后调整到8最近的整数倍
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)                      #全连接层1，输入输出的channel
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)                      #全连接层2

    def forward(self, x: Tensor) -> Tensor:                              #定义正向传播
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))             #对输入的特征矩阵的每一个维度进行池化操作
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x                                                 #利用得到的权重scale与对应的channel维度的输入相乘

class InvertedResidualConfig:                                            #对应MobileNetv3中的每一个bneck结构的参数配置
    def __init__(self,
                 input_c: int,                                           #对应网络结构中的input
                 kernel: int,                                            #对应网络结构中的Operator
                 expanded_c: int,                                        #对应网络结构中的exp size
                 out_c: int,                                             #对应网络结构中的#out
                 use_se: bool,                                           #对应网络结构中的SE
                 activation: str,                                        #对应网络结构中的NL
                 stride: int,                                            #对应网络结构中的s
                 width_multi: float):                                    #对应mobilenetv2中提到的α参数，调节每一个卷积层所使用channel的倍率银子
        self.input_c = self.adjust_channels(input_c, width_multi)        #调节每一个卷积层所使用channel
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):                                             #mobilenetv3的倒残差结构
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:                                           #判断对应某一层的步长是否为1或2(网络结构中步长只有1和2)
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)  #判断是否使用捷径分支(当s=1且input_c=output_c时使用)

        layers: List[nn.Module] = []                                           #定义layers空列表
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU             #判断使用哪个激活函数

        # expand                                                               #倒残差结构中第一个1x1卷积层，用来升维处理
        if cnf.expanded_c != cnf.input_c:                                      #判断exp size是否等于输入特征矩阵channel
            layers.append(ConvBNActivation(cnf.input_c,                        #两者不相等才有1x1卷积层
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwise                                                           #dw卷积
        layers.append(ConvBNActivation(cnf.expanded_c,                        #上一层输出的特征矩阵channel
                                       cnf.expanded_c,                        #dw卷积的输入输出特征矩阵channel一直
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_c,                 #dw卷积对每一个channel都单独使用chennel为1卷积核来进行卷积处理,其group数等于channel
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        if cnf.use_se:                                                        #判断当前层结构是否使用SE模块
            layers.append(SqueezeExcitation(cnf.expanded_c))                  #如果使用传入SE模块中的expanded_c

        # project                                                             #最后一个卷积层，降维
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:             #定义正向传播过程
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result


class MobileNetV3(nn.Module):                                                   #定义MobileNetv3网络结构
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],       #对应bneck结构的参数列表
                 last_channel: int,                                             #对应倒数第二个全连接层
                 num_classes: int = 1000,                                       #类别个数，默认1000
                 block: Optional[Callable[..., nn.Module]] = None,              #定义的mobilenetv3的倒残差结构
                 norm_layer: Optional[Callable[..., nn.Module]] = None):        #BN层
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:            #数据格式检查
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []                                        #构建网络

        # building first layer                                              #第一个卷积层，conv2d
        firstconv_output_c = inverted_residual_setting[0].input_c           #获取第一个bneck结构的input-channel，对应第一个卷积层的输出channel
        layers.append(ConvBNActivation(3,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # building inverted residual blocks
        for cnf in inverted_residual_setting:                              #遍历每一个bneck结构
            layers.append(block(cnf, norm_layer))                          #将每一层的配置文件和norm_layer传给block，然后将创建好的block(即InvertedResidual模块)添加到layers中

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c             #构建bneck之后的一层网络
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)                             #特征提取主干部分
        self.avgpool = nn.AdaptiveAvgPool2d(1)                             #平均池化
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),   #全连接层，分类部分
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0              #α参数
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1       #如果设置为1不会对参数进行调整，设置为2会对参数调整减小网络大小

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)
